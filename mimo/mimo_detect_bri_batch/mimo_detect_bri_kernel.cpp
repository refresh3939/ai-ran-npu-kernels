/**
 * @file mimo_detect_bri_kernel.cpp — massive-MIMO Block-Richardson LMMSE 检测器 (Ascend 310P1)
 *
 * 五段全组批流水 (默认 GB=8 个独立 RE 一组, 4 核):
 *   ① Gram          A = HᴴH + σ²I      —— 瓦片 TᵀT 取对角块; Aim 用反对称 X−Xᵀ 还原
 *   ② MatchFilter   m = Hᴴ·yv          —— 与 Gram 同构, 复用它留在 L0A 的 Hrᵀ/Hiᵀ
 *   ③ BlockInv      M⁻¹ (Neumann×NEU)  —— 块对角预条件, A 侧 (E) 循环不变常驻 L0A
 *   ④ BRI           X += M⁻¹(I−AX)     —— 重构成 X + M⁻¹ − B·X, B = M⁻¹A 循环外算一次
 *   ⑤ Detect        x̂ = X·m, unbias    —— 每 64 RE 转置并连续写回 GM
 *
 * 关键手法: 跨 RE 组批 (指令数 ÷GB) / L0C 只搬有效对角块 / load3dv2 img2col 转置 /
 *           裸 TBuf + 手工事件 (绕开 TQue 簿记) / Hermitian 对称省转置 /
 *           GM/L2→L1 环形多缓冲 / Gram 与后续 cmm 复用同一块 L0C /
 *           NL<16 时打包 P=16/NL_REAL 个 RE 进一个瓦片 (零补零, 指令数不变)
 *
 * 当前验证 (4 核, 23296 RE, NR=64, NL=16, NEU=3, BRI_L=5):
 *     p50 中位约 9.060 ms，EVM 1.76%，no_eff relative median 1.41e-3.
 *   本最终实现固定使用 PACK=0、GROUP_BATCH=8、L1 双缓冲和 5824-RE 调度窗。
 */
#include "kernel_operator.h"
#include "mimo_detect_bri.h"
using namespace AscendC;
namespace {
using namespace airan;
constexpr uint32_t NEU=3;       // 不减少 Neumann 迭代次数
constexpr uint32_t MB_DEPTH=2;  // 已验证的最终 GM/L2 -> L1 ring 深度
constexpr uint32_t M=NR, K=NL, B=BLK, GB=GROUP_BATCH;
// ── 尺寸自洽性 (改 GB 时这些必须全部成立) ──
static_assert(GB==8, "clean 最终实现固定 GROUP_BATCH=8");
static_assert(BATCH % GB == 0,        "BATCH 必须被 GB 整除");
static_assert((GB*K) % 8 == 0,        "Brcb 的 repeatTimes = MG2/8 必须整除");
static_assert(GB*K <= 255,            "WholeReduceSum 的 repeatTimes 上限 255");
static_assert(2*GB*GB <= 255,         "DataCopy 的 blockLen 上限");
static_assert(MB_DEPTH == 2, "clean 最终实现固定 L1 ring depth=2");
static_assert(3*BATCH*K <= GB*GB*K*K, "bAim_ 不足以容纳整批 xhat/no_eff staging");
static_assert(BATCH*K <= GB*K*K, "转置写回暂存只支持 BATCH<=GB*K");
constexpr int32_t E0=0;
}

class MimoBRI {
public:
    __aicore__ inline MimoBRI() {}
    __aicore__ inline void Init(GM_ADDR h_re,GM_ADDR h_im,GM_ADDR y_re,GM_ADDR y_im,
        GM_ADDR no,GM_ADDR mask,GM_ADDR xhat_re,GM_ADDR xhat_im,GM_ADDR no_eff,GM_ADDR yv_re,GM_ADDR yv_im,TPipe*p);
    __aicore__ inline void Run();
private:
    // ── Run() 的分段 (纯等价拆分; TBuf 都是成员, 各自 Get<> 拿视图) ──
    __aicore__ inline void BuildConst();               // 循环外: Id / Ibat  (×1)
    __aicore__ inline void LoadRE(uint32_t re,uint32_t slot); // 预取 H/Y -> L1 ring
    __aicore__ inline void Gram(uint32_t slot);        // ② A = HᴴH + σ²I -> Are/Aim
    __aicore__ inline void BlockInvG();
    __aicore__ inline void cmmGF(const LocalTensor<half>&la0,const LocalTensor<half>&la1,const LocalTensor<half>&la2,
        const LocalTensor<half>&Br,const LocalTensor<half>&Bi,const LocalTensor<half>&Cr,const LocalTensor<half>&Ci);
    __aicore__ inline void BriG();
    __aicore__ inline void DetectG(uint32_t g);
    __aicore__ inline void cmmGR(const LoadData3DParamsV2Pro&pb,const MmadParams&mp,
        const LocalTensor<half>&Br,const LocalTensor<half>&Bi,
        const LocalTensor<half>&Cr,const LocalTensor<half>&Ci);
    __aicore__ inline void MatchFilterG(uint32_t re0,uint32_t slot);
    uint32_t aiv_, core_re0_;
    GlobalTensor<half> gHr_,gHi_,gNo_,gXr_,gXi_,gNe_,gYvr_,gYvi_;
    TPipe* pipe_;
    TBuf<TPosition::A2> bA2_; TBuf<TPosition::B2> bB2_; TBuf<TPosition::CO1> bCO_;
    TBuf<TPosition::VECCALC> bGmr_,bGmi_;                      // MatchFilterG 的输出 m
    // ── BlockInvG: 组版缓冲, 每个都是 GB 个 [16,16] 连续 (= img2col 的平面堆叠) ──
    TBuf<TPosition::VECCALC> bGAr_,bGAi_,bGMbr_,bGMbi_,bGEr_,bGEi_,bGErT_,bGEiT_;
    TBuf<TPosition::VECCALC> bGSr_,bGSi_,bGTr_,bGTi_,bGMvr_,bGMvi_,bGtmp_,bGId_,bGMblk_,bGMpk_,bGdcol_,bGdcolT_;
    TBuf<TPosition::VECCALC> bGDinv_,bGf_;
    TBuf<TPosition::VECCALC> bDvh_,bB0r_,bB0i_,bSigE_,bDiagIdx_,bSkinnyIdx_,bMatchIdx_; // 对角/瘦GEMM RHS+输出索引
    TBuf<TPosition::A1> bL1ar_,bL1ai_,bL1br_,bL1bn_,bL1bi_;
    // A 侧 6 块 L1 瓦片 (M⁻¹ᵀ 三件套 + Bᵀ 三件套). 每个操作数独立 buffer ——
    // L1 内的源偏移未标定过, 共用一块再切偏移不可靠.
    TBuf<TPosition::A1> bL1a0_,bL1a1_,bL1a2_,bL1a3_,bL1a4_,bL1a5_;
    TBuf<TPosition::A2> bGA2_; TBuf<TPosition::B2> bGB2_;
    TBuf<TPosition::A1> bHrL1_,bHiL1_;
    TBuf<TPosition::VECCALC> bAre_,bAim_,bGT_,bNo_,bCr_,bCi_,bId_,bIdS_;

    TBuf<TPosition::A1> bYvL1r_,bYvL1i_;

    event_t evG_,e1G_,emG_,ebG_;   // 四个常用事件 ID: Init 取一次, 全 kernel 复用
};

__aicore__ inline void MimoBRI::Init(GM_ADDR h_re,GM_ADDR h_im,GM_ADDR y_re,GM_ADDR y_im,
    GM_ADDR no,GM_ADDR mask,GM_ADDR xhat_re,GM_ADDR xhat_im,GM_ADDR no_eff,GM_ADDR yv_re,GM_ADDR yv_im,TPipe*p)
{
    pipe_=p; aiv_=GetBlockIdx(); core_re0_=aiv_*U_PER_CORE;   // ★ 打包后是 unit 索引 (P=1 时与 RE 索引相同)
    // GM 输入按 unit 计 (P=1 时 N_UNIT == N_RE); no 在打包模式是逐槽 [N_UNIT,K]
    gHr_.SetGlobalBuffer((__gm__ half*)h_re,(uint64_t)N_UNIT*M*K);
    gHi_.SetGlobalBuffer((__gm__ half*)h_im,(uint64_t)N_UNIT*M*K);
    gNo_.SetGlobalBuffer((__gm__ half*)no,(P>1)?(uint64_t)N_UNIT*K:(uint64_t)N_RE);
    gXr_.SetGlobalBuffer((__gm__ half*)xhat_re,(uint64_t)K*N_RE);
    gXi_.SetGlobalBuffer((__gm__ half*)xhat_im,(uint64_t)K*N_RE);
    gNe_.SetGlobalBuffer((__gm__ half*)no_eff,(uint64_t)K*N_RE);
    gYvr_.SetGlobalBuffer((__gm__ half*)yv_re,(uint64_t)(N_UNIT/GB)*M*K);
    gYvi_.SetGlobalBuffer((__gm__ half*)yv_im,(uint64_t)(N_UNIT/GB)*M*K);
    evG_=static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    e1G_=static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE1));
    emG_=static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE1_M));
    ebG_=static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::M_MTE1));
    pipe_->InitBuffer(bHrL1_,MB_DEPTH*M*GB*K*sizeof(half));
    pipe_->InitBuffer(bHiL1_,MB_DEPTH*M*GB*K*sizeof(half));
    pipe_->InitBuffer(bA2_,2*GB*K*M*sizeof(half)); pipe_->InitBuffer(bB2_,2*GB*K*M*sizeof(half));   // 裸 L0A/L0B: 各 2 个操作数
    // Gram/MatchFilter 与后续 cmm 的 L0C 生命周期不重叠，共享这一块。
    // GB=8 时 3*(128x128)*fp32=192KiB，低于 256KiB；若分成两块会超容量。
    pipe_->InitBuffer(bCO_,3*GB*K*GB*K*sizeof(float));
    // L0C 只取 GB 个 16x16 对角块，FixPipe 直接 FP32->FP16 落到最终 UB。
    pipe_->InitBuffer(bGmr_,GB*K*K*sizeof(half)); pipe_->InitBuffer(bGmi_,GB*K*K*sizeof(half));
    // ── BlockInvG 缓冲 ──
    { constexpr uint32_t GB2=GB*K*K, MG2=GB*K;
      pipe_->InitBuffer(bGAr_,GB2*sizeof(half));  pipe_->InitBuffer(bGAi_,GB2*sizeof(half));
      pipe_->InitBuffer(bGMbr_,GB2*sizeof(half)); pipe_->InitBuffer(bGMbi_,GB2*sizeof(half));
      pipe_->InitBuffer(bGEr_,GB2*sizeof(half));  pipe_->InitBuffer(bGEi_,GB2*sizeof(half));
      pipe_->InitBuffer(bGErT_,GB2*sizeof(half)); pipe_->InitBuffer(bGEiT_,GB2*sizeof(half));
      pipe_->InitBuffer(bGSr_,GB2*sizeof(half));  pipe_->InitBuffer(bGSi_,GB2*sizeof(half));
      pipe_->InitBuffer(bGTr_,GB2*sizeof(half));  pipe_->InitBuffer(bGTi_,GB2*sizeof(half));
      pipe_->InitBuffer(bGMvr_,GB2*sizeof(half)); pipe_->InitBuffer(bGMvi_,GB2*sizeof(half));
      pipe_->InitBuffer(bGtmp_,GB2*sizeof(half)); pipe_->InitBuffer(bGId_,GB2*sizeof(half));
      pipe_->InitBuffer(bGMblk_,GB2*sizeof(half));pipe_->InitBuffer(bGMpk_,GB2*sizeof(half));pipe_->InitBuffer(bGdcol_,GB2*sizeof(half));
      pipe_->InitBuffer(bGdcolT_,GB2*sizeof(half));
      pipe_->InitBuffer(bGDinv_,MG2*sizeof(half));
      pipe_->InitBuffer(bDvh_,MG2*sizeof(half)); pipe_->InitBuffer(bB0r_,(MG2+16)*sizeof(half));
      pipe_->InitBuffer(bB0i_,(MG2+16)*sizeof(half)); pipe_->InitBuffer(bSigE_,8*K*sizeof(half));   // Brcb 半精度: 1 repeat = 8 元素 -> 8 block = 128 half
      pipe_->InitBuffer(bDiagIdx_,MG2*sizeof(uint32_t));
      pipe_->InitBuffer(bSkinnyIdx_,(MG2+K*K)*sizeof(uint32_t));
      pipe_->InitBuffer(bMatchIdx_,MG2*sizeof(uint32_t));
      pipe_->InitBuffer(bGf_,6*MG2*sizeof(float));
      pipe_->InitBuffer(bL1ar_,GB2*sizeof(half)); pipe_->InitBuffer(bL1ai_,GB2*sizeof(half));
      pipe_->InitBuffer(bL1br_,GB2*sizeof(half)); pipe_->InitBuffer(bL1bn_,GB2*sizeof(half)); pipe_->InitBuffer(bL1bi_,GB2*sizeof(half));
      pipe_->InitBuffer(bL1a0_,GB2*sizeof(half)); pipe_->InitBuffer(bL1a1_,GB2*sizeof(half));
      pipe_->InitBuffer(bL1a2_,GB2*sizeof(half)); pipe_->InitBuffer(bL1a3_,GB2*sizeof(half));
      pipe_->InitBuffer(bL1a4_,GB2*sizeof(half)); pipe_->InitBuffer(bL1a5_,GB2*sizeof(half));
      pipe_->InitBuffer(bGA2_,3*MG2*K*sizeof(half));  pipe_->InitBuffer(bGB2_,2*MG2*K*sizeof(half)); } // 常驻: A 侧 Ar/Ai/-Ai, B 侧 Br/Bi
    pipe_->InitBuffer(bAre_,2*GB*K*K*sizeof(half));   // 紧凑 [re GBx16x16][im GBx16x16]
    pipe_->InitBuffer(bAim_,GB*K*GB*K*sizeof(half));  // bt 输出 staging（最大 BATCH=64）
    pipe_->InitBuffer(bGT_,4*GB*K*K*sizeof(half));   // BriG 切 2 块 (Xr/Xi), 各 GB*K*K; 留 4 块余量
    pipe_->InitBuffer(bIdS_,P*K*K*sizeof(half));   // P 个槽对角掩码 (逐槽 σ²)
    #if AIRAN_PACK && (MIMO_K > MIMO_KR)
    pipe_->InitBuffer(bNo_,BATCH*K*sizeof(half));   // 逐槽 σ²: [BATCH unit][K 槽]
#else
    pipe_->InitBuffer(bNo_,BATCH*sizeof(half));    // P=1: 当前 staging batch 的逐 unit 标量
#endif
    pipe_->InitBuffer(bCr_,K*K*sizeof(half)); pipe_->InitBuffer(bCi_,K*K*sizeof(half)); pipe_->InitBuffer(bId_,K*K*sizeof(half));
    pipe_->InitBuffer(bYvL1r_,MB_DEPTH*M*K*sizeof(half));
    pipe_->InitBuffer(bYvL1i_,MB_DEPTH*M*K*sizeof(half));
}

// 复数 16×16 UB-matmul: C=A·B (照 channel_est: 一个co+累加+预取负). A,B,C 在 UB(ND).

// batch cmm16: BATCH 个 [16,16] 复数 C=A·B, 逐 batch 自包含 (§4B.1 + §4C 骨架).
//   C_re=Ar·Br+Ai·(-Bi), C_im=Ar·Bi+Ai·Br. A/B/C 在 UB [BATCH,16,16].

// ═══════════════ ① 循环外常量: Id / Ibat (整个 kernel ×1) ═══════════════
__aicore__ inline void MimoBRI::BuildConst()
{
    // fractal 寄存器是全局的且参数恒定 -> 整个 kernel 设一次 (原来每组一次)
    { constexpr uint8_t padList[4]={0,0,0,0}; Load3DSetFMatrixCal(1, M, padList); }
    // 组版常量: GB 份 [16,16] 连续
    { constexpr uint32_t GB2=GB*K*K;
      auto IdG=bGId_.Get<half>(); auto MblkG=bGMblk_.Get<half>();
      auto MpkG=bGMpk_.Get<half>();
      Duplicate(IdG,(half)0,GB2); Duplicate(MblkG,(half)0,GB2);
      Duplicate(MpkG,(half)0,GB2); PipeBarrier<PIPE_ALL>();
      for (uint32_t g=0;g<GB;++g){ uint32_t o=g*K*K;
        for (uint32_t i=0;i<K;++i){ IdG.SetValue(o+i*K+i,(half)1.0);
          for (uint32_t j=0;j<K;++j){
            if (i/B == j/B)     MblkG.SetValue(o+i*K+j,(half)1.0);   // 预条件块
            if (i/NLR == j/NLR) MpkG.SetValue (o+i*K+j,(half)1.0);   // RE 分隔 (P=1 时全 1)
          } } }
      PipeBarrier<PIPE_ALL>(); }

    auto Id=bId_.Get<half>();
    // 单位阵 I: 对角=1, 其余=0
    Duplicate(Id,(half)0,K*K); PipeBarrier<PIPE_ALL>();
    for (uint32_t i=0;i<K;++i){ Id.SetValue(i*K+i,(half)1.0); }
    PipeBarrier<PIPE_ALL>();

    // [GB,K,K] 行主序对角的 byte-offset，供 Gather 直接抽取；整个 kernel 只生成一次。
    { auto idx=bDiagIdx_.Get<uint32_t>();
      for (uint32_t g=0;g<GB;++g) for (uint32_t i=0;i<K;++i)
          idx.SetValue(g*K+i,(g*K*K+i*K+i)*sizeof(half));
      PipeBarrier<PIPE_ALL>(); }

    // 瘦 GEMM 输出 [GB*K,K]：第 g 个 RE 只取第 g 列。默认路径不分配/生成它。
    { constexpr uint32_t MG2=GB*K;
      auto idx=bSkinnyIdx_.Get<uint32_t>();
      for (uint32_t g=0;g<GB;++g) for (uint32_t i=0;i<K;++i)
          idx.SetValue(g*K+i,((g*K+i)*K+g)*sizeof(half));
      // mvec=[GB,K] -> [K,K] sparse RHS：前 GB 列分别放一个 RE 的 m，
      // 其余列指向 mvec 尾部的常驻零。避免每组两次整矩阵清零和 Transpose。
      auto rhsIdx=idx[MG2];
      Duplicate(rhsIdx,(uint32_t)(MG2*sizeof(half)),K*K); PipeBarrier<PIPE_V>();
      for (uint32_t g=0;g<GB;++g) for (uint32_t i=0;i<K;++i)
          rhsIdx.SetValue(i*K+g,(g*K+i)*sizeof(half));
      Duplicate(bB0r_.Get<half>()[MG2],(half)0,16);
      Duplicate(bB0i_.Get<half>()[MG2],(half)0,16);
      PipeBarrier<PIPE_ALL>(); }

    // MatchFilter 输出 [GB*K,K] 中第 g 个行块只取第 g 列。
    { constexpr uint32_t MG2=GB*K;
      auto idx=bMatchIdx_.Get<uint32_t>();
      for (uint32_t g=0;g<GB;++g) for (uint32_t i=0;i<K;++i)
          idx.SetValue(g*K+i,((g*K+i)*K+g)*sizeof(half));
      // 生产态的 skinny Detect 直接消费紧凑 mvec；尾部 16 个零作为
      // sparse RHS Gather 的常驻零源。
      Duplicate(bGmr_.Get<half>()[MG2],(half)0,16);
      Duplicate(bGmi_.Get<half>()[MG2],(half)0,16);
      PipeBarrier<PIPE_ALL>(); }

    // 槽对角掩码 IdS[p]: 只有槽 p 的对角为 1. 逐槽 σ² 用 P 次 Axpy 累加,
    // 只用已验证过的原语 (Axpy 纯 V 流水 + 标量读), 不引入新语义.
    { auto IdS=bIdS_.Get<half>();
      Duplicate(IdS,(half)0,P*K*K); PipeBarrier<PIPE_ALL>();
      for (uint32_t p=0;p<P;++p)
        for (uint32_t i=p*NLR;i<(p+1)*NLR;++i) IdS.SetValue(p*K*K+i*K+i,(half)1.0);
      PipeBarrier<PIPE_ALL>(); }

    // 块对角 mask (借 bCi_ —— 原本清零后从未被读, 是死 buffer; 零新增, UB 布局不动)
    // 块内=1 块外=0, 一次 Mul 完成 "复制 Are + 抹掉块外"
    auto Mblk=bCi_.Get<half>();
    Duplicate(Mblk,(half)0,K*K); PipeBarrier<PIPE_ALL>();
    for (uint32_t i=0;i<K;++i) for (uint32_t j=0;j<K;++j)
        if (i/B == j/B) Mblk.SetValue(i*K+j,(half)1.0);
    PipeBarrier<PIPE_ALL>();

    // Ibat: BATCH 个块对角单位阵 (batch BRI 的 I)
}

// ═══════════════ ② 预取 GB 个独立 RE 的 H/Y -> L1 ring ═══════════════
__aicore__ inline void MimoBRI::LoadRE(uint32_t re0,uint32_t slot)
{
    constexpr uint32_t H_SLOT=GB*M*K;
    auto HrL1=bHrL1_.Get<half>()[slot*H_SLOT];
    auto HiL1=bHiL1_.Get<half>()[slot*H_SLOT];
    const uint64_t goH=(uint64_t)re0*M*K;
    DataCopy(HrL1,gHr_[goH],H_SLOT);
    DataCopy(HiL1,gHi_[goH],H_SLOT);
    constexpr uint32_t Y_SLOT=M*K;
    auto YrL1=bYvL1r_.Get<half>()[slot*Y_SLOT];
    auto YiL1=bYvL1i_.Get<half>()[slot*Y_SLOT];
    const uint64_t goY=(uint64_t)(re0/GB)*Y_SLOT;
    DataCopy(YrL1,gYvr_[goY],Y_SLOT);
    DataCopy(YiL1,gYvi_[goY],Y_SLOT);
    // 无 barrier：future slot 的 MTE2 与 current slot 的 MTE1/Cube/Vector 重叠。
}

// ═══════════════ ③ Gram: A = HᴴH + σ²I  (4 gmm1 + 3 flush) ═══════════════
__aicore__ inline void MimoBRI::Gram(uint32_t slot)
{
    // L0A/L0B 用裸 TBuf + 手工事件, 绕开 TQue 的 Alloc/EnQue/DeQue/Free (每组省 24 条标量).
    // 三个乘积只需 4 个不同操作数: A={Hrᵀ,Hiᵀ}, B={Hr,Hi}, 原来 6 次 LoadData 有 2 次是重复的.
    constexpr uint32_t MG = GB*K, OPS = GB*K*M;      // OPS = 单个操作数的元素数
    constexpr uint64_t EXTCFG = ((uint64_t)0<<48)|((uint64_t)0<<32)|((uint64_t)M<<16)|(uint64_t)MG;
    constexpr uint32_t SLOT_ELEMS=GB*M*K;
    auto HrL1=bHrL1_.Get<half>()[slot*SLOT_ELEMS];
    auto HiL1=bHiL1_.Get<half>()[slot*SLOT_ELEMS];
    auto Are=bAre_.Get<half>();   // Are 在 [0], Aim 在 [NN]
    auto a2=bA2_.Get<half>(); auto b2=bB2_.Get<half>();

    { constexpr uint8_t padList[4]={0,0,0,0}; Load3DSetFMatrixCal(1, M, padList); }   // cmm16 会改全局 FMatrix
    LoadData3DParamsV2Pro pa; pa.channelSize=MG; pa.extConfig=EXTCFG; pa.enTranspose=true;
    LoadData3DParamsV2Pro pb; pb.channelSize=MG; pb.extConfig=EXTCFG;
    LoadData<half>(a2,       HrL1, pa);              // A0 = Hrᵀ
    LoadData<half>(a2[OPS],  HiL1, pa);              // A1 = Hiᵀ
    LoadData<half>(b2,       HrL1, pb);              // B0 = Hr
    LoadData<half>(b2[OPS],  HiL1, pb);              // B1 = Hi
    SetFlag<HardEvent::MTE1_M>(E0); WaitFlag<HardEvent::MTE1_M>(E0);

    // L0C 也用裸 TBuf: Are 放 co[0], X 放 co[MG*MG], 一次 DataCopy 把两个结果一起搬出.
    // (同时重测"Mmad 忽略目标偏移"—— 那个结论是 TQue 语境下得到的)
    constexpr uint32_t NN = MG*MG, GBKK=GB*K*K;
    auto co=bCO_.Get<float>();
    MmadParams mp; mp.m=MG; mp.n=MG; mp.k=M;
    mp.cmatrixInitVal=true;  Mmad(co,     a2,      b2,      mp);    // TrᵀTr
    mp.cmatrixInitVal=false; Mmad(co,     a2[OPS], b2[OPS], mp);    // + TiᵀTi -> Are
    mp.cmatrixInitVal=true;  Mmad(co[NN], a2,      b2[OPS], mp);    // X = TrᵀTi
    PipeBarrier<PIPE_ALL>();
    { DataCopyParams dcp; dcp.blockCount=(uint16_t)GB; dcp.blockLen=1;
      dcp.srcStride=(uint16_t)GB; dcp.dstStride=0;
      DataCopyEnhancedParams ep; ep.blockMode=BlockMode::BLOCK_MODE_MATRIX;
      DataCopy(Are,       co,     dcp, ep);
      DataCopy(Are[GBKK], co[NN], dcp, ep); }
    PipeBarrier<PIPE_ALL>();                      // FixPipe 直转 FP16，无 Vector Cast
}

// ═══════════════ ④ 块逆: 置零块外 + Dinv + E + Neumann×NEU -> M⁻¹ ═══════════════

// ═══════ 组版匹配滤波: m = Hᴴ·yv  (GB 个 RE 一起) ═══════
//   m_re = Hrᵀ·yvr + Hiᵀ·yvi ;  m_im = Hrᵀ·yvi − Hiᵀ·yvr
//   与 Gram 同构, 只是第二操作数换成 yv —— A 侧 (Hrᵀ/Hiᵀ) 直接复用 Gram 留在 L0A 的, 一次都不重载
__aicore__ inline void MimoBRI::MatchFilterG(uint32_t re0,uint32_t slot)
{
    // 8 个独立 y 直接占 [M,K] RHS 的前 GB 列：Mmad 从 128x128x64
    // 收窄为 128x16x64。只取 (第 j 个 H 行块, 第 j 列)，其 K=64
    // 累加顺序与旧方阵路径相同；结果保持紧凑并直接交给 Detect。
    constexpr uint32_t MG=GB*K, OPS=MG*M, RHS=M*K, GB2=GB*K*K;
    constexpr uint64_t EXTCFG=((uint64_t)0<<48)|((uint64_t)0<<32)|((uint64_t)M<<16)|(uint64_t)K;
    auto Yr=bYvL1r_.Get<half>()[slot*RHS];
    auto Yi=bYvL1i_.Get<half>()[slot*RHS];
    auto a2=bA2_.Get<half>(); auto b2=bB2_.Get<half>();
    auto co=bCO_.Get<float>();
    auto sr=bGTr_.Get<half>(); auto sip=bGTi_.Get<half>(); auto sin=bGtmp_.Get<half>();
    auto vr=bB0r_.Get<half>(); auto vi=bB0i_.Get<half>(); auto bc=bGdcol_.Get<half>();
    auto mr=bGmr_.Get<half>(); auto mi=bGmi_.Get<half>();

    (void)re0;
    { constexpr uint8_t padList[4]={0,0,0,0}; Load3DSetFMatrixCal(1, M, padList); }
    LoadData3DParamsV2Pro pb; pb.channelSize=K; pb.extConfig=EXTCFG;
    LoadData<half>(b2, Yr, pb); LoadData<half>(b2[RHS], Yi, pb);
    SetFlag<HardEvent::MTE1_M>(emG_); WaitFlag<HardEvent::MTE1_M>(emG_);

    MmadParams mp; mp.m=MG; mp.n=K; mp.k=M;
    mp.cmatrixInitVal=true;  Mmad(co,        a2,      b2,      mp);  // Hr^T*y_re
    mp.cmatrixInitVal=false; Mmad(co,        a2[OPS],b2[RHS], mp);  // +Hi^T*y_im
    mp.cmatrixInitVal=true;  Mmad(co[GB2],   a2,      b2[RHS], mp);  // Hr^T*y_im
    mp.cmatrixInitVal=true;  Mmad(co[2*GB2], a2[OPS],b2,      mp);  // Hi^T*y_re
    SetFlag<HardEvent::M_MTE1>(ebG_); WaitFlag<HardEvent::M_MTE1>(ebG_);

    PipeBarrier<PIPE_ALL>();
    { DataCopyParams dcp; dcp.blockCount=(uint16_t)GB; dcp.blockLen=1;
      dcp.srcStride=0; dcp.dstStride=0;
      DataCopyEnhancedParams ep; ep.blockMode=BlockMode::BLOCK_MODE_MATRIX;
      DataCopy(sr,  co,        dcp,ep);
      DataCopy(sip, co[GB2],   dcp,ep);
      DataCopy(sin, co[2*GB2], dcp,ep); }
    PipeBarrier<PIPE_ALL>();
    Sub(sip,sip,sin,GB2); PipeBarrier<PIPE_V>();

    // [MG,K] 中第 g 个行块只取第 g 列，得到 GB 个真实 m 向量。
    Gather(vr,sr, bMatchIdx_.Get<uint32_t>(),(uint32_t)0,MG);
    Gather(vi,sip,bMatchIdx_.Get<uint32_t>(),(uint32_t)0,MG); PipeBarrier<PIPE_V>();
    // skinny Detect 直接消费 [GB,K]，不再先广播 K 列又马上压缩回来。
    DataCopy(mr,vr,MG); DataCopy(mi,vi,MG); PipeBarrier<PIPE_V>();
}

// ═══════ 组版块逆: GB 个 RE 一起做 ═══════
__aicore__ inline void MimoBRI::BlockInvG()
{
    constexpr uint32_t GB2=GB*K*K, MG2=GB*K;
    auto Are=bGAr_.Get<half>();  auto Aim=bGAi_.Get<half>();
    auto IdG=bGId_.Get<half>();  auto MblkG=bGMblk_.Get<half>();
    auto Mbr=bGMbr_.Get<half>(); auto Mbi=bGMbi_.Get<half>();
    auto Er=bGEr_.Get<half>();   auto Ei=bGEi_.Get<half>();
    auto ErT=bGErT_.Get<half>(); auto EiT=bGEiT_.Get<half>();
    auto Sr=bGSr_.Get<half>();   auto Si=bGSi_.Get<half>();
    auto Tr=bGTr_.Get<half>();   auto Ti=bGTi_.Get<half>();
    auto Mvr=bGMvr_.Get<half>(); auto Mvi=bGMvi_.Get<half>();
    auto tmp=bGtmp_.Get<half>(); auto dcol=bGdcol_.Get<half>(); auto dcolT=bGdcolT_.Get<half>();
    auto Dinv=bGDinv_.Get<half>();

    Mul(Mbr,Are,MblkG,GB2); Mul(Mbi,Aim,MblkG,GB2); PipeBarrier<PIPE_V>();   // 置零块外
    // Dinv = 1/diag: 固定 byte-offset 直接抽对角，避免整张矩阵乘 I 再 reduce。
    Gather(Dinv,Mbr,bDiagIdx_.Get<uint32_t>(),(uint32_t)0,MG2); PipeBarrier<PIPE_V>();
    // Dinv 最终以 FP16 参与后续 BRI；直接在 half 上求倒数，删除
    // half->float->half 两次 Cast 和 FP32 Div。
    Maxs(Dinv,Dinv,(half)DET_EPS,MG2); PipeBarrier<PIPE_V>();
    Reciprocal(Dinv,Dinv,MG2); PipeBarrier<PIPE_V>();
    Brcb(dcol,Dinv,MG2/8,{1,8}); PipeBarrier<PIPE_V>();      // dcol[i*K+j] = Dinv[i] (行广播)
    Mul(Er,Mbr,dcol,GB2); Mul(Ei,Mbi,dcol,GB2); PipeBarrier<PIPE_V>();
    Sub(Er,IdG,Er,GB2);                                      // E_re = I − Dinv·M (省掉一条 Muls)
    Muls(Ei,Ei,(half)-1.0,GB2); PipeBarrier<PIPE_V>();       // E_im = −Dinv·M

    #pragma unroll
    for (uint32_t j=0;j<GB;++j){ Transpose(ErT[j*K*K],Er[j*K*K]); Transpose(EiT[j*K*K],Ei[j*K*K]); }
    PipeBarrier<PIPE_V>();
    { // A 侧 Ar/Ai/-Ai 在全部 NEU 项里不变 -> 循环外备好, 送 L1 并常驻 L0A
      constexpr uint32_t MG2=GB*K, OP=MG2*K;
      auto nAi=bGtmp_.Get<half>();
      Muls(nAi,EiT,(half)-1.0,GB2); PipeBarrier<PIPE_V>();          // -Aiᵀ
      SetFlag<HardEvent::V_MTE3>(evG_);WaitFlag<HardEvent::V_MTE3>(evG_);
      DataCopy(bL1ar_.Get<half>(), ErT, GB2);
      DataCopy(bL1ai_.Get<half>(), EiT, GB2);
      DataCopy(bL1bn_.Get<half>(), nAi, GB2);
      SetFlag<HardEvent::MTE3_MTE1>(e1G_);WaitFlag<HardEvent::MTE3_MTE1>(e1G_);
      { constexpr uint8_t padList[4]={0,0,0,0}; Load3DSetFMatrixCal(1, K, padList); }   // 提出 cmmG
      LoadData3DParamsV2Pro pa; pa.channelSize=MG2; pa.enTranspose=true;
      pa.extConfig=((uint64_t)0<<48)|((uint64_t)0<<32)|((uint64_t)K<<16)|(uint64_t)MG2;
      auto a2=bGA2_.Get<half>();
      LoadData<half>(a2,       bL1ar_.Get<half>(), pa);
      LoadData<half>(a2[OP],   bL1ai_.Get<half>(), pa);
      LoadData<half>(a2[2*OP], bL1bn_.Get<half>(), pa); }

    // E*I=E：直接从 S=I+E 起步，仍完整计算 I+E+E^2+E^3，
    // 只消除原先首轮 E*I 的单位矩阵 Cube 乘法，不改变 Neumann 阶数。
    Add(Sr,IdG,Er,GB2); DataCopy(Si,Ei,GB2); PipeBarrier<PIPE_ALL>();
    // ── cmmG 内联进循环, 所有不变量提到循环外 (局部变量, 只跨这一组的生命周期) ──
    //    事件 ID / 参数结构 / 张量视图 原来每轮各重建一次, 现在每组一次
    {
      constexpr uint32_t MG2=GB*K, NN2=MG2*MG2, OP=MG2*K;
      auto l1br=bL1br_.Get<half>(); auto l1bi=bL1bi_.Get<half>();
      auto a2=bGA2_.Get<half>();    auto b2=bGB2_.Get<half>();
      auto co=bCO_.Get<float>();
      LoadData3DParamsV2Pro pb; pb.channelSize=MG2;
      pb.extConfig=((uint64_t)0<<48)|((uint64_t)0<<32)|((uint64_t)K<<16)|(uint64_t)MG2;
      MmadParams mp; mp.m=MG2; mp.n=MG2; mp.k=K;
      DataCopyParams dcp; dcp.blockCount=(uint16_t)GB; dcp.blockLen=1;
      dcp.srcStride=(uint16_t)GB; dcp.dstStride=0;
      DataCopyEnhancedParams ep; ep.blockMode=BlockMode::BLOCK_MODE_MATRIX;

      #pragma unroll
      for (uint32_t it=1; it<NEU; ++it){
        SetFlag<HardEvent::V_MTE3>(evG_); WaitFlag<HardEvent::V_MTE3>(evG_);
        DataCopy(l1br, Sr, GB2); DataCopy(l1bi, Si, GB2);              // B 侧每轮变
        SetFlag<HardEvent::MTE3_MTE1>(e1G_); WaitFlag<HardEvent::MTE3_MTE1>(e1G_);
        LoadData<half>(b2, l1br, pb); LoadData<half>(b2[OP], l1bi, pb);
        SetFlag<HardEvent::MTE1_M>(emG_); WaitFlag<HardEvent::MTE1_M>(emG_);
        mp.cmatrixInitVal=true;  Mmad(co,      a2,        b2,      mp);   // Ar·Br
        mp.cmatrixInitVal=false; Mmad(co,      a2[2*OP],  b2[OP],  mp);   // + (-Ai)·Bi
        mp.cmatrixInitVal=true;  Mmad(co[NN2], a2,        b2[OP],  mp);   // Ar·Bi
        mp.cmatrixInitVal=false; Mmad(co[NN2], a2[OP],    b2,      mp);   // + Ai·Br
        SetFlag<HardEvent::M_MTE1>(ebG_); WaitFlag<HardEvent::M_MTE1>(ebG_);
        PipeBarrier<PIPE_ALL>();
        DataCopy(Tr,co,       dcp, ep);
        DataCopy(Si,co[NN2],  dcp, ep);                // FixPipe 直转 FP16
        PipeBarrier<PIPE_ALL>();
        Add(Sr,Tr,IdG,GB2); PipeBarrier<PIPE_V>();
      }
    }
    // M⁻¹ = S ⊙ Dinv_mat, Dinv_mat[i][j]=Dinv[j] = dcol 的转置
    #pragma unroll
    for (uint32_t j=0;j<GB;++j) Transpose(dcolT[j*K*K],dcol[j*K*K]);
    PipeBarrier<PIPE_V>();
    Mul(Mvr,Sr,dcolT,GB2); Mul(Mvi,Si,dcolT,GB2);
    PipeBarrier<PIPE_V>();     // BriG 先在 Vector 流水读 Mvr/Mvi
}

// ═══════ 自包含的组版复数乘 (照 19.43ms 那版已验证的模式) ═══════
//   每个 Mmad 前紧接着载自己的 A 和 B —— 一次只驻留一个操作数, 最保守
__aicore__ inline void MimoBRI::cmmGF(const LocalTensor<half>&la0,const LocalTensor<half>&la1,const LocalTensor<half>&la2,
    const LocalTensor<half>&Br,const LocalTensor<half>&Bi,const LocalTensor<half>&Cr,const LocalTensor<half>&Ci)
{
    // 照 BlockInvG 那个已跑通的循环体: A 侧 3 个 + B 侧 2 个常驻, 4 个 Mmad, 一次 drain.
    // A 侧瓦片 (Ar/Ai/-Ai) 由调用方在循环外备好 —— 负号在不变的 A 侧, B 侧不用算 -Bi.
    constexpr uint32_t GB2=GB*K*K, MG2=GB*K, NN2=MG2*MG2, OP=MG2*K;
    auto l1br=bL1br_.Get<half>(); auto l1bi=bL1bi_.Get<half>();
    auto a2=bGA2_.Get<half>(); auto b2=bGB2_.Get<half>();
    auto co=bCO_.Get<float>();

    SetFlag<HardEvent::V_MTE3>(evG_);WaitFlag<HardEvent::V_MTE3>(evG_);
    DataCopy(l1br,Br,GB2); DataCopy(l1bi,Bi,GB2);
    SetFlag<HardEvent::MTE3_MTE1>(e1G_);WaitFlag<HardEvent::MTE3_MTE1>(e1G_);

    // FMatrix 由 BriG 在循环外设好 (1, K)
    LoadData3DParamsV2Pro pa; pa.channelSize=MG2; pa.enTranspose=true;
    pa.extConfig=((uint64_t)0<<48)|((uint64_t)0<<32)|((uint64_t)K<<16)|(uint64_t)MG2;
    LoadData3DParamsV2Pro pb; pb.channelSize=MG2; pb.extConfig=pa.extConfig;
    LoadData<half>(a2,       la0, pa);   // Ar
    LoadData<half>(a2[OP],   la1, pa);   // Ai
    LoadData<half>(a2[2*OP], la2, pa);   // -Ai
    LoadData<half>(b2,       l1br, pb);  // Br
    LoadData<half>(b2[OP],   l1bi, pb);  // Bi
    SetFlag<HardEvent::MTE1_M>(emG_); WaitFlag<HardEvent::MTE1_M>(emG_);

    MmadParams mp; mp.m=MG2; mp.n=MG2; mp.k=K;
    mp.cmatrixInitVal=true;  Mmad(co,      a2,        b2,      mp);   // Ar·Br
    mp.cmatrixInitVal=false; Mmad(co,      a2[2*OP],  b2[OP],  mp);   // + (-Ai)·Bi -> C_re
    mp.cmatrixInitVal=true;  Mmad(co[NN2], a2,        b2[OP],  mp);   // Ar·Bi
    mp.cmatrixInitVal=false; Mmad(co[NN2], a2[OP],    b2,      mp);   // + Ai·Br    -> C_im
    SetFlag<HardEvent::M_MTE1>(ebG_); WaitFlag<HardEvent::M_MTE1>(ebG_);

    PipeBarrier<PIPE_ALL>();
    { DataCopyParams dcp; dcp.blockCount=(uint16_t)GB; dcp.blockLen=1;
      dcp.srcStride=(uint16_t)GB; dcp.dstStride=0;
      DataCopyEnhancedParams ep; ep.blockMode=BlockMode::BLOCK_MODE_MATRIX;
      DataCopy(Cr,co,      dcp, ep);
      DataCopy(Ci,co[NN2],dcp, ep); }
    PipeBarrier<PIPE_ALL>();
}

// ═══════ A 侧已常驻 L0A 的版本 (与 BlockInvG 的循环体逐条一致) ═══════
__aicore__ inline void MimoBRI::cmmGR(const LoadData3DParamsV2Pro&pb,const MmadParams&mp0,
    const LocalTensor<half>&Br,const LocalTensor<half>&Bi,
    const LocalTensor<half>&Cr,const LocalTensor<half>&Ci)
{
    constexpr uint32_t GB2=GB*K*K, MG2=GB*K, NN2=MG2*MG2, OP=MG2*K;
    auto l1br=bL1br_.Get<half>(); auto l1bi=bL1bi_.Get<half>();
    auto a2=bGA2_.Get<half>(); auto b2=bGB2_.Get<half>();
    auto co=bCO_.Get<float>();

    SetFlag<HardEvent::V_MTE3>(evG_);WaitFlag<HardEvent::V_MTE3>(evG_);
    DataCopy(l1br,Br,GB2); DataCopy(l1bi,Bi,GB2);
    SetFlag<HardEvent::MTE3_MTE1>(e1G_);WaitFlag<HardEvent::MTE3_MTE1>(e1G_);

    LoadData<half>(b2, l1br, pb); LoadData<half>(b2[OP], l1bi, pb);
    SetFlag<HardEvent::MTE1_M>(emG_); WaitFlag<HardEvent::MTE1_M>(emG_);

    MmadParams mp=mp0;
    mp.cmatrixInitVal=true;  Mmad(co,      a2,        b2,      mp);
    mp.cmatrixInitVal=false; Mmad(co,      a2[2*OP],  b2[OP],  mp);
    mp.cmatrixInitVal=true;  Mmad(co[NN2], a2,        b2[OP],  mp);
    mp.cmatrixInitVal=false; Mmad(co[NN2], a2[OP],    b2,      mp);
    SetFlag<HardEvent::M_MTE1>(ebG_); WaitFlag<HardEvent::M_MTE1>(ebG_);

    PipeBarrier<PIPE_ALL>();
    { DataCopyParams dcp; dcp.blockCount=(uint16_t)GB; dcp.blockLen=1;
      dcp.srcStride=(uint16_t)GB; dcp.dstStride=0;
      DataCopyEnhancedParams ep; ep.blockMode=BlockMode::BLOCK_MODE_MATRIX;
      DataCopy(Cr,co,      dcp, ep);
      DataCopy(Ci,co[NN2],dcp, ep); }
    PipeBarrier<PIPE_ALL>();
}

// ═══════ 组版 BRI: X = M⁻¹; ×NLAY: X += M⁻¹(I − A·X)  [保守版, 全用已验证构件] ═══════
__aicore__ inline void MimoBRI::BriG()
{
    // 数学重构: X += M⁻¹(I − A·X) = X + M⁻¹ − (M⁻¹A)·X
    //   令 B = M⁻¹A (循环不变, 只算一次) -> 每轮只需 1 次复数乘, 而不是 2 次
    //   每组 cmmGF 调用由旧代数形式的 10 次降为 1+NLAY=6 次；NLAY 仍固定为 5，
    //   减少的是重复矩阵乘，不是 BRI 迭代次数，且循环里 A 侧恒为 B。
    constexpr uint32_t GB2=GB*K*K;
    auto Ar=bGAr_.Get<half>();  auto Ai=bGAi_.Get<half>();
    auto Mr=bGMvr_.Get<half>(); auto Mi=bGMvi_.Get<half>();
    auto Br_=bGMbr_.Get<half>();auto Bi_=bGMbi_.Get<half>();   // B = M⁻¹A
    auto PrT=bGEr_.Get<half>(); auto PiT=bGEi_.Get<half>();    // 转置暂存 (先 M⁻¹ᵀ, 后 Bᵀ)
    auto nPiT=bGEiT_.Get<half>();
    auto GT=bGT_.Get<half>();
    auto Xr=GT;          auto Xi=GT[GB2];
    auto Tr=bGTr_.Get<half>();  auto Ti=bGTi_.Get<half>();

    { constexpr uint8_t padList[4]={0,0,0,0}; Load3DSetFMatrixCal(1, K, padList); }   // 循环外设一次

    // ── ① B = M⁻¹·A ──
    //   A = HᴴH + σ²I 是 Hermitian ⇒ M⁻¹ 也是 ⇒ Mrᵀ = Mr, Miᵀ = −Mi
    //   转置退化成取负号: 三件套 (Prᵀ, Piᵀ, −Piᵀ) = (Mr, −Mi, Mi), 省 2*GB 次 Transpose
    Muls(nPiT,Mi,(half)-1.0,GB2); PipeBarrier<PIPE_V>();
    { SetFlag<HardEvent::V_MTE3>(evG_);WaitFlag<HardEvent::V_MTE3>(evG_);
      DataCopy(bL1a0_.Get<half>(),Mr,GB2); DataCopy(bL1a1_.Get<half>(),nPiT,GB2); DataCopy(bL1a2_.Get<half>(),Mi,GB2);
      SetFlag<HardEvent::MTE3_MTE1>(e1G_);WaitFlag<HardEvent::MTE3_MTE1>(e1G_); }
    cmmGF(bL1a0_.Get<half>(),bL1a1_.Get<half>(),bL1a2_.Get<half>(), Ar,Ai, Br_,Bi_);

    // ── ② C = I − M⁻¹A；Cᵀ 三件套在 Richardson 循环里常驻 L0A ──
    // X <- M⁻¹ + C·X 与 X <- X + M⁻¹ − (M⁻¹A)·X 数学等价，
    // 每轮仍做一次复矩阵乘，但全矩阵 Add/Sub 从四条减为两条。
    Sub(Br_,bGId_.Get<half>(),Br_,GB2);
    Muls(Bi_,Bi_,(half)-1.0,GB2); PipeBarrier<PIPE_V>();
    #pragma unroll
    for (uint32_t j=0;j<GB;++j){ uint32_t o=j*K*K; Transpose(PrT[o],Br_[o]); Transpose(PiT[o],Bi_[o]); }
    PipeBarrier<PIPE_V>();
    Muls(nPiT,PiT,(half)-1.0,GB2); PipeBarrier<PIPE_V>();
    { SetFlag<HardEvent::V_MTE3>(evG_);WaitFlag<HardEvent::V_MTE3>(evG_);
      DataCopy(bL1a3_.Get<half>(),PrT,GB2); DataCopy(bL1a4_.Get<half>(),PiT,GB2); DataCopy(bL1a5_.Get<half>(),nPiT,GB2);
      SetFlag<HardEvent::MTE3_MTE1>(e1G_);WaitFlag<HardEvent::MTE3_MTE1>(e1G_);
      // Bᵀ 三件套常驻 L0A —— 循环里 A 侧恒定, 正是 BlockInvG 的用法
      constexpr uint32_t MG2b=GB*K, OPb=MG2b*K;
      LoadData3DParamsV2Pro pa; pa.channelSize=MG2b; pa.enTranspose=true;
      pa.extConfig=((uint64_t)0<<48)|((uint64_t)0<<32)|((uint64_t)K<<16)|(uint64_t)MG2b;
      auto a2b=bGA2_.Get<half>();
      LoadData<half>(a2b,        bL1a3_.Get<half>(), pa);
      LoadData<half>(a2b[OPb],   bL1a4_.Get<half>(), pa);
      LoadData<half>(a2b[2*OPb], bL1a5_.Get<half>(), pa); }

    // ── ③ X = M⁻¹; ×NLAY: X ← M⁻¹ + C·X ──
    DataCopy(Xr,Mr,GB2); DataCopy(Xi,Mi,GB2); PipeBarrier<PIPE_ALL>();
    LoadData3DParamsV2Pro pbR; pbR.channelSize=GB*K;
    pbR.extConfig=((uint64_t)0<<48)|((uint64_t)0<<32)|((uint64_t)K<<16)|(uint64_t)(GB*K);
    MmadParams mpR; mpR.m=GB*K; mpR.n=GB*K; mpR.k=K;
    #pragma unroll
    for (uint32_t it=0; it<NLAY; ++it){
        cmmGR(pbR,mpR,Xr,Xi, Tr,Ti);                            // T = C·X (A 侧常驻)
        Add(Xr,Tr,Mr,GB2); Add(Xi,Ti,Mi,GB2); PipeBarrier<PIPE_V>();
    }
}

// ═══════ 组版检测: x̂ = X·m, unbias, no_eff  (GB 个 RE 一起) ═══════
//   结果存进 per-bt 的 [BATCH][K] 暂存 (偏移 g*GB*K), 由 bt 结尾统一转置写 GM
//   —— 因为 GM 里固定 k 时同批 RE 连续, 一次只写 GB=4 个 half (8B) 不满足 32B 对齐
__aicore__ inline void MimoBRI::DetectG(uint32_t g)
{
    // 瘦 GEMM：仍只发射 4 次 Cube，但把 8 个独立 RHS 放入 16 列，
    // 从 [128,128,16] 改成 [128,16,16]。K 维累加顺序与旧方阵路径一致。
    constexpr uint32_t GB2=GB*K*K, MG2=GB*K, OP=MG2*K, KK=K*K;
    auto GT=bGT_.Get<half>(); auto Xr=GT; auto Xi=GT[GB2];
    auto mr=bGmr_.Get<half>(); auto mi=bGmi_.Get<half>();
    auto Tr=bGTr_.Get<half>(); auto Ti=bGTi_.Get<half>();
    auto rhsR=bGdcol_.Get<half>(); auto rhsI=bGdcolT_.Get<half>(); auto nRhs=bGtmp_.Get<half>();
    auto dvh=bDvh_.Get<half>(); auto b0r=bB0r_.Get<half>(); auto b0i=bB0i_.Get<half>();
    auto sigE=bSigE_.Get<half>(); auto no_=bNo_.Get<half>();
    auto f=bGf_.Get<float>();
    auto dvf=f; auto omf=f[MG2]; auto b0rf=f[2*MG2]; auto b0if=f[3*MG2]; auto sigf=f[4*MG2];
    auto a2=bGA2_.Get<half>(); auto b2=bGB2_.Get<half>(); auto co=bCO_.Get<float>();
    auto l1br=bL1br_.Get<half>(); auto l1bi=bL1bi_.Get<half>();

    // m 由 [GB,K,K] 的重复列压成 [K,K] 稀疏 RHS。预计算 byte index
    // 直接把第 g 个 RE 的 m 放进第 g 列，零尾跨组常驻。
    auto rhsIdx=bSkinnyIdx_.Get<uint32_t>()[MG2];
    Gather(rhsR,mr,rhsIdx,(uint32_t)0,KK);
    Gather(rhsI,mi,rhsIdx,(uint32_t)0,KK); PipeBarrier<PIPE_V>();

    // 复数乘法的负号移到 16x16 RHS：只处理 256 个元素，
    // 替代对 8x16x16 的 Xi 共 2048 个元素取负。
    Muls(nRhs,rhsR,(half)-1.0,KK); PipeBarrier<PIPE_V>();
    SetFlag<HardEvent::V_MTE3>(evG_); WaitFlag<HardEvent::V_MTE3>(evG_);
    DataCopy(bL1a0_.Get<half>(),Xr,GB2); DataCopy(bL1a1_.Get<half>(),Xi,GB2);
    DataCopy(l1br,rhsR,KK); DataCopy(l1bi,rhsI,KK); DataCopy(bL1bn_.Get<half>(),nRhs,KK);
    SetFlag<HardEvent::MTE3_MTE1>(e1G_); WaitFlag<HardEvent::MTE3_MTE1>(e1G_);
    { LoadData3DParamsV2Pro pa; pa.channelSize=MG2; pa.enTranspose=true;
      pa.extConfig=((uint64_t)0<<48)|((uint64_t)0<<32)|((uint64_t)K<<16)|(uint64_t)MG2;
      LoadData3DParamsV2Pro pb; pb.channelSize=K;
      pb.extConfig=((uint64_t)0<<48)|((uint64_t)0<<32)|((uint64_t)K<<16)|(uint64_t)K;
      LoadData<half>(a2,      bL1a0_.Get<half>(),pa);
      LoadData<half>(a2[OP],  bL1a1_.Get<half>(),pa);
      LoadData<half>(b2,      l1br,pb); LoadData<half>(b2[KK],l1bi,pb);
      LoadData<half>(b2[2*KK],bL1bn_.Get<half>(),pb); }
    SetFlag<HardEvent::MTE1_M>(emG_); WaitFlag<HardEvent::MTE1_M>(emG_);
    { MmadParams mp; mp.m=MG2; mp.n=K; mp.k=K;
      mp.cmatrixInitVal=true;  Mmad(co,      a2,     b2,       mp);
      mp.cmatrixInitVal=false; Mmad(co,      a2[OP],b2[KK],   mp);
      mp.cmatrixInitVal=true;  Mmad(co[GB2], a2,     b2[KK],   mp);
      mp.cmatrixInitVal=false; Mmad(co[GB2], a2[OP],b2[2*KK],mp); }
    SetFlag<HardEvent::M_MTE1>(ebG_); WaitFlag<HardEvent::M_MTE1>(ebG_);

    // Cube 运行期间计算与它无依赖的 no_eff。
    Gather(dvh,Xr,bDiagIdx_.Get<uint32_t>(),(uint32_t)0,MG2); PipeBarrier<PIPE_V>();
    #if AIRAN_PACK && (MIMO_K > MIMO_KR)
    DataCopy(sigE,no_[(uint64_t)g*GB*K],GB*K); PipeBarrier<PIPE_V>();
    #else
    Brcb(sigE,no_[g*GB],GB/8>0?GB/8:1,{1,8}); PipeBarrier<PIPE_V>();
    #endif
    Cast(dvf,dvh,RoundMode::CAST_NONE,MG2); Cast(sigf,sigE,RoundMode::CAST_NONE,MG2); PipeBarrier<PIPE_V>();
    Mul(dvf,dvf,sigf,MG2); PipeBarrier<PIPE_V>(); Muls(dvf,dvf,-1.0f,MG2); PipeBarrier<PIPE_V>();
    Adds(dvf,dvf,1.0f,MG2); PipeBarrier<PIPE_V>(); Maxs(dvf,dvf,DET_EPS,MG2); PipeBarrier<PIPE_V>();
    Duplicate(omf,1.0f,MG2); PipeBarrier<PIPE_V>(); Sub(omf,omf,dvf,MG2); PipeBarrier<PIPE_V>();
    Div(omf,omf,dvf,MG2); PipeBarrier<PIPE_V>();

    PipeBarrier<PIPE_ALL>();
    { DataCopyParams dcp; dcp.blockCount=(uint16_t)GB; dcp.blockLen=1; dcp.srcStride=0; dcp.dstStride=0;
      DataCopyEnhancedParams ep; ep.blockMode=BlockMode::BLOCK_MODE_MATRIX;
      DataCopy(Tr,co,dcp,ep); DataCopy(Ti,co[GB2],dcp,ep); }
    PipeBarrier<PIPE_ALL>();
    Gather(b0r,Tr,bSkinnyIdx_.Get<uint32_t>(),(uint32_t)0,MG2);
    Gather(b0i,Ti,bSkinnyIdx_.Get<uint32_t>(),(uint32_t)0,MG2); PipeBarrier<PIPE_V>();
    Cast(b0rf,b0r,RoundMode::CAST_NONE,MG2); Cast(b0if,b0i,RoundMode::CAST_NONE,MG2); PipeBarrier<PIPE_V>();
    Div(b0rf,b0rf,dvf,MG2); Div(b0if,b0if,dvf,MG2); PipeBarrier<PIPE_V>();
    { auto SB=bAim_.Get<half>(); auto xhB=SB; auto xhiB=SB[BATCH*K]; auto nefB=SB[2*BATCH*K];
      uint32_t off=g*GB*K;
      Cast(xhB[off],b0rf,RoundMode::CAST_NONE,MG2); Cast(xhiB[off],b0if,RoundMode::CAST_NONE,MG2);
      Cast(nefB[off],omf,RoundMode::CAST_NONE,MG2); }
    PipeBarrier<PIPE_V>();
}


// ═══════════════ 主流程 ═══════════════
__aicore__ inline void MimoBRI::Run()
{
    BuildConst();
    constexpr uint32_t SCHEDULE_BATCH=RE_WINDOW;
    constexpr uint32_t N_SCHEDULE_PER_CORE=N_WINDOW_PER_CORE;
    for (uint32_t bt=0; bt<N_SCHEDULE_PER_CORE; ++bt) {
        uint32_t schedule_re0=core_re0_+bt*RE_WINDOW;
        constexpr uint32_t NG=SCHEDULE_BATCH/GB;
        constexpr uint32_t PRELOAD=(NG<MB_DEPTH)?NG:MB_DEPTH;
        for (uint32_t t=0;t<PRELOAD;++t)
            LoadRE(schedule_re0+t*GB,t);
        PipeBarrier<PIPE_ALL>();
        for (uint32_t g=0; g<NG; ++g) {
            if (g>0) {
                uint32_t future=g+MB_DEPTH-1;
                if (future<NG) {
                    LoadRE(schedule_re0+future*GB,(g-1)%MB_DEPTH);
                }
            }
            uint32_t slot=g%MB_DEPTH;
            uint32_t re0 = schedule_re0 + g*GB;
            uint32_t output_g = g%(BATCH/GB);
            uint32_t output_re0 = re0-output_g*GB;
            if (output_g==0) {
#if AIRAN_PACK && (MIMO_K > MIMO_KR)
                DataCopy(bNo_.Get<half>(),gNo_[(uint64_t)output_re0*K],BATCH*K);
#else
                DataCopy(bNo_.Get<half>(),gNo_[output_re0],BATCH);
#endif
            }
            Gram(slot);
            MatchFilterG(re0,slot);   // 必须紧跟 Gram: 复用它留在 L0A 的 Hrᵀ/Hiᵀ
            auto Areg=bAre_.Get<half>();  constexpr uint32_t NN=GB*K*K;   // 紧凑 Aim 在 Areg[NN]
            auto Id=bId_.Get<half>(); auto no_=bNo_.Get<half>(); auto xt=bCr_.Get<half>();
            // 提对角块 + 加 σ²I + 由反对称还原 Aim, 原地做后直接落到组版输入.
            //   · 对角块本身就是连续的 K*K, 不用中转暂存
            //   · Muls/Add/Transpose/Sub 同在 Vector 流水且顺序执行 -> 中间无需屏障
            auto GAr=bGAr_.Get<half>(); auto GAi=bGAi_.Get<half>();
            #pragma unroll
            for (uint32_t j=0;j<GB;++j){
                uint64_t off=(uint64_t)j*K*K;                 // 已紧凑的第 j 个对角块
                uint64_t go =(uint64_t)j*K*K;
                #if AIRAN_PACK && (MIMO_K > MIMO_KR)
                // 逐槽 σ²: P 次 Axpy, 每次只往槽 p 的对角加它自己的 σ²
                { auto IdS=bIdS_.Get<half>();
                  for (uint32_t p=0;p<P;++p)
                    Axpy(Areg[off], IdS[p*K*K], no_.GetValue((uint64_t)(output_g*GB+j)*K+p*NLR), K*K); }
#else
                Axpy(Areg[off],Id,no_.GetValue(output_g*GB+j),K*K);   // P=1: no.bin 仍是逐 unit 标量
#endif
                Transpose(xt,Areg[NN+off]);
                Sub(Areg[NN+off],Areg[NN+off],xt,K*K);        // A_im = X − Xᵀ
                DataCopy(GAr[go], Areg[off],    K*K);
                DataCopy(GAi[go], Areg[NN+off], K*K);
            }
#if AIRAN_PACK && (MIMO_K > MIMO_KR)
            // 强制块对角: 抹掉跨 RE 的互相关. 对独立 RE 是【精确】的, 不是近似.
            { PipeBarrier<PIPE_V>(); auto Mpk=bGMpk_.Get<half>(); constexpr uint32_t GB2=GB*K*K;
              Mul(GAr,GAr,Mpk,GB2); Mul(GAi,GAi,Mpk,GB2); }
#endif
            PipeBarrier<PIPE_V>();
            BlockInvG();                                      // GB 个 RE 一起做块逆
            BriG();                                           // Richardson: X ≈ A⁻¹
            DetectG(output_g);                                // x̂ / no_eff -> 64-unit staging
        if (output_g+1==BATCH/GB) {
        // ── 调度窗内每攒满 BATCH 个 unit 就统一写 GM ──
        // 每 16 unit 做一次 16x16 转置，但所有转置完成后只做一次全流水同步/写回。
        { auto SB=bAim_.Get<half>();
          auto xhB=SB; auto xhiB=SB[BATCH*K]; auto nefB=SB[2*BATCH*K];
          auto xhT=bGMbr_.Get<half>(); auto xhiT=bGMbi_.Get<half>(); auto nefT=bGEr_.Get<half>();
          #pragma unroll
          for (uint32_t q=0;q<BATCH/16;++q) {
            uint32_t bo=q*16*K;
            Transpose(xhT[bo],xhB[bo]); Transpose(xhiT[bo],xhiB[bo]); Transpose(nefT[bo],nefB[bo]);
          }
          PipeBarrier<PIPE_ALL>();
          #pragma unroll
          for (uint32_t q=0;q<BATCH/16;++q) {
            uint64_t u0q=output_re0+q*16;
            uint64_t req=u0q*P;
            uint32_t bo=q*16*K;
            #pragma unroll
            for (uint32_t p=0;p<P;++p) {
              #pragma unroll
              for (uint32_t l=0;l<NLR;++l){
              uint32_t k=p*NLR+l; uint64_t off=(uint64_t)l*N_RE+req+(uint64_t)p*16;
              DataCopy(gXr_[off],xhT[bo+k*16],16);
              DataCopy(gXi_[off],xhiT[bo+k*16],16);
              DataCopy(gNe_[off],nefT[bo+k*16],16); }
            }
          }
          PipeBarrier<PIPE_ALL>(); }
        }
        }
    }
}

extern "C" __global__ __aicore__ void mimo_detect_bri_kernel(
    GM_ADDR h_re,GM_ADDR h_im,GM_ADDR y_re,GM_ADDR y_im,GM_ADDR no,GM_ADDR mask,
    GM_ADDR xhat_re,GM_ADDR xhat_im,GM_ADDR no_eff,GM_ADDR yv_re,GM_ADDR yv_im,GM_ADDR workspace,GM_ADDR tiling)
{
    if (GetBlockIdx() >= airan::BLOCK_DIM) return;
    TPipe pipe; MimoBRI op;
    op.Init(h_re,h_im,y_re,y_im,no,mask,xhat_re,xhat_im,no_eff,yv_re,yv_im,&pipe);
    op.Run();
}
