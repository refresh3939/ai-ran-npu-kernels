


















#include "kernel_operator.h"
#include "precode_zf.h"
using namespace AscendC;
#ifndef GB
#if AIRAN_PACK && (MIMO_KR < MIMO_K)
#define GB 4
#else
#define GB 8
#endif
#endif
#ifndef ZF_EPS
#define ZF_EPS 1.0e-3f
#endif
#ifndef NEU
#define NEU 3
#endif

namespace {
using namespace airan;
constexpr uint32_t M=NT, K=NL, B=BLK;
constexpr uint32_t MB_DEPTH=2;
constexpr uint32_t MBLK = NT/16;
static_assert(GB==4 || GB==8, "当前调度只支持 GB=4/8");
static_assert(BATCH % GB == 0,        "BATCH 必须被 GB 整除");
static_assert(RE_BATCH % GB == 0,     "RE_BATCH 必须被 GB 整除");
static_assert(MB_DEPTH == 2,          "L1 ring 固定为双缓冲");
static_assert((GB*K) % 8 == 0,        "Brcb 的 repeatTimes = MG2/8 必须整除");
static_assert(GB*K <= 255,            "WholeReduceSum 的 repeatTimes 上限 255");
static_assert(2*GB*GB <= 255,         "DataCopy 的 blockLen 上限");

static_assert(2u*GB*K*M + 3u*GB*K*K <= 32768u, "L0A 超限: 减小 NT 或 GB");
static_assert(2u*GB*K*M + 2u*GB*K*K <= 32768u, "L0B 超限: 减小 NT 或 GB");
static_assert(2u*(GB*K)*(GB*K) <= 65536u, "L0C 超限: 减小 NL 或 GB");
constexpr int32_t E0=0;
}

class PrecodeZF {
public:
    __aicore__ inline PrecodeZF() {}
    __aicore__ inline void Init(GM_ADDR w_re,GM_ADDR w_im,GM_ADDR s_re,GM_ADDR s_im,
        GM_ADDR mask,GM_ADDR x_re,GM_ADDR x_im,TPipe*p);
    __aicore__ inline void Run();
private:
    __aicore__ inline void BuildConst();
    __aicore__ inline void LoadRE(uint32_t re,uint32_t slot);
    __aicore__ inline void Gram(uint32_t slot,uint32_t re);
    __aicore__ inline void BlockInvG();
    __aicore__ inline void cmmGF(const LocalTensor<half>&la0,const LocalTensor<half>&la1,const LocalTensor<half>&la2,
        const LocalTensor<half>&Br,const LocalTensor<half>&Bi,const LocalTensor<half>&Cr,const LocalTensor<half>&Ci);
    __aicore__ inline void cmmGR(const LoadData3DParamsV2Pro&pb,const MmadParams&mp,
        const LocalTensor<half>&Br,const LocalTensor<half>&Bi,
        const LocalTensor<half>&Cr,const LocalTensor<half>&Ci);
    __aicore__ inline void BriG();
    __aicore__ inline void PrecodeG(uint32_t g,uint32_t re0,uint32_t slot);
    __aicore__ inline void StoreBatch(uint32_t unit0);

    __aicore__ inline void RowBcastK(const LocalTensor<half>&dst,const LocalTensor<half>&v,
                                     const LocalTensor<half>&tmp);
    uint32_t aiv_, core_re0_;
    GlobalTensor<half> gWr_,gWi_,gSr_,gSi_,gXr_,gXi_;
    TPipe* pipe_;
    TBuf<TPosition::A2> bA2_; TBuf<TPosition::B2> bB2_; TBuf<TPosition::CO1> bCO_;
    TBuf<TPosition::VECCALC> bGAr_,bGAi_,bGMbr_,bGMbi_,bGEr_,bGEi_,bGErT_,bGEiT_;
    TBuf<TPosition::VECCALC> bGSr_,bGSi_,bGTr_,bGTi_,bGMvr_,bGMvi_,bGtmp_,bGId_,bGMblk_,bGMpack_,bGdcol_,bGdcolT_;
    TBuf<TPosition::VECCALC> bGDinv_,bDiagIdx_,bRowIdx_,bOutIdx_;
    TBuf<TPosition::A1> bL1ar_,bL1ai_,bL1br_,bL1bn_,bL1bi_;
    TBuf<TPosition::A1> bL1a0_,bL1a1_,bL1a2_,bL1a3_,bL1a4_,bL1a5_;
    TBuf<TPosition::A2> bGA2_; TBuf<TPosition::B2> bGB2_;
    TBuf<TPosition::A1> bWrL1_,bWiL1_;
    TBuf<TPosition::VECCALC> bAre_,bGT_,bT1_,bT2_;

    TBuf<TPosition::VECCALC> bWrU_,bWiU_;
    TBuf<TPosition::VECCALC> bUTr_,bUTi_;
    TBuf<TPosition::VECCALC> bURr_,bURi_;
    TBuf<TPosition::VECCALC> bUr_,bUi_,bSr_,bSi_,bRs_,bDvh_;
    TBuf<TPosition::VECCALC> bXar_,bXai_,bXtr_,bXti_,bXtmp_;
    TBuf<TPosition::VECCALC> bTmpA_,bTmpB_;
    event_t evG_,e1G_,emG_,ebG_;
};

__aicore__ inline void PrecodeZF::Init(GM_ADDR w_re,GM_ADDR w_im,GM_ADDR s_re,GM_ADDR s_im,
    GM_ADDR mask,GM_ADDR x_re,GM_ADDR x_im,TPipe*p)
{
    pipe_=p; aiv_=GetBlockIdx(); core_re0_=aiv_*RE_PER_CORE;

    gWr_.SetGlobalBuffer((__gm__ half*)w_re,(uint64_t)N_UNIT*M*K);
    gWi_.SetGlobalBuffer((__gm__ half*)w_im,(uint64_t)N_UNIT*M*K);
    gSr_.SetGlobalBuffer((__gm__ half*)s_re,(uint64_t)N_UNIT*K);
    gSi_.SetGlobalBuffer((__gm__ half*)s_im,(uint64_t)N_UNIT*K);
    gXr_.SetGlobalBuffer((__gm__ half*)x_re,(uint64_t)M*N_RE);
    gXi_.SetGlobalBuffer((__gm__ half*)x_im,(uint64_t)M*N_RE);
    evG_=static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    e1G_=static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE1));
    emG_=static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE1_M));
    ebG_=static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::M_MTE1));
    pipe_->InitBuffer(bWrL1_,MB_DEPTH*M*GB*K*sizeof(half));
    pipe_->InitBuffer(bWiL1_,MB_DEPTH*M*GB*K*sizeof(half));
    pipe_->InitBuffer(bA2_,2*GB*K*M*sizeof(half)); pipe_->InitBuffer(bB2_,2*GB*K*M*sizeof(half));

    pipe_->InitBuffer(bCO_,2*GB*K*GB*K*sizeof(float));
    { constexpr uint32_t GB2=GB*K*K, MG2=GB*K;
      pipe_->InitBuffer(bGAr_,GB2*sizeof(half));  pipe_->InitBuffer(bGAi_,GB2*sizeof(half));
      pipe_->InitBuffer(bGMbr_,GB2*sizeof(half)); pipe_->InitBuffer(bGMbi_,GB2*sizeof(half));
      pipe_->InitBuffer(bGEr_,GB2*sizeof(half));  pipe_->InitBuffer(bGEi_,GB2*sizeof(half));
      pipe_->InitBuffer(bGErT_,GB2*sizeof(half)); pipe_->InitBuffer(bGEiT_,GB2*sizeof(half));
      pipe_->InitBuffer(bGSr_,GB2*sizeof(half));  pipe_->InitBuffer(bGSi_,GB2*sizeof(half));
      pipe_->InitBuffer(bGTr_,GB2*sizeof(half));  pipe_->InitBuffer(bGTi_,GB2*sizeof(half));
      pipe_->InitBuffer(bGMvr_,GB2*sizeof(half)); pipe_->InitBuffer(bGMvi_,GB2*sizeof(half));
      pipe_->InitBuffer(bGtmp_,GB2*sizeof(half)); pipe_->InitBuffer(bGId_,GB2*sizeof(half));
      pipe_->InitBuffer(bGMblk_,GB2*sizeof(half));pipe_->InitBuffer(bGdcol_,GB2*sizeof(half));

      constexpr uint32_t MPACK_BYTES=(PACK==1)?(GB2*sizeof(uint32_t)):(GB2*sizeof(half));
      pipe_->InitBuffer(bGMpack_,MPACK_BYTES);
      pipe_->InitBuffer(bGdcolT_,GB2*sizeof(half));
      pipe_->InitBuffer(bGDinv_,MG2*sizeof(half));
      pipe_->InitBuffer(bDiagIdx_,MG2*sizeof(uint32_t));
      pipe_->InitBuffer(bRowIdx_,MG2*sizeof(uint32_t));
      pipe_->InitBuffer(bOutIdx_,MG2*sizeof(uint32_t));
      pipe_->InitBuffer(bL1ar_,GB2*sizeof(half)); pipe_->InitBuffer(bL1ai_,GB2*sizeof(half));
      pipe_->InitBuffer(bL1br_,GB2*sizeof(half)); pipe_->InitBuffer(bL1bn_,GB2*sizeof(half)); pipe_->InitBuffer(bL1bi_,GB2*sizeof(half));
      pipe_->InitBuffer(bL1a0_,GB2*sizeof(half)); pipe_->InitBuffer(bL1a1_,GB2*sizeof(half));
      pipe_->InitBuffer(bL1a2_,GB2*sizeof(half)); pipe_->InitBuffer(bL1a3_,GB2*sizeof(half));
      pipe_->InitBuffer(bL1a4_,GB2*sizeof(half)); pipe_->InitBuffer(bL1a5_,GB2*sizeof(half));
      pipe_->InitBuffer(bGA2_,3*MG2*K*sizeof(half));  pipe_->InitBuffer(bGB2_,2*MG2*K*sizeof(half)); }
    pipe_->InitBuffer(bAre_,GB*K*K*sizeof(half));
    pipe_->InitBuffer(bGT_,4*GB*K*K*sizeof(half));
    pipe_->InitBuffer(bT1_,K*K*sizeof(half)); pipe_->InitBuffer(bT2_,K*K*sizeof(half));

    { constexpr uint32_t WU=MBLK*GB*16*K, GB2=GB*K*K, MG2=GB*K, BN=BATCH*M;
      constexpr uint32_t WUS=(PACK==1)?1:WU, URS=(PACK==1)?(M*K):WU, TMS=(PACK==1)?GB2:WU;
      pipe_->InitBuffer(bWrU_,WUS*sizeof(half));  pipe_->InitBuffer(bWiU_,WUS*sizeof(half));
      pipe_->InitBuffer(bUTr_,GB2*sizeof(half)); pipe_->InitBuffer(bUTi_,GB2*sizeof(half));
      pipe_->InitBuffer(bURr_,URS*sizeof(half));  pipe_->InitBuffer(bURi_,URS*sizeof(half));
      constexpr uint32_t UVS=MG2+((PACK==1)?16:0);
      pipe_->InitBuffer(bUr_,UVS*sizeof(half));  pipe_->InitBuffer(bUi_,UVS*sizeof(half));
      pipe_->InitBuffer(bSr_,MG2*sizeof(half));  pipe_->InitBuffer(bSi_,MG2*sizeof(half));
      pipe_->InitBuffer(bRs_,MG2*sizeof(half));  pipe_->InitBuffer(bDvh_,MG2*sizeof(half));
      pipe_->InitBuffer(bXar_,PACK*BN*sizeof(half));  pipe_->InitBuffer(bXai_,PACK*BN*sizeof(half));
      pipe_->InitBuffer(bXtr_,PACK*BN*sizeof(half));  pipe_->InitBuffer(bXti_,PACK*BN*sizeof(half));
      pipe_->InitBuffer(bXtmp_,2*BN*sizeof(half));
      pipe_->InitBuffer(bTmpA_,TMS*sizeof(half)); pipe_->InitBuffer(bTmpB_,TMS*sizeof(half)); }
}


__aicore__ inline void PrecodeZF::BuildConst()
{
    { constexpr uint8_t padList[4]={0,0,0,0}; Load3DSetFMatrixCal(1, M, padList); }
    { constexpr uint32_t GB2=GB*K*K;
      auto IdG=bGId_.Get<half>(); auto MblkG=bGMblk_.Get<half>();
      Duplicate(IdG,(half)0,GB2); Duplicate(MblkG,(half)0,GB2);
      PipeBarrier<PIPE_ALL>();
      for (uint32_t g=0;g<GB;++g){ uint32_t o=g*K*K;
        for (uint32_t i=0;i<K;++i){ IdG.SetValue(o+i*K+i,(half)1.0);
          for (uint32_t j=0;j<K;++j){
            if ((i/B)==(j/B)) MblkG.SetValue(o+i*K+j,(half)1.0);
          } } }
      PipeBarrier<PIPE_ALL>(); }
    if (PACK>1) {
      constexpr uint32_t GB2=GB*K*K;
      auto MpackG=bGMpack_.Get<half>(); Duplicate(MpackG,(half)0,GB2); PipeBarrier<PIPE_V>();
      for (uint32_t g=0;g<GB;++g) for (uint32_t i=0;i<K;++i) for (uint32_t j=0;j<K;++j)
        if ((i/NLR)==(j/NLR)) MpackG.SetValue(g*K*K+i*K+j,(half)1.0);
      PipeBarrier<PIPE_ALL>();
    } else {
      constexpr uint32_t GB2=GB*K*K, MG2=GB*K;
      auto idx=bGMpack_.Get<uint32_t>();
      Duplicate(idx,(uint32_t)(MG2*sizeof(half)),GB2); PipeBarrier<PIPE_V>();
      for (uint32_t g=0;g<GB;++g) for (uint32_t i=0;i<K;++i)
        idx.SetValue(g*K*K+i*K+g,(g*K+i)*sizeof(half));
      Duplicate(bUr_.Get<half>()[MG2],(half)0,16);
      Duplicate(bUi_.Get<half>()[MG2],(half)0,16); PipeBarrier<PIPE_ALL>();
    }

    Duplicate(bT1_.Get<half>(),(half)0,K*K);
    Duplicate(bT2_.Get<half>(),(half)0,K*K); PipeBarrier<PIPE_ALL>();

    { auto idx=bDiagIdx_.Get<uint32_t>();
      for (uint32_t g=0;g<GB;++g) for (uint32_t i=0;i<K;++i)
          idx.SetValue(g*K+i,(g*K*K+i*K+i)*sizeof(half));
      PipeBarrier<PIPE_ALL>(); }

    { auto idx=bRowIdx_.Get<uint32_t>();
      for (uint32_t g=0;g<GB;++g) for (uint32_t i=0;i<K;++i)
          idx.SetValue(g*K+i,(g*K*K+i*K+g)*sizeof(half));
      PipeBarrier<PIPE_ALL>(); }

    { auto idx=bOutIdx_.Get<uint32_t>();
      for (uint32_t g=0;g<GB;++g) for (uint32_t i=0;i<K;++i)
          idx.SetValue(g*K+i,(i*K+g)*sizeof(half));
      PipeBarrier<PIPE_ALL>(); }
}


__aicore__ inline void PrecodeZF::LoadRE(uint32_t re0,uint32_t slot)
{
    constexpr uint32_t WS=GB*M*K;
    auto WrL1=bWrL1_.Get<half>()[slot*WS]; auto WiL1=bWiL1_.Get<half>()[slot*WS];
    DataCopy(WrL1, gWr_[(uint64_t)re0*M*K], GB*M*K);
    DataCopy(WiL1, gWi_[(uint64_t)re0*M*K], GB*M*K);
}


__aicore__ inline void PrecodeZF::Gram(uint32_t slot,uint32_t re0)
{
    constexpr uint32_t MG = GB*K, OPS = GB*K*M;
    constexpr uint64_t EXTCFG = ((uint64_t)0<<48)|((uint64_t)0<<32)|((uint64_t)M<<16)|(uint64_t)MG;
    auto WrL1=bWrL1_.Get<half>()[slot*OPS]; auto WiL1=bWiL1_.Get<half>()[slot*OPS];
    auto WrU=bWrU_.Get<half>(); auto WiU=bWiU_.Get<half>();
    auto Are=bGAr_.Get<half>(); auto Aim=bGAi_.Get<half>();
    auto a2=bA2_.Get<half>(); auto b2=bB2_.Get<half>();

    { constexpr uint8_t padList[4]={0,0,0,0}; Load3DSetFMatrixCal(1, M, padList); }
    LoadData3DParamsV2Pro pa; pa.channelSize=MG; pa.extConfig=EXTCFG; pa.enTranspose=true;
    LoadData3DParamsV2Pro pb; pb.channelSize=MG; pb.extConfig=EXTCFG;
    LoadData<half>(a2,       WrL1, pa);
    LoadData<half>(a2[OPS],  WiL1, pa);
    LoadData<half>(b2,       WrL1, pb);
    LoadData<half>(b2[OPS],  WiL1, pb);
    SetFlag<HardEvent::MTE1_M>(E0); WaitFlag<HardEvent::MTE1_M>(E0);

    constexpr uint32_t NN = MG*MG, GBKK=GB*K*K;
    auto co=bCO_.Get<float>();
    MmadParams mp; mp.m=MG; mp.n=MG; mp.k=M;
    mp.cmatrixInitVal=true;  Mmad(co,     a2,      b2,      mp);
    mp.cmatrixInitVal=false; Mmad(co,     a2[OPS], b2[OPS], mp);
    mp.cmatrixInitVal=true;  Mmad(co[NN], a2,      b2[OPS], mp);

    if (PACK>1) {
        DataCopy(WrU,gWr_[(uint64_t)re0*M*K],GB*M*K);
        DataCopy(WiU,gWi_[(uint64_t)re0*M*K],GB*M*K);
    }
    PipeBarrier<PIPE_ALL>();

    { DataCopyParams dcp; dcp.blockCount=(uint16_t)GB; dcp.blockLen=1;
      dcp.srcStride=(uint16_t)GB; dcp.dstStride=0;
      DataCopyEnhancedParams ep; ep.blockMode=BlockMode::BLOCK_MODE_MATRIX;
      DataCopy(Are,co,dcp,ep);
      DataCopy(Aim,co[NN],dcp,ep); }
    PipeBarrier<PIPE_ALL>();

    if (PACK > 1) {
        auto MpackG=bGMpack_.Get<half>();
        Mul(Are,Are,MpackG,GBKK);
        Mul(Aim,Aim,MpackG,GBKK);
        PipeBarrier<PIPE_V>();
    }
}


__aicore__ inline void PrecodeZF::RowBcastK(const LocalTensor<half>&dst,const LocalTensor<half>&v,
                                            const LocalTensor<half>&tmp)
{
    constexpr uint32_t MG2=GB*K;
    Brcb(tmp, v, MG2/8, {1,8});
    PipeBarrier<PIPE_V>();
    for (uint32_t j=0;j<GB;++j) Transpose(dst[j*K*K], tmp[j*K*K]);
    PipeBarrier<PIPE_V>();
}



__aicore__ inline void PrecodeZF::BlockInvG()
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

    Mul(Mbr,Are,MblkG,GB2); Mul(Mbi,Aim,MblkG,GB2); PipeBarrier<PIPE_V>();
    Gather(Dinv,Mbr,bDiagIdx_.Get<uint32_t>(),(uint32_t)0,MG2); PipeBarrier<PIPE_V>();

    Maxs(Dinv,Dinv,(half)DET_EPS,MG2); PipeBarrier<PIPE_V>();
    Reciprocal(Dinv,Dinv,MG2); PipeBarrier<PIPE_V>();
    Brcb(dcol,Dinv,MG2/8,{1,8}); PipeBarrier<PIPE_V>();
    Mul(Er,Mbr,dcol,GB2); Mul(Ei,Mbi,dcol,GB2); PipeBarrier<PIPE_V>();
    Sub(Er,IdG,Er,GB2);
    Muls(Ei,Ei,(half)-1.0,GB2); PipeBarrier<PIPE_V>();

    #pragma unroll
    for (uint32_t j=0;j<GB;++j){ Transpose(ErT[j*K*K],Er[j*K*K]); Transpose(EiT[j*K*K],Ei[j*K*K]); }
    PipeBarrier<PIPE_V>();
    { constexpr uint32_t MG2b=GB*K, OP=MG2b*K;
      auto nAi=bGtmp_.Get<half>();
      Muls(nAi,EiT,(half)-1.0,GB2); PipeBarrier<PIPE_V>();
      SetFlag<HardEvent::V_MTE3>(evG_);WaitFlag<HardEvent::V_MTE3>(evG_);
      DataCopy(bL1ar_.Get<half>(), ErT, GB2);
      DataCopy(bL1ai_.Get<half>(), EiT, GB2);
      DataCopy(bL1bn_.Get<half>(), nAi, GB2);
      SetFlag<HardEvent::MTE3_MTE1>(e1G_);WaitFlag<HardEvent::MTE3_MTE1>(e1G_);
      { constexpr uint8_t padList[4]={0,0,0,0}; Load3DSetFMatrixCal(1, K, padList); }
      LoadData3DParamsV2Pro pa; pa.channelSize=MG2b; pa.enTranspose=true;
      pa.extConfig=((uint64_t)0<<48)|((uint64_t)0<<32)|((uint64_t)K<<16)|(uint64_t)MG2b;
      auto a2=bGA2_.Get<half>();
      LoadData<half>(a2,       bL1ar_.Get<half>(), pa);
      LoadData<half>(a2[OP],   bL1ai_.Get<half>(), pa);
      LoadData<half>(a2[2*OP], bL1bn_.Get<half>(), pa); }



    Add(Sr,IdG,Er,GB2); DataCopy(Si,Ei,GB2); PipeBarrier<PIPE_ALL>();
    {
      constexpr uint32_t MG2b=GB*K, NN2=MG2b*MG2b, OP=MG2b*K;
      auto l1br=bL1br_.Get<half>(); auto l1bi=bL1bi_.Get<half>();
      auto a2=bGA2_.Get<half>();    auto b2=bGB2_.Get<half>();
      auto co=bCO_.Get<float>();
      LoadData3DParamsV2Pro pb; pb.channelSize=MG2b;
      pb.extConfig=((uint64_t)0<<48)|((uint64_t)0<<32)|((uint64_t)K<<16)|(uint64_t)MG2b;
      MmadParams mp; mp.m=MG2b; mp.n=MG2b; mp.k=K;
      #pragma unroll
      for (uint32_t it=1; it<NEU; ++it){
        SetFlag<HardEvent::V_MTE3>(evG_); WaitFlag<HardEvent::V_MTE3>(evG_);
        DataCopy(l1br, Sr, GB2); DataCopy(l1bi, Si, GB2);
        SetFlag<HardEvent::MTE3_MTE1>(e1G_); WaitFlag<HardEvent::MTE3_MTE1>(e1G_);
        LoadData<half>(b2, l1br, pb); LoadData<half>(b2[OP], l1bi, pb);
        SetFlag<HardEvent::MTE1_M>(emG_); WaitFlag<HardEvent::MTE1_M>(emG_);
        mp.cmatrixInitVal=true;  Mmad(co,      a2,        b2,      mp);
        mp.cmatrixInitVal=false; Mmad(co,      a2[2*OP],  b2[OP],  mp);
        mp.cmatrixInitVal=true;  Mmad(co[NN2], a2,        b2[OP],  mp);
        mp.cmatrixInitVal=false; Mmad(co[NN2], a2[OP],    b2,      mp);
        SetFlag<HardEvent::M_MTE1>(ebG_); WaitFlag<HardEvent::M_MTE1>(ebG_);
        PipeBarrier<PIPE_ALL>();

        { DataCopyParams dcp2; dcp2.blockCount=(uint16_t)GB; dcp2.blockLen=1;
          dcp2.srcStride=(uint16_t)GB; dcp2.dstStride=0;
          DataCopyEnhancedParams ep2; ep2.blockMode=BlockMode::BLOCK_MODE_MATRIX;
          DataCopy(Tr, co,      dcp2, ep2);
          DataCopy(Si, co[NN2], dcp2, ep2); }
        PipeBarrier<PIPE_ALL>();
        Add(Sr,Tr,IdG,GB2); PipeBarrier<PIPE_V>();
      }
    }
    #pragma unroll
    for (uint32_t j=0;j<GB;++j) Transpose(dcolT[j*K*K],dcol[j*K*K]);
    PipeBarrier<PIPE_V>();
    Mul(Mvr,Sr,dcolT,GB2); Mul(Mvi,Si,dcolT,GB2);
    PipeBarrier<PIPE_V>();
}


__aicore__ inline void PrecodeZF::cmmGF(const LocalTensor<half>&la0,const LocalTensor<half>&la1,const LocalTensor<half>&la2,
    const LocalTensor<half>&Br,const LocalTensor<half>&Bi,const LocalTensor<half>&Cr,const LocalTensor<half>&Ci)
{
    constexpr uint32_t GB2=GB*K*K, MG2=GB*K, NN2=MG2*MG2, OP=MG2*K;
    auto l1br=bL1br_.Get<half>(); auto l1bi=bL1bi_.Get<half>();
    auto a2=bGA2_.Get<half>(); auto b2=bGB2_.Get<half>();
    auto co=bCO_.Get<float>();
    SetFlag<HardEvent::V_MTE3>(evG_);WaitFlag<HardEvent::V_MTE3>(evG_);
    DataCopy(l1br,Br,GB2); DataCopy(l1bi,Bi,GB2);
    SetFlag<HardEvent::MTE3_MTE1>(e1G_);WaitFlag<HardEvent::MTE3_MTE1>(e1G_);
    LoadData3DParamsV2Pro pa; pa.channelSize=MG2; pa.enTranspose=true;
    pa.extConfig=((uint64_t)0<<48)|((uint64_t)0<<32)|((uint64_t)K<<16)|(uint64_t)MG2;
    LoadData3DParamsV2Pro pb; pb.channelSize=MG2; pb.extConfig=pa.extConfig;
    LoadData<half>(a2,       la0, pa);
    LoadData<half>(a2[OP],   la1, pa);
    LoadData<half>(a2[2*OP], la2, pa);
    LoadData<half>(b2,       l1br, pb);
    LoadData<half>(b2[OP],   l1bi, pb);
    SetFlag<HardEvent::MTE1_M>(emG_); WaitFlag<HardEvent::MTE1_M>(emG_);
    MmadParams mp; mp.m=MG2; mp.n=MG2; mp.k=K;
    mp.cmatrixInitVal=true;  Mmad(co,      a2,        b2,      mp);
    mp.cmatrixInitVal=false; Mmad(co,      a2[2*OP],  b2[OP],  mp);
    mp.cmatrixInitVal=true;  Mmad(co[NN2], a2,        b2[OP],  mp);
    mp.cmatrixInitVal=false; Mmad(co[NN2], a2[OP],    b2,      mp);
    SetFlag<HardEvent::M_MTE1>(ebG_); WaitFlag<HardEvent::M_MTE1>(ebG_);
    PipeBarrier<PIPE_ALL>();

    { DataCopyParams dcp; dcp.blockCount=(uint16_t)GB; dcp.blockLen=1;
      dcp.srcStride=(uint16_t)GB; dcp.dstStride=0;
      DataCopyEnhancedParams ep; ep.blockMode=BlockMode::BLOCK_MODE_MATRIX;
      DataCopy(Cr, co,      dcp, ep);
      DataCopy(Ci, co[NN2], dcp, ep); }
    PipeBarrier<PIPE_ALL>();
}


__aicore__ inline void PrecodeZF::cmmGR(const LoadData3DParamsV2Pro&pb,const MmadParams&mp0,
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
      DataCopy(Cr, co,       dcp, ep);
      DataCopy(Ci, co[NN2],  dcp, ep); }
    PipeBarrier<PIPE_ALL>();
}


__aicore__ inline void PrecodeZF::BriG()
{
    constexpr uint32_t GB2=GB*K*K;
    auto Ar=bGAr_.Get<half>();  auto Ai=bGAi_.Get<half>();
    auto Mr=bGMvr_.Get<half>(); auto Mi=bGMvi_.Get<half>();
    auto Br_=bGMbr_.Get<half>();auto Bi_=bGMbi_.Get<half>();
    auto PrT=bGEr_.Get<half>(); auto PiT=bGEi_.Get<half>();
    auto nPiT=bGEiT_.Get<half>();
    auto GT=bGT_.Get<half>();
    auto Xr=GT;          auto Xi=GT[GB2];
    auto Tr=bGTr_.Get<half>();  auto Ti=bGTi_.Get<half>();

    { constexpr uint8_t padList[4]={0,0,0,0}; Load3DSetFMatrixCal(1, K, padList); }

    Muls(nPiT,Mi,(half)-1.0,GB2); PipeBarrier<PIPE_V>();
    { SetFlag<HardEvent::V_MTE3>(evG_);WaitFlag<HardEvent::V_MTE3>(evG_);
      DataCopy(bL1a0_.Get<half>(),Mr,GB2); DataCopy(bL1a1_.Get<half>(),nPiT,GB2); DataCopy(bL1a2_.Get<half>(),Mi,GB2);
      SetFlag<HardEvent::MTE3_MTE1>(e1G_);WaitFlag<HardEvent::MTE3_MTE1>(e1G_); }
    cmmGF(bL1a0_.Get<half>(),bL1a1_.Get<half>(),bL1a2_.Get<half>(), Ar,Ai, Br_,Bi_);



    Sub(Br_,bGId_.Get<half>(),Br_,GB2);
    Muls(Bi_,Bi_,(half)-1.0,GB2); PipeBarrier<PIPE_V>();
    #pragma unroll
    for (uint32_t j=0;j<GB;++j){ uint32_t o=j*K*K; Transpose(PrT[o],Br_[o]); Transpose(PiT[o],Bi_[o]); }
    PipeBarrier<PIPE_V>();
    Muls(nPiT,PiT,(half)-1.0,GB2); PipeBarrier<PIPE_V>();
    { SetFlag<HardEvent::V_MTE3>(evG_);WaitFlag<HardEvent::V_MTE3>(evG_);
      DataCopy(bL1a3_.Get<half>(),PrT,GB2); DataCopy(bL1a4_.Get<half>(),PiT,GB2); DataCopy(bL1a5_.Get<half>(),nPiT,GB2);
      SetFlag<HardEvent::MTE3_MTE1>(e1G_);WaitFlag<HardEvent::MTE3_MTE1>(e1G_);
      constexpr uint32_t MG2b=GB*K, OPb=MG2b*K;
      LoadData3DParamsV2Pro pa; pa.channelSize=MG2b; pa.enTranspose=true;
      pa.extConfig=((uint64_t)0<<48)|((uint64_t)0<<32)|((uint64_t)K<<16)|(uint64_t)MG2b;
      auto a2b=bGA2_.Get<half>();
      LoadData<half>(a2b,        bL1a3_.Get<half>(), pa);
      LoadData<half>(a2b[OPb],   bL1a4_.Get<half>(), pa);
      LoadData<half>(a2b[2*OPb], bL1a5_.Get<half>(), pa); }

    DataCopy(Xr,Mr,GB2); DataCopy(Xi,Mi,GB2); PipeBarrier<PIPE_ALL>();

    LoadData3DParamsV2Pro pbR; pbR.channelSize=GB*K;
    pbR.extConfig=((uint64_t)0<<48)|((uint64_t)0<<32)|((uint64_t)K<<16)|(uint64_t)(GB*K);
    MmadParams mpR; mpR.m=GB*K; mpR.n=GB*K; mpR.k=K;
    #pragma unroll
    for (uint32_t it=0; it<NLAY; ++it){
        cmmGR(pbR,mpR, Xr,Xi, Tr,Ti);
        Add(Xr,Tr,Mr,GB2); Add(Xi,Ti,Mi,GB2); PipeBarrier<PIPE_V>();
    }
}



__aicore__ inline void PrecodeZF::PrecodeG(uint32_t g,uint32_t re0,uint32_t slot)
{
    constexpr uint32_t GB2=GB*K*K, MG2=GB*K, BN=BATCH*M;
    auto GT=bGT_.Get<half>();
    auto Xr=GT;  auto Xi=GT[GB2];
    auto tmp=bGtmp_.Get<half>();  auto t2=bGTr_.Get<half>();  auto t3=bGTi_.Get<half>();
    auto dvh=bDvh_.Get<half>();   auto rs=bRs_.Get<half>();
    auto sr=bSr_.Get<half>();     auto si=bSi_.Get<half>();
    auto ur=bUr_.Get<half>();     auto ui=bUi_.Get<half>();
    auto UTr=bUTr_.Get<half>();   auto UTi=bUTi_.Get<half>();

    if (PACK != 0) {

    Gather(dvh,Xr,bDiagIdx_.Get<uint32_t>(),(uint32_t)0,MG2); PipeBarrier<PIPE_V>();

    Maxs(dvh,dvh,(half)DET_EPS,MG2); PipeBarrier<PIPE_V>();
    Rsqrt(rs,dvh,MG2); PipeBarrier<PIPE_V>();

    DataCopy(sr, gSr_[(uint64_t)re0*K], MG2);
    DataCopy(si, gSi_[(uint64_t)re0*K], MG2);
    PipeBarrier<PIPE_ALL>();
    Mul(sr,sr,rs,MG2); Mul(si,si,rs,MG2); PipeBarrier<PIPE_V>();




    { constexpr uint32_t KK=K*K;
      auto rhsR=bT1_.Get<half>(); auto rhsI=bT2_.Get<half>();
      DataCopy(rhsR,sr,MG2); DataCopy(rhsI,si,MG2); PipeBarrier<PIPE_ALL>();
      Transpose(UTr,rhsR); Transpose(UTi,rhsI); PipeBarrier<PIPE_V>();


      auto nUr=tmp;
      Muls(nUr,UTr,(half)-1.0,KK); PipeBarrier<PIPE_V>();
      SetFlag<HardEvent::V_MTE3>(evG_); WaitFlag<HardEvent::V_MTE3>(evG_);
      DataCopy(bL1a0_.Get<half>(),Xr,GB2);
      DataCopy(bL1a1_.Get<half>(),Xi,GB2);
      DataCopy(bL1br_.Get<half>(),UTr,KK);
      DataCopy(bL1bi_.Get<half>(),UTi,KK);
      DataCopy(bL1bn_.Get<half>(),nUr,KK);
      SetFlag<HardEvent::MTE3_MTE1>(e1G_); WaitFlag<HardEvent::MTE3_MTE1>(e1G_);
      auto a2=bGA2_.Get<half>(); auto b2=bGB2_.Get<half>(); auto co=bCO_.Get<float>();
      { constexpr uint8_t padList[4]={0,0,0,0};
        Load3DSetFMatrixCal(1,K,padList);
        LoadData3DParamsV2Pro pa; pa.channelSize=MG2; pa.enTranspose=true;
        pa.extConfig=((uint64_t)0<<48)|((uint64_t)0<<32)|((uint64_t)K<<16)|(uint64_t)MG2;
        LoadData<half>(a2,bL1a0_.Get<half>(),pa);
        LoadData<half>(a2[GB2],bL1a1_.Get<half>(),pa);
        Load3DSetFMatrixCal(1,K,padList);
        LoadData3DParamsV2Pro pb; pb.channelSize=K;
        pb.extConfig=((uint64_t)0<<48)|((uint64_t)0<<32)|((uint64_t)K<<16)|(uint64_t)K;
        LoadData<half>(b2,bL1br_.Get<half>(),pb);
        LoadData<half>(b2[KK],bL1bi_.Get<half>(),pb);
        LoadData<half>(b2[2*KK],bL1bn_.Get<half>(),pb); }
      SetFlag<HardEvent::MTE1_M>(emG_); WaitFlag<HardEvent::MTE1_M>(emG_);
      MmadParams mp; mp.m=MG2; mp.n=K; mp.k=K;
      mp.cmatrixInitVal=true;  Mmad(co,a2,b2,mp);
      mp.cmatrixInitVal=false; Mmad(co,a2[GB2],b2[KK],mp);
      mp.cmatrixInitVal=true;  Mmad(co[GB2],a2,b2[KK],mp);
      mp.cmatrixInitVal=false; Mmad(co[GB2],a2[GB2],b2[2*KK],mp);
      SetFlag<HardEvent::M_MTE1>(ebG_); WaitFlag<HardEvent::M_MTE1>(ebG_);
      PipeBarrier<PIPE_ALL>();
      { DataCopyParams dcp; dcp.blockCount=(uint16_t)GB; dcp.blockLen=1;
        dcp.srcStride=0; dcp.dstStride=0;
        DataCopyEnhancedParams ep; ep.blockMode=BlockMode::BLOCK_MODE_MATRIX;
        DataCopy(t2,co,dcp,ep); DataCopy(t3,co[GB2],dcp,ep); }
      PipeBarrier<PIPE_ALL>(); }
    Gather(ur,t2,bRowIdx_.Get<uint32_t>(),(uint32_t)0,MG2);
    Gather(ui,t3,bRowIdx_.Get<uint32_t>(),(uint32_t)0,MG2); PipeBarrier<PIPE_V>();
    }


    { auto Xsr=bXar_.Get<half>(); auto Xsi=bXai_.Get<half>();
      auto ta=bTmpA_.Get<half>(); auto tb=bTmpB_.Get<half>();
      auto Ur2=bURr_.Get<half>(); auto Ui2=bURi_.Get<half>();
      constexpr uint32_t WU=GB*M*K;
      if (PACK==1) {


        auto rhsIdx=bGMpack_.Get<uint32_t>();
        Gather(UTr,ur,rhsIdx,(uint32_t)0,GB2);
        Gather(UTi,ui,rhsIdx,(uint32_t)0,GB2); PipeBarrier<PIPE_V>();
        auto WrL1=bWrL1_.Get<half>()[slot*WU]; auto WiL1=bWiL1_.Get<half>()[slot*WU];
        auto a2=bA2_.Get<half>(); auto b2=bB2_.Get<half>(); auto co=bCO_.Get<float>();
        auto l1ur=bL1br_.Get<half>(); auto l1ui=bL1bi_.Get<half>();
        SetFlag<HardEvent::V_MTE3>(evG_); WaitFlag<HardEvent::V_MTE3>(evG_);
        DataCopy(l1ur,UTr,GB2); DataCopy(l1ui,UTi,GB2);
        SetFlag<HardEvent::MTE3_MTE1>(e1G_); WaitFlag<HardEvent::MTE3_MTE1>(e1G_);
        { constexpr uint8_t padList[4]={0,0,0,0};
          Load3DSetFMatrixCal(1,M,padList);
          LoadData3DParamsV2Pro pa; pa.channelSize=MG2;
          pa.extConfig=((uint64_t)0<<48)|((uint64_t)0<<32)|((uint64_t)M<<16)|(uint64_t)MG2;
          LoadData<half>(a2,WrL1,pa); LoadData<half>(a2[WU],WiL1,pa);
          Load3DSetFMatrixCal(1,MG2,padList);
          LoadData3DParamsV2Pro pb; pb.channelSize=K;
          pb.extConfig=((uint64_t)0<<48)|((uint64_t)0<<32)|((uint64_t)MG2<<16)|(uint64_t)K;
          LoadData<half>(b2,l1ur,pb); LoadData<half>(b2[GB2],l1ui,pb);
        }
        SetFlag<HardEvent::MTE1_M>(emG_); WaitFlag<HardEvent::MTE1_M>(emG_);
        constexpr uint32_t CN=M*K;
        MmadParams mp; mp.m=M; mp.n=K; mp.k=MG2;
        mp.cmatrixInitVal=true;  Mmad(co,a2,b2,mp);
        mp.cmatrixInitVal=false; Mmad(co,a2[WU],b2[GB2],mp);
        mp.cmatrixInitVal=true;  Mmad(co[CN],a2,b2[GB2],mp);
        mp.cmatrixInitVal=true;  Mmad(co[2*CN],a2[WU],b2,mp);
        SetFlag<HardEvent::M_MTE1>(ebG_); WaitFlag<HardEvent::M_MTE1>(ebG_);
        PipeBarrier<PIPE_ALL>();
        { DataCopyParams dcp; dcp.blockCount=(uint16_t)MBLK; dcp.blockLen=1;
          dcp.srcStride=0; dcp.dstStride=0;
          DataCopyEnhancedParams ep; ep.blockMode=BlockMode::BLOCK_MODE_MATRIX;
          DataCopy(Ur2,co,dcp,ep); DataCopy(Ui2,co[CN],dcp,ep); DataCopy(ta,co[2*CN],dcp,ep); }
        PipeBarrier<PIPE_ALL>();
        Sub(Ui2,Ui2,ta,CN); PipeBarrier<PIPE_V>();

        constexpr uint32_t KK=K*K;
        auto outIdx=bOutIdx_.Get<uint32_t>();
        #pragma unroll
        for (uint32_t mb=0;mb<MBLK;++mb) {
          const uint32_t dst=mb*BATCH*K+g*GB*K;
          Gather(Xsr[dst],Ur2[mb*KK],outIdx,(uint32_t)0,GB*K);
          Gather(Xsi[dst],Ui2[mb*KK],outIdx,(uint32_t)0,GB*K);
        }
        PipeBarrier<PIPE_V>();
      } else {
        auto WrU=bWrU_.Get<half>(); auto WiU=bWiU_.Get<half>();
        auto bc=bGdcol_.Get<half>(); auto bctmp=bGdcolT_.Get<half>();
        RowBcastK(bc,ur,bctmp);
        for (uint32_t j=0;j<GB;++j) for (uint32_t r=0;r<M/K;++r)
            DataCopy(Ur2[j*M*K+r*K*K],bc[j*K*K],K*K);
        RowBcastK(bc,ui,bctmp);
        for (uint32_t j=0;j<GB;++j) for (uint32_t r=0;r<M/K;++r)
            DataCopy(Ui2[j*M*K+r*K*K],bc[j*K*K],K*K);
        PipeBarrier<PIPE_ALL>();
        Mul(ta,WrU,Ur2,WU); Mul(tb,WiU,Ui2,WU); PipeBarrier<PIPE_V>();
        Add(ta,ta,tb,WU); PipeBarrier<PIPE_V>();
        for (uint32_t j0=0;j0<GB;j0+=3) { const uint32_t nr=(j0+3<=GB)?3:(GB-j0);
          for (uint32_t p=0;p<PACK;++p) WholeReduceSum<half>(tmp[p*GB*M+j0*M],ta[j0*M*K+p*NLR],NLR,nr*M,1,1,1); }
        PipeBarrier<PIPE_V>();
        for (uint32_t j=0;j<GB;++j) for (uint32_t p=0;p<PACK;++p) { DataCopyParams dcs;
          dcs.blockCount=(uint16_t)MBLK; dcs.blockLen=1; dcs.srcStride=0; dcs.dstStride=(uint16_t)(BATCH-1);
          DataCopy(Xsr[p*BN+(g*GB+j)*16],tmp[p*GB*M+j*M],dcs); }
        PipeBarrier<PIPE_ALL>();
        Mul(ta,WrU,Ui2,WU); Mul(tb,WiU,Ur2,WU); PipeBarrier<PIPE_V>();
        Sub(ta,ta,tb,WU); PipeBarrier<PIPE_V>();
        for (uint32_t j0=0;j0<GB;j0+=3) { const uint32_t nr=(j0+3<=GB)?3:(GB-j0);
          for (uint32_t p=0;p<PACK;++p) WholeReduceSum<half>(tmp[p*GB*M+j0*M],ta[j0*M*K+p*NLR],NLR,nr*M,1,1,1); }
        PipeBarrier<PIPE_V>();
        for (uint32_t j=0;j<GB;++j) for (uint32_t p=0;p<PACK;++p) { DataCopyParams dcs;
          dcs.blockCount=(uint16_t)MBLK; dcs.blockLen=1; dcs.srcStride=0; dcs.dstStride=(uint16_t)(BATCH-1);
          DataCopy(Xsi[p*BN+(g*GB+j)*16],tmp[p*GB*M+j*M],dcs); }
        PipeBarrier<PIPE_ALL>();
      } }
}


__aicore__ inline void PrecodeZF::StoreBatch(uint32_t unit0)
{
    auto Xsr=bXar_.Get<half>(); auto Xsi=bXai_.Get<half>();
    auto Xtr=bXtr_.Get<half>(); auto Xti=bXti_.Get<half>();
    auto Xtmp=bXtmp_.Get<half>();
    constexpr uint32_t BN=BATCH*M, PB=PACK*BATCH;
    const uint64_t re_base=(uint64_t)unit0*PACK;
    constexpr uint32_t KK=K*K, NQ=BATCH/K;
    #pragma unroll
    for (uint32_t p=0;p<PACK;++p){
        #pragma unroll
        for (uint32_t q=0;q<NQ;++q) {
          #pragma unroll
          for (uint32_t mb=0;mb<MBLK;++mb){
            const uint32_t to=(q*MBLK+mb)*KK;
            const uint32_t so=p*BN+mb*BATCH*K+q*KK;
            Transpose(Xtmp[to],   Xsr[so]);
            Transpose(Xtmp[BN+to],Xsi[so]);
          }
        }
        PipeBarrier<PIPE_V>();
        DataCopyParams dcr; dcr.blockCount=(uint16_t)K; dcr.blockLen=1;
        dcr.srcStride=0; dcr.dstStride=(uint16_t)(PB/K-1);
        #pragma unroll
        for (uint32_t q=0;q<NQ;++q) {
          #pragma unroll
          for (uint32_t mb=0;mb<MBLK;++mb){
            const uint32_t so=(q*MBLK+mb)*KK;
            const uint32_t to=mb*K*PB+p*BATCH+q*K;
            DataCopy(Xtr[to],Xtmp[so],dcr);
            DataCopy(Xti[to],Xtmp[BN+so],dcr);
          }
        }
    }
    PipeBarrier<PIPE_ALL>();
    #pragma unroll
    for (uint32_t m=0;m<M;++m){
        const uint64_t off=(uint64_t)m*N_RE+re_base;
        DataCopy(gXr_[off],Xtr[m*PB],PB);
        DataCopy(gXi_[off],Xti[m*PB],PB);
    }
    PipeBarrier<PIPE_ALL>();
}


__aicore__ inline void PrecodeZF::Run()
{
    BuildConst();
    for (uint32_t bt=0; bt<N_BATCH_PER_CORE; ++bt) {
        const uint32_t batch_re0=core_re0_+bt*RE_BATCH;
        constexpr uint32_t NG=RE_BATCH/GB;
        constexpr uint32_t PRELOAD=(NG<MB_DEPTH)?NG:MB_DEPTH;
        for (uint32_t t=0;t<PRELOAD;++t) LoadRE(batch_re0+t*GB,t);
        PipeBarrier<PIPE_ALL>();
        for (uint32_t g=0;g<NG;++g) {
            if (g>0) {
                const uint32_t future=g+MB_DEPTH-1;
                if (future<NG) LoadRE(batch_re0+future*GB,(g-1)%MB_DEPTH);
            }
            const uint32_t slot=g%MB_DEPTH;
            const uint32_t re0=batch_re0+g*GB;
            {
                Gram(slot,re0);
                constexpr uint32_t GBKK=GB*K*K;
                auto Areg=bAre_.Get<half>(); auto IdG=bGId_.Get<half>();
                auto GAr=bGAr_.Get<half>(); auto GAi=bGAi_.Get<half>();

                Axpy(GAr,IdG,(half)ZF_EPS,GBKK);
                #pragma unroll
                for (uint32_t j=0;j<GB;++j){
                    uint64_t off=(uint64_t)j*K*K;
                    Transpose(Areg[off],GAi[off]);
                }
                Sub(GAi,Areg,GAi,GBKK);
                PipeBarrier<PIPE_V>();
                BlockInvG();
                BriG();
            }
            const uint32_t output_g=g%(BATCH/GB);
            PrecodeG(output_g,re0,slot);
            if (((g+1)*GB)%BATCH==0) StoreBatch(re0+GB-BATCH);
        }
    }
}

extern "C" __global__ __aicore__ void precode_zf_kernel(
    GM_ADDR w_re,GM_ADDR w_im,GM_ADDR s_re,GM_ADDR s_im,GM_ADDR mask,
    GM_ADDR x_re,GM_ADDR x_im,GM_ADDR workspace,GM_ADDR tiling)
{
    if (GetBlockIdx() >= airan::BLOCK_DIM) return;
    TPipe pipe; PrecodeZF op;
    op.Init(w_re,w_im,s_re,s_im,mask,x_re,x_im,&pipe);
    op.Run();
}
