#pragma once
#include <cstddef>
#include <cstdint>
// ============================================================================
//  mimo_detect_bri — massive MIMO LMMSE, Block-Richardson 迭代求逆 (Cube)
//   A = HᴴH + σ²I, 块对角预条件 M (BRI_B×BRI_B 块) + Richardson 迭代
//   Production host contract selects io_pack storage at build time: NR is a
//   16-aligned execution bucket, NL=16, N_RE=23296 and PACK=0. NL_REAL is the active layer count; upstream must
//   zero hrm columns [NL_REAL,16). The C++ host checks this before launch.
//
//  ── 收敛法则 (numpy f64 扫描得出, 违反必挂; 见文件末尾 static_assert) ──────
//     B = NL_REAL/2   且   NR >= 8*B      <=>   NR >= 4*NL_REAL
//   违反的代价:  32×8 用 B=8 -> EVM 190%   16×4 用 B=8 -> 60%   64×16 用 B=4 -> 9.4%
//
//  ── 实测 (310P1, 一个 5G slot: 14 符号 × 1664 子载波 = 23296 RE, 30kHz SCS) ──
//     配置      P  BRI_B   p50        EVM
//     16×4      4    2     4.975 ms   1.97%      P = 16/NL_REAL, 一个瓦片装 P 个 RE
//     32×8      2    4     9.684 ms   1.84%      打包后时延与层数成正比
//     64×16     1    8    19.795 ms   1.77%      天线数翻倍仅 +5.6%
//     128×16    1    8    21.174 ms   0.27%      过定比越高精度越好
// ============================================================================
#ifndef MIMO_M
#define MIMO_M 64
#endif
#ifndef MIMO_K
#define MIMO_K 16
#endif
#ifndef MIMO_KR
#define MIMO_KR 16
#endif
#ifndef AIRAN_PACK
#define AIRAN_PACK 0
#endif
#ifndef BRI_B
#define BRI_B 8
#endif
#define BRI_L 5
#define AIRAN_RULE_CHECK 1

#define AIRAN_STR2(x) #x
#define AIRAN_STR(x)  AIRAN_STR2(x)
#ifndef N_SC_PAD_VAL
#define N_SC_PAD_VAL 1664
#endif
// NL_REAL < NL 加 _r 后缀; P>1 再加 _pk (打包与补零的 hrm 布局不同, 不能共用目录)
#define AIRAN_BASE "case_0_m" AIRAN_STR(MIMO_M) "_k" AIRAN_STR(MIMO_K) "_sc" AIRAN_STR(N_SC_PAD_VAL)
#if MIMO_KR == MIMO_K
#define CASE_NAME AIRAN_BASE
#elif AIRAN_PACK
#define CASE_NAME AIRAN_BASE "_r" AIRAN_STR(MIMO_KR) "_pk"
#else
#define CASE_NAME AIRAN_BASE "_r" AIRAN_STR(MIMO_KR)
#endif

namespace airan {
constexpr uint32_t N_SYMBOL = 14, N_SC_PAD = N_SC_PAD_VAL;
constexpr uint32_t N_RE = N_SYMBOL * N_SC_PAD;
constexpr uint32_t NR = MIMO_M, NL = MIMO_K, NLR = MIMO_KR;
constexpr uint32_t BLK = BRI_B, NLAY = BRI_L;
constexpr uint32_t F = 16;                          // fractal 边长 (fp16), 硬件常量

// ── 打包: P 个 RE 共用一个 [M,16] 瓦片 (= 一个 "unit"). P=1 时全部退化成原来的定义 ──
constexpr uint32_t P = AIRAN_PACK ? (MIMO_K / MIMO_KR) : 1u;
constexpr uint32_t N_UNIT = N_RE / P;               // GM 里 hrm/yvpad 的瓦片数
constexpr uint32_t BLOCK_DIM   = 4;   // 4 个 AI Core (310P1 有 8 个, 未用满)
constexpr uint32_t U_PER_CORE  = N_UNIT / BLOCK_DIM;
constexpr uint32_t BATCH = 16;        // 一个 bt 的 unit 数 (写回 GM 时 32B 对齐要 BATCH*2B >= 32)
constexpr uint32_t N_BATCH_PER_CORE = U_PER_CORE / BATCH;
// 打包排列 (与 ref.py 的 re_of 一致):
//   unit u = b*16+f 的第 p 个槽  ->  RE = b*(16P) + 16p + f
//   于是攒 16 个 unit 做【一次】16x16 转置后, 第 p*NLR+l 行沿 f 展开
//   = 连续 16 个 RE 的第 l 层  ->  x̂ 直写标准 [NL][N_RE], 32B 对齐天然满足

constexpr float DET_EPS = 1.0e-3f;
constexpr size_t TILING_TOTAL_SIZE = 2048;

// ── 维度可行性 (编译期拦截, 别等运行时静默算错) ──────────────────────────
static_assert(MIMO_M % F == 0,      "NR 必须是 16 的倍数: img2col 的 W 和 Mmad 的 k 都要 fractal 对齐");
static_assert(MIMO_K == F,          "NL 目前只支持 16 (= fractal 边长). NL<16 请用 NL_REAL 补零");
static_assert(MIMO_KR <= MIMO_K,    "NL_REAL 不能超过 NL");
static_assert(MIMO_KR >= 1,         "NL_REAL 至少是 1");
static_assert(MIMO_K % BRI_B == 0,  "NL 必须被预条件块大小整除");
static_assert(U_PER_CORE % BATCH == 0,
    "打包后每核 unit 数必须被 BATCH=16 整除. N_RE/(P*BLOCK_DIM*BATCH) 要整除 —— "
    "N_SC_PAD=1664 时 P<=4 可以(NL_REAL>=4), P=8/16 需换 N_SC_PAD");
static_assert(!AIRAN_PACK || (MIMO_K % MIMO_KR == 0),
    "打包要求 NL_REAL 整除 16: 可用 1/2/4/8/16");
static_assert(!AIRAN_PACK || (MIMO_KR % BRI_B == 0),
    "打包要求 BRI_B 整除 NL_REAL, 否则预条件块会跨 RE (B=NL_REAL/2 自动满足)");
static_assert(BATCH * sizeof(int16_t) >= 32, "写回 GM 需 32B 对齐");

// ── 收敛法则 (2026-07-26: 整晚在追一个算法上无解的配置, 教训编译期化) ────
#if AIRAN_RULE_CHECK
static_assert(MIMO_M >= 8u * (BRI_B < MIMO_KR ? BRI_B : MIMO_KR),
    "违反 Neumann 收敛法则 NR >= 8*min(BRI_B, NL_REAL) —— 块对角缩放的谱半径 "
    "rho ~ 2*sqrt(B/NR) 会 >=1, M^-1 发散. 减小 BRI_B 或增大 NR. "
    "(诊断用途可把 AIRAN_RULE_CHECK 设 0)");
static_assert(2u * BRI_B >= MIMO_KR,
    "违反 BRI 收敛法则 BRI_B >= NL_REAL/2 —— 预条件子盖不住足够的 A, "
    "Richardson 迭代收敛太慢 (B=NL/4 时 L=5 只到 9.4%). 增大 BRI_B. "
    "(诊断用途可把 AIRAN_RULE_CHECK 设 0)");
#endif

// L0A/L0B 容量 (各 64KB = 32768 half), GB=4 时:
//   bA2_/bB2_ = 2*GB*K*M half,  bGA2_ = 3*GB*K*K,  bGB2_ = 2*GB*K*K
static_assert(2u*4*MIMO_K*MIMO_M + 3u*4*MIMO_K*MIMO_K <= 32768u, "L0A 超限: 减小 NR 或 GB");
static_assert(2u*4*MIMO_K*MIMO_M + 2u*4*MIMO_K*MIMO_K <= 32768u, "L0B 超限: 减小 NR 或 GB");
// L0C 容量 (256KB = 65536 float): bCO_ = 3*(GB*K)², bGCO_ = 2*(GB*K)²
static_assert(5u*(4*MIMO_K)*(4*MIMO_K) <= 65536u, "L0C 超限: 减小 NL 或 GB");
}  // namespace airan

// ── 关于 NL < 16 ────────────────────────────────────────────────────────────
//  16 是 fp16 fractal 的边长, 也是 Transpose/Brcb/WholeReduceSum 的粒度 ——
//  算 4 列还是 16 列, 指令条数一样. 所以 NL<16 时那 16 条通道必须填满:
//
//  生产链路只接受 PACK=0 (补零): 用【零】填满不活跃 H 列.
//  PACK=1 是历史独立实验模式, 与 mimo_detect_io_pack 的物理 ABI 不兼容;
//  当前 host 会拒绝该配置. 历史模式曾用【别的 RE】填 -> P=16/NL_REAL 个 RE 共用一个瓦片,
//                  指令数不变而每组 RE 数 ×P  ->  实测 P=4 给 3.98×, P=2 给 2.03×
//
//  打包为何不损精度: A = tileᴴ·tile 会算出跨 RE 的互相关, 用 Mpack 掩码抹掉 ——
//  不同 RE 本就该独立检测, 所以这是【精确】的, 不是近似. 实测 EVM 与补零版逐位相同.
//
//  支持的 NL_REAL: 1/2/4/8/16 (整除 16). 其中 P>4 受 U_PER_CORE%BATCH 限制,
//  N_SC_PAD=1664 时只到 P=4 (即 NL_REAL>=4), 见上面的 static_assert.
