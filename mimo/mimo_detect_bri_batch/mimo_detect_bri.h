#pragma once
#include <cstddef>
#include <cstdint>


















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
constexpr uint32_t F = 16;


constexpr uint32_t P = AIRAN_PACK ? (MIMO_K / MIMO_KR) : 1u;
constexpr uint32_t N_UNIT = N_RE / P;
constexpr uint32_t BLOCK_DIM   = 4;
constexpr uint32_t U_PER_CORE  = N_UNIT / BLOCK_DIM;
constexpr uint32_t BATCH = 16;
constexpr uint32_t N_BATCH_PER_CORE = U_PER_CORE / BATCH;





constexpr float DET_EPS = 1.0e-3f;
constexpr size_t TILING_TOTAL_SIZE = 2048;


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



static_assert(2u*4*MIMO_K*MIMO_M + 3u*4*MIMO_K*MIMO_K <= 32768u, "L0A 超限: 减小 NR 或 GB");
static_assert(2u*4*MIMO_K*MIMO_M + 2u*4*MIMO_K*MIMO_K <= 32768u, "L0B 超限: 减小 NR 或 GB");

static_assert(5u*(4*MIMO_K)*(4*MIMO_K) <= 65536u, "L0C 超限: 减小 NL 或 GB");
}















