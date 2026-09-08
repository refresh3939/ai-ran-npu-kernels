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
#ifndef AIRAN_BLOCK_DIM
#define AIRAN_BLOCK_DIM 4
#endif
#ifndef AIRAN_RE_BATCH
#define AIRAN_RE_BATCH 5824
#endif
#define BRI_B  8
#define BRI_L  5


#define AIRAN_STR2(x) #x
#define AIRAN_STR(x)  AIRAN_STR2(x)

#define N_SC_PAD_VAL 1664

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
constexpr uint32_t NT = MIMO_M, NL = MIMO_K, NLR = MIMO_KR;
constexpr uint32_t PACK = AIRAN_PACK ? (NL / NLR) : 1u;
constexpr uint32_t N_UNIT = N_RE / PACK;
constexpr uint32_t BLK = BRI_B, NLAY = BRI_L;
constexpr uint32_t F = 16;
constexpr uint32_t MB = MIMO_M / F;
constexpr uint32_t NBLK = MIMO_K / BRI_B;

constexpr uint32_t BLOCK_DIM   = AIRAN_BLOCK_DIM;
static_assert(BLOCK_DIM == 4, "本优化版本按要求固定只使用 4 个核");
constexpr uint32_t UNIT_PER_CORE = N_UNIT / BLOCK_DIM;
constexpr uint32_t RE_PER_CORE = UNIT_PER_CORE;
constexpr uint32_t BATCH = (PACK==1) ? 64u : 16u;
constexpr uint32_t RE_BATCH = AIRAN_RE_BATCH;
constexpr uint32_t N_BATCH_PER_CORE = UNIT_PER_CORE / RE_BATCH;

constexpr float DET_EPS = 1.0e-3f;
constexpr size_t TILING_TOTAL_SIZE = 2048;


static_assert(MIMO_M % F == 0,      "NT 必须是 16 的倍数: img2col 的 W 和 Mmad 的 k 都要 fractal 对齐");
static_assert(MIMO_K == F,          "NL 目前只支持 16 (= fractal 边长). 见 header 末尾说明");
static_assert(MIMO_KR >= 1 && MIMO_KR <= MIMO_K, "NLR 必须在 [1,NL] 范围内");
static_assert(MIMO_K % BRI_B == 0,  "NL 必须被预条件块大小整除");
static_assert(N_RE % PACK == 0,     "N_RE 必须被 PACK 整除");
static_assert(N_UNIT % BLOCK_DIM == 0, "N_UNIT 必须被 BLOCK_DIM 整除");
static_assert(RE_BATCH % BATCH == 0, "RE_BATCH 必须是输出 BATCH=16 的整数倍");
static_assert(UNIT_PER_CORE % RE_BATCH == 0, "RE_BATCH 需整除每核瓦片数");
static_assert(!AIRAN_PACK || NL % NLR == 0, "PACK 要求 NLR 整除 16: 1/2/4/8/16");
static_assert(NLR % BLK == 0 || NLR < BLK, "预条件块不应跨 RE (NLR>=BLK 时 BLK 整除 NLR)");
static_assert(MIMO_K >= BRI_B,      "块不能大于矩阵");

static_assert(MIMO_M <= 255,        "末段 WholeReduceSum repeatTimes 上限 255");

}






