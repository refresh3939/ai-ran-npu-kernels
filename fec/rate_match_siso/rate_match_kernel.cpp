







#include "kernel_operator.h"
#include "rate_match.h"

using namespace AscendC;

namespace {
constexpr uint32_t C_NUM           = airan::C_NUM;
constexpr uint32_t N_CB_BUF        = airan::N_CB_BUF;
constexpr uint32_t BLOCK_DIM       = airan::BLOCK_DIM;
constexpr uint32_t N_SYM           = airan::N_SYM;
constexpr uint32_t N_SYM_PAD       = airan::N_SYM_PAD;
constexpr uint32_t NFLOOR          = airan::NFLOOR;
constexpr uint32_t EQ_LO           = airan::EQ_LO;
constexpr uint32_t STREAMS_PER_AIV = airan::STREAMS_PER_AIV;
constexpr uint32_t SLOT_PAIR_ELEMS = airan::SLOT_PAIR_ELEMS;
constexpr uint32_t PAIR_INPUT_PAD  = airan::PAIR_INPUT_PAD;
constexpr uint32_t PAIR_RING_DEPTH = airan::PAIR_RING_DEPTH;
constexpr uint32_t I8_ALIGN        = 32;
constexpr uint32_t I16_ALIGN       = 16;

static_assert(BLOCK_DIM == 4, "v5c scheduling requires four AIVs");
static_assert(STREAMS_PER_AIV == 2, "each AIV owns two adjacent streams");
static_assert(airan::SLOT_RING_DEPTH == 1, "v5c uses one full-slot buffer");
static_assert(PAIR_RING_DEPTH == 2 || PAIR_RING_DEPTH == 4,
              "validated A/B ring depths are two and four");
static_assert(N_SYM % I16_ALIGN == 0, "valid slot must be 32B aligned");
static_assert((N_SYM_PAD - N_SYM) % I16_ALIGN == 0,
              "padding must be an integer number of datablocks");
static_assert(PAIR_INPUT_PAD >= 2 * airan::EQ_HI + I8_ALIGN - 1,
              "input pair buffer is too small");
}

extern "C" __global__ __aicore__ void rate_match_kernel(
    GM_ADDR codeword_gm, GM_ADDR unused1_gm, GM_ADDR unused2_gm, GM_ADDR layout_gm)
{
    const uint32_t aivId = GetBlockIdx() ^ 2u;
    if (aivId >= BLOCK_DIM) return;
    (void)unused1_gm;
    (void)unused2_gm;

    const uint32_t firstStream = aivId * STREAMS_PER_AIV;

    GlobalTensor<int8_t> codewordG;
    GlobalTensor<int16_t> layoutG;
    codewordG.SetGlobalBuffer((__gm__ int8_t*)codeword_gm, airan::CODEWORD_LEN);
    layoutG.SetGlobalBuffer((__gm__ int16_t*)layout_gm, airan::LAYOUT_LEN);

    TPipe pipe;
    TBuf<TPosition::VECCALC> slotBuf, pairBuf, pair8Buf;
    pipe.InitBuffer(slotBuf, SLOT_PAIR_ELEMS * sizeof(int16_t));
    pipe.InitBuffer(pairBuf,
                    PAIR_RING_DEPTH * PAIR_INPUT_PAD * sizeof(int16_t));
    pipe.InitBuffer(pair8Buf,
                    PAIR_RING_DEPTH * PAIR_INPUT_PAD * sizeof(int8_t));

    auto slotPair = slotBuf.Get<int16_t>();
    auto pairI16  = pairBuf.Get<int16_t>();
    auto pairHalf = pairBuf.Get<half>();
    auto pairI8   = pair8Buf.Get<int8_t>();

    DataCopyParams writeParams;
    writeParams.blockCount = (uint16_t)STREAMS_PER_AIV;
    writeParams.blockLen = (uint16_t)(N_SYM / I16_ALIGN);
    writeParams.srcStride = 0;
    writeParams.dstStride =
        (uint16_t)((N_SYM_PAD - N_SYM) / I16_ALIGN);

    uint32_t slot = 0;
    uint32_t c0 = 0;
    for (uint32_t cb = 0; cb < C_NUM; ++cb) {
        const uint32_t eq = cb < NFLOOR ? EQ_LO : (EQ_LO + 1u);
        const uint32_t re0 = c0 - slot * N_SYM;
        const uint32_t room = N_SYM - re0;
        const uint32_t count0 = eq < room ? eq : room;
        const uint32_t count1 = eq - count0;

        const uint64_t pairStart = (uint64_t)cb * N_CB_BUF + firstStream * eq;
        const uint32_t srcDelta = (uint32_t)(pairStart & (I8_ALIGN - 1u));
        const uint64_t pairAligned = pairStart - srcDelta;
        const uint32_t pairCount =
            (srcDelta + STREAMS_PER_AIV * eq + I8_ALIGN - 1u) &
            ~(I8_ALIGN - 1u);
        const uint32_t pairBase = (cb % PAIR_RING_DEPTH) * PAIR_INPUT_PAD;

        DataCopy(pairI8[pairBase], codewordG[pairAligned], pairCount);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
        Cast(pairHalf[pairBase], pairI8[pairBase], RoundMode::CAST_NONE, pairCount);
        Cast(pairI16[pairBase], pairHalf[pairBase], RoundMode::CAST_RINT, pairCount);
        PipeBarrier<PIPE_V>();

        Adds(slotPair[re0], pairI16[pairBase + srcDelta],
             (int16_t)0, (int32_t)count0);
        Adds(slotPair[N_SYM + re0], pairI16[pairBase + srcDelta + eq],
             (int16_t)0, (int32_t)count0);

        if (re0 + count0 == N_SYM) {
            SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
            WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
            const uint64_t outBase =
                (uint64_t)slot * airan::SLOT_STRIDE +
                (uint64_t)firstStream * airan::STREAM_STRIDE;
            DataCopy(layoutG[outBase], slotPair, writeParams);
            SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
            WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
            ++slot;

            if (count1 != 0u) {
                Adds(slotPair, pairI16[pairBase + srcDelta + count0],
                     (int16_t)0, (int32_t)count1);
                Adds(slotPair[N_SYM], pairI16[pairBase + srcDelta + eq + count0],
                     (int16_t)0, (int32_t)count1);
            }
        }

        if ((cb % PAIR_RING_DEPTH) == PAIR_RING_DEPTH - 1u || cb + 1u == C_NUM)
            PipeBarrier<PIPE_ALL>();
        c0 += eq;
    }
}
