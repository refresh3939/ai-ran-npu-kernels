














#include "kernel_operator.h"
#include "rate_dematch.h"
using namespace AscendC;

namespace {
constexpr uint32_t C_NUM       = airan::C_NUM;
constexpr uint32_t LDPC_N      = airan::LDPC_N;
constexpr uint32_t N_2Z        = airan::N_2Z;
constexpr uint32_t BLOCK_DIM   = airan::BLOCK_DIM;
constexpr uint32_t N_STREAMS   = airan::N_STREAMS;
constexpr uint32_t N_SYM       = airan::N_SYM;
constexpr int16_t  SCALE       = airan::SCALE;
constexpr int16_t  CLIP        = airan::LLR_CLIP;
constexpr uint32_t NFLOOR      = airan::NFLOOR;
constexpr uint32_t EQ_LO       = airan::EQ_LO;
constexpr uint32_t EPAD_LO     = airan::EPAD_LO;
constexpr uint32_t EPAD_HI     = airan::EPAD_HI;
constexpr uint32_t LPM         = airan::LPM;
constexpr uint32_t ALIGN       = 16;
constexpr uint32_t INPUT_SLOTS = airan::INPUT_RING_DEPTH;



constexpr uint64_t MASK128 = 128;
constexpr uint32_t REPS193 = 193;
static const UnaryRepeatParams UREP = {1, 1, 8, 8};

constexpr uint32_t E_ELEMS   = EPAD_HI + 64;
constexpr uint32_t UB_BYTES  = INPUT_SLOTS * airan::UB_SRC_MAX * sizeof(int16_t)
                             + E_ELEMS * sizeof(int16_t);
static_assert(UB_BYTES <= 256 * 1024, "rate_dematch v5 exceeds dav_m200 UB");

__aicore__ inline bool IsLowClass(uint32_t cb)
{
    return cb < NFLOOR;
}




__aicore__ inline void LoadCb(const GlobalTensor<int16_t>& descramG,
                              const LocalTensor<int16_t>& ubSrc,
                              uint32_t cb)
{
    const bool lo       = IsLowClass(cb);
    const uint32_t EQ   = lo ? EQ_LO : (EQ_LO + 1);
    const uint32_t c0   = lo ? cb * EQ_LO
                             : NFLOOR * EQ_LO + (cb - NFLOOR) * (EQ_LO + 1);
    const uint32_t s0   = c0 / N_SYM;
    const uint32_t re0  = c0 % N_SYM;
    const uint32_t endS = (c0 + EQ - 1) / N_SYM;

    uint32_t bs[2], ba[2], bl[2], bp[2], nblk;
    if (endS == s0) {
        const uint32_t reA = (re0 / ALIGN) * ALIGN;
        const uint32_t lp  = ((re0 + EQ - reA + ALIGN - 1) / ALIGN) * ALIGN;
        bs[0] = s0; ba[0] = reA; bl[0] = lp; bp[0] = 0; nblk = 1;
    } else {
        const uint32_t reA0 = (re0 / ALIGN) * ALIGN;
        const uint32_t lp0  = N_SYM - reA0;
        const uint32_t reHi = (c0 + EQ) - (s0 + 1) * N_SYM;
        const uint32_t lp1  = ((reHi + ALIGN - 1) / ALIGN) * ALIGN;
        bs[0] = s0;     ba[0] = reA0; bl[0] = lp0; bp[0] = 0;
        bs[1] = s0 + 1; ba[1] = 0;    bl[1] = lp1; bp[1] = lp0;
        nblk = 2;
    }

    for (uint32_t bk = 0; bk < nblk; ++bk) {
        DataCopyParams cp;
        cp.blockCount = static_cast<uint16_t>(N_STREAMS);
        cp.blockLen   = static_cast<uint16_t>(bl[bk] / ALIGN);
        cp.srcStride  = static_cast<uint16_t>((airan::STREAM_STRIDE - bl[bk]) / ALIGN);
        cp.dstStride  = static_cast<uint16_t>((LPM - bl[bk]) / ALIGN);
        const uint64_t g0 = static_cast<uint64_t>(bs[bk]) * airan::SLOT_STRIDE + ba[bk];
        DataCopy(ubSrc[bp[bk]], descramG[g0], cp);
    }
}

}

extern "C" __global__ __aicore__ void rate_dematch_kernel(
    GM_ADDR descram_gm, GM_ADDR aux0_gm, GM_ADDR aux1_gm, GM_ADDR lam_out_gm)
{
    const uint32_t aivId = GetBlockIdx() ^ 2;
    if (aivId >= BLOCK_DIM) return;
    (void)aux0_gm;
    (void)aux1_gm;

    GlobalTensor<int16_t> descramG, lamG;
    descramG.SetGlobalBuffer(reinterpret_cast<__gm__ int16_t*>(descram_gm), airan::DESCRAM_LEN);
    lamG.SetGlobalBuffer(reinterpret_cast<__gm__ int16_t*>(lam_out_gm),
                         static_cast<uint64_t>(C_NUM) * LDPC_N);

    TPipe pipe;
    TBuf<TPosition::VECCALC> srcRing, eBuf;
    pipe.InitBuffer(srcRing, INPUT_SLOTS * airan::UB_SRC_MAX * sizeof(int16_t));
    pipe.InitBuffer(eBuf, E_ELEMS * sizeof(int16_t));

    auto ubSrcRing = srcRing.Get<int16_t>();
    auto ubE       = eBuf.Get<int16_t>();
    auto ubEH      = ubE.ReinterpretCast<half>();

    const uint32_t firstCb = aivId;
    if (firstCb >= C_NUM) return;

    LoadCb(descramG, ubSrcRing, firstCb);
    SetFlag<HardEvent::MTE2_V>(EVENT_ID0);

    uint32_t currentSlot = 0;

    for (uint32_t cb = firstCb; cb < C_NUM; cb += BLOCK_DIM) {
        if (currentSlot == 0) {
            WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
        } else {
            WaitFlag<HardEvent::MTE2_V>(EVENT_ID1);
        }

        const bool lo = IsLowClass(cb);
        const uint32_t EQ   = lo ? EQ_LO : (EQ_LO + 1u);
        const uint32_t EPad = lo ? EPAD_LO : EPAD_HI;
        const uint32_t c0   = lo ? cb * EQ_LO
                                 : NFLOOR * EQ_LO + (cb - NFLOOR) * (EQ_LO + 1);
        const uint32_t delta = (c0 % N_SYM) % ALIGN;

        const uint32_t nextCb = cb + BLOCK_DIM;
        const uint32_t nextSlot = currentSlot ^ 1u;
        if (nextCb < C_NUM) {
            auto ubNext = ubSrcRing[nextSlot * airan::UB_SRC_MAX];
            LoadCb(descramG, ubNext, nextCb);
            if (nextSlot == 0) {
                SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
            } else {
                SetFlag<HardEvent::MTE2_V>(EVENT_ID1);
            }
        }

        if (!lo) {
            Duplicate(ubEH[EPAD_LO], static_cast<half>(0), ALIGN);
        }

        auto ubSrc = ubSrcRing[currentSlot * airan::UB_SRC_MAX];
        for (uint32_t stream = 0; stream < N_STREAMS; ++stream) {
            Adds(ubE[stream * EQ], ubSrc[stream * LPM + delta],
                 static_cast<int16_t>(0), static_cast<int32_t>(EQ));
        }

        Muls(ubE, ubE, SCALE, MASK128, REPS193, UREP);
        Mins(ubE, ubE, CLIP, MASK128, REPS193, UREP);
        Maxs(ubE, ubE, static_cast<int16_t>(-CLIP), MASK128, REPS193, UREP);

        PipeBarrier<PIPE_ALL>();
        DataCopy(lamG[static_cast<uint64_t>(cb) * LDPC_N + N_2Z], ubE, EPad);
        PipeBarrier<PIPE_ALL>();
        currentSlot = nextSlot;
    }
}
