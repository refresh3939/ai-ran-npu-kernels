

















#include "kernel_operator.h"
#include "decimate.h"

using namespace AscendC;
using namespace decimate;


class Decimate {
public:
    __aicore__ inline Decimate() {}

    __aicore__ inline void Init(GM_ADDR in_iq_gm, GM_ADDR fir_taps_gm,
                                 GM_ADDR out_iq_gm, TPipe *pipe)
    {
        pipe_    = pipe;
        blockId_ = GetBlockIdx();
        outStart_ = blockId_ * OUT_PER_CORE;

        inG_  .SetGlobalBuffer(reinterpret_cast<__gm__ int16_t *>(in_iq_gm),   IN_GM_I16_LEN);
        tapsG_.SetGlobalBuffer(reinterpret_cast<__gm__ half    *>(fir_taps_gm), L_TAP);
        outIQG_.SetGlobalBuffer(reinterpret_cast<__gm__ int16_t *>(out_iq_gm), OUT_GM_I16_LEN);


        pipe_->InitBuffer(bufLoad_, LOAD_I16 * sizeof(int16_t));
        pipe_->InitBuffer(bufHalf_, LOAD_I16 * sizeof(half));
        pipe_->InitBuffer(bufXi_,   LOAD_CPX * sizeof(half));
        pipe_->InitBuffer(bufXq_,   LOAD_CPX * sizeof(half));
        pipe_->InitBuffer(bufSri_,  DECIM * STREAM_LEN * sizeof(half));
        pipe_->InitBuffer(bufSrq_,  DECIM * STREAM_LEN * sizeof(half));

        pipe_->InitBuffer(bufScrE_,  (LOAD_CPX / 2) * sizeof(half));
        pipe_->InitBuffer(bufScrO_,  (LOAD_CPX / 2) * sizeof(half));
        pipe_->InitBuffer(bufScrEE_, (LOAD_CPX / 4) * sizeof(half));
        pipe_->InitBuffer(bufScrEO_, (LOAD_CPX / 4) * sizeof(half));
        pipe_->InitBuffer(bufScrOE_, (LOAD_CPX / 4) * sizeof(half));
        pipe_->InitBuffer(bufScrOO_, (LOAD_CPX / 4) * sizeof(half));
        pipe_->InitBuffer(bufYi_,   OUT_TILE * sizeof(half));
        pipe_->InitBuffer(bufYq_,   OUT_TILE * sizeof(half));
        pipe_->InitBuffer(bufTi_,   OUT_TILE * sizeof(int32_t));
        pipe_->InitBuffer(bufTq_,   OUT_TILE * sizeof(int32_t));
        pipe_->InitBuffer(bufTaps_, L_TAP    * sizeof(half));
    }

    __aicore__ inline void Process()
    {
        auto taps = bufTaps_.Get<half>();
        DataCopy(taps, tapsG_, L_TAP);
        Wait<HardEvent::MTE2_S>();


        for (uint32_t k = 0; k < L_TAP; ++k) {
            uint32_t b = k & 7u, a = k >> 3;
            uint32_t r = (b == 0) ? 0u : (8u - b);
            uint32_t O = (b == 0) ? (12u - a) : (11u - a);
            sOff_[k] = r * STREAM_LEN + O;
            htap_[k] = taps.GetValue(k);
        }
        for (uint32_t t = 0; t < N_TILES; ++t) ProcessTile(t);
    }

private:
    template <HardEvent EV>
    __aicore__ inline void Wait()
    {
        event_t e = static_cast<event_t>(GetTPipePtr()->FetchEventID(EV));
        SetFlag<EV>(e); WaitFlag<EV>(e);
    }


    __aicore__ inline void LoadChunked(const LocalTensor<int16_t> &dst, uint32_t dstOff,
                                       uint32_t srcOff, uint32_t cnt)
    {
        for (uint32_t done = 0; done < cnt; ) {
            uint32_t c = (cnt - done > LOAD_CHUNK_I16) ? LOAD_CHUNK_I16 : (cnt - done);
            DataCopy(dst[dstOff + done], inG_[srcOff + done], c);
            done += c;
        }
    }


    __aicore__ inline void ExtractResidues8(const LocalTensor<half> &src,
                                             const LocalTensor<half> &srbase)
    {
        auto scrE  = bufScrE_.Get<half>();   auto scrO  = bufScrO_.Get<half>();
        auto scrEE = bufScrEE_.Get<half>();  auto scrEO = bufScrEO_.Get<half>();
        auto scrOE = bufScrOE_.Get<half>();  auto scrOO = bufScrOO_.Get<half>();
        uint64_t rsvd = 0;
        GatherMaskParams p;
        p.src0BlockStride = 1; p.src0RepeatStride = 8; p.src1RepeatStride = 0;


        p.repeatTimes = LOAD_CPX / 128;
        GatherMask(scrE, src, (uint8_t)1, false, 0, p, rsvd);
        GatherMask(scrO, src, (uint8_t)2, false, 0, p, rsvd);
        PipeBarrier<PIPE_V>();

        p.repeatTimes = (LOAD_CPX / 2) / 128;
        GatherMask(scrEE, scrE, (uint8_t)1, false, 0, p, rsvd);
        GatherMask(scrEO, scrE, (uint8_t)2, false, 0, p, rsvd);
        GatherMask(scrOE, scrO, (uint8_t)1, false, 0, p, rsvd);
        GatherMask(scrOO, scrO, (uint8_t)2, false, 0, p, rsvd);
        PipeBarrier<PIPE_V>();

        p.repeatTimes = (LOAD_CPX / 4) / 128;
        auto r0 = srbase[0 * STREAM_LEN]; GatherMask(r0, scrEE, (uint8_t)1, false, 0, p, rsvd);
        auto r4 = srbase[4 * STREAM_LEN]; GatherMask(r4, scrEE, (uint8_t)2, false, 0, p, rsvd);
        auto r2 = srbase[2 * STREAM_LEN]; GatherMask(r2, scrEO, (uint8_t)1, false, 0, p, rsvd);
        auto r6 = srbase[6 * STREAM_LEN]; GatherMask(r6, scrEO, (uint8_t)2, false, 0, p, rsvd);
        auto r1 = srbase[1 * STREAM_LEN]; GatherMask(r1, scrOE, (uint8_t)1, false, 0, p, rsvd);
        auto r5 = srbase[5 * STREAM_LEN]; GatherMask(r5, scrOE, (uint8_t)2, false, 0, p, rsvd);
        auto r3 = srbase[3 * STREAM_LEN]; GatherMask(r3, scrOO, (uint8_t)1, false, 0, p, rsvd);
        auto r7 = srbase[7 * STREAM_LEN]; GatherMask(r7, scrOO, (uint8_t)2, false, 0, p, rsvd);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void ProcessTile(uint32_t t)
    {
        auto load = bufLoad_.Get<int16_t>();
        auto xh   = bufHalf_.Get<half>();
        auto xi   = bufXi_.Get<half>();
        auto xq   = bufXq_.Get<half>();
        auto sri  = bufSri_.Get<half>();
        auto srq  = bufSrq_.Get<half>();


        uint32_t m0  = outStart_ + t * OUT_TILE;
        uint32_t sIn = m0 * DECIM;
        int32_t  gStart = (int32_t)sIn - (int32_t)HALO;

        Duplicate(load, (int16_t)0, LOAD_I16);
        PipeBarrier<PIPE_V>();
        Wait<HardEvent::V_MTE2>();
        if (gStart >= 0) {
            LoadChunked(load, 0, (uint32_t)gStart * 2, WINDOW_CPX * 2);
        } else {
            LoadChunked(load, HALO * 2, 0, IN_PER_TILE * 2);
        }
        Wait<HardEvent::MTE2_V>();
        Cast(xh, load, RoundMode::CAST_NONE, LOAD_I16);
        PipeBarrier<PIPE_V>();


        uint64_t rsvd = 0;
        GatherMaskParams gmp;
        gmp.src0BlockStride = 1; gmp.repeatTimes = LOAD_I16 / 128;
        gmp.src0RepeatStride = 8; gmp.src1RepeatStride = 0;
        GatherMask(xi, xh, (uint8_t)1, false, 0, gmp, rsvd);
        GatherMask(xq, xh, (uint8_t)2, false, 0, gmp, rsvd);
        PipeBarrier<PIPE_V>();


        ExtractResidues8(xi, sri);
        ExtractResidues8(xq, srq);


        auto yI = bufYi_.Get<half>();
        auto yQ = bufYq_.Get<half>();
        Duplicate(yI, (half)0.0, OUT_TILE);
        Duplicate(yQ, (half)0.0, OUT_TILE);
        PipeBarrier<PIPE_V>();
        for (uint32_t k = 0; k < L_TAP; ++k) {
            auto srIk = sri[sOff_[k]];
            auto srQk = srq[sOff_[k]];
            Axpy(yI, srIk, htap_[k], OUT_TILE);
            Axpy(yQ, srQk, htap_[k], OUT_TILE);
            PipeBarrier<PIPE_V>();
        }






        auto tI = bufTi_.Get<int32_t>();
        auto tQ = bufTq_.Get<int32_t>();
        auto fa = bufXi_.Get<float>();
        auto fb = bufXq_.Get<float>();

        Cast(tI, yI, RoundMode::CAST_RINT, OUT_TILE);
        PipeBarrier<PIPE_V>();
        Cast(fa, tI, RoundMode::CAST_NONE, OUT_TILE);
        PipeBarrier<PIPE_V>();
        Muls(fb, fa, (float)(1.0f / 65536.0f), OUT_TILE);
        PipeBarrier<PIPE_V>();
        Cast(tQ, fb, RoundMode::CAST_FLOOR, OUT_TILE);
        PipeBarrier<PIPE_V>();
        Cast(fb, tQ, RoundMode::CAST_NONE, OUT_TILE);
        PipeBarrier<PIPE_V>();
        Axpy(fa, fb, (float)(-65536.0f), OUT_TILE);
        PipeBarrier<PIPE_V>();
        Cast(tI, fa, RoundMode::CAST_RINT, OUT_TILE);

        Cast(tQ, yQ, RoundMode::CAST_RINT, OUT_TILE);
        PipeBarrier<PIPE_V>();
        for (int s = 0; s < 16; ++s) {
            Add(tQ, tQ, tQ, OUT_TILE);
            PipeBarrier<PIPE_V>();
        }
        Add(tI, tI, tQ, OUT_TILE);
        PipeBarrier<PIPE_V>();
        Wait<HardEvent::V_MTE3>();
        DataCopy(outIQG_[m0 * 2], tI.ReinterpretCast<int16_t>(), OUT_TILE * 2);
        Wait<HardEvent::MTE3_MTE2>();
    }

    TPipe *pipe_;
    uint32_t blockId_;
    uint32_t outStart_;

    GlobalTensor<int16_t> inG_, outIQG_;
    GlobalTensor<half>    tapsG_;

    TBuf<TPosition::VECCALC> bufLoad_, bufHalf_, bufXi_, bufXq_, bufSri_, bufSrq_;
    TBuf<TPosition::VECCALC> bufScrE_, bufScrO_, bufScrEE_, bufScrEO_, bufScrOE_, bufScrOO_;
    TBuf<TPosition::VECCALC> bufYi_, bufYq_, bufTi_, bufTq_, bufTaps_;

    half     htap_[L_TAP];
    uint32_t sOff_[L_TAP];
};


extern "C" __global__ __aicore__ void decimate_kernel(
    GM_ADDR in_iq_gm,
    GM_ADDR fir_taps_gm,
    GM_ADDR out_iq_gm,
    GM_ADDR ws_gm,
    GM_ADDR tiling_gm)
{
    TPipe pipe;
    Decimate op;
    op.Init(in_iq_gm, fir_taps_gm, out_iq_gm, &pipe);
    op.Process();
}