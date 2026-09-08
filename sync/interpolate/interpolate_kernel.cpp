






























#include "kernel_operator.h"
#include "interpolate.h"

using namespace AscendC;
using namespace interpolate;


class Interpolate {
public:
    __aicore__ inline Interpolate() {}

    __aicore__ inline void Init(GM_ADDR in_iq_gm, GM_ADDR fir_taps_gm,
                                 GM_ADDR out_iq_gm, GM_ADDR tiling_gm, TPipe *pipe)
    {
        pipe_    = pipe;
        blockId_ = GetBlockIdx();
        inStart_ = blockId_ * IN_PER_CORE;

        inG_   .SetGlobalBuffer(reinterpret_cast<__gm__ int16_t  *>(in_iq_gm),    IN_GM_I16_LEN);
        tapsG_ .SetGlobalBuffer(reinterpret_cast<__gm__ half     *>(fir_taps_gm),  L_TAP);
        outIQG_.SetGlobalBuffer(reinterpret_cast<__gm__ int16_t  *>(out_iq_gm),   OUT_GM_I16_LEN);
        idxTG_ .SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t *>(tiling_gm),    IDX_GM_LEN);


        pipe_->InitBuffer(bufLoad_,  LOAD_I16  * sizeof(int16_t));
        pipe_->InitBuffer(bufHalf_,  LOAD_I16  * sizeof(half));
        pipe_->InitBuffer(bufXi_,    LOAD_CPX  * sizeof(half));
        pipe_->InitBuffer(bufXq_,    LOAD_CPX  * sizeof(half));
        pipe_->InitBuffer(bufYi_,    IN_TILE   * sizeof(half));
        pipe_->InitBuffer(bufYqall_, CMAT_I32  * sizeof(half));
        pipe_->InitBuffer(bufFa_,    IN_TILE   * sizeof(float));
        pipe_->InitBuffer(bufFb_,    IN_TILE   * sizeof(float));
        pipe_->InitBuffer(bufCmat_,  CMAT_I32  * sizeof(int32_t));
        pipe_->InitBuffer(bufTQall_, CMAT_I32  * sizeof(int32_t));
        pipe_->InitBuffer(bufIdx_,   IDX_LEN   * sizeof(uint32_t));
        pipe_->InitBuffer(bufOut_,   OUT_TILE  * sizeof(int32_t));
        pipe_->InitBuffer(bufTaps_,  L_TAP     * sizeof(half));
    }

    __aicore__ inline void Process()
    {
        auto taps = bufTaps_.Get<half>();
        DataCopy(taps, tapsG_, L_TAP);
        Wait<HardEvent::MTE2_S>();

        for (uint32_t p = 0; p < N_PHASE; ++p)
            for (uint32_t n = 0; n < NPT; ++n)
                hpn_[p * NPT + n] = taps.GetValue(N_PHASE * n + p);


        auto idx = bufIdx_.Get<uint32_t>();
        for (uint32_t done = 0; done < IDX_LEN; ) {
            uint32_t c = (IDX_LEN - done > IDX_CHUNK_I32) ? IDX_CHUNK_I32 : (IDX_LEN - done);
            DataCopy(idx[done], idxTG_[done], c);
            done += c;
        }
        Wait<HardEvent::MTE2_V>();

        for (uint32_t t = 0; t < N_TILES; ++t) ProcessTile(t);
    }

private:
    template <HardEvent EV>
    __aicore__ inline void Wait()
    {
        event_t e = static_cast<event_t>(GetTPipePtr()->FetchEventID(EV));
        SetFlag<EV>(e); WaitFlag<EV>(e);
    }



    __aicore__ inline void MaskI(const LocalTensor<int32_t> &cp, const LocalTensor<half> &yI,
                                 const LocalTensor<float> &fa, const LocalTensor<float> &fb)
    {
        Cast(cp, yI, RoundMode::CAST_RINT, IN_TILE);
        PipeBarrier<PIPE_V>();
        Cast(fa, cp, RoundMode::CAST_NONE, IN_TILE);
        PipeBarrier<PIPE_V>();
        Muls(fb, fa, (float)(1.0f / 65536.0f), IN_TILE);
        PipeBarrier<PIPE_V>();
        Cast(cp, fb, RoundMode::CAST_FLOOR, IN_TILE);
        PipeBarrier<PIPE_V>();
        Cast(fb, cp, RoundMode::CAST_NONE, IN_TILE);
        PipeBarrier<PIPE_V>();
        Axpy(fa, fb, (float)(-65536.0f), IN_TILE);
        PipeBarrier<PIPE_V>();
        Cast(cp, fa, RoundMode::CAST_RINT, IN_TILE);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void ProcessTile(uint32_t t)
    {
        auto load  = bufLoad_.Get<int16_t>();
        auto xh    = bufHalf_.Get<half>();
        auto xi    = bufXi_.Get<half>();
        auto xq    = bufXq_.Get<half>();
        auto yI    = bufYi_.Get<half>();
        auto yQall = bufYqall_.Get<half>();
        auto fa    = bufFa_.Get<float>();
        auto fb    = bufFb_.Get<float>();
        auto cmat  = bufCmat_.Get<int32_t>();
        auto tQall = bufTQall_.Get<int32_t>();


        uint32_t m0     = inStart_ + t * IN_TILE;
        int32_t  gStart = (int32_t)m0 - (int32_t)HALO_IN;

        Duplicate(load, (int16_t)0, LOAD_I16);
        PipeBarrier<PIPE_V>();
        Wait<HardEvent::V_MTE2>();
        if (gStart >= 0) {
            DataCopy(load, inG_[(uint32_t)gStart * 2], WINDOW_CPX * 2);
        } else {
            DataCopy(load[HALO_IN * 2], inG_[0], IN_TILE * 2);
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


        for (uint32_t p = 0; p < N_PHASE; ++p) {
            auto yQp = yQall[p * IN_TILE];
            Duplicate(yI,  (half)0.0, IN_TILE);
            Duplicate(yQp, (half)0.0, IN_TILE);
            PipeBarrier<PIPE_V>();
            for (uint32_t n = 0; n < NPT; ++n) {
                half h = hpn_[p * NPT + n];
                uint32_t off = HALO_IN - n;
                Axpy(yI,  xi[off], h, IN_TILE);
                Axpy(yQp, xq[off], h, IN_TILE);
                PipeBarrier<PIPE_V>();
            }
            auto cp = cmat[p * IN_TILE];
            MaskI(cp, yI, fa, fb);
        }

        Cast(tQall, yQall, RoundMode::CAST_RINT, CMAT_I32);
        PipeBarrier<PIPE_V>();
        for (int s = 0; s < 16; ++s) {
            Add(tQall, tQall, tQall, CMAT_I32);
            PipeBarrier<PIPE_V>();
        }
        Add(cmat, cmat, tQall, CMAT_I32);
        PipeBarrier<PIPE_V>();


        auto idx   = bufIdx_.Get<uint32_t>();
        auto out32 = bufOut_.Get<int32_t>();
        Gather(out32, cmat, idx, (uint32_t)0, OUT_TILE);
        PipeBarrier<PIPE_V>();


        auto outI16 = out32.ReinterpretCast<int16_t>();
        Wait<HardEvent::V_MTE3>();
        uint32_t dstBase = m0 * INTERP * 2;
        for (uint32_t done = 0; done < OUT_I16; ) {
            uint32_t c = (OUT_I16 - done > OUT_CHUNK_I16) ? OUT_CHUNK_I16 : (OUT_I16 - done);
            DataCopy(outIQG_[dstBase + done], outI16[done], c);
            done += c;
        }
        Wait<HardEvent::MTE3_MTE2>();
    }

    TPipe *pipe_;
    uint32_t blockId_;
    uint32_t inStart_;

    GlobalTensor<int16_t>  inG_, outIQG_;
    GlobalTensor<half>     tapsG_;
    GlobalTensor<uint32_t> idxTG_;

    TBuf<TPosition::VECCALC> bufLoad_, bufHalf_, bufXi_, bufXq_;
    TBuf<TPosition::VECCALC> bufYi_, bufYqall_, bufFa_, bufFb_;
    TBuf<TPosition::VECCALC> bufCmat_, bufTQall_, bufIdx_, bufOut_, bufTaps_;

    half hpn_[L_TAP];
};


extern "C" __global__ __aicore__ void interpolate_kernel(
    GM_ADDR in_iq_gm,
    GM_ADDR fir_taps_gm,
    GM_ADDR out_iq_gm,
    GM_ADDR ws_gm,
    GM_ADDR tiling_gm)
{
    TPipe pipe;
    Interpolate op;
    op.Init(in_iq_gm, fir_taps_gm, out_iq_gm, tiling_gm, &pipe);
    op.Process();
}