









#include "kernel_operator.h"
using namespace AscendC;

namespace {
constexpr int32_t  N_SAMP_SLOT   = 30720;
constexpr int32_t  IQ_LEN        = 2 * N_SAMP_SLOT;
constexpr int32_t  TILE          = 7680;
constexpr int32_t  N_TILE        = IQ_LEN / TILE;
}

class WinSlice {
public:
    __aicore__ inline WinSlice() {}

    __aicore__ inline void Init(GM_ADDR capture_gm, GM_ADDR delta_T_gm,
                                GM_ADDR slot_iq_gm, GM_ADDR rd_ptr_gm) {
        capBase_ = reinterpret_cast<__gm__ int16_t *>(capture_gm);
        dtG_ .SetGlobalBuffer(reinterpret_cast<__gm__ float   *>(delta_T_gm), 8);
        outG_.SetGlobalBuffer(reinterpret_cast<__gm__ int16_t *>(slot_iq_gm), IQ_LEN);
        rdG_ .SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(rd_ptr_gm),  8);


        pipe_.InitBuffer(buf0_, TILE * sizeof(int16_t));
        pipe_.InitBuffer(buf1_, TILE * sizeof(int16_t));
        pipe_.InitBuffer(bufH0_, TILE * sizeof(half));
        pipe_.InitBuffer(bufH1_, TILE * sizeof(half));
        pipe_.InitBuffer(bufO_,  TILE * sizeof(int16_t));
        pipe_.InitBuffer(qRd_,  1, 8   * sizeof(int32_t));
    }

    __aicore__ inline void Process() {
        if (GetBlockIdx() != 0) return;


        int32_t rd_in   = rdG_.GetValue(0);
        int32_t frac_q  = rdG_.GetValue(1);
        float   frac_in = (float)frac_q / 65536.0f;
        float   dt      = dtG_.GetValue(0);
        float   rd_f    = (float)rd_in + frac_in + (float)N_SAMP_SLOT + dt;
        int32_t rd_int  = (int32_t)(rd_f);
        if ((float)rd_int > rd_f) rd_int -= 1;
        float   frac    = rd_f - (float)rd_int;
        int32_t base    = 2 * rd_int;

        half frac_h = (half)frac;
        half omf_h  = (half)(1.0f - frac);


        GlobalTensor<int16_t> src0, src1;
        src0.SetGlobalBuffer(capBase_ + base,     IQ_LEN + 2);
        src1.SetGlobalBuffer(capBase_ + base + 2, IQ_LEN);

        LocalTensor<int16_t> ub0 = buf0_.Get<int16_t>();
        LocalTensor<int16_t> ub1 = buf1_.Get<int16_t>();
        LocalTensor<half>    h0  = bufH0_.Get<half>();
        LocalTensor<half>    h1  = bufH1_.Get<half>();
        LocalTensor<int16_t> uo  = bufO_.Get<int16_t>();

        for (int32_t t = 0; t < N_TILE; ++t) {

            DataCopy(ub0, src0[t * TILE], TILE);
            DataCopy(ub1, src1[t * TILE], TILE);
            SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
            WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);


            Cast(h0, ub0, RoundMode::CAST_NONE, TILE);
            Cast(h1, ub1, RoundMode::CAST_NONE, TILE);
            PipeBarrier<PIPE_V>();
            Muls(h0, h0, omf_h, TILE);
            Muls(h1, h1, frac_h, TILE);
            PipeBarrier<PIPE_V>();
            Add(h0, h0, h1, TILE);
            PipeBarrier<PIPE_V>();
            Cast(uo, h0, RoundMode::CAST_RINT, TILE);
            SetFlag<HardEvent::V_MTE3>(EVENT_ID1);
            WaitFlag<HardEvent::V_MTE3>(EVENT_ID1);


            DataCopy(outG_[t * TILE], uo, TILE);


            SetFlag<HardEvent::MTE3_V>(EVENT_ID3);
            WaitFlag<HardEvent::MTE3_V>(EVENT_ID3);
            if (t + 1 < N_TILE) {
                SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID2);
                WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID2);
            }
        }


        int32_t new_frac_q = (int32_t)(frac * 65536.0f + 0.5f);
        LocalTensor<int32_t> rb = qRd_.AllocTensor<int32_t>();
        for (int i = 0; i < 8; ++i) rb.SetValue(i, 0);
        rb.SetValue(0, rd_int);
        rb.SetValue(1, new_frac_q);
        qRd_.EnQue(rb);
        LocalTensor<int32_t> rb2 = qRd_.DeQue<int32_t>();
        DataCopy(rdG_, rb2, 8);
        qRd_.FreeTensor(rb2);
    }

private:
    TPipe pipe_;
    __gm__ int16_t* capBase_;
    GlobalTensor<int16_t> outG_;
    GlobalTensor<float>   dtG_;
    GlobalTensor<int32_t> rdG_;
    TQue<QuePosition::VECOUT, 1> qRd_;
    TBuf<TPosition::VECCALC> buf0_, buf1_, bufH0_, bufH1_, bufO_;
};

extern "C" __global__ __aicore__ void win_slice_kernel(
    GM_ADDR capture_gm, GM_ADDR delta_T_gm,
    GM_ADDR slot_iq_gm, GM_ADDR rd_ptr_gm,
    GM_ADDR ws, GM_ADDR tiling) {
    (void)ws; (void)tiling;
    WinSlice op;
    op.Init(capture_gm, delta_T_gm, slot_iq_gm, rd_ptr_gm);
    op.Process();
}
