






















#include "kernel_operator.h"
#include "cfo_compensate.h"

using namespace AscendC;
using namespace cfo_compensate;


class CfoCompensate {
public:
    __aicore__ inline CfoCompensate() {}

    __aicore__ inline void Init(GM_ADDR in_iq_gm, GM_ADDR swap_idx_gm,
                                 GM_ADDR a2_gm, GM_ADDR b2_gm,
                                 GM_ADDR out_iq_gm, TPipe *pipe)
    {
        pipe_ = pipe;
        const uint32_t raw_bid = GetBlockIdx();
        aiv_id_ = raw_bid ^ 2;

        if      (aiv_id_ == 0) { st_start_ = 0;  st_cnt_ = 4; }
        else if (aiv_id_ == 1) { st_start_ = 4;  st_cnt_ = 3; }
        else if (aiv_id_ == 2) { st_start_ = 7;  st_cnt_ = 4; }
        else                   { st_start_ = 11; st_cnt_ = 4; }

        inG_  .SetGlobalBuffer(reinterpret_cast<__gm__ int16_t *>(in_iq_gm),   IN_INT16_LEN);

        swapIdxG_.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t *>(swap_idx_gm), 2 * SUB_TILE);
        a2G_  .SetGlobalBuffer(reinterpret_cast<__gm__ half *>(a2_gm),         OUT_INT16_LEN);
        b2G_  .SetGlobalBuffer(reinterpret_cast<__gm__ half *>(b2_gm),         OUT_INT16_LEN);
        outG_ .SetGlobalBuffer(reinterpret_cast<__gm__ int16_t *>(out_iq_gm),  OUT_INT16_LEN);

        pipe_->InitBuffer(bufXi16_,  2 * SUB_TILE * sizeof(int16_t));
        pipe_->InitBuffer(bufSwapIdx_, 2 * SUB_TILE * sizeof(uint32_t));
        pipe_->InitBuffer(bufXfull_, 2 * SUB_TILE * sizeof(half));
        pipe_->InitBuffer(bufXswap_, 2 * SUB_TILE * sizeof(half));
        pipe_->InitBuffer(bufA2_,    2 * SUB_TILE * sizeof(half));
        pipe_->InitBuffer(bufB2_,    2 * SUB_TILE * sizeof(half));
        pipe_->InitBuffer(bufP_,     2 * SUB_TILE * sizeof(half));
        pipe_->InitBuffer(bufOIQ16_, 2 * SUB_TILE * sizeof(int16_t));


        auto swidx = bufSwapIdx_.Get<uint32_t>();
        DataCopy(swidx, swapIdxG_, 2 * SUB_TILE);
    }

    __aicore__ inline void Process()
    {
        for (uint32_t i = 0; i < st_cnt_; ++i) {
            ProcessSubTile(st_start_ + i);
        }
    }

private:
    __aicore__ inline void ProcessSubTile(uint32_t st)
    {
        const uint32_t g2 = st * SUB_TILE * 2;

        auto xi16  = bufXi16_.Get<int16_t>();
        auto xfull = bufXfull_.Get<half>();
        auto xswap = bufXswap_.Get<half>();
        auto swidx = bufSwapIdx_.Get<uint32_t>();
        auto a2    = bufA2_.Get<half>();
        auto b2    = bufB2_.Get<half>();
        auto p     = bufP_.Get<half>();
        auto oiq16 = bufOIQ16_.Get<int16_t>();


        DataCopy(xi16, inG_[g2], 2 * SUB_TILE);
        DataCopy(a2,   a2G_[g2], 2 * SUB_TILE);
        DataCopy(b2,   b2G_[g2], 2 * SUB_TILE);
        event_t eIn = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(eIn); WaitFlag<HardEvent::MTE2_V>(eIn);



        Cast(xfull, xi16, RoundMode::CAST_NONE, 2 * SUB_TILE);
        PipeBarrier<PIPE_V>();
        Gather(xswap, xfull, swidx, (uint32_t)0, 2 * SUB_TILE);
        PipeBarrier<PIPE_V>();


        Mul(p, xfull, a2, 2 * SUB_TILE);        PipeBarrier<PIPE_V>();
        MulAddDst(p, xswap, b2, 2 * SUB_TILE);  PipeBarrier<PIPE_V>();


        Cast(oiq16, p, RoundMode::CAST_RINT, 2 * SUB_TILE);  PipeBarrier<PIPE_V>();


        event_t eOut = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(eOut); WaitFlag<HardEvent::V_MTE3>(eOut);
        DataCopy(outG_[g2], oiq16, 2 * SUB_TILE);
        event_t eNext = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
        SetFlag<HardEvent::MTE3_MTE2>(eNext); WaitFlag<HardEvent::MTE3_MTE2>(eNext);
    }

    TPipe   *pipe_;
    uint32_t aiv_id_;
    uint32_t st_start_, st_cnt_;

    GlobalTensor<int16_t>  inG_, outG_;
    GlobalTensor<uint32_t> swapIdxG_;
    GlobalTensor<half>    a2G_, b2G_;

    TBuf<TPosition::VECCALC> bufXi16_,  bufSwapIdx_;
    TBuf<TPosition::VECCALC> bufXfull_, bufXswap_;
    TBuf<TPosition::VECCALC> bufA2_,    bufB2_;
    TBuf<TPosition::VECCALC> bufP_,     bufOIQ16_;
};





extern "C" __global__ __aicore__ void cfo_compensate_kernel(
    GM_ADDR in_iq_gm, GM_ADDR swap_idx_gm,
    GM_ADDR a2_gm, GM_ADDR b2_gm,
    GM_ADDR out_iq_gm,
    GM_ADDR ws, GM_ADDR tilingGm)
{
    TPipe pipe;
    CfoCompensate op;
    op.Init(in_iq_gm, swap_idx_gm, a2_gm, b2_gm, out_iq_gm, &pipe);
    op.Process();
}
