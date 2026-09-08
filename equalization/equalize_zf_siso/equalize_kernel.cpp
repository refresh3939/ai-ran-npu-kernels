





























#include "kernel_operator.h"
#include "equalize.h"

using namespace AscendC;

namespace {
using namespace airan;
constexpr uint32_t N_PER_CORE = SC_PER_CORE * N_SYMBOL;
}


class Equalize {
public:
    __aicore__ inline Equalize() {}

    __aicore__ inline void Init(GM_ADDR y_re_gm, GM_ADDR y_im_gm,
                                GM_ADDR h_re_gm, GM_ADDR h_im_gm,
                                GM_ADDR n0_gm,
                                GM_ADDR x_re_gm, GM_ADDR x_im_gm,
                                GM_ADDR no_eff_gm,
                                TPipe *pipe);
    __aicore__ inline void Process();

private:
    TPipe   *pipe_;
    uint32_t blockId_;
    uint32_t sc_start_;

    GlobalTensor<half> yReG_, yImG_, hReG_, hImG_, n0G_;
    GlobalTensor<half> xReG_, xImG_, neG_;


    TBuf<TPosition::VECCALC> bufYre_, bufYim_, bufHre_, bufHim_, bufN0_;



    TBuf<TPosition::VECCALC> bufHr32_, bufHi32_, bufYr32_, bufYi32_, bufG32_, bufAcc32_;
};


__aicore__ inline void Equalize::Init(
    GM_ADDR y_re_gm, GM_ADDR y_im_gm,
    GM_ADDR h_re_gm, GM_ADDR h_im_gm,
    GM_ADDR n0_gm,
    GM_ADDR x_re_gm, GM_ADDR x_im_gm,
    GM_ADDR no_eff_gm,
    TPipe *pipe)
{
    pipe_    = pipe;
    blockId_ = GetBlockIdx();
    sc_start_ = blockId_ * SC_PER_CORE;

    yReG_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(y_re_gm),  STREAM_HALF_LEN);
    yImG_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(y_im_gm),  STREAM_HALF_LEN);
    hReG_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(h_re_gm),  STREAM_HALF_LEN);
    hImG_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(h_im_gm),  STREAM_HALF_LEN);
    n0G_ .SetGlobalBuffer(reinterpret_cast<__gm__ half *>(n0_gm),    STREAM_HALF_LEN);
    xReG_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(x_re_gm),  STREAM_HALF_LEN);
    xImG_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(x_im_gm),  STREAM_HALF_LEN);
    neG_ .SetGlobalBuffer(reinterpret_cast<__gm__ half *>(no_eff_gm),STREAM_HALF_LEN);

    pipe_->InitBuffer(bufYre_, N_PER_CORE * sizeof(half));
    pipe_->InitBuffer(bufYim_, N_PER_CORE * sizeof(half));
    pipe_->InitBuffer(bufHre_, N_PER_CORE * sizeof(half));
    pipe_->InitBuffer(bufHim_, N_PER_CORE * sizeof(half));
    pipe_->InitBuffer(bufN0_,  N_PER_CORE * sizeof(half));
    pipe_->InitBuffer(bufHr32_,  N_PER_CORE * sizeof(float));
    pipe_->InitBuffer(bufHi32_,  N_PER_CORE * sizeof(float));
    pipe_->InitBuffer(bufYr32_,  N_PER_CORE * sizeof(float));
    pipe_->InitBuffer(bufYi32_,  N_PER_CORE * sizeof(float));
    pipe_->InitBuffer(bufG32_,   N_PER_CORE * sizeof(float));
    pipe_->InitBuffer(bufAcc32_, N_PER_CORE * sizeof(float));

}


__aicore__ inline void Equalize::Process()
{
    auto y_re = bufYre_.Get<half>();
    auto y_im = bufYim_.Get<half>();
    auto h_re = bufHre_.Get<half>();
    auto h_im = bufHim_.Get<half>();
    auto n0   = bufN0_ .Get<half>();
    auto hr   = bufHr32_ .Get<float>();
    auto hi   = bufHi32_ .Get<float>();
    auto yr   = bufYr32_ .Get<float>();
    auto yi   = bufYi32_ .Get<float>();
    auto g    = bufG32_  .Get<float>();
    auto acc  = bufAcc32_.Get<float>();
    const uint32_t L = SC_PER_CORE;
    const uint32_t N = N_PER_CORE;


    for (uint32_t sym = 0; sym < N_SYMBOL; ++sym) {
        uint32_t g_off = sym * N_SC_PAD + sc_start_;
        uint32_t u_off = sym * L;
        DataCopy(y_re[u_off], yReG_[g_off], L);
        DataCopy(y_im[u_off], yImG_[g_off], L);
        DataCopy(h_re[u_off], hReG_[g_off], L);
        DataCopy(h_im[u_off], hImG_[g_off], L);
        DataCopy(n0  [u_off], n0G_ [g_off], L);
    }
    PipeBarrier<PIPE_ALL>();



    Cast(hr, h_re, RoundMode::CAST_NONE, N);
    Cast(hi, h_im, RoundMode::CAST_NONE, N);
    Cast(yr, y_re, RoundMode::CAST_NONE, N);
    Cast(yi, y_im, RoundMode::CAST_NONE, N);
    PipeBarrier<PIPE_V>();


    Mul(g, hr, hr, N);
    PipeBarrier<PIPE_V>();
    MulAddDst(g, hi, hi, N);
    PipeBarrier<PIPE_V>();
    Maxs(g, g, (float)G_FLOOR, N);
    PipeBarrier<PIPE_V>();


    Mul(acc, hr, yr, N);
    PipeBarrier<PIPE_V>();
    MulAddDst(acc, hi, yi, N);
    PipeBarrier<PIPE_V>();
    Div(acc, acc, g, N);
    PipeBarrier<PIPE_V>();
    Cast(h_re, acc, RoundMode::CAST_NONE, N);
    PipeBarrier<PIPE_V>();


    Mul(acc, hr, yi, N);
    PipeBarrier<PIPE_V>();
    Mul(yi, hi, yr, N);
    PipeBarrier<PIPE_V>();
    Sub(acc, acc, yi, N);
    PipeBarrier<PIPE_V>();
    Div(acc, acc, g, N);
    PipeBarrier<PIPE_V>();
    Cast(h_im, acc, RoundMode::CAST_NONE, N);
    PipeBarrier<PIPE_V>();


    Cast(yr, n0, RoundMode::CAST_NONE, N);
    PipeBarrier<PIPE_V>();
    Div(yr, yr, g, N);
    PipeBarrier<PIPE_V>();
    Cast(n0, yr, RoundMode::CAST_NONE, N);
    PipeBarrier<PIPE_ALL>();


    for (uint32_t sym = 0; sym < N_SYMBOL; ++sym) {
        uint32_t g_off = sym * N_SC_PAD + sc_start_;
        uint32_t u_off = sym * L;
        DataCopy(xReG_[g_off], h_re[u_off], L);
        DataCopy(xImG_[g_off], h_im[u_off], L);
        DataCopy(neG_ [g_off], n0  [u_off], L);
    }
    PipeBarrier<PIPE_ALL>();
}


extern "C" __global__ __aicore__ void equalize_kernel(
    GM_ADDR y_re_gm, GM_ADDR y_im_gm,
    GM_ADDR h_re_gm, GM_ADDR h_im_gm,
    GM_ADDR n0_gm,
    GM_ADDR x_re_gm, GM_ADDR x_im_gm,
    GM_ADDR no_eff_gm,
    GM_ADDR ws,
    GM_ADDR tilingGm)
{
    (void)ws;
    (void)tilingGm;
    TPipe pipe;
    Equalize op;
    op.Init(y_re_gm, y_im_gm, h_re_gm, h_im_gm, n0_gm,
            x_re_gm, x_im_gm, no_eff_gm, &pipe);
    op.Process();
}
