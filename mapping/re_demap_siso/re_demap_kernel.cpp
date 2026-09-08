













#include "kernel_operator.h"
#include "re_demap.h"

using namespace AscendC;
using namespace re_demap;


class ReDemap {
public:
    __aicore__ inline ReDemap() {}

    __aicore__ inline void Init(GM_ADDR in_re_gm, GM_ADDR in_im_gm, GM_ADDR idx_gm,
                                 GM_ADDR out_re_gm, GM_ADDR out_im_gm, TPipe *pipe);
    __aicore__ inline void Process();

private:
    __aicore__ inline void GatherOne(const GlobalTensor<half> &inG,
                                      const GlobalTensor<half> &outG, uint32_t sym);

    TPipe   *pipe_;
    uint32_t blockId_;
    uint32_t sym_start_;

    GlobalTensor<half>     inReG_, inImG_;
    GlobalTensor<half>     outReG_, outImG_;
    GlobalTensor<uint32_t> idxG_;

    TBuf<TPosition::VECCALC> bufSrc_;
    TBuf<TPosition::VECCALC> bufDst_;
    TBuf<TPosition::VECCALC> bufIdx_;
};


__aicore__ inline void ReDemap::Init(
    GM_ADDR in_re_gm, GM_ADDR in_im_gm, GM_ADDR idx_gm,
    GM_ADDR out_re_gm, GM_ADDR out_im_gm, TPipe *pipe)
{
    pipe_      = pipe;
    blockId_   = GetBlockIdx();
    sym_start_ = blockId_ * SYMBOLS_PER_CORE;

    inReG_ .SetGlobalBuffer(reinterpret_cast<__gm__ half *>(in_re_gm),  N_SYMBOL * N_FFT);
    inImG_ .SetGlobalBuffer(reinterpret_cast<__gm__ half *>(in_im_gm),  N_SYMBOL * N_FFT);
    outReG_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(out_re_gm), N_SYMBOL * N_SC_PAD);
    outImG_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(out_im_gm), N_SYMBOL * N_SC_PAD);
    idxG_  .SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t *>(idx_gm), N_SC_PAD);

    pipe_->InitBuffer(bufSrc_, N_FFT    * sizeof(half));
    pipe_->InitBuffer(bufDst_, N_SC_PAD * sizeof(half));
    pipe_->InitBuffer(bufIdx_, N_SC_PAD * sizeof(uint32_t));
}


__aicore__ inline void ReDemap::GatherOne(
    const GlobalTensor<half> &inG, const GlobalTensor<half> &outG, uint32_t sym)
{
    auto src   = bufSrc_.Get<half>();
    auto dst   = bufDst_.Get<half>();
    auto idxUB = bufIdx_.Get<uint32_t>();


    DataCopy(src, inG[sym * N_FFT], N_FFT);
    event_t e0 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(e0); WaitFlag<HardEvent::MTE2_V>(e0);



    Gather(dst, src, idxUB, static_cast<uint32_t>(0), N_SC_PAD);
    PipeBarrier<PIPE_V>();


    event_t e1 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(e1); WaitFlag<HardEvent::V_MTE3>(e1);
    DataCopy(outG[sym * N_SC_PAD], dst, N_SC_PAD);
    event_t e2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
    SetFlag<HardEvent::MTE3_MTE2>(e2); WaitFlag<HardEvent::MTE3_MTE2>(e2);
}


__aicore__ inline void ReDemap::Process()
{

    auto idxUB = bufIdx_.Get<uint32_t>();
    DataCopy(idxUB, idxG_, N_SC_PAD);
    event_t ei = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(ei); WaitFlag<HardEvent::MTE2_V>(ei);

    for (uint32_t s = 0; s < SYMBOLS_PER_CORE; ++s) {
        uint32_t sym = sym_start_ + s;
        if (sym >= N_SYMBOL) break;
        GatherOne(inReG_, outReG_, sym);
        GatherOne(inImG_, outImG_, sym);
    }
}


extern "C" __global__ __aicore__ void re_demap_kernel(
    GM_ADDR in_re_gm, GM_ADDR in_im_gm, GM_ADDR idx_gm,
    GM_ADDR out_re_gm, GM_ADDR out_im_gm,
    GM_ADDR ws_gm, GM_ADDR tiling_gm)
{
    (void)ws_gm; (void)tiling_gm;
    TPipe pipe;
    ReDemap op;
    op.Init(in_re_gm, in_im_gm, idx_gm, out_re_gm, out_im_gm, &pipe);
    op.Process();
}