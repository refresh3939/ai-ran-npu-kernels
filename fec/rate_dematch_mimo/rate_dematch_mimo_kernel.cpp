









#include "kernel_operator.h"
#include "rate_dematch_mimo.h"

using namespace AscendC;
namespace rdm = airan::rate_dematch_mimo;

namespace {
constexpr uint32_t ALIGN_ELEMS = 16;
constexpr uint32_t UB_BYTES =
    rdm::META_WORDS * sizeof(uint32_t) +
    rdm::DESC_PAD_WORDS * sizeof(uint32_t) +
    rdm::LDPC_N * sizeof(int16_t) +
    rdm::TILE_PAD_ELEMS * sizeof(int16_t);
static_assert(UB_BYTES <= 128u * 1024u,
              "rate_dematch_mimo UB working set exceeds dav_m200 capacity");

__aicore__ inline uint32_t Min2(uint32_t a, uint32_t b)
{
    return a < b ? a : b;
}
}

extern "C" __global__ __aicore__ void rate_dematch_mimo_kernel(
    GM_ADDR cw_llr_gm, GM_ADDR descriptor_gm, GM_ADDR ldpc_llr_gm,
    GM_ADDR workspace_gm, GM_ADDR tiling_gm)
{
    (void)workspace_gm;
    const uint32_t core = GetBlockIdx() ^ 2;
    if (core >= rdm::BLOCK_DIM) return;

    TPipe pipe;
    TBuf<TPosition::VECCALC> meta_buf, descriptor_buf, lam_buf, source_buf;
    pipe.InitBuffer(meta_buf, rdm::META_WORDS * sizeof(uint32_t));
    pipe.InitBuffer(descriptor_buf, rdm::DESC_PAD_WORDS * sizeof(uint32_t));
    pipe.InitBuffer(lam_buf, rdm::LDPC_N * sizeof(int16_t));
    pipe.InitBuffer(source_buf, rdm::TILE_PAD_ELEMS * sizeof(int16_t));

    GlobalTensor<uint32_t> metadata_g, descriptor_g;
    metadata_g.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t *>(tiling_gm),
                               rdm::META_WORDS);
    descriptor_g.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t *>(descriptor_gm),
                                 rdm::DESC_PAD_WORDS);
    auto metadata = meta_buf.Get<uint32_t>();
    auto descriptor = descriptor_buf.Get<uint32_t>();
    DataCopy(metadata, metadata_g, rdm::META_WORDS);
    DataCopy(descriptor, descriptor_g, rdm::DESC_PAD_WORDS);
    PipeBarrier<PIPE_ALL>();

    if (metadata.GetValue(0) != rdm::META_MAGIC) return;
    const uint32_t num_slots = metadata.GetValue(1);
    const uint32_t num_layers = metadata.GetValue(2);
    const uint32_t qm = metadata.GetValue(3);
    const uint32_t num_data_re = metadata.GetValue(4);
    const uint32_t codeword_symbols = metadata.GetValue(5);
    const uint32_t codeword_stride = metadata.GetValue(6);
    const uint32_t code_blocks = metadata.GetValue(8);
    const uint32_t ldpc_n = metadata.GetValue(9);
    const uint32_t n_2z = metadata.GetValue(10);
    const uint32_t profile_ncb = metadata.GetValue(11);
    const int16_t scale = static_cast<int16_t>(metadata.GetValue(12));
    const int16_t clip = static_cast<int16_t>(metadata.GetValue(13));
    const uint32_t tile_elems = metadata.GetValue(14);
    const uint32_t max_slots = metadata.GetValue(15);
    const uint32_t descriptor_words = metadata.GetValue(16);
    if (num_slots == 0 || num_slots > rdm::MAX_SLOTS ||
        max_slots != rdm::MAX_SLOTS || num_layers == 0 ||
        num_layers > rdm::MAX_LAYERS || qm != rdm::Q_M ||
        num_data_re != rdm::N_DATA_RE ||
        codeword_symbols != num_layers * rdm::N_DATA_RE ||
        codeword_stride != num_layers * rdm::N_DATA_PAD ||
        code_blocks != rdm::C_NUM || ldpc_n != rdm::LDPC_N ||
        n_2z != rdm::N_2Z || profile_ncb != rdm::N_CB_BUF ||
        scale != rdm::LLR_SCALE || clip != rdm::LLR_CLIP ||
        tile_elems != rdm::TILE_ELEMS ||
        descriptor_words != rdm::DESC_TABLE_WORDS) return;

    const uint32_t input_elems = num_slots * qm * codeword_stride;
    GlobalTensor<int16_t> input_g, output_g;
    input_g.SetGlobalBuffer(reinterpret_cast<__gm__ int16_t *>(cw_llr_gm),
                            input_elems);
    output_g.SetGlobalBuffer(reinterpret_cast<__gm__ int16_t *>(ldpc_llr_gm),
                             rdm::C_NUM * rdm::LDPC_N);
    auto lam = lam_buf.Get<int16_t>();
    auto source = source_buf.Get<int16_t>();

    for (uint32_t cb = core; cb < code_blocks; cb += rdm::BLOCK_DIM) {
        const uint32_t d = cb * rdm::DESC_WORDS;
        const uint32_t e = descriptor.GetValue(d);
        const uint32_t k0 = descriptor.GetValue(d + 1);
        const uint32_t ncb = descriptor.GetValue(d + 2);
        const uint32_t cw_symbol_offset = descriptor.GetValue(d + 3);
        if (e == 0 || e % qm != 0 || k0 >= ncb || ncb != profile_ncb) return;
        const uint32_t eq = e / qm;

        Duplicate(lam, static_cast<int16_t>(0), ldpc_n);
        PipeBarrier<PIPE_ALL>();

        for (uint32_t bit = 0; bit < qm; ++bit) {
            uint32_t consumed = 0;
            while (consumed < eq) {
                const uint32_t logical = cw_symbol_offset + consumed;
                const uint32_t slot = logical / codeword_symbols;
                const uint32_t symbol = logical - slot * codeword_symbols;
                const uint32_t circular = (k0 + bit * eq + consumed) % ncb;
                uint32_t count = Min2(eq - consumed, tile_elems);
                count = Min2(count, codeword_symbols - symbol);
                count = Min2(count, ncb - circular);

                const uint32_t input_offset =
                    slot * qm * codeword_stride + bit * codeword_stride + symbol;
                const uint32_t aligned_offset =
                    (input_offset / ALIGN_ELEMS) * ALIGN_ELEMS;
                const uint32_t delta = input_offset - aligned_offset;
                const uint32_t copy_count =
                    ((delta + count + ALIGN_ELEMS - 1) / ALIGN_ELEMS) * ALIGN_ELEMS;
                DataCopy(source, input_g[aligned_offset], copy_count);
                PipeBarrier<PIPE_ALL>();
                Muls(source[delta], source[delta], scale, count);
                Mins(source[delta], source[delta], clip, count);
                Maxs(source[delta], source[delta], static_cast<int16_t>(-clip), count);
                Add(lam[n_2z + circular], lam[n_2z + circular], source[delta], count);
                PipeBarrier<PIPE_ALL>();
                consumed += count;
            }
        }

        Mins(lam[n_2z], lam[n_2z], clip, ncb);
        Maxs(lam[n_2z], lam[n_2z], static_cast<int16_t>(-clip), ncb);
        PipeBarrier<PIPE_ALL>();
        DataCopy(output_g[cb * ldpc_n], lam, ldpc_n);
        PipeBarrier<PIPE_ALL>();
    }
}
