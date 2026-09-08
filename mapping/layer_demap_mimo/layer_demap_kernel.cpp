






#include "kernel_operator.h"
#include "layer_demap.h"

using namespace AscendC;
namespace ldm = airan::layer_demap;

extern "C" __global__ __aicore__ void layer_demap_kernel(
    GM_ADDR layer_llr_gm, GM_ADDR cw_llr_gm, GM_ADDR workspace_gm, GM_ADDR tiling_gm)
{
    (void)workspace_gm;
    const uint32_t core = GetBlockIdx();
    if (core >= ldm::BLOCK_DIM) return;

    TPipe pipe;
    TBuf<TPosition::VECCALC> meta_buf, row_buf, source_buf, output_buf, index_buf;
    pipe.InitBuffer(meta_buf, ldm::META_WORDS * sizeof(uint32_t));
    pipe.InitBuffer(row_buf, ldm::N_SC_LLR_PAD * sizeof(int16_t));
    pipe.InitBuffer(source_buf, ldm::SOURCE_ELEMS * sizeof(int16_t));
    pipe.InitBuffer(output_buf, ldm::OUTPUT_ELEMS * sizeof(int16_t));
    pipe.InitBuffer(index_buf, ldm::MAX_INDEX_ELEMS * sizeof(uint32_t));

    GlobalTensor<uint32_t> tiling_g;
    tiling_g.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t *>(tiling_gm),
                            ldm::META_WORDS + ldm::MAX_INDEX_ELEMS);
    auto meta = meta_buf.Get<uint32_t>();
    DataCopy(meta, tiling_g, ldm::META_WORDS);
    PipeBarrier<PIPE_ALL>();

    if (meta.GetValue(0) != ldm::META_MAGIC) return;
    const uint32_t num_layers = meta.GetValue(1);
    const uint32_t qm = meta.GetValue(2);
    const uint32_t num_data_re = meta.GetValue(3);
    const uint32_t data_stride = meta.GetValue(4);
    const uint32_t codeword_symbols = meta.GetValue(5);
    const uint32_t codeword_stride = meta.GetValue(6);
    const uint32_t input_symbol_stride = meta.GetValue(7);
    const uint32_t symbols_per_group = meta.GetValue(8);
    const uint32_t group_data_re = meta.GetValue(9);
    const uint32_t gather_chunk = meta.GetValue(10);
    const uint32_t index_elems = meta.GetValue(11);
    if (num_layers == 0 || num_layers > ldm::MAX_LAYERS || qm != ldm::Q_M ||
        num_data_re != ldm::N_DATA_RE || data_stride != ldm::N_DATA_PAD ||
        codeword_symbols != num_layers * num_data_re ||
        codeword_stride != num_layers * ldm::N_DATA_PAD ||
        input_symbol_stride != ldm::N_SC_LLR_PAD ||
        symbols_per_group != ldm::SYMBOLS_PER_GROUP ||
        group_data_re != ldm::GROUP_DATA_RE || gather_chunk != ldm::GATHER_CHUNK ||
        index_elems != num_layers * gather_chunk) return;

    GlobalTensor<int16_t> layer_g, cw_g;
    layer_g.SetGlobalBuffer(reinterpret_cast<__gm__ int16_t *>(layer_llr_gm),
                            ldm::MAX_LAYERS * ldm::Q_M * ldm::N_DATA_PAD);
    cw_g.SetGlobalBuffer(reinterpret_cast<__gm__ int16_t *>(cw_llr_gm),
                         ldm::Q_M * ldm::MAX_LAYERS * ldm::N_DATA_PAD);
    auto row = row_buf.Get<int16_t>();
    auto source = source_buf.Get<int16_t>();
    auto output = output_buf.Get<int16_t>();
    auto index = index_buf.Get<uint32_t>();
    DataCopy(index, tiling_g[ldm::META_WORDS], index_elems);
    PipeBarrier<PIPE_ALL>();

    const uint32_t layer_stride = qm * data_stride;
    const uint32_t q_values[2] = {core, core + ldm::BLOCK_DIM};
    for (uint32_t owned = 0; owned < 2; ++owned) {
        const uint32_t q = q_values[owned];
        const uint32_t q_base = q * data_stride;
        for (uint32_t group = 0; group < ldm::NUM_GROUPS; ++group) {

            for (uint32_t layer = 0; layer < num_layers; ++layer) {
                const uint32_t layer_base = layer * layer_stride + q_base;
                const uint32_t source_base = layer * ldm::GROUP_DATA_RE;
                for (uint32_t local_symbol = 0;
                     local_symbol < ldm::SYMBOLS_PER_GROUP; ++local_symbol) {
                    const uint32_t data_symbol = group * ldm::SYMBOLS_PER_GROUP + local_symbol;
                    DataCopy(row,
                             layer_g[layer_base + data_symbol * ldm::N_SC_LLR_PAD],
                             ldm::N_SC_LLR_PAD);
                    PipeBarrier<PIPE_ALL>();
                    Adds(source[source_base + local_symbol * ldm::N_SC_USED], row,
                         static_cast<int16_t>(0), ldm::N_SC_USED);
                    PipeBarrier<PIPE_ALL>();
                }
            }

            for (uint32_t base = 0; base < ldm::GROUP_DATA_RE;
                 base += ldm::GATHER_CHUNK) {
                const bool full = base + ldm::GATHER_CHUNK <= ldm::GROUP_DATA_RE;
                if (num_layers == 1) {
                    if (full) {
                        constexpr uint32_t count = ldm::GATHER_CHUNK;
                        Gather(output[base], source[base], index, static_cast<uint32_t>(0), count);
                    } else {
                        constexpr uint32_t count = ldm::GATHER_TAIL;
                        Gather(output[base], source[base], index, static_cast<uint32_t>(0), count);
                    }
                } else if (num_layers == 2) {
                    if (full) {
                        constexpr uint32_t count = 2 * ldm::GATHER_CHUNK;
                        Gather(output[2 * base], source[base], index, static_cast<uint32_t>(0), count);
                    } else {
                        constexpr uint32_t count = 2 * ldm::GATHER_TAIL;
                        Gather(output[2 * base], source[base], index, static_cast<uint32_t>(0), count);
                    }
                } else if (num_layers == 3) {
                    if (full) {
                        constexpr uint32_t count = 3 * ldm::GATHER_CHUNK;
                        Gather(output[3 * base], source[base], index, static_cast<uint32_t>(0), count);
                    } else {
                        constexpr uint32_t count = 3 * ldm::GATHER_TAIL;
                        Gather(output[3 * base], source[base], index, static_cast<uint32_t>(0), count);
                    }
                } else {
                    if (full) {
                        constexpr uint32_t count = 4 * ldm::GATHER_CHUNK;
                        Gather(output[4 * base], source[base], index, static_cast<uint32_t>(0), count);
                    } else {
                        constexpr uint32_t count = 4 * ldm::GATHER_TAIL;
                        Gather(output[4 * base], source[base], index, static_cast<uint32_t>(0), count);
                    }
                }
                PipeBarrier<PIPE_ALL>();
            }
            DataCopy(cw_g[q * codeword_stride + group * num_layers * ldm::GROUP_DATA_RE],
                     output, num_layers * ldm::GROUP_DATA_RE);
            PipeBarrier<PIPE_ALL>();
        }

        const uint32_t tail = codeword_stride - num_layers * num_data_re;
        Duplicate(output, static_cast<int16_t>(0), tail);
        PipeBarrier<PIPE_ALL>();
        DataCopy(cw_g[q * codeword_stride + num_layers * num_data_re], output, tail);
        PipeBarrier<PIPE_ALL>();
    }
}
