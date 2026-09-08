



#pragma once

#include <cstddef>
#include <cstdint>

#include "../../common/pusch_mimo_types.h"

namespace airan::re_map_batch {

constexpr uint32_t ABI_VERSION = PUSCH_MIMO_ABI_VERSION;
constexpr uint32_t MAX_PORTS = 4;
constexpr uint32_t N_SYMBOLS = 14;
constexpr uint32_t N_FFT = 2048;
constexpr uint32_t N_SC_USED = 1596;
constexpr uint32_t N_SC_PAD = 1664;
constexpr uint32_t GRID_PORT_STRIDE = N_SYMBOLS * N_SC_PAD;
constexpr uint32_t FFT_PORT_STRIDE = N_SYMBOLS * N_FFT;
constexpr uint32_t SISO_BLOCK_DIM = 4;
constexpr uint32_t SCATTER_INDEX_ELEMS = N_FFT;
constexpr uint32_t ZERO_SLOT = N_SC_PAD;


constexpr size_t DUMMY_WORKSPACE_BYTES = 1024;
constexpr size_t DUMMY_TILING_BYTES = 1024;

using PuschMimoConfig = ::airan::PuschMimoConfig;
using PuschMimoLayout = ::airan::PuschMimoLayout;

enum Status : int32_t {
    OK = 0,
    INVALID_ARGUMENT = -1,
    UNSUPPORTED_PROFILE = -2,
    LAYOUT_MISMATCH = -3,
    RESOURCE_TOO_SMALL = -4,
    LAUNCH_FAILED = -5,
};





struct ReMapBatchOpArgsV1 {
    uint16_t abi_version;
    uint16_t struct_size;
    const void *port_grid_re;
    const void *port_grid_im;
    void *fft_grid_re;
    void *fft_grid_im;
    const PuschMimoConfig *config;
    const PuschMimoLayout *layout;
    void *stream;
};




Status BuildCurrentProfile(const PuschMimoConfig &config,
                           PuschMimoLayout *layout,
                           uint32_t *scatter_index,
                           size_t scatter_index_elems);

Status ValidateOpArgs(const ReMapBatchOpArgsV1 &args);



Status ReferenceMap(const uint16_t *port_grid_re,
                    const uint16_t *port_grid_im,
                    const PuschMimoConfig &config,
                    const PuschMimoLayout &layout,
                    const uint32_t *scatter_index,
                    size_t scatter_index_elems,
                    uint16_t *fft_grid_re,
                    uint16_t *fft_grid_im);




Status Enqueue(const ReMapBatchOpArgsV1 &args,
               const void *scatter_index,
               size_t scatter_index_bytes,
               void *workspace,
               size_t workspace_bytes,
               void *tiling,
               size_t tiling_bytes);

constexpr size_t PortGridElems(uint32_t ports) {
    return static_cast<size_t>(ports) * GRID_PORT_STRIDE;
}

constexpr size_t FftGridElems(uint32_t ports) {
    return static_cast<size_t>(ports) * FFT_PORT_STRIDE;
}

}
