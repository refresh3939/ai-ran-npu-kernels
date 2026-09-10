#pragma once

#include <cstddef>
#include <cstdint>

namespace airan::pusch_mimo_tx_chain {

constexpr uint32_t NUM_SLOTS = 23;
constexpr uint32_t MAX_LAYERS = 4;
constexpr uint32_t MAX_LOGICAL_PORTS = 4;
constexpr uint32_t MAX_TX_ANTENNAS = 16;
constexpr uint32_t N_DATA_PAD = 19200;
constexpr uint32_t N_GRID = 14 * 1664;
constexpr uint32_t N_FFT_GRID = 14 * 2048;
constexpr uint32_t N_TIME_SAMPLES = 30720;

struct RunOptions {
    const char *input_root = nullptr;
    const char *output_iq = nullptr;
    const char *receipt_json = nullptr;
    uint16_t rank = 1;
    uint16_t num_slots = NUM_SLOTS;
    uint16_t data_scrambling_id = 321;
    uint16_t dmrs_scrambling_id = 321;
    uint16_t rnti = 12345;
    bool cell_id_override = false;
    bool rnti_override = false;
};

// Runs one complete MIMO TX transport block in a single ACL context and
// stream. Intermediate tensors never leave device memory. output_iq contains
// int16 interleaved IQ in [slot,physical_tx_antenna,time_sample,iq] order.
int Run(const RunOptions &options);

}  // namespace airan::pusch_mimo_tx_chain
