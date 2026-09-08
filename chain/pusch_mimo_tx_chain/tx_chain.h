#pragma once

#include <cstddef>
#include <cstdint>

namespace airan::pusch_mimo_tx_chain {

constexpr uint32_t NUM_SLOTS = 23;
constexpr uint32_t MAX_LAYERS = 4;
constexpr uint32_t MAX_PORTS = 4;
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
};




int Run(const RunOptions &options);

}
