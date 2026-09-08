#include <cstdint>
#include <cstdio>
#include <cstring>

#include "channel_est_lmmse.h"

extern "C" void GenerateTiling(const char *, uint8_t *buf)
{
    std::memset(buf, 0, airan::channel_est_lmmse::TILING_BYTES);
    std::printf("[tiling] m%u_k%u_r%u, blockDim=%u\n",
                airan::channel_est_lmmse::NR, airan::channel_est_lmmse::NL,
                airan::channel_est_lmmse::RANK, airan::channel_est_lmmse::BLOCK_DIM);
}
