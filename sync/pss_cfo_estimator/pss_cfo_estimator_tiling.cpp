













#include <cstdint>
#include <cstdio>
#include <cstring>
#include "pss_cfo_estimator.h"

extern "C" void GenerateTiling(const char *  , uint8_t *buf)
{
    std::memset(buf, 0, airan::TILING_TOTAL_SIZE);
    std::printf("[tiling] pss_cfo_estimator: empty placeholder (%zu B, all zero)\n",
                airan::TILING_TOTAL_SIZE);
}
