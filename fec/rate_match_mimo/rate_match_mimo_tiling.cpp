#include "rate_match_mimo.h"

#include <cstdio>
#include <cstring>

extern "C" void GenerateTiling(const char *  , uint8_t *buffer)
{
    std::memset(buffer, 0, airan::rate_match_mimo::TILING_BYTES);
    std::printf("[tiling] rate_match_mimo: %zu-byte runtime metadata buffer\n",
                airan::rate_match_mimo::TILING_BYTES);
}
