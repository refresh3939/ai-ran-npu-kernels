#include "scramble_mimo.h"

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>

extern "C" void GenerateTiling(const char *  , uint8_t *buf)
{
    std::memset(buf, 0, airan::scramble_mimo::TILING_BYTES);
    std::printf("[tiling] scramble_mimo: %zu-byte runtime metadata buffer\n",
                airan::scramble_mimo::TILING_BYTES);
}
