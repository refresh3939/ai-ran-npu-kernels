#include "descramble_mimo.h"

#include <cstdio>
#include <cstring>

extern "C" void GenerateTiling(const char * /*socVersion*/, uint8_t *buffer)
{
    std::memset(buffer, 0, airan::descramble_mimo::TILING_BYTES);
    std::printf("[tiling] descramble_mimo: %zu-byte runtime descriptor\n",
                airan::descramble_mimo::TILING_BYTES);
}
