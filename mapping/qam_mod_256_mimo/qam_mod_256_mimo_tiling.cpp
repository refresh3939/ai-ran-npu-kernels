#include "qam_mod_256_mimo.h"

#include <cstdio>
#include <cstring>

extern "C" void GenerateTiling(const char *  , uint8_t *buffer)
{
    std::memset(buffer, 0, airan::qam_mod_256_mimo::TILING_BYTES);
    std::printf("[tiling] qam_mod_256_mimo: %zu-byte runtime metadata buffer\n",
                airan::qam_mod_256_mimo::TILING_BYTES);
}
