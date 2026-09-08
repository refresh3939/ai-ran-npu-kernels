#include "qam256_demod_batch.h"

#include <cstdio>
#include <cstring>

extern "C" void GenerateTiling(const char *  , uint8_t *buffer)
{
    std::memset(buffer, 0, airan::qam256_demod_batch::TILING_BYTES);
    std::printf("[tiling] qam_demod_256_batch: %zu-byte batch descriptor\n",
                airan::qam256_demod_batch::TILING_BYTES);
}
