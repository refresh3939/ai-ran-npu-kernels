





#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "pss_acquisition.h"

extern "C" void GenerateTiling(const char *  , uint8_t *buf)
{
    std::memset(buf, 0, airan::TILING_TOTAL_SIZE);
    std::printf("[tiling] pss_acquisition: vector-only kernel, no cube tiling\n");
}
