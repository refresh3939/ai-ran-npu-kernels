





#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "cfo_estimate.h"

extern "C" void GenerateTiling(const char *  , uint8_t *buf)
{

    std::memset(buf, 0, airan::TILING_TOTAL_SIZE);
    std::printf("[tiling] cfo_estimate: vector-only kernel, no cube tiling\n");
}