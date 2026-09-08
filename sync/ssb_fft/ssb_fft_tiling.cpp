




#include <cstring>
#include <cstdio>
#include "tiling/platform/platform_ascendc.h"
#include "kernel_tiling/kernel_tiling.h"
#include "ssb_fft.h"

extern "C" void GenerateTiling(const char *  , uint8_t *buf)
{
    std::memset(buf, 0, 2 * sizeof(TCubeTiling));
    printf("[tiling] placeholder (kernel uses raw Mmad ISA, tiling unused)\n");
}
