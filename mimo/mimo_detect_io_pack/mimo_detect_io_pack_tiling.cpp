#include "mimo_detect_io_pack.h"

#include <cstdio>
#include <cstring>

extern "C" void GenerateTiling(const char * /*socVersion*/, uint8_t *buf)
{
    std::memset(buf, 0, airan::mimo_detect_io_pack::TILING_BYTES);
    std::printf("[tiling] metadata placeholder; host applies common layer plan, "
                "4 AI Cores, %u RE/core\n",
                airan::mimo_detect_io_pack::RE_PER_CORE);
}
