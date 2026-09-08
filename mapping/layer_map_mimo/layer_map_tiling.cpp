#include "layer_map.h"

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>

extern "C" void GenerateTiling(const char *  , uint8_t *buf)
{
    std::memset(buf, 0, airan::layer_map::TILING_BYTES);
    std::printf("[tiling] layer_map: %zu-byte runtime metadata/index buffer\n",
                airan::layer_map::TILING_BYTES);
}
