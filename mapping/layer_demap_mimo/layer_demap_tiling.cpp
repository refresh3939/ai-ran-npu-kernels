#include "layer_demap.h"

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>

extern "C" void GenerateTiling(const char * /*socVersion*/, uint8_t *buf)
{
    std::memset(buf, 0, airan::layer_demap::TILING_BYTES);
    std::printf("[tiling] layer_demap: %zu-byte runtime metadata/index buffer\n",
                airan::layer_demap::TILING_BYTES);
}
