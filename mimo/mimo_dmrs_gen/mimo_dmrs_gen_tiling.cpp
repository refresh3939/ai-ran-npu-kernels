




#include <cstdint>
#include <cstdio>
#include <cstring>

namespace {
constexpr size_t TILING_TOTAL_SIZE = 128;
}

extern "C" void GenerateTiling(const char *  , uint8_t *buf)
{
    std::memset(buf, 0, TILING_TOTAL_SIZE);
    std::printf("[tiling] mimo_dmrs_gen: 128-byte runtime metadata buffer\n");
}
