










#include <cstdint>
#include "interpolate.h"

extern "C" void GenerateTiling(const char *  , uint8_t *buf)
{
    if (buf == nullptr) return;
    uint32_t *idx = reinterpret_cast<uint32_t *>(buf);
    for (uint32_t k = 0; k < interpolate::IDX_LEN; ++k) {
        uint32_t p = k % interpolate::INTERP;
        uint32_t m = k / interpolate::INTERP;
        idx[k] = (p * interpolate::IN_TILE + m) * static_cast<uint32_t>(sizeof(int32_t));
    }
}