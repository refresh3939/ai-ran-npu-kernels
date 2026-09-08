#include "rate_dematch_mimo.h"

#include <cstdint>
#include <cstring>

extern "C" void GenerateTiling(const char *  , uint8_t *buffer)
{
    std::memset(buffer, 0, airan::rate_dematch_mimo::TILING_BYTES);
}
