


#include <cstdint>
#include <cstring>
#include "precode_zf.h"

extern "C" void GenerateTiling(const char*  , uint8_t* buf)
{
    std::memset(buf, 0, airan::TILING_TOTAL_SIZE);
}