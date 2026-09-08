



#include <cstdlib>
#include <cstring>
#include "cfo_compensate.h"

namespace cfo_compensate {

extern "C" void GenerateTiling(const char *  , uint8_t *buf)
{
    std::memset(buf, 0, TILING_TOTAL_SIZE);
}

extern "C" size_t GetTilingSize()    { return TILING_TOTAL_SIZE; }
extern "C" size_t GetWorkspaceSize() { return WS_TOTAL; }

}
