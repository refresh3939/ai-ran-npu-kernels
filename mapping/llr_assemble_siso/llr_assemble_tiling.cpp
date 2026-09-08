


#include <cstdlib>
#include <cstring>
#include "llr_assemble.h"

namespace airan {

extern "C" uint8_t* GenerateTiling(const char*  , uint32_t  ) {
    uint8_t* buf = (uint8_t*)std::malloc(TILING_TOTAL_SIZE);
    std::memset(buf, 0, TILING_TOTAL_SIZE);
    return buf;
}

extern "C" size_t GetTilingSize()    { return TILING_TOTAL_SIZE; }
extern "C" size_t GetWorkspaceSize() { return WS_TOTAL; }

}
