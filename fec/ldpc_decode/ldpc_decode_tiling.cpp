/**
 * @file ldpc_decode_tiling.cpp
 * @brief Tiling stub. Matches encoder pattern 1:1.
 */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "ldpc_decode.h"

namespace airan {

extern "C" uint8_t* GenerateTiling(const char* socVersion, uint32_t blockDim)
{
    (void)socVersion; (void)blockDim;
    uint8_t* buf = static_cast<uint8_t*>(std::malloc(TILING_TOTAL_SIZE));
    if (!buf) return nullptr;
    std::memset(buf, 0, TILING_TOTAL_SIZE);
    return buf;
}

extern "C" size_t GetTilingSize()    { return TILING_TOTAL_SIZE; }
extern "C" size_t GetWorkspaceSize() { return WS_TOTAL; }

}  // namespace airan