





#include <cstdint>
#include <cstring>

extern "C" void GenerateTiling(const char *  , uint8_t *buf)
{
    if (buf != nullptr) {
        std::memset(buf, 0, 256);
    }
}
