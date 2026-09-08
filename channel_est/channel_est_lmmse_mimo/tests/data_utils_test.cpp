#include "data_utils.h"

#include <array>
#include <cstdio>
#include <filesystem>
#include <string>
#include <unistd.h>

int main()
{
    const std::filesystem::path path = std::filesystem::temp_directory_path() /
        ("channel_est_lmmse_io_" + std::to_string(static_cast<long long>(::getpid())) + ".bin");
    const std::filesystem::path missing = path.string() + ".missing";
    const std::array<unsigned char, 32> expected = {
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
        16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
    };
    std::array<unsigned char, 32> actual {};
    size_t actualSize = 999;
    bool pass = WriteFile(path.string(), expected.data(), expected.size());
    pass &= ReadFile(path.string(), actualSize, actual.data(), actual.size());
    pass &= actualSize == expected.size() && actual == expected;
    pass &= !ReadFile(path.string(), actualSize, actual.data(), actual.size() - 1);
    pass &= actualSize == 0;
    pass &= !ReadFile(missing.string(), actualSize, actual.data(), actual.size());
    pass &= actualSize == 0;
    if (std::remove(path.c_str()) != 0) pass = false;
    std::printf("=== %s exact I/O and missing-file contract ===\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}
