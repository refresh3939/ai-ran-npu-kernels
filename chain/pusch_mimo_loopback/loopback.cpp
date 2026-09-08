#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

static bool ExactNonzero(const fs::path& path, std::uintmax_t bytes) {
    if (!fs::is_regular_file(path) || fs::file_size(path) != bytes) return false;
    std::ifstream in(path, std::ios::binary);
    std::vector<char> data(static_cast<std::size_t>(bytes));
    in.read(data.data(), static_cast<std::streamsize>(data.size()));
    if (!in) return false;
    for (char value : data) if (value != 0) return true;
    return false;
}

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "usage: pusch_mimo_loopback_check WORK_ROOT\n";
        return 2;
    }
    const fs::path root(argv[1]);
    constexpr std::uintmax_t kTxBytes = 1u * 30720u * 2u * sizeof(std::int16_t);
    constexpr std::uintmax_t kRxBytes = 64u * 30720u * 2u * sizeof(std::int16_t);
    constexpr std::uintmax_t kGridPlaneBytes = 64u * 14u * 1664u * sizeof(std::uint16_t);
    const bool ok =
        ExactNonzero(root / "artifacts/tx_iq_rank1_slot0.bin", kTxBytes) &&
        ExactNonzero(root / "artifacts/rx_iq_rank1_slot0.bin", kRxBytes) &&
        ExactNonzero(root / "re_demap/data/ascend_output/rx_grid_re.bin", kGridPlaneBytes) &&
        fs::is_regular_file(root / "weight_manifest.json") &&
        fs::is_regular_file(root / "artifacts/milestone_result.json");
    if (!ok) {
        std::cerr << "[HARD_FAIL] Rank1 device milestone artifact contract failed\n";
        return 2;
    }
    std::cout << "[PASS] Rank1 single-slot device OFDM subchain artifacts are exact-size and nonzero\n";
    std::cout << "[NOTE] this is not the 23-slot coded TX-to-LDPC-decode E2E gate\n";
    return 0;
}
