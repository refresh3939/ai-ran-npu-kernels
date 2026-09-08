#include <acl/acl.h>
#include "aclrtlaunch_mimo_dmrs_ls_kernel.h"
#include "mimo_dmrs_ls.h"
#include "pusch_mimo_runtime_config.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;
namespace ls = airan::mimo_dmrs_ls;

#define ACL_OK(call) do { const aclError e_ = (call); if (e_ != ACL_ERROR_NONE) { \
  std::fprintf(stderr, "[HARD_FAIL] %s=%d at %s:%d\n", #call, e_, __FILE__, __LINE__); \
  return 1; } } while (0)

template <class T>
static bool ReadExact(const fs::path &path, std::vector<T> *value, size_t count) {
  std::ifstream input(path, std::ios::binary | std::ios::ate);
  if (!input || static_cast<size_t>(input.tellg()) != count * sizeof(T)) {
    std::fprintf(stderr, "[HARD_FAIL] %s expected exactly %zu bytes\n",
                 path.c_str(), count * sizeof(T));
    return false;
  }
  value->resize(count);
  input.seekg(0);
  input.read(reinterpret_cast<char *>(value->data()),
             static_cast<std::streamsize>(count * sizeof(T)));
  return input.good();
}

template <class T>
static bool WriteExact(const fs::path &path, const std::vector<T> &value) {
  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  output.write(reinterpret_cast<const char *>(value.data()),
               static_cast<std::streamsize>(value.size() * sizeof(T)));
  return output.good();
}

static airan::PuschMimoConfig LegacyConfig(uint16_t slot, uint16_t rank) {
  airan::PuschMimoConfig c{};
  c.abi_version = ls::ABI_VERSION;
  c.struct_size = sizeof(c);
  c.num_layers = rank;
  c.num_tx_ports = rank == 3 ? 4 : rank;
  c.num_rx_antennas = 64;
  c.qm = 8;
  c.num_symbols = 14;
  c.fft_size = 2048;
  c.num_rb = 133;
  c.slot_number = slot;
  c.num_allocated_symbols = 14;
  c.used_subcarriers = 1596;
  c.padded_subcarriers = 1664;
  c.dmrs_symbol_mask = static_cast<uint16_t>((1u << 2) | (1u << 11));
  for (uint16_t layer = 0; layer < rank; ++layer) {
    c.dmrs_ports[layer] = (rank == 2 && layer == 1)
                              ? 1002
                              : static_cast<uint16_t>(1000 + layer);
  }
  c.dmrs_scrambling_id = 321;
  c.data_scrambling_id = 321;
  c.rnti = 12345;
  c.dmrs_type = 1;
  c.dmrs_length = 1;
  c.num_cdm_groups_without_data = 2;
  return c;
}

int main(int argc, char **argv) {
  if (argc != 5) {
    std::fprintf(stderr, "usage: %s RX_GRID_DIR DMRS_DIR OUTPUT_DIR RANK\n", argv[0]);
    return 2;
  }
  const uint16_t rank = static_cast<uint16_t>(std::stoul(argv[4]));
  if (rank < 1 || rank > 4) return 2;
  airan::PuschMimoRuntimeConfig runtime{};
  bool runtime_enabled = false;
  std::string why;
  if (airan::LoadMimoRuntimeConfigFromEnv(rank, &runtime, &runtime_enabled,
                                          &why) !=
      airan::MimoRuntimeConfigStatus::kSuccess) {
    std::fprintf(stderr, "[HARD_FAIL] runtime config: %s\n", why.c_str());
    return 1;
  }
  if (runtime_enabled &&
      (runtime.rx_bucket != ls::NR_CURRENT || runtime.layer_bucket != 16 ||
       runtime.num_dmrs_symbols != ls::CURRENT_DMRS_SYMBOLS ||
       runtime.dmrs_port_count != rank || runtime.num_symbols != ls::N_SYMBOLS ||
       runtime.used_subcarriers != ls::N_SC_USED ||
       runtime.padded_subcarriers != ls::N_SC_PAD)) {
    std::fprintf(stderr,
                 "[HARD_FAIL] runtime config exceeds current DMRS-LS profile\n");
    return 1;
  }
  const fs::path rx(argv[1]), dmrs(argv[2]), output(argv[3]);
  fs::create_directories(output);
  constexpr size_t rx_elements = static_cast<size_t>(ls::NR_CURRENT) * 14u * 1664u;
  const size_t ref_elements = static_cast<size_t>(rank) * 2u * 896u;
  const size_t h_elements = ls::NaturalElems(rank);
  const size_t sc_elements = static_cast<size_t>(rank) * 2u * 832u;

  ACL_OK(aclInit(nullptr)); ACL_OK(aclrtSetDevice(0));
  aclrtStream stream = nullptr; ACL_OK(aclrtCreateStream(&stream));
  void *drr = nullptr, *dri = nullptr, *dref_r = nullptr, *dref_i = nullptr;
  void *dhr = nullptr, *dhi = nullptr, *dsc = nullptr, *dcount = nullptr;
  void *dnoise = nullptr, *dworkspace = nullptr, *dmetadata = nullptr;
#define ALLOC(pointer, bytes) ACL_OK(aclrtMalloc(&(pointer), (bytes), ACL_MEM_MALLOC_HUGE_FIRST))
  ALLOC(drr, rx_elements * 2); ALLOC(dri, rx_elements * 2);
  ALLOC(dref_r, ref_elements * 2); ALLOC(dref_i, ref_elements * 2);
  ALLOC(dhr, h_elements * 2); ALLOC(dhi, h_elements * 2);
  ALLOC(dsc, sc_elements * 2); ALLOC(dcount, ls::COUNT_PAD * 2);
  ALLOC(dnoise, ls::NR_CURRENT * 2); ALLOC(dworkspace, 128); ALLOC(dmetadata, ls::TILING_BYTES);

  std::vector<uint16_t> rr, ri, ref_r, ref_i;
  std::vector<uint16_t> hr(h_elements), hi(h_elements), sc(sc_elements);
  std::vector<uint16_t> count(ls::COUNT_PAD), noise(ls::NR_CURRENT);
  std::vector<uint16_t> expected_sc(ls::MAX_LAYERS * 2 * 832);
  std::vector<uint16_t> expected_count(ls::COUNT_PAD);
  for (uint16_t slot = 0; slot < 23; ++slot) {
    char name[32];
    std::snprintf(name, sizeof(name), "slot%02u_re.bin", slot);
    if (!ReadExact(rx / name, &rr, rx_elements) ||
        !ReadExact(dmrs / name, &ref_r, ref_elements)) return 1;
    std::snprintf(name, sizeof(name), "slot%02u_im.bin", slot);
    if (!ReadExact(rx / name, &ri, rx_elements) ||
        !ReadExact(dmrs / name, &ref_i, ref_elements)) return 1;
    airan::PuschMimoConfig config{};
    if (runtime_enabled) {
      if (airan::DerivePuschMimoConfig(runtime, runtime.rx_bucket,
                                       &config, &why, slot) !=
          airan::MimoRuntimeConfigStatus::kSuccess) {
        std::fprintf(stderr, "[HARD_FAIL] runtime config derivation: %s\n",
                     why.c_str());
        return 1;
      }
    } else {
      config = LegacyConfig(slot, rank);
    }
    airan::PuschMimoLayout layout{};
    ls::KernelMetadata metadata{};
    if (ls::BuildCurrentProfile(config, &layout, &metadata,
                                expected_count.data(), expected_sc.data()) != ls::OK ||
        ls::ValidateNaturalLmmseContract(expected_count.data(), expected_sc.data(),
                                         rank, 2) != ls::OK) return 1;
    for (uint16_t layer = 0; layer < rank; ++layer) {
      const uint16_t expected = metadata.observation_model[layer] == ls::FD_OCC2_399
                                    ? 399 : 798;
      if (expected_count[layer * 2] != expected ||
          expected_count[layer * 2 + 1] != expected) return 1;
    }
    ls::MimoDmrsLsOpArgsV1 args{};
    args.abi_version = ls::ABI_VERSION; args.struct_size = sizeof(args);
    args.rx_grid_re = drr; args.rx_grid_im = dri;
    args.dmrs_ref_re = dref_r; args.dmrs_ref_im = dref_i;
    args.h_ls_re = dhr; args.h_ls_im = dhi; args.pilot_sc = dsc;
    args.pilot_count = dcount; args.noise_var_rx = dnoise;
    args.config = &config; args.layout = &layout; args.stream = stream;
    if (ls::ValidateOpArgs(args) != ls::OK) return 1;
    ACL_OK(aclrtMemcpy(drr, rx_elements * 2, rr.data(), rx_elements * 2, ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_OK(aclrtMemcpy(dri, rx_elements * 2, ri.data(), rx_elements * 2, ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_OK(aclrtMemcpy(dref_r, ref_elements * 2, ref_r.data(), ref_elements * 2, ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_OK(aclrtMemcpy(dref_i, ref_elements * 2, ref_i.data(), ref_elements * 2, ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_OK(aclrtMemcpy(dmetadata, ls::TILING_BYTES, &metadata, ls::TILING_BYTES, ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_OK(aclrtMemset(dhr, h_elements * 2, 0x5a, h_elements * 2));
    ACL_OK(aclrtMemset(dhi, h_elements * 2, 0xa5, h_elements * 2));
    ACL_OK(aclrtMemset(dsc, sc_elements * 2, 0, sc_elements * 2));
    ACL_OK(aclrtMemset(dcount, ls::COUNT_PAD * 2, 0, ls::COUNT_PAD * 2));
    ACL_OK(aclrtMemset(dnoise, ls::NR_CURRENT * 2, 0, ls::NR_CURRENT * 2));
    const uint32_t launch = ACLRT_LAUNCH_KERNEL(mimo_dmrs_ls_kernel)(
        ls::BLOCK_DIM, stream, drr, dri, dref_r, dref_i, dhr, dhi,
        dsc, dcount, dnoise, dworkspace, dmetadata);
    if (launch != ACL_ERROR_NONE) return 1;
    ACL_OK(aclrtSynchronizeStream(stream));
    ACL_OK(aclrtMemcpy(hr.data(), h_elements * 2, dhr, h_elements * 2, ACL_MEMCPY_DEVICE_TO_HOST));
    ACL_OK(aclrtMemcpy(hi.data(), h_elements * 2, dhi, h_elements * 2, ACL_MEMCPY_DEVICE_TO_HOST));
    ACL_OK(aclrtMemcpy(sc.data(), sc_elements * 2, dsc, sc_elements * 2, ACL_MEMCPY_DEVICE_TO_HOST));
    ACL_OK(aclrtMemcpy(count.data(), ls::COUNT_PAD * 2, dcount, ls::COUNT_PAD * 2, ACL_MEMCPY_DEVICE_TO_HOST));
    ACL_OK(aclrtMemcpy(noise.data(), ls::NR_CURRENT * 2, dnoise,
                       ls::NR_CURRENT * 2, ACL_MEMCPY_DEVICE_TO_HOST));
    if (!std::equal(sc.begin(), sc.end(), expected_sc.begin()) || count != expected_count ||
        !std::any_of(hr.begin(), hr.end(), [](uint16_t v) { return v != 0; })) return 1;
    std::snprintf(name, sizeof(name), "slot%02u_h_re.bin", slot); if (!WriteExact(output / name, hr)) return 1;
    std::snprintf(name, sizeof(name), "slot%02u_h_im.bin", slot); if (!WriteExact(output / name, hi)) return 1;
    std::snprintf(name, sizeof(name), "slot%02u_noise.bin", slot); if (!WriteExact(output / name, noise)) return 1;
  }
  if (!WriteExact(output / "pilot_sc.bin", sc) ||
      !WriteExact(output / "pilot_count.bin", count)) return 1;
  for (void *pointer : {dmetadata, dworkspace, dnoise, dcount, dsc, dhi, dhr,
                        dref_i, dref_r, dri, drr}) aclrtFree(pointer);
  ACL_OK(aclrtDestroyStream(stream)); ACL_OK(aclrtResetDevice(0)); ACL_OK(aclFinalize());
  std::printf("[PASS] mimo_dmrs_ls Rank%u slots=23 natural observation contract "
              "runtime=%s\n", rank, runtime_enabled ? "profile" : "legacy");
  return 0;
}
