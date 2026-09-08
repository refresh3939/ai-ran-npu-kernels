#include <acl/acl.h>
#include "qam256_demod_batch.h"
#include "pusch_mimo_runtime_config.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <utility>
#include <vector>

namespace fs = std::filesystem;
namespace q = airan::qam256_demod_batch;
#define ACL_OK(call) do { const aclError e_ = (call); if (e_ != ACL_ERROR_NONE) { \
  std::fprintf(stderr, "[HARD_FAIL] %s=%d\n", #call, e_); return 1; } } while (0)

static bool ReadExact(const fs::path &path, std::vector<uint16_t> *value, size_t count) {
  std::ifstream input(path, std::ios::binary | std::ios::ate);
  if (!input || static_cast<size_t>(input.tellg()) != count * 2) return false;
  value->resize(count); input.seekg(0);
  input.read(reinterpret_cast<char *>(value->data()), static_cast<std::streamsize>(count * 2));
  return input.good();
}

static float Half(uint16_t bits) {
  const float sign = (bits & 0x8000u) ? -1.0f : 1.0f;
  const unsigned exponent = (bits >> 10) & 31u, fraction = bits & 1023u;
  if (exponent == 0) return sign * std::ldexp(static_cast<float>(fraction), -24);
  if (exponent == 31) return fraction ? NAN : sign * INFINITY;
  return sign * std::ldexp(static_cast<float>(1024u + fraction),
                           static_cast<int>(exponent) - 25);
}

static airan::PuschMimoConfig LegacyConfig(uint16_t rank) {
  airan::PuschMimoConfig c{};
  c.abi_version = 1; c.struct_size = sizeof(c); c.num_layers = rank;
  c.num_tx_ports = rank == 3 ? 4 : rank; c.num_rx_antennas = 64; c.qm = 8;
  c.num_symbols = 14; c.fft_size = 2048; c.num_rb = 133;
  c.num_allocated_symbols = 14; c.used_subcarriers = 1596; c.padded_subcarriers = 1664;
  c.dmrs_symbol_mask = static_cast<uint16_t>((1u << 2) | (1u << 11));
  for (uint16_t layer = 0; layer < rank; ++layer)
    c.dmrs_ports[layer] = rank == 2 && layer == 1 ? 1002 : static_cast<uint16_t>(1000 + layer);
  c.dmrs_type = 1; c.dmrs_length = 1; c.num_cdm_groups_without_data = 2;
  c.codebook_enabled = rank == 3 ? 1 : 0; c.tpmi = rank == 3 ? 6 : 0;
  c.prg_size_rb = rank == 3 ? 4 : 133;
  return c;
}

static bool LoadActiveConfig(uint16_t rank,
                             airan::PuschMimoRuntimeConfig *runtime,
                             bool *runtime_enabled,
                             airan::PuschMimoConfig *config) {
  std::string why;
  if (airan::LoadMimoRuntimeConfigFromEnv(rank, runtime, runtime_enabled, &why) !=
      airan::MimoRuntimeConfigStatus::kSuccess) {
    std::fprintf(stderr, "[HARD_FAIL] runtime config: %s\n", why.c_str());
    return false;
  }
  if (!*runtime_enabled) {
    *config = LegacyConfig(rank);
    return true;
  }
  if (runtime->rx_bucket < runtime->num_rx_antennas ||
      runtime->layer_bucket != 16 ||
      runtime->qm != 8 || runtime->num_symbols != 14 ||
      runtime->used_subcarriers != 1596 ||
      runtime->padded_subcarriers != 1664 ||
      runtime->grid_re_per_port != 23296) {
    std::fprintf(stderr,
                 "[HARD_FAIL] runtime config exceeds current QAM kernel profile\n");
    return false;
  }
  if (airan::DerivePuschMimoConfig(*runtime, runtime->rx_bucket,
                                   config, &why) !=
      airan::MimoRuntimeConfigStatus::kSuccess) {
    std::fprintf(stderr, "[HARD_FAIL] runtime config derivation: %s\n", why.c_str());
    return false;
  }
  return true;
}

int main(int argc, char **argv) {
  if (argc != 4) return 2;
  const uint16_t rank = static_cast<uint16_t>(std::stoul(argv[3]));
  if (rank < 1 || rank > 4) return 2;
  const fs::path input(argv[1]), output(argv[2]);
  airan::PuschMimoRuntimeConfig runtime{};
  airan::PuschMimoConfig config{};
  bool runtime_enabled = false;
  if (!LoadActiveConfig(rank, &runtime, &runtime_enabled, &config)) return 1;
  const size_t grid_elements = runtime_enabled ? runtime.grid_re_per_port : 23296u;
  const size_t layer_capacity = runtime_enabled ? runtime.layer_bucket : 16u;
  const size_t physical_elements = layer_capacity * grid_elements;
  const size_t active_elements = static_cast<size_t>(rank) * grid_elements;
  std::vector<uint16_t> all_re, all_im, all_noise;
  if (!ReadExact(input / "xhat_re.bin", &all_re, physical_elements) ||
      !ReadExact(input / "xhat_im.bin", &all_im, physical_elements) ||
      !ReadExact(input / "no_eff.bin", &all_noise, physical_elements)) {
    std::fprintf(stderr, "[HARD_FAIL] detector physical output shape\n"); return 1;
  }
  std::vector<uint16_t> re(all_re.begin(), all_re.begin() + active_elements);
  std::vector<uint16_t> im(all_im.begin(), all_im.begin() + active_elements);
  std::vector<uint16_t> noise(all_noise.begin(), all_noise.begin() + active_elements);
  for (uint16_t layer = 0; layer < rank; ++layer) {
    for (unsigned symbol = 0; symbol < config.num_symbols; ++symbol) {
      if (((config.dmrs_symbol_mask >> symbol) & 1u) != 0) continue;
      for (unsigned subcarrier = 0; subcarrier < config.used_subcarriers; ++subcarrier) {
        const size_t at = static_cast<size_t>(layer) * grid_elements +
                          symbol * config.padded_subcarriers + subcarrier;
        const float no_eff = Half(noise[at]);
        const float xr = Half(re[at]), xi = Half(im[at]);
        const float scale = 2.45428796f / no_eff;
        const float threshold = 0.188235294f / no_eff;
        if (!(no_eff > 0.0f) || !std::isfinite(scale) || !std::isfinite(threshold) ||
            std::abs(xr) * scale >= 999.0f || std::abs(xi) * scale >= 999.0f) {
          std::fprintf(stderr,
              "[HARD_FAIL] qam numeric domain layer=%u RE=%u,%u no_eff=%g scale=%g x=(%g,%g)\n",
              layer, symbol, subcarrier, no_eff, scale, xr, xi);
          return 1;
        }
      }
    }
  }
  airan::PuschMimoLayout layout{}; q::BatchTilingData tiling{};
  if (q::BuildCurrentProfile(config, &layout, &tiling) != q::OK) return 1;
  ACL_OK(aclInit(nullptr)); ACL_OK(aclrtSetDevice(0));
  aclrtStream stream = nullptr; ACL_OK(aclrtCreateStream(&stream));
  void *dre = nullptr, *dim = nullptr, *dnoise = nullptr, *dllr = nullptr;
  void *dworkspace = nullptr, *dtiling = nullptr;
  ACL_OK(aclrtMalloc(&dre, active_elements * 2, ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_OK(aclrtMalloc(&dim, active_elements * 2, ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_OK(aclrtMalloc(&dnoise, active_elements * 2, ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_OK(aclrtMalloc(&dllr, q::LlrElems(rank) * 2, ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_OK(aclrtMalloc(&dworkspace, q::WORKSPACE_BYTES, ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_OK(aclrtMalloc(&dtiling, q::TILING_BYTES, ACL_MEM_MALLOC_HUGE_FIRST));
  for (auto value : {std::pair<void *, std::vector<uint16_t> *>{dre, &re},
                     {dim, &im}, {dnoise, &noise}})
    ACL_OK(aclrtMemcpy(value.first, value.second->size() * 2, value.second->data(),
                       value.second->size() * 2, ACL_MEMCPY_HOST_TO_DEVICE));
  ACL_OK(aclrtMemcpy(dtiling, sizeof(tiling), &tiling, sizeof(tiling), ACL_MEMCPY_HOST_TO_DEVICE));
  q::QamDemod256BatchOpArgsV1 args{};
  args.abi_version = 1; args.struct_size = sizeof(args);
  args.x_re = dre; args.x_im = dim; args.no_eff = dnoise; args.layer_llr = dllr;
  args.config = &config; args.layout = &layout; args.stream = stream;
  if (q::Enqueue(args, dworkspace, q::WORKSPACE_BYTES, dtiling, q::TILING_BYTES) != q::OK) return 1;
  ACL_OK(aclrtSynchronizeStream(stream));
  std::vector<int16_t> llr(q::LlrElems(rank));
  ACL_OK(aclrtMemcpy(llr.data(), llr.size() * 2, dllr, llr.size() * 2, ACL_MEMCPY_DEVICE_TO_HOST));
  std::ofstream stream_out(output, std::ios::binary | std::ios::trunc);
  stream_out.write(reinterpret_cast<const char *>(llr.data()),
                   static_cast<std::streamsize>(llr.size() * 2));
  if (!stream_out || !std::any_of(llr.begin(), llr.end(), [](int16_t v) { return v != 0; })) return 1;
  for (void *pointer : {dtiling, dworkspace, dllr, dnoise, dim, dre}) aclrtFree(pointer);
  ACL_OK(aclrtDestroyStream(stream)); ACL_OK(aclrtResetDevice(0)); ACL_OK(aclFinalize());
  std::printf("[PASS] qam_demod_256_mimo_batch Rank%u full-grid runtime=%s\n",
              rank, runtime_enabled ? "profile" : "legacy");
  return 0;
}
