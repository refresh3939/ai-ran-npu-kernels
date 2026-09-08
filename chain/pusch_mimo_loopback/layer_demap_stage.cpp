#include <acl/acl.h>
#include "aclrtlaunch_layer_demap_kernel.h"
#include "layer_demap.h"
#include "pusch_mimo_runtime_config.h"

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

namespace l = airan::layer_demap;
#define ACL_OK(call) do { const aclError e_ = (call); if (e_ != ACL_ERROR_NONE) { \
  std::fprintf(stderr, "[HARD_FAIL] %s=%d\n", #call, e_); return 1; } } while (0)

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
      runtime->data_re_per_layer != 19152 ||
      runtime->data_stride_per_layer != 19200) {
    std::fprintf(stderr,
                 "[HARD_FAIL] runtime config exceeds current layer-demap profile\n");
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
  airan::PuschMimoRuntimeConfig runtime{};
  airan::PuschMimoConfig config{};
  bool runtime_enabled = false;
  if (!LoadActiveConfig(rank, &runtime, &runtime_enabled, &config)) return 1;
  std::ifstream input(argv[1], std::ios::binary | std::ios::ate);
  if (!input || static_cast<size_t>(input.tellg()) != l::LayerElems(rank) * 2) return 1;
  std::vector<int16_t> layer_llr(l::LayerElems(rank));
  input.seekg(0); input.read(reinterpret_cast<char *>(layer_llr.data()),
                             static_cast<std::streamsize>(layer_llr.size() * 2));
  airan::PuschMimoLayout layout{};
  std::vector<uint8_t> tiling(l::TILING_BYTES);
  auto *metadata = reinterpret_cast<l::KernelMetadata *>(tiling.data());
  auto *gather = reinterpret_cast<uint32_t *>(tiling.data() + sizeof(*metadata));
  if (l::BuildCurrentProfile(config, &layout, metadata, gather) != l::OK) return 1;
  std::vector<int16_t> reference(l::CodewordElems(rank));
  if (l::ReferenceDemap(layer_llr.data(), config, layout, reference.data()) != l::OK) return 1;
  ACL_OK(aclInit(nullptr)); ACL_OK(aclrtSetDevice(0));
  aclrtStream stream = nullptr; ACL_OK(aclrtCreateStream(&stream));
  void *dinput = nullptr, *doutput = nullptr, *dtiling = nullptr;
  ACL_OK(aclrtMalloc(&dinput, layer_llr.size() * 2, ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_OK(aclrtMalloc(&doutput, reference.size() * 2, ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_OK(aclrtMalloc(&dtiling, tiling.size(), ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_OK(aclrtMemcpy(dinput, layer_llr.size() * 2, layer_llr.data(), layer_llr.size() * 2, ACL_MEMCPY_HOST_TO_DEVICE));
  ACL_OK(aclrtMemcpy(dtiling, tiling.size(), tiling.data(), tiling.size(), ACL_MEMCPY_HOST_TO_DEVICE));
  const uint32_t launch = ACLRT_LAUNCH_KERNEL(layer_demap_kernel)(
      l::BLOCK_DIM, stream, dinput, doutput, nullptr, dtiling);
  if (launch != ACL_ERROR_NONE) return 1;
  ACL_OK(aclrtSynchronizeStream(stream));
  std::vector<int16_t> result(reference.size());
  ACL_OK(aclrtMemcpy(result.data(), result.size() * 2, doutput, result.size() * 2, ACL_MEMCPY_DEVICE_TO_HOST));
  if (result != reference) return 1;
  std::ofstream output(argv[2], std::ios::binary | std::ios::trunc);
  output.write(reinterpret_cast<const char *>(result.data()),
               static_cast<std::streamsize>(result.size() * 2));
  if (!output) return 1;
  for (void *pointer : {dtiling, doutput, dinput}) aclrtFree(pointer);
  ACL_OK(aclrtDestroyStream(stream)); ACL_OK(aclrtResetDevice(0)); ACL_OK(aclFinalize());
  std::printf("[PASS] layer_demap_mimo Rank%u device/reference bit-exact runtime=%s\n",
              rank, runtime_enabled ? "profile" : "legacy");
  return 0;
}
