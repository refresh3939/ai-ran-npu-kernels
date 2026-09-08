#include <acl/acl.h>
#include "mimo_detect_io_pack.h"
#include "pusch_mimo_runtime_config.h"

#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <utility>
#include <vector>

namespace fs = std::filesystem;
namespace io = airan::mimo_detect_io_pack;

#define ACL_OK(call) do { \
  const aclError error_ = (call); \
  if (error_ != ACL_ERROR_NONE) { \
    std::fprintf(stderr, "[HARD_FAIL] %s=%d\n", #call, error_); \
    return 1; \
  } \
} while (0)

template <class T>
bool Read(const fs::path &path, std::vector<T> *value, size_t count) {
  std::ifstream stream(path, std::ios::binary | std::ios::ate);
  if (!stream || static_cast<size_t>(stream.tellg()) != count * sizeof(T)) {
    std::fprintf(stderr, "[HARD_FAIL] %s expected %zu bytes\n",
                 path.c_str(), count * sizeof(T));
    return false;
  }
  value->resize(count);
  stream.seekg(0);
  stream.read(reinterpret_cast<char *>(value->data()), count * sizeof(T));
  return stream.good();
}

template <class T>
bool Write(const fs::path &path, const std::vector<T> &value) {
  std::ofstream stream(path, std::ios::binary | std::ios::trunc);
  stream.write(reinterpret_cast<const char *>(value.data()),
               value.size() * sizeof(T));
  return stream.good();
}

airan::PuschMimoConfig LegacyConfig(uint16_t rank) {
  airan::PuschMimoConfig config{};
  config.abi_version = airan::PUSCH_MIMO_ABI_VERSION;
  config.struct_size = sizeof(config);
  config.num_layers = rank;
  config.num_tx_ports = rank == 3 ? 4 : rank;
  config.num_rx_antennas = 64;
  config.qm = 8;
  config.num_symbols = 14;
  config.fft_size = 2048;
  config.num_rb = 133;
  config.num_allocated_symbols = 14;
  config.used_subcarriers = 1596;
  config.padded_subcarriers = 1664;
  config.dmrs_symbol_mask = (1u << 2) | (1u << 11);
  for (uint16_t layer = 0; layer < rank; ++layer) {
    config.dmrs_ports[layer] =
        rank == 2 && layer == 1 ? 1002 : static_cast<uint16_t>(1000 + layer);
  }
  config.dmrs_type = 1;
  config.dmrs_length = 1;
  config.num_cdm_groups_without_data = 2;
  config.codebook_enabled = rank == 3 ? 1 : 0;
  config.tpmi = rank == 3 ? 6 : 0;
  config.prg_size_rb = rank == 3 ? 4 : 133;
  return config;
}

bool LoadActiveConfig(uint16_t requested_rank,
                      airan::PuschMimoRuntimeConfig *runtime,
                      bool *runtime_enabled,
                      airan::PuschMimoConfig *config) {
  std::string why;
  if (airan::LoadMimoRuntimeConfigFromEnv(
          requested_rank, runtime, runtime_enabled, &why) !=
      airan::MimoRuntimeConfigStatus::kSuccess) {
    std::fprintf(stderr, "[HARD_FAIL] runtime config: %s\n", why.c_str());
    return false;
  }
  if (!*runtime_enabled) {
    *config = LegacyConfig(requested_rank);
    return true;
  }
  if (runtime->rx_bucket != io::NR ||
      runtime->layer_bucket != io::NL ||
      runtime->num_dmrs_symbols != 2 ||
      runtime->dmrs_port_count > 4 ||
      runtime->num_symbols != 14 ||
      runtime->used_subcarriers != 1596 ||
      runtime->padded_subcarriers != 1664 ||
      runtime->grid_re_per_port != io::N_RE) {
    std::fprintf(stderr,
                 "[HARD_FAIL] runtime config exceeds current io_pack kernel profile\n");
    return false;
  }


  if (airan::DerivePuschMimoConfig(
          *runtime, runtime->rx_bucket, config, &why) !=
      airan::MimoRuntimeConfigStatus::kSuccess) {
    std::fprintf(stderr, "[HARD_FAIL] runtime config derivation: %s\n", why.c_str());
    return false;
  }
  return true;
}

int main(int argc, char **argv) {
  if (argc != 7) {
    std::fprintf(stderr, "usage: %s RX_RE RX_IM H_PREFIX NOISE OUT_DIR RANK\n", argv[0]);
    return 2;
  }
  const uint16_t rank = static_cast<uint16_t>(std::stoul(argv[6]));
  if (rank < 1 || rank > 4) return 2;
  const fs::path output(argv[5]);
  fs::create_directories(output);

  std::vector<uint16_t> rx_re, rx_im, h_re, h_im, noise_var;
  if (!Read(argv[1], &rx_re, io::RX_ELEMS) ||
      !Read(argv[2], &rx_im, io::RX_ELEMS) ||
      !Read(std::string(argv[3]) + "_re.bin", &h_re, io::H_ELEMS) ||
      !Read(std::string(argv[3]) + "_im.bin", &h_im, io::H_ELEMS) ||
      !Read(argv[4], &noise_var, io::NOISE_ELEMS)) {
    return 1;
  }

  airan::PuschMimoRuntimeConfig runtime{};
  airan::PuschMimoConfig config{};
  bool runtime_enabled = false;
  if (!LoadActiveConfig(rank, &runtime, &runtime_enabled, &config)) return 1;
  if (runtime_enabled) {
    const size_t physical_rx = runtime.num_rx_antennas;
    if (physical_rx == 0 || noise_var.size() != io::NR ||
        io::NR % physical_rx != 0) {
      std::fprintf(stderr, "[HARD_FAIL] physical RX noise-capacity adapter\n");
      return 1;
    }
    for (size_t rx = physical_rx; rx < noise_var.size(); ++rx) {
      noise_var[rx] = noise_var[rx % physical_rx];
    }
  }

  airan::MimoDetectLayerPlan plan{};
  plan.abi_version = airan::PUSCH_MIMO_ABI_VERSION;
  plan.struct_size = sizeof(plan);
  plan.num_rx_antennas = config.num_rx_antennas;
  plan.layer_capacity = runtime_enabled ? runtime.layer_bucket : 16;
  plan.total_layers = rank;
  plan.num_allocations = 1;
  plan.allocations[0] = {0, 0, rank, config.num_tx_ports};
  io::KernelMetadata metadata{};
  if (io::BuildCurrentProfile(&config, 1, plan, &metadata) != io::OK ||
      metadata.active_layers != rank) {
    std::fprintf(stderr, "[HARD_FAIL] io_pack profile\n");
    return 1;
  }

  ACL_OK(aclInit(nullptr));
  ACL_OK(aclrtSetDevice(0));
  aclrtStream stream = nullptr;
  ACL_OK(aclrtCreateStream(&stream));
  void *d_rx_re = nullptr, *d_rx_im = nullptr, *d_h_re = nullptr, *d_h_im = nullptr;
  void *d_noise = nullptr, *d_hrm_re = nullptr, *d_hrm_im = nullptr;
  void *d_y_re = nullptr, *d_y_im = nullptr, *d_no = nullptr, *d_metadata = nullptr;
#define ALLOC(value, bytes) ACL_OK(aclrtMalloc(&(value), (bytes), ACL_MEM_MALLOC_HUGE_FIRST))
  ALLOC(d_rx_re, io::RX_ELEMS * 2);
  ALLOC(d_rx_im, io::RX_ELEMS * 2);
  ALLOC(d_h_re, io::H_ELEMS * 2);
  ALLOC(d_h_im, io::H_ELEMS * 2);
  ALLOC(d_noise, io::NOISE_ELEMS * 2);
  ALLOC(d_hrm_re, io::PACKED_ELEMS * 2);
  ALLOC(d_hrm_im, io::PACKED_ELEMS * 2);
  ALLOC(d_y_re, io::PACKED_ELEMS * 2);
  ALLOC(d_y_im, io::PACKED_ELEMS * 2);
  ALLOC(d_no, io::NO_ELEMS * 2);
  ALLOC(d_metadata, io::TILING_BYTES);
#define UPLOAD(device, host) \
  ACL_OK(aclrtMemcpy((device), (host).size() * 2, (host).data(), \
                     (host).size() * 2, ACL_MEMCPY_HOST_TO_DEVICE))
  UPLOAD(d_rx_re, rx_re);
  UPLOAD(d_rx_im, rx_im);
  UPLOAD(d_h_re, h_re);
  UPLOAD(d_h_im, h_im);
  UPLOAD(d_noise, noise_var);
  ACL_OK(aclrtMemcpy(d_metadata, sizeof(metadata), &metadata, sizeof(metadata),
                     ACL_MEMCPY_HOST_TO_DEVICE));

  io::MimoDetectIoPackOpArgsV1 args{};
  args.abi_version = airan::PUSCH_MIMO_ABI_VERSION;
  args.struct_size = sizeof(args);
  args.rx_grid_re = d_rx_re;
  args.rx_grid_im = d_rx_im;
  args.h_grid_re = d_h_re;
  args.h_grid_im = d_h_im;
  args.noise_var_rx = d_noise;
  args.hrm_re = d_hrm_re;
  args.hrm_im = d_hrm_im;
  args.yvpad_re = d_y_re;
  args.yvpad_im = d_y_im;
  args.no = d_no;
  args.configs = &config;
  args.num_configs = 1;
  args.layer_plan = &plan;
  args.stream = stream;
  if (io::ValidateOpArgs(args) != io::OK ||
      io::Enqueue(args, d_metadata, io::TILING_BYTES) != io::OK) {
    std::fprintf(stderr, "[HARD_FAIL] io_pack enqueue\n");
    return 1;
  }
  ACL_OK(aclrtSynchronizeStream(stream));

  std::vector<uint16_t> packed(io::PACKED_ELEMS), no(io::NO_ELEMS);
  for (auto item : {std::pair<void *, const char *>{d_hrm_re, "hrm_re.bin"},
                    {d_hrm_im, "hrm_im.bin"}, {d_y_re, "yvpad_re.bin"},
                    {d_y_im, "yvpad_im.bin"}}) {
    ACL_OK(aclrtMemcpy(packed.data(), packed.size() * 2, item.first,
                       packed.size() * 2, ACL_MEMCPY_DEVICE_TO_HOST));
    if (!Write(output / item.second, packed)) return 1;
  }
  ACL_OK(aclrtMemcpy(no.data(), no.size() * 2, d_no, no.size() * 2,
                     ACL_MEMCPY_DEVICE_TO_HOST));
  if (!Write(output / "no.bin", no)) return 1;

  for (void *pointer : {d_metadata, d_no, d_y_im, d_y_re, d_hrm_im, d_hrm_re,
                        d_noise, d_h_im, d_h_re, d_rx_im, d_rx_re}) {
    aclrtFree(pointer);
  }
  ACL_OK(aclrtDestroyStream(stream));
  ACL_OK(aclrtResetDevice(0));
  ACL_OK(aclFinalize());
  std::printf("[PASS] mimo_detect_io_pack activeL=%u physicalRx=%u capacityRx=%u runtime=%s\n",
              rank, runtime_enabled ? runtime.num_rx_antennas : config.num_rx_antennas,
              config.num_rx_antennas, runtime_enabled ? "profile" : "legacy");
  return 0;
}
