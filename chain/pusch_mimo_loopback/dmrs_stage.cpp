#include <acl/acl.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "mimo_dmrs_gen.h"
#include "pusch_mimo_runtime_config.h"

namespace fs = std::filesystem;
namespace mdg = airan::mimo_dmrs_gen;

#define ACL_OK(call) do { const aclError e_ = (call); if (e_ != ACL_ERROR_NONE) { \
  std::fprintf(stderr, "[HARD_FAIL] %s returned %d at %s:%d\n", #call, e_, __FILE__, __LINE__); \
  return 1; } } while (0)

static bool WriteExact(const fs::path &path, const std::vector<uint16_t> &value) {
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  out.write(reinterpret_cast<const char *>(value.data()),
            static_cast<std::streamsize>(value.size() * sizeof(uint16_t)));
  return out.good();
}

static airan::PuschMimoConfig LegacyConfig(uint16_t slot,uint16_t rank) {
  airan::PuschMimoConfig config{};
  config.abi_version = mdg::ABI_VERSION;
  config.struct_size = sizeof(config);
  config.num_layers = rank;
  config.num_tx_ports = rank==1?1:(rank==2?2:4);
  config.num_rx_antennas = 64;
  config.qm = 8;
  config.num_symbols = 14;
  config.fft_size = 2048;
  config.num_rb = 133;
  config.slot_number = slot;
  config.num_allocated_symbols = 14;
  config.used_subcarriers = 1596;
  config.padded_subcarriers = 1664;
  config.dmrs_symbol_mask = static_cast<uint16_t>((1u << 2) | (1u << 11));
  const uint16_t ports[4]={1000,1001,1002,1003};
  for(uint16_t i=0;i<rank;++i)config.dmrs_ports[i]=(rank==2&&i==1)?1002:ports[i];
  config.dmrs_scrambling_id = 321;
  config.data_scrambling_id = 321;
  config.rnti = 12345;
  config.dmrs_type = 1;
  config.dmrs_length = 1;
  config.num_cdm_groups_without_data = 2;
  return config;
}

int main(int argc, char **argv) {
  if (argc != 3) {
    std::fprintf(stderr, "usage: %s OUTPUT_DIR RANK\n", argv[0]);
    return 2;
  }
  const fs::path root(argv[1]);
  const uint16_t rank=static_cast<uint16_t>(std::stoul(argv[2]));if(rank<1||rank>4)return 2;
  airan::PuschMimoRuntimeConfig profile{};
  bool profile_enabled=false;
  std::string why;
  if(airan::LoadMimoRuntimeConfigFromEnv(rank,&profile,&profile_enabled,&why)!=
      airan::MimoRuntimeConfigStatus::kSuccess){
    std::fprintf(stderr,"[HARD_FAIL] runtime config: %s\n",why.c_str());return 1;}
  if(profile_enabled&&(profile.max_rx_antennas!=64||profile.max_layers!=16||
      profile.num_dmrs_symbols!=2||profile.dmrs_port_count!=rank||
      profile.num_symbols!=14||profile.used_subcarriers!=1596||
      profile.padded_subcarriers!=1664)){
    std::fprintf(stderr,"[HARD_FAIL] runtime config exceeds current DMRS profile\n");return 1;}
  fs::create_directories(root);
  ACL_OK(aclInit(nullptr));
  ACL_OK(aclrtSetDevice(0));
  aclrtStream stream = nullptr;
  ACL_OK(aclrtCreateStream(&stream));
  mdg::MimoDmrsGenRuntimeV1 *runtime = nullptr;
  if (mdg::CreateRuntime(&runtime) != mdg::OK || runtime == nullptr) {
    std::fprintf(stderr, "[HARD_FAIL] mimo_dmrs_gen CreateRuntime failed\n");
    return 1;
  }
  const size_t elems = static_cast<size_t>(rank) * 2u * 896u;
  const size_t bytes = elems * sizeof(uint16_t);
  void *device_re = nullptr;
  void *device_im = nullptr;
  ACL_OK(aclrtMalloc(&device_re, bytes, ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_OK(aclrtMalloc(&device_im, bytes, ACL_MEM_MALLOC_HUGE_FIRST));
  std::vector<uint16_t> host_re(elems), host_im(elems);
  for (uint16_t slot = 0; slot < 23; ++slot) {
    airan::PuschMimoConfig config{};
    if(profile_enabled){
      if(airan::DerivePuschMimoConfig(profile,profile.max_rx_antennas,&config,&why,slot)!=
          airan::MimoRuntimeConfigStatus::kSuccess){
        std::fprintf(stderr,"[HARD_FAIL] runtime config derivation: %s\n",why.c_str());return 1;}


      config.dmrs_scrambling_id=321;config.data_scrambling_id=321;config.rnti=12345;
    }else{config=LegacyConfig(slot,rank);}
    airan::PuschMimoLayout layout{};
    mdg::KernelMetadata metadata{};
    int32_t cinit[mdg::CINIT_PAD]{};
    if (mdg::BuildCurrentProfile(config, &layout, &metadata, cinit) != mdg::OK ||
        layout.num_dmrs_symbols != 2 || layout.num_data_re != 19152 ||
        layout.data_stride != 19200) {
      std::fprintf(stderr, "[HARD_FAIL] slot %u DMRS profile mismatch\n", slot);
      return 1;
    }
    ACL_OK(aclrtMemset(device_re, bytes, 0x5a, bytes));
    ACL_OK(aclrtMemset(device_im, bytes, 0xa5, bytes));
    mdg::MimoDmrsGenOpArgsV1 args{};
    args.abi_version = mdg::ABI_VERSION;
    args.struct_size = sizeof(args);
    args.dmrs_re = device_re;
    args.dmrs_im = device_im;
    args.config = &config;
    args.layout = &layout;
    args.stream = stream;
    const auto status = mdg::Enqueue(runtime, args);
    if (status != mdg::OK) {
      std::fprintf(stderr, "[HARD_FAIL] slot %u DMRS enqueue status=%d\n", slot,
                   static_cast<int>(status));
      return 1;
    }
    ACL_OK(aclrtSynchronizeStream(stream));
    ACL_OK(aclrtMemcpy(host_re.data(), bytes, device_re, bytes,
                       ACL_MEMCPY_DEVICE_TO_HOST));
    ACL_OK(aclrtMemcpy(host_im.data(), bytes, device_im, bytes,
                       ACL_MEMCPY_DEVICE_TO_HOST));
    char name[32];
    std::snprintf(name, sizeof(name), "slot%02u_re.bin", slot);
    if (!WriteExact(root / name, host_re)) return 1;
    std::snprintf(name, sizeof(name), "slot%02u_im.bin", slot);
    if (!WriteExact(root / name, host_im)) return 1;
  }
  ACL_OK(aclrtFree(device_im));
  ACL_OK(aclrtFree(device_re));
  mdg::DestroyRuntime(runtime);
  ACL_OK(aclrtDestroyStream(stream));
  ACL_OK(aclrtResetDevice(0));
  ACL_OK(aclFinalize());
  std::printf("[PASS] mimo_dmrs_gen Rank%u slots=23 logical=[L,2,896] runtime=%s\n",
              rank,profile_enabled?"profile":"legacy");
  return 0;
}
