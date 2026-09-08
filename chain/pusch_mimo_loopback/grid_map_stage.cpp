#include <acl/acl.h>
#include "aclrtlaunch_mimo_resource_grid_map_kernel.h"
#include "mimo_resource_grid_map.h"
#include "pusch_mimo_runtime_config.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;
namespace rgm = airan::mimo_resource_grid_map;

#define ACL_OK(call) do { const aclError e_ = (call); if (e_ != ACL_ERROR_NONE) { \
  std::fprintf(stderr, "[HARD_FAIL] %s returned %d at %s:%d\n", #call, e_, __FILE__, __LINE__); \
  return 1; } } while (0)

template <typename T>
static bool ReadExact(const fs::path &path, std::vector<T> &value, size_t count) {
  std::ifstream in(path, std::ios::binary | std::ios::ate);
  if (!in || static_cast<size_t>(in.tellg()) != count * sizeof(T)) {
    std::fprintf(stderr, "[HARD_FAIL] %s expected %zu bytes\n", path.c_str(), count * sizeof(T));
    return false;
  }
  value.resize(count); in.seekg(0);
  in.read(reinterpret_cast<char *>(value.data()), static_cast<std::streamsize>(count * sizeof(T)));
  return in.good();
}

template <typename T>
static bool WriteExact(const fs::path &path, const std::vector<T> &value) {
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  out.write(reinterpret_cast<const char *>(value.data()),
            static_cast<std::streamsize>(value.size() * sizeof(T)));
  return out.good();
}

static airan::PuschMimoConfig LegacyConfig(uint16_t slot,uint16_t rank) {
  airan::PuschMimoConfig c{};
  c.abi_version = rgm::ABI_VERSION; c.struct_size = sizeof(c);
  c.num_layers = rank; c.num_tx_ports = rank==1?1:(rank==2?2:4); c.num_rx_antennas = 64; c.qm = 8;
  c.num_symbols = 14; c.fft_size = 2048; c.num_rb = 133; c.slot_number = slot;
  c.num_allocated_symbols = 14; c.used_subcarriers = 1596; c.padded_subcarriers = 1664;
  c.dmrs_symbol_mask = static_cast<uint16_t>((1u << 2) | (1u << 11));
  for (uint16_t i = 0; i < rank; ++i) {
    c.dmrs_ports[i] = (rank == 2 && i == 1)
                          ? 1002
                          : static_cast<uint16_t>(1000 + i);
  }
  c.dmrs_scrambling_id = 321;
  c.data_scrambling_id = 321;
  c.rnti = 12345; c.dmrs_type = 1; c.dmrs_length = 1;
  c.num_cdm_groups_without_data = 2;
  return c;
}

int main(int argc, char **argv) {
  if (argc != 5) {
    std::fprintf(stderr, "usage: %s LAYER_DIR DMRS_DIR OUTPUT_DIR RANK\n", argv[0]); return 2;
  }
  const fs::path layer_root(argv[1]), dmrs_root(argv[2]), output_root(argv[3]);
  const uint16_t rank=static_cast<uint16_t>(std::stoul(argv[4]));if(rank<1||rank>4)return 2;
  airan::PuschMimoRuntimeConfig profile{};
  bool profile_enabled=false;
  std::string why;
  if(airan::LoadMimoRuntimeConfigFromEnv(rank,&profile,&profile_enabled,&why)!=
      airan::MimoRuntimeConfigStatus::kSuccess){
    std::fprintf(stderr,"[HARD_FAIL] runtime config: %s\n",why.c_str());return 1;}
  if(profile_enabled&&(profile.max_rx_antennas!=64||profile.max_layers!=16||
      profile.num_dmrs_symbols!=2||profile.dmrs_port_count!=rank||
      profile.data_re_per_layer!=19152||profile.data_stride_per_layer!=19200||
      profile.grid_re_per_port!=14*1664)){
    std::fprintf(stderr,"[HARD_FAIL] runtime config exceeds current grid-map profile\n");return 1;}
  fs::create_directories(output_root);
  const size_t data_stride=profile_enabled?profile.data_stride_per_layer:19200;
  const size_t grid_stride=profile_enabled?profile.grid_re_per_port:14*1664;
  const size_t data_elems = static_cast<size_t>(rank)*data_stride, dmrs_elems = static_cast<size_t>(rank)*1792, grid_elems = static_cast<size_t>(rank)*grid_stride;
  constexpr size_t data_offset_elems = 19200, dmrs_offset_elems = 4 * 2 * 896;
  ACL_OK(aclInit(nullptr)); ACL_OK(aclrtSetDevice(0));
  aclrtStream stream = nullptr; ACL_OK(aclrtCreateStream(&stream));
  void *d_lr=nullptr,*d_li=nullptr,*d_dr=nullptr,*d_di=nullptr,*d_gr=nullptr,*d_gi=nullptr;
  void *d_do=nullptr,*d_dmo=nullptr,*d_meta=nullptr,*d_ws=nullptr;
  ACL_OK(aclrtMalloc(&d_lr,data_elems*2,ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_OK(aclrtMalloc(&d_li,data_elems*2,ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_OK(aclrtMalloc(&d_dr,dmrs_elems*2,ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_OK(aclrtMalloc(&d_di,dmrs_elems*2,ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_OK(aclrtMalloc(&d_gr,grid_elems*2,ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_OK(aclrtMalloc(&d_gi,grid_elems*2,ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_OK(aclrtMalloc(&d_do,data_offset_elems*4,ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_OK(aclrtMalloc(&d_dmo,dmrs_offset_elems*4,ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_OK(aclrtMalloc(&d_meta,rgm::TILING_BYTES,ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_OK(aclrtMalloc(&d_ws,128,ACL_MEM_MALLOC_HUGE_FIRST));
  std::vector<uint16_t> lr,li,dr,di,gr(grid_elems),gi(grid_elems);
  std::vector<uint16_t> ref_r(grid_elems),ref_i(grid_elems);
  std::vector<uint32_t> data_offset(data_offset_elems),dmrs_offset(dmrs_offset_elems);
  for (uint16_t slot=0; slot<23; ++slot) {
    char name[32];
    std::snprintf(name,sizeof(name),"slot%02u_re.bin",slot);
    if (!ReadExact(layer_root/name,lr,data_elems) || !ReadExact(dmrs_root/name,dr,dmrs_elems)) return 1;
    std::snprintf(name,sizeof(name),"slot%02u_im.bin",slot);
    if (!ReadExact(layer_root/name,li,data_elems) || !ReadExact(dmrs_root/name,di,dmrs_elems)) return 1;
    airan::PuschMimoConfig config{};
    if(profile_enabled){
      if(airan::DerivePuschMimoConfig(profile,profile.max_rx_antennas,&config,&why,slot)!=
          airan::MimoRuntimeConfigStatus::kSuccess){
        std::fprintf(stderr,"[HARD_FAIL] runtime config derivation: %s\n",why.c_str());return 1;}
    }else{config=LegacyConfig(slot,rank);}
    airan::PuschMimoLayout layout{}; rgm::KernelMetadata meta{};
    if (rgm::BuildCurrentProfile(config,&layout,&meta,data_offset.data(),dmrs_offset.data())!=rgm::OK ||
        layout.num_data_re!=19152 || layout.data_stride!=19200) {
      std::fprintf(stderr,"[HARD_FAIL] slot %u resource-grid profile mismatch\n",slot); return 1;
    }
    rgm::MimoResourceGridMapOpArgsV1 args{};
    args.abi_version=rgm::ABI_VERSION; args.struct_size=sizeof(args);
    args.layer_re=d_lr; args.layer_im=d_li; args.dmrs_re=d_dr; args.dmrs_im=d_di;
    args.layer_grid_re=d_gr; args.layer_grid_im=d_gi; args.config=&config; args.layout=&layout; args.stream=stream;
    if (rgm::ValidateOpArgs(args)!=rgm::OK) { std::fprintf(stderr,"[HARD_FAIL] slot %u grid args rejected\n",slot); return 1; }
    if (rgm::ReferenceMap(lr.data(),li.data(),dr.data(),di.data(),config,layout,
                          data_offset.data(),dmrs_offset.data(),ref_r.data(),ref_i.data())!=rgm::OK) return 1;
    ACL_OK(aclrtMemcpy(d_lr,data_elems*2,lr.data(),data_elems*2,ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_OK(aclrtMemcpy(d_li,data_elems*2,li.data(),data_elems*2,ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_OK(aclrtMemcpy(d_dr,dmrs_elems*2,dr.data(),dmrs_elems*2,ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_OK(aclrtMemcpy(d_di,dmrs_elems*2,di.data(),dmrs_elems*2,ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_OK(aclrtMemcpy(d_do,data_offset_elems*4,data_offset.data(),data_offset_elems*4,ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_OK(aclrtMemcpy(d_dmo,dmrs_offset_elems*4,dmrs_offset.data(),dmrs_offset_elems*4,ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_OK(aclrtMemcpy(d_meta,rgm::TILING_BYTES,&meta,rgm::TILING_BYTES,ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_OK(aclrtMemset(d_gr,grid_elems*2,0x5a,grid_elems*2)); ACL_OK(aclrtMemset(d_gi,grid_elems*2,0xa5,grid_elems*2));
    const uint32_t launch=ACLRT_LAUNCH_KERNEL(mimo_resource_grid_map_kernel)(
      rgm::BLOCK_DIM,stream,d_lr,d_li,d_dr,d_di,d_do,d_dmo,d_gr,d_gi,d_ws,d_meta);
    if (launch!=ACL_ERROR_NONE) { std::fprintf(stderr,"[HARD_FAIL] slot %u launch=%u\n",slot,launch); return 1; }
    ACL_OK(aclrtSynchronizeStream(stream));
    ACL_OK(aclrtMemcpy(gr.data(),grid_elems*2,d_gr,grid_elems*2,ACL_MEMCPY_DEVICE_TO_HOST));
    ACL_OK(aclrtMemcpy(gi.data(),grid_elems*2,d_gi,grid_elems*2,ACL_MEMCPY_DEVICE_TO_HOST));
    if (gr!=ref_r || gi!=ref_i) { std::fprintf(stderr,"[HARD_FAIL] slot %u grid device/reference mismatch\n",slot); return 1; }
    std::snprintf(name,sizeof(name),"slot%02u_re.bin",slot); if(!WriteExact(output_root/name,gr)) return 1;
    std::snprintf(name,sizeof(name),"slot%02u_im.bin",slot); if(!WriteExact(output_root/name,gi)) return 1;
  }
  aclrtFree(d_ws); aclrtFree(d_meta); aclrtFree(d_dmo); aclrtFree(d_do); aclrtFree(d_gi);
  aclrtFree(d_gr); aclrtFree(d_di); aclrtFree(d_dr); aclrtFree(d_li); aclrtFree(d_lr);
  ACL_OK(aclrtDestroyStream(stream)); ACL_OK(aclrtResetDevice(0)); ACL_OK(aclFinalize());
  std::printf("[PASS] mimo_resource_grid_map Rank%u slots=23 device/reference bit-exact runtime=%s\n",
              rank,profile_enabled?"profile":"legacy"); return 0;
}
