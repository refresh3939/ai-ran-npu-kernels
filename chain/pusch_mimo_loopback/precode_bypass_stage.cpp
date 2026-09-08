#include <acl/acl.h>
#include "pusch_codebook_precode_runtime.h"
#include "pusch_mimo_runtime_config.h"

#include <cstdint>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;
namespace pc = airan::pusch_precode;
#define ACL_OK(call) do { const aclError e_=(call); if(e_!=ACL_ERROR_NONE){ \
  std::fprintf(stderr,"[HARD_FAIL] %s returned %d at %s:%d\n",#call,e_,__FILE__,__LINE__); return 1;} } while(0)

static bool Read(const fs::path &p,std::vector<uint16_t>&v,size_t n){
  std::ifstream f(p,std::ios::binary|std::ios::ate);
  if(!f||static_cast<size_t>(f.tellg())!=n*2){std::fprintf(stderr,"[HARD_FAIL] bad input %s\n",p.c_str());return false;}
  v.resize(n);f.seekg(0);f.read(reinterpret_cast<char*>(v.data()),n*2);return f.good();}
static bool Write(const fs::path&p,const std::vector<uint16_t>&v){
  std::ofstream f(p,std::ios::binary|std::ios::trunc);f.write(reinterpret_cast<const char*>(v.data()),v.size()*2);return f.good();}

int main(int argc,char**argv){
  if(argc!=4){std::fprintf(stderr,"usage: %s INPUT_DIR OUTPUT_DIR RANK\n",argv[0]);return 2;}
  fs::path in(argv[1]),out(argv[2]);fs::create_directories(out);
  const uint16_t rank=static_cast<uint16_t>(std::stoul(argv[3]));if(rank<1||rank>4)return 2;
  airan::PuschMimoRuntimeConfig profile{};bool profile_enabled=false;std::string why;
  if(airan::LoadMimoRuntimeConfigFromEnv(rank,&profile,&profile_enabled,&why)!=airan::MimoRuntimeConfigStatus::kSuccess){std::fprintf(stderr,"[HARD_FAIL] runtime config: %s\n",why.c_str());return 1;}
  if(profile_enabled&&(profile.max_rx_antennas!=64||profile.max_layers!=16||profile.num_layers>pc::MAX_LAYERS||profile.num_tx_ports>pc::MAX_PORTS||profile.num_symbols!=pc::N_SYMBOLS||profile.num_rb!=pc::N_RB||profile.used_subcarriers!=pc::N_SC_USED||profile.padded_subcarriers!=pc::N_SC_PAD||profile.grid_re_per_port!=pc::N_RE_GRID)){std::fprintf(stderr,"[HARD_FAIL] runtime config exceeds current precoder profile\n");return 1;}
  const uint16_t ports=profile_enabled?profile.num_tx_ports:(rank==3?4:rank);
  ACL_OK(aclInit(nullptr));ACL_OK(aclrtSetDevice(0));aclrtStream stream=nullptr;ACL_OK(aclrtCreateStream(&stream));
  pc::RuntimeContext runtime{};if(pc::RuntimeInit(&runtime)!=pc::Status::kSuccess){std::fprintf(stderr,"[HARD_FAIL] precoder runtime init\n");return 1;}
  void *d_lr=nullptr,*d_li=nullptr,*d_pr=nullptr,*d_pi=nullptr,*d_ws=nullptr;
  const size_t in_bytes=static_cast<size_t>(rank)*pc::N_RE_GRID*2,out_bytes=static_cast<size_t>(ports)*pc::N_RE_GRID*2;
  ACL_OK(aclrtMalloc(&d_lr,in_bytes,ACL_MEM_MALLOC_HUGE_FIRST));ACL_OK(aclrtMalloc(&d_li,in_bytes,ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_OK(aclrtMalloc(&d_pr,out_bytes,ACL_MEM_MALLOC_HUGE_FIRST));ACL_OK(aclrtMalloc(&d_pi,out_bytes,ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_OK(aclrtMalloc(&d_ws,1,ACL_MEM_MALLOC_HUGE_FIRST));
  airan::PuschMimoConfig config{};
  if(profile_enabled){if(airan::DerivePuschMimoConfig(profile,profile.max_rx_antennas,&config,&why)!=airan::MimoRuntimeConfigStatus::kSuccess){std::fprintf(stderr,"[HARD_FAIL] runtime config derivation: %s\n",why.c_str());return 1;}}
  else{config=pc::MakeDefaultConfig(rank,ports);config.codebook_enabled=rank==3?1:0;config.tpmi=rank==3?6:0;config.prg_size_rb=rank==3?4:133;}
  if(pc::ValidateConfig(config,&why)!=pc::Status::kSuccess){std::fprintf(stderr,"[HARD_FAIL] precode config: %s\n",why.c_str());return 1;}
  std::vector<uint16_t> lr,li,pr(static_cast<size_t>(ports)*pc::N_RE_GRID),pi(static_cast<size_t>(ports)*pc::N_RE_GRID);
  for(unsigned slot=0;slot<23;++slot){char name[32];std::snprintf(name,sizeof(name),"slot%02u_re.bin",slot);if(!Read(in/name,lr,static_cast<size_t>(rank)*pc::N_RE_GRID))return 1;
    std::snprintf(name,sizeof(name),"slot%02u_im.bin",slot);if(!Read(in/name,li,static_cast<size_t>(rank)*pc::N_RE_GRID))return 1;
    ACL_OK(aclrtMemcpy(d_lr,in_bytes,lr.data(),in_bytes,ACL_MEMCPY_HOST_TO_DEVICE));ACL_OK(aclrtMemcpy(d_li,in_bytes,li.data(),in_bytes,ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_OK(aclrtMemset(d_pr,out_bytes,0x5a,out_bytes));ACL_OK(aclrtMemset(d_pi,out_bytes,0xa5,out_bytes));
    auto status=pc::Launch(&runtime,d_lr,d_li,config,d_pr,d_pi,d_ws,stream,&why);
    if(status!=pc::Status::kSuccess){std::fprintf(stderr,"[HARD_FAIL] slot %u precode bypass: %s (%d)\n",slot,why.c_str(),static_cast<int>(status));return 1;}
    ACL_OK(aclrtSynchronizeStream(stream));ACL_OK(aclrtMemcpy(pr.data(),out_bytes,d_pr,out_bytes,ACL_MEMCPY_DEVICE_TO_HOST));ACL_OK(aclrtMemcpy(pi.data(),out_bytes,d_pi,out_bytes,ACL_MEMCPY_DEVICE_TO_HOST));
    if(config.codebook_enabled){
      std::vector<float> in_r(lr.size()),in_i(li.size()),ref_r(pr.size()),ref_i(pi.size());
      for(size_t j=0;j<lr.size();++j){in_r[j]=aclFloat16ToFloat(lr[j]);in_i[j]=aclFloat16ToFloat(li[j]);}
      if(pc::Reference(in_r.data(),in_i.data(),config,ref_r.data(),ref_i.data(),&why)!=pc::Status::kSuccess)return 1;
      float max_error=0.0f;for(size_t j=0;j<pr.size();++j){max_error=std::max(max_error,std::fabs(aclFloat16ToFloat(pr[j])-ref_r[j]));max_error=std::max(max_error,std::fabs(aclFloat16ToFloat(pi[j])-ref_i[j]));}
      if(max_error>6.0e-3f){std::fprintf(stderr,"[HARD_FAIL] slot %u Rank3 codebook max_error=%g\n",slot,max_error);return 1;}
    }else if(pr!=lr||pi!=li){std::fprintf(stderr,"[HARD_FAIL] slot %u bypass is not bit-exact identity\n",slot);return 1;}
    std::snprintf(name,sizeof(name),"slot%02u_re.bin",slot);if(!Write(out/name,pr))return 1;
    std::snprintf(name,sizeof(name),"slot%02u_im.bin",slot);if(!Write(out/name,pi))return 1;}
  if(runtime.cache_misses!=1||runtime.cache_hits!=22){std::fprintf(stderr,"[HARD_FAIL] precoder cache contract miss=%u hit=%u\n",runtime.cache_misses,runtime.cache_hits);return 1;}
  aclrtFree(d_ws);aclrtFree(d_pi);aclrtFree(d_pr);aclrtFree(d_li);aclrtFree(d_lr);pc::RuntimeDestroy(&runtime);
  ACL_OK(aclrtDestroyStream(stream));ACL_OK(aclrtResetDevice(0));ACL_OK(aclFinalize());
  std::printf("[PASS] pusch_codebook_precode Rank%u slots=23 mode=%s runtime=%s\n",rank,config.codebook_enabled?"codebook device/reference":"bypass bit-exact",profile_enabled?"profile":"legacy");return 0;}
