#include <acl/acl.h>
#include "aclrtlaunch_ldpc_decode_kernel.h"
#include "ldpc_decode.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <vector>

namespace fs=std::filesystem;
#define ACL_OK(call) do{const aclError e_=(call);if(e_!=ACL_ERROR_NONE){std::fprintf(stderr,"[HARD_FAIL] %s=%d\n",#call,e_);return 1;}}while(0)
template<class T>bool Read(const fs::path&p,std::vector<T>*v,size_t n){std::ifstream f(p,std::ios::binary|std::ios::ate);if(!f||static_cast<size_t>(f.tellg())!=n*sizeof(T)){std::fprintf(stderr,"[HARD_FAIL] %s expected %zu bytes\n",p.c_str(),n*sizeof(T));return false;}v->resize(n);f.seekg(0);f.read(reinterpret_cast<char*>(v->data()),n*sizeof(T));return f.good();}
int main(int argc,char**argv){
 if(argc!=4){std::fprintf(stderr,"usage: %s DATA_DIR WEIGHT_DIR OUTPUT\n",argv[0]);return 2;}fs::path data(argv[1]),weights(argv[2]),output(argv[3]);
 std::vector<int16_t> lam,shift,degree;std::vector<int32_t> edge;
 if(!Read(data/"lam_in.bin",&lam,airan::LDPC_C_NUM*airan::LAM_ELEMS_PER_CB)||!Read(weights/"shift_table.bin",&shift,airan::SHIFT_ELEMS)||!Read(data/"degrees.bin",&degree,airan::LDPC_MB)||!Read(data/"edge_offsets.bin",&edge,airan::LDPC_MB+1))return 1;
 if(!std::any_of(lam.begin(),lam.end(),[](int16_t v){return v!=0;})||edge.back()!=static_cast<int32_t>(airan::LDPC_TOTAL_EDGES)){std::fprintf(stderr,"[HARD_FAIL] LDPC input all-zero or edge count mismatch\n");return 1;}
 std::vector<int16_t> bc(airan::PACKED_ELEMS_TOTAL,0),ps(airan::PACKED_ELEMS_TOTAL,0);
 for(uint32_t br=0;br<airan::LDPC_MB;++br){uint32_t k=0;for(uint32_t col=0;col<airan::LDPC_NFULL;++col){const int16_t s=shift[br*airan::LDPC_NFULL+col];if(s<0)continue;if(k>=airan::LDPC_MAX_DEG){std::fprintf(stderr,"[HARD_FAIL] LDPC degree overflow\n");return 1;}bc[br*airan::LDPC_MAX_DEG+k]=static_cast<int16_t>(col);ps[br*airan::LDPC_MAX_DEG+k]=s;++k;}if(k!=static_cast<uint32_t>(degree[br])){std::fprintf(stderr,"[HARD_FAIL] LDPC degree/table mismatch\n");return 1;}}
 ACL_OK(aclInit(nullptr));ACL_OK(aclrtSetDevice(0));aclrtStream stream=nullptr;ACL_OK(aclrtCreateStream(&stream));
 auto alloc=[](void**p,size_t n){return aclrtMalloc(p,n,ACL_MEM_MALLOC_HUGE_FIRST);};void *dl,*dbc,*dps,*dd,*de,*dp,*db,*dlo,*scratch;
 ACL_OK(alloc(&dl,airan::LAM_BYTES));ACL_OK(alloc(&dbc,airan::PACKED_BYTES));ACL_OK(alloc(&dps,airan::PACKED_BYTES));ACL_OK(alloc(&dd,airan::LDPC_MB*2));ACL_OK(alloc(&de,(airan::LDPC_MB+1)*4));ACL_OK(alloc(&dp,airan::PREV_BYTES));ACL_OK(alloc(&db,airan::BITS_BYTES));ACL_OK(alloc(&dlo,airan::LAM_BYTES));ACL_OK(alloc(&scratch,airan::LAM_SCRATCH_BYTES));
 ACL_OK(aclrtMemcpy(dl,airan::LAM_BYTES,lam.data(),airan::LAM_BYTES,ACL_MEMCPY_HOST_TO_DEVICE));ACL_OK(aclrtMemcpy(dbc,airan::PACKED_BYTES,bc.data(),airan::PACKED_BYTES,ACL_MEMCPY_HOST_TO_DEVICE));ACL_OK(aclrtMemcpy(dps,airan::PACKED_BYTES,ps.data(),airan::PACKED_BYTES,ACL_MEMCPY_HOST_TO_DEVICE));ACL_OK(aclrtMemcpy(dd,airan::LDPC_MB*2,degree.data(),airan::LDPC_MB*2,ACL_MEMCPY_HOST_TO_DEVICE));ACL_OK(aclrtMemcpy(de,(airan::LDPC_MB+1)*4,edge.data(),(airan::LDPC_MB+1)*4,ACL_MEMCPY_HOST_TO_DEVICE));ACL_OK(aclrtMemset(dp,airan::PREV_BYTES,0,airan::PREV_BYTES));ACL_OK(aclrtMemset(db,airan::BITS_BYTES,0xff,airan::BITS_BYTES));ACL_OK(aclrtMemset(dlo,airan::LAM_BYTES,0,airan::LAM_BYTES));ACL_OK(aclrtMemset(scratch,airan::LAM_SCRATCH_BYTES,0,airan::LAM_SCRATCH_BYTES));
 const uint32_t status=ACLRT_LAUNCH_KERNEL(ldpc_decode_kernel)(4,stream,reinterpret_cast<uint8_t*>(dl),reinterpret_cast<uint8_t*>(dbc),reinterpret_cast<uint8_t*>(dps),reinterpret_cast<uint8_t*>(dd),reinterpret_cast<uint8_t*>(de),reinterpret_cast<uint8_t*>(dp),reinterpret_cast<uint8_t*>(db),reinterpret_cast<uint8_t*>(dlo),reinterpret_cast<uint8_t*>(scratch));if(status!=ACL_ERROR_NONE){std::fprintf(stderr,"[HARD_FAIL] ldpc launch=%u\n",status);return 1;}ACL_OK(aclrtSynchronizeStream(stream));
 std::vector<int8_t> bits(airan::LDPC_C_NUM*airan::LDPC_K);ACL_OK(aclrtMemcpy(bits.data(),bits.size(),db,bits.size(),ACL_MEMCPY_DEVICE_TO_HOST));if(!std::all_of(bits.begin(),bits.end(),[](int8_t v){return v==0||v==1;})){std::fprintf(stderr,"[HARD_FAIL] decoded output is not binary\n");return 1;}std::ofstream out(output,std::ios::binary|std::ios::trunc);out.write(reinterpret_cast<const char*>(bits.data()),bits.size());if(!out)return 1;
 for(void*p:{scratch,dlo,db,dp,de,dd,dps,dbc,dl}){aclrtFree(p);}ACL_OK(aclrtDestroyStream(stream));ACL_OK(aclrtResetDevice(0));ACL_OK(aclFinalize());std::printf("[PASS] ldpc_decode actual kernel single launch bits=%zu\n",bits.size());return 0;}
