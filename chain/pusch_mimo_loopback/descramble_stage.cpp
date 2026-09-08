#include <acl/acl.h>
#include "descramble_mimo.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

namespace op = airan::descramble_mimo;

#define ACL_OK(call) do { const aclError e_=(call); if(e_!=ACL_ERROR_NONE){ \
    std::fprintf(stderr,"[HARD_FAIL] %s=%d\n",#call,e_); return 1; } } while(0)

static airan::PuschMimoConfig Config(uint16_t cell_id, uint16_t rank)
{
    airan::PuschMimoConfig c{};
    c.abi_version=1;c.struct_size=sizeof(c);c.num_layers=rank;c.num_tx_ports=rank==3?4:rank;
    c.num_rx_antennas=64;c.qm=8;c.num_symbols=14;c.fft_size=2048;c.num_rb=133;
    c.num_allocated_symbols=14;c.used_subcarriers=1596;c.padded_subcarriers=1664;
    c.dmrs_symbol_mask=(1u<<2)|(1u<<11);for(uint16_t i=0;i<rank;++i)c.dmrs_ports[i]=(rank==2&&i==1)?1002:static_cast<uint16_t>(1000+i);c.dmrs_type=1;
    c.dmrs_length=1;c.num_cdm_groups_without_data=2;c.data_scrambling_id=cell_id;
    c.dmrs_scrambling_id=cell_id;c.rnti=12345;c.codebook_enabled=rank==3?1:0;
    c.tpmi=rank==3?6:0;c.prg_size_rb=rank==3?4:133;
    return c;
}

int main(int argc,char **argv)
{
    if(argc!=5){std::fprintf(stderr,"usage: %s INPUT OUTPUT CELL_ID RANK\n",argv[0]);return 2;}
    const uint16_t cell_id=static_cast<uint16_t>(std::stoul(argv[3]));
    const uint16_t rank=static_cast<uint16_t>(std::stoul(argv[4]));if(rank<1||rank>4)return 2;
    const size_t elems=op::BufferElems(23,rank);
    std::ifstream stream(argv[1],std::ios::binary|std::ios::ate);
    if(!stream||static_cast<size_t>(stream.tellg())!=elems*sizeof(int16_t)){
        std::fprintf(stderr,"[HARD_FAIL] descramble input shape\n");return 1;}
    std::vector<int16_t> input(elems);stream.seekg(0);stream.read(reinterpret_cast<char*>(input.data()),elems*2);
    if(!stream||!std::any_of(input.begin(),input.end(),[](int16_t v){return v!=0;})){
        std::fprintf(stderr,"[HARD_FAIL] descramble input read/all-zero\n");return 1;}
    for(uint32_t slot=0;slot<23;++slot)for(uint32_t q=0;q<8;++q){
        const size_t tail=(static_cast<size_t>(slot)*8+q)*rank*19200+rank*19152;
        if(std::any_of(input.begin()+tail,input.begin()+tail+rank*48,[](int16_t v){return v!=0;})){
            std::fprintf(stderr,"[HARD_FAIL] descramble input padding nonzero\n");return 1;}}
    auto cfg=Config(cell_id,rank);airan::PuschMimoLayout layout{};op::KernelMetadata meta{};
    if(op::BuildCurrentProfile(cfg,23,&layout,&meta)!=op::OK)return 1;
    std::vector<int16_t> sign(elems),reference(elems),actual(elems);
    if(op::BuildGoldSign(cfg,layout,23,sign.data(),sign.size())!=op::OK||
       op::ReferenceDescramble(input.data(),sign.data(),cfg,layout,23,reference.data())!=op::OK){return 1;}
    ACL_OK(aclInit(nullptr));ACL_OK(aclrtSetDevice(0));aclrtStream acl_stream=nullptr;ACL_OK(aclrtCreateStream(&acl_stream));
    void *di=nullptr,*ds=nullptr,*doo=nullptr,*dw=nullptr,*dt=nullptr;
    ACL_OK(aclrtMalloc(&di,elems*2,ACL_MEM_MALLOC_HUGE_FIRST));ACL_OK(aclrtMalloc(&ds,elems*2,ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_OK(aclrtMalloc(&doo,elems*2,ACL_MEM_MALLOC_HUGE_FIRST));ACL_OK(aclrtMalloc(&dw,op::WORKSPACE_BYTES,ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_OK(aclrtMalloc(&dt,op::TILING_BYTES,ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_OK(aclrtMemcpy(di,elems*2,input.data(),elems*2,ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_OK(aclrtMemcpy(ds,elems*2,sign.data(),elems*2,ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_OK(aclrtMemcpy(dt,sizeof(meta),&meta,sizeof(meta),ACL_MEMCPY_HOST_TO_DEVICE));
    op::DescrambleMimoOpArgsV1 args{};args.abi_version=1;args.struct_size=sizeof(args);args.llr_qam=di;
    args.llr_nr=doo;args.config=&cfg;args.layout=&layout;args.num_slots=23;args.stream=acl_stream;
    if(op::Enqueue(args,ds,elems*2,dw,op::WORKSPACE_BYTES,dt,op::TILING_BYTES)!=op::OK){
        std::fprintf(stderr,"[HARD_FAIL] descramble enqueue\n");return 1;}
    ACL_OK(aclrtSynchronizeStream(acl_stream));ACL_OK(aclrtMemcpy(actual.data(),elems*2,doo,elems*2,ACL_MEMCPY_DEVICE_TO_HOST));
    if(actual!=reference){std::fprintf(stderr,"[HARD_FAIL] descramble device/reference mismatch\n");return 1;}
    std::ofstream out(argv[2],std::ios::binary|std::ios::trunc);out.write(reinterpret_cast<const char*>(actual.data()),elems*2);
    if(!out)return 1;
    for(void *p:{dt,dw,doo,ds,di}){aclrtFree(p);}
    ACL_OK(aclrtDestroyStream(acl_stream));ACL_OK(aclrtResetDevice(0));ACL_OK(aclFinalize());
    std::printf("[PASS] descramble_mimo Rank%u slots=23 cell_id=%u bit-exact\n",rank,cell_id);return 0;
}
