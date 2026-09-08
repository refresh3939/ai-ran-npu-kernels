





#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include "data_utils.h"
#include "tiling/platform/platform_ascendc.h"
#include "precode_zf.h"
#ifndef ASCENDC_CPU_DEBUG
#include "acl/acl.h"
#include "aclrtlaunch_precode_zf_kernel.h"
#endif
extern "C" void GenerateTiling(const char* socVersion, uint8_t* buf);
using namespace airan;

namespace {
int EnvInt(const char* k,int d){const char*e=std::getenv(k);return e?std::atoi(e):d;}
std::string Dir(){const char*e=std::getenv("AIRAN_DATA_DIR");return e?std::string(e):std::string(".");}
std::string Case(){return Dir()+"/data/golden/"+std::string(CASE_NAME);}
std::string Out(){return Dir()+"/data/ascend_output/"+std::string(CASE_NAME);}
std::string GIn(const char*n){return Case()+"/"+n;}
std::string GOut(const char*n){return Out()+"/"+n;}
#ifndef ASCENDC_CPU_DEBUG
void Load(const std::string&p,size_t b,uint8_t**h,uint8_t**d){
    CHECK_ACL(aclrtMallocHost((void**)h,b));CHECK_ACL(aclrtMalloc((void**)d,b,ACL_MEM_MALLOC_HUGE_FIRST));
    ReadFile(p.c_str(),b,*h,b);CHECK_ACL(aclrtMemcpy(*d,b,*h,b,ACL_MEMCPY_HOST_TO_DEVICE));}
void AllocOut(size_t b,uint8_t**h,uint8_t**d){
    CHECK_ACL(aclrtMallocHost((void**)h,b));CHECK_ACL(aclrtMalloc((void**)d,b,ACL_MEM_MALLOC_HUGE_FIRST));
    memset(*h,0,b);CHECK_ACL(aclrtMemcpy(*d,b,*h,b,ACL_MEMCPY_HOST_TO_DEVICE));}
#endif
}

int32_t main(){
    const int N_WARMUP=std::max(0,EnvInt("AIRAN_WARMUP",3));
    const int N_TIMED=std::max(1,EnvInt("AIRAN_TIMED",20));
    const char*soc=SOC_VERSION;
    auto plat=platform_ascendc::PlatformAscendCManager::GetInstance(soc);
    const size_t wBytes=(size_t)N_UNIT*NT*NL*sizeof(int16_t);
    const size_t sBytes=(size_t)N_UNIT*NL*sizeof(int16_t);
    const size_t mBytes=(size_t)N_RE*sizeof(int16_t);
    const size_t oBytes=(size_t)NT*N_RE*sizeof(int16_t);
    const size_t tilBytes=TILING_TOTAL_SIZE;
    const size_t wsBytes=(size_t)plat->GetLibApiWorkSpaceSize();
    uint8_t*tilBuf=(uint8_t*)malloc(tilBytes); GenerateTiling(soc,tilBuf);
    uint32_t blockDim=BLOCK_DIM;
    printf("[precode_zf_opt] SOC=%s blockDim=%u NT=%u NL=%u NLR=%u P=%u unit=%u RE_BATCH=%u B=%u L=%u case=%s\n",
           soc,blockDim,NT,NL,NLR,PACK,N_UNIT,RE_BATCH,BLK,NLAY,CASE_NAME);
#ifndef ASCENDC_CPU_DEBUG
    CHECK_ACL(aclInit(nullptr));CHECK_ACL(aclrtSetDevice(0));
    aclrtStream stream=nullptr;CHECK_ACL(aclrtCreateStream(&stream));
    uint8_t *wrH,*wrD,*wiH,*wiD,*srH,*srD,*siH,*siD,*mkH,*mkD,*xrH,*xrD,*xiH,*xiD,*wsD,*tH,*tD;
    Load(GIn("wrm_re.bin"),wBytes,&wrH,&wrD); Load(GIn("wrm_im.bin"),wBytes,&wiH,&wiD);
    Load(GIn("srm_re.bin"),sBytes,&srH,&srD); Load(GIn("srm_im.bin"),sBytes,&siH,&siD);
    Load(GIn("mask.bin"),  mBytes,&mkH,&mkD);
    AllocOut(oBytes,&xrH,&xrD); AllocOut(oBytes,&xiH,&xiD);
    CHECK_ACL(aclrtMalloc((void**)&wsD,wsBytes,ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMallocHost((void**)&tH,tilBytes));CHECK_ACL(aclrtMalloc((void**)&tD,tilBytes,ACL_MEM_MALLOC_HUGE_FIRST));
    memcpy(tH,tilBuf,tilBytes);CHECK_ACL(aclrtMemcpy(tD,tilBytes,tH,tilBytes,ACL_MEMCPY_HOST_TO_DEVICE));
    auto launch=[&](){ACLRT_LAUNCH_KERNEL(precode_zf_kernel)(blockDim,stream,wrD,wiD,srD,siD,mkD,xrD,xiD,wsD,tD);};
    printf("[precode_zf_opt] warm-up %d\n",N_WARMUP);
    for(int i=0;i<N_WARMUP;++i){launch();CHECK_ACL(aclrtSynchronizeStream(stream));}
    printf("[precode_zf_opt] timed %d\n",N_TIMED);
    std::vector<double> us;us.reserve(N_TIMED);
    for(int i=0;i<N_TIMED;++i){auto t0=std::chrono::high_resolution_clock::now();launch();CHECK_ACL(aclrtSynchronizeStream(stream));
        auto t1=std::chrono::high_resolution_clock::now();us.push_back(std::chrono::duration<double,std::micro>(t1-t0).count());}
    std::sort(us.begin(),us.end());double sum=0;for(double v:us)sum+=v;
    const double p50=us[(size_t)N_TIMED/2];
    const double p99=us[(size_t)((N_TIMED-1)*0.99)];
    printf("[precode_zf_opt] latency: min %.1f avg %.1f p50 %.1f p99 %.1f max %.1f us (spread %.1f%%)\n",
           us.front(),sum/N_TIMED,p50,p99,us.back(),100.0*(us.back()-us.front())/p50);
    printf("[precode_zf_opt] report p50 %.3f ms; 500us reference ratio %.3fx\n",p50/1000.0,500.0/p50);
    CHECK_ACL(aclrtMemcpy(xrH,oBytes,xrD,oBytes,ACL_MEMCPY_DEVICE_TO_HOST));
    CHECK_ACL(aclrtMemcpy(xiH,oBytes,xiD,oBytes,ACL_MEMCPY_DEVICE_TO_HOST));
    WriteFile(GOut("x_re.bin").c_str(),xrH,oBytes);WriteFile(GOut("x_im.bin").c_str(),xiH,oBytes);
    aclrtFree(wrD);aclrtFreeHost(wrH);aclrtFree(wiD);aclrtFreeHost(wiH);
    aclrtFree(srD);aclrtFreeHost(srH);aclrtFree(siD);aclrtFreeHost(siH);
    aclrtFree(mkD);aclrtFreeHost(mkH);
    aclrtFree(xrD);aclrtFreeHost(xrH);aclrtFree(xiD);aclrtFreeHost(xiH);
    aclrtFree(wsD);aclrtFree(tD);aclrtFreeHost(tH);
    CHECK_ACL(aclrtDestroyStream(stream));CHECK_ACL(aclrtResetDevice(0));CHECK_ACL(aclFinalize());
#endif
    free(tilBuf);return 0;
}
