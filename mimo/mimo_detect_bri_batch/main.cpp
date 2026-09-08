












#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include "data_utils.h"
#include "tiling/platform/platform_ascendc.h"
#include "mimo_detect_bri.h"
#include "mimo_detect_bri_host_contract.h"
#include "bri_grouped_rhs_adapter.h"
#ifndef ASCENDC_CPU_DEBUG
#include "acl/acl.h"
#include "aclrtlaunch_mimo_detect_bri_kernel.h"
#endif
extern "C" void GenerateTiling(const char* socVersion, uint8_t* buf);
using namespace airan;

namespace {
int EnvInt(const char* k, int d){ const char* e=std::getenv(k); return e? std::atoi(e): d; }
std::string Dir(){const char*e=std::getenv("AIRAN_DATA_DIR");return e?std::string(e):std::string(".");}
std::string Case(){return Dir()+"/data/golden/"+std::string(CASE_NAME);}
std::string Out(){return Dir()+"/data/ascend_output/"+std::string(CASE_NAME);}
std::string GIn(const char*n){return Case()+"/"+n;}
std::string GOut(const char*n){return Out()+"/"+n;}
#ifndef ASCENDC_CPU_DEBUG
void LoadHostExact(const std::string&p,size_t b,uint8_t**h){
    struct stat info{};
    const int statStatus=stat(p.c_str(),&info);
    const bool regular=statStatus==0&&S_ISREG(info.st_mode);
    const size_t fileBytes=regular?static_cast<size_t>(info.st_size):0;
    if(!regular||fileBytes!=b){
        fprintf(stderr,"[mimo_bri][ABI] %s: got %zu bytes, expected exactly %zu\n",p.c_str(),fileBytes,b);
        std::exit(EXIT_FAILURE);
    }
    CHECK_ACL(aclrtMallocHost((void**)h,b));
    size_t readBytes=b;
    if(!ReadFile(p.c_str(),readBytes,*h,b)||readBytes!=b){
        fprintf(stderr,"[mimo_bri][ABI] %s: got %zu bytes, expected exactly %zu\n",p.c_str(),readBytes,b);
        std::exit(EXIT_FAILURE);
    }
}
void Upload(size_t b,const uint8_t*h,uint8_t**d){
    CHECK_ACL(aclrtMalloc((void**)d,b,ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMemcpy(*d,b,h,b,ACL_MEMCPY_HOST_TO_DEVICE));
}
void AllocOut(size_t b,uint8_t**h,uint8_t**d){
    CHECK_ACL(aclrtMallocHost((void**)h,b));CHECK_ACL(aclrtMalloc((void**)d,b,ACL_MEM_MALLOC_HUGE_FIRST));
    memset(*h,0,b);CHECK_ACL(aclrtMemcpy(*d,b,*h,b,ACL_MEMCPY_HOST_TO_DEVICE));}
#endif
}

int32_t main(){
    const int N_WARMUP = EnvInt("AIRAN_WARMUP", 3);
    const int N_TIMED  = std::max(1, EnvInt("AIRAN_TIMED", 20));
    const char*soc=SOC_VERSION;
    const MimoDetectBriHostContract contract{
        MIMO_IO_ABI_VERSION, NR, NL, NLR, N_SYMBOL, N_SC_PAD, BLK, AIRAN_PACK};
    std::string contractError;
    if(!ValidateMimoDetectBriHostContract(contract,&contractError)){
        fprintf(stderr,"[mimo_bri][contract] FAIL: %s\n",contractError.c_str());
        return EXIT_FAILURE;
    }
    auto plat=platform_ascendc::PlatformAscendCManager::GetInstance(soc);
    const size_t hBytes=MimoIoMatrixElements()*sizeof(uint16_t);
    const size_t groupedYElements=airan::bri_grouped_rhs_adapter::GROUPED_ELEMS;
    const size_t groupedYBytes=groupedYElements*sizeof(uint16_t);
    const size_t noBytes=MimoIoNoiseElements()*sizeof(uint16_t);
    const size_t oBytes=MimoIoOutputElements()*sizeof(uint16_t);
    const size_t tilBytes=TILING_TOTAL_SIZE;
    const size_t wsBytes=(size_t)plat->GetLibApiWorkSpaceSize();
    uint8_t*tilBuf=(uint8_t*)malloc(tilBytes); GenerateTiling(soc,tilBuf);
    uint32_t blockDim=BLOCK_DIM;
    printf("[mimo_bri] SOC=%s blockDim=%u M=%u K=%u activeL=%u P=%u unit=%u case=%s\n",
           soc,blockDim,NR,NL,NLR,P,N_UNIT,CASE_NAME);
    printf("[mimo_bri][contract] io_pack ABI v%u: hrm/yvpad=[%u,%u,%u], no=[%u], output=[%u,%u]\n",
           contract.abi_version,N_RE,NR,NL,N_RE,NL,N_RE);
#ifndef ASCENDC_CPU_DEBUG
    CHECK_ACL(aclInit(nullptr));CHECK_ACL(aclrtSetDevice(0));
    aclrtStream stream=nullptr;CHECK_ACL(aclrtCreateStream(&stream));
    uint8_t *hrH,*hrD,*hiH,*hiD,*yrH,*yrD,*yiH,*yiD,*noH,*noD,*mkH,*mkD,*xrH,*xrD,*xiH,*xiD,*neH,*neD,*wsD,*tH,*tD;
    const size_t yvBytes=MimoIoMatrixElements()*sizeof(uint16_t);
    uint8_t *yvrH,*yvrD,*yviH,*yviD;

    LoadHostExact(GIn("hrm_re.bin"),hBytes,&hrH); LoadHostExact(GIn("hrm_im.bin"),hBytes,&hiH);
    LoadHostExact(GIn("yvpad_re.bin"),yvBytes,&yvrH); LoadHostExact(GIn("yvpad_im.bin"),yvBytes,&yviH);
    LoadHostExact(GIn("no.bin"),noBytes,&noH);
    if(!ValidateMimoDetectBriPhysicalInputs(
           contract,reinterpret_cast<const uint16_t*>(hrH),hBytes/sizeof(uint16_t),
           reinterpret_cast<const uint16_t*>(hiH),hBytes/sizeof(uint16_t),
           reinterpret_cast<const uint16_t*>(yvrH),yvBytes/sizeof(uint16_t),
           reinterpret_cast<const uint16_t*>(yviH),yvBytes/sizeof(uint16_t),
           reinterpret_cast<const uint16_t*>(noH),noBytes/sizeof(uint16_t),&contractError)){
        fprintf(stderr,"[mimo_bri][ABI] FAIL: %s\n",contractError.c_str());
        return EXIT_FAILURE;
    }
    printf("[mimo_bri][ABI] PASS: exact sizes, active-L padding, repeated yvpad and noise checked\n");






    std::vector<uint16_t> groupedYr(groupedYElements,0);
    std::vector<uint16_t> groupedYi(groupedYElements,0);
    const auto* canonicalYr=reinterpret_cast<const uint16_t*>(yvrH);
    const auto* canonicalYi=reinterpret_cast<const uint16_t*>(yviH);
    for(uint32_t re=0;re<N_RE;++re){
        for(uint32_t rx=0;rx<NR;++rx){
            const size_t canonical=airan::bri_grouped_rhs_adapter::CanonicalIndex(re,rx,0);
            const size_t grouped=airan::bri_grouped_rhs_adapter::GroupedIndex(re,rx);
            groupedYr[grouped]=canonicalYr[canonical];
            groupedYi[grouped]=canonicalYi[canonical];
        }
    }
    printf("[mimo_bri][adapter] canonical yvpad -> grouped RHS: %zu -> %zu fp16/plane\n",
           MimoIoMatrixElements(),groupedYElements);
    Upload(hBytes,hrH,&hrD); Upload(hBytes,hiH,&hiD);
    Upload(groupedYBytes,reinterpret_cast<const uint8_t*>(groupedYr.data()),&yvrD);
    Upload(groupedYBytes,reinterpret_cast<const uint8_t*>(groupedYi.data()),&yviD);
    Upload(noBytes,noH,&noD);


    AllocOut(64,&yrH,&yrD); AllocOut(64,&yiH,&yiD); AllocOut(64,&mkH,&mkD);
    AllocOut(oBytes,&xrH,&xrD);AllocOut(oBytes,&xiH,&xiD);AllocOut(oBytes,&neH,&neD);
    CHECK_ACL(aclrtMalloc((void**)&wsD,wsBytes,ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMallocHost((void**)&tH,tilBytes));CHECK_ACL(aclrtMalloc((void**)&tD,tilBytes,ACL_MEM_MALLOC_HUGE_FIRST));
    memcpy(tH,tilBuf,tilBytes);CHECK_ACL(aclrtMemcpy(tD,tilBytes,tH,tilBytes,ACL_MEMCPY_HOST_TO_DEVICE));
    auto launch=[&](){ACLRT_LAUNCH_KERNEL(mimo_detect_bri_kernel)(blockDim,stream,hrD,hiD,yrD,yiD,noD,mkD,xrD,xiD,neD,yvrD,yviD,wsD,tD);};
    printf("[mimo_bri] warm-up %d\n",N_WARMUP);
    for(int i=0;i<N_WARMUP;++i){launch();CHECK_ACL(aclrtSynchronizeStream(stream));}
    printf("[mimo_bri] timed %d\n",N_TIMED);
    std::vector<double> us;us.reserve(N_TIMED);
    for(int i=0;i<N_TIMED;++i){auto t0=std::chrono::high_resolution_clock::now();launch();CHECK_ACL(aclrtSynchronizeStream(stream));
        auto t1=std::chrono::high_resolution_clock::now();us.push_back(std::chrono::duration<double,std::micro>(t1-t0).count());}
    std::sort(us.begin(),us.end());double sum=0;for(double v:us)sum+=v;
    const double p50=us[N_TIMED/2], p99=us[(size_t)((N_TIMED-1)*0.99)];
    printf("[mimo_bri] latency: min %.1f avg %.1f p50 %.1f p99 %.1f max %.1f us  (spread %.1f%%)\n",
           us.front(),sum/N_TIMED,p50,p99,us.back(),100.0*(us.back()-us.front())/p50);
    printf("[mimo_bri] ★ 出表格用 p50 = %.3f ms\n", p50/1000.0);
    printf("[mimo_bri] real-time headroom (500us slot): %.1fx\n",500.0/p50);
    CHECK_ACL(aclrtMemcpy(xrH,oBytes,xrD,oBytes,ACL_MEMCPY_DEVICE_TO_HOST));
    CHECK_ACL(aclrtMemcpy(xiH,oBytes,xiD,oBytes,ACL_MEMCPY_DEVICE_TO_HOST));
    CHECK_ACL(aclrtMemcpy(neH,oBytes,neD,oBytes,ACL_MEMCPY_DEVICE_TO_HOST));
    WriteFile(GOut("xhat_re.bin").c_str(),xrH,oBytes);WriteFile(GOut("xhat_im.bin").c_str(),xiH,oBytes);WriteFile(GOut("no_eff.bin").c_str(),neH,oBytes);
    aclrtFree(hrD);aclrtFreeHost(hrH);aclrtFree(hiD);aclrtFreeHost(hiH);aclrtFree(yrD);aclrtFreeHost(yrH);aclrtFree(yiD);aclrtFreeHost(yiH);
    aclrtFree(noD);aclrtFreeHost(noH);aclrtFree(mkD);aclrtFreeHost(mkH);aclrtFree(xrD);aclrtFreeHost(xrH);aclrtFree(xiD);aclrtFreeHost(xiH);
    aclrtFree(neD);aclrtFreeHost(neH);aclrtFree(yvrD);aclrtFreeHost(yvrH);aclrtFree(yviD);aclrtFreeHost(yviH);
    aclrtFree(wsD);aclrtFree(tD);aclrtFreeHost(tH);
    CHECK_ACL(aclrtDestroyStream(stream));CHECK_ACL(aclrtResetDevice(0));CHECK_ACL(aclFinalize());
#endif
    free(tilBuf);return 0;
}
