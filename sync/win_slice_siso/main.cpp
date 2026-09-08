

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include "acl/acl.h"
#include "aclrtlaunch_win_slice_kernel.h"

#define CK(x) do{ aclError e=(x); if(e!=ACL_SUCCESS){ std::printf("ACL err %d @ %s:%d\n",e,__FILE__,__LINE__); std::exit(1);} }while(0)

static constexpr int32_t N_SAMP_SLOT   = 30720;
static constexpr int32_t IQ_LEN        = 2 * N_SAMP_SLOT;
static constexpr int32_t CAP_SAMPLES   = 1228800;
static constexpr int32_t CAP_LEN       = 2 * CAP_SAMPLES;
static constexpr uint32_t BD           = 4;

static std::vector<uint8_t> readbin(const std::string& p){
    FILE* f=std::fopen(p.c_str(),"rb"); if(!f){ std::printf("missing %s\n",p.c_str()); std::exit(1);}
    std::fseek(f,0,SEEK_END); long n=std::ftell(f); std::fseek(f,0,SEEK_SET);
    std::vector<uint8_t> b(n); if(std::fread(b.data(),1,n,f)!=(size_t)n){ std::exit(1);} std::fclose(f); return b;
}
static void writebin(const std::string& p,const void* d,size_t n){
    FILE* f=std::fopen(p.c_str(),"wb"); if(!f){ std::printf("cant write %s\n",p.c_str()); std::exit(1);}
    std::fwrite(d,1,n,f); std::fclose(f);
}

int main(int argc,char** argv){
    const char* dd = std::getenv("AIRAN_DATA_DIR");
    std::string data = dd ? dd : "./data";
    int ncase = (argc>1)? std::atoi(argv[1]) : 5;

    CK(aclInit(nullptr));
    CK(aclrtSetDevice(0));
    aclrtStream stm; CK(aclrtCreateStream(&stm));


    uint8_t *capD,*dtD,*outD,*rdD,*wsD,*tilD;
    CK(aclrtMalloc((void**)&capD, CAP_LEN*sizeof(int16_t), ACL_MEM_MALLOC_HUGE_FIRST));
    CK(aclrtMalloc((void**)&dtD,  8*sizeof(float),         ACL_MEM_MALLOC_HUGE_FIRST));
    CK(aclrtMalloc((void**)&outD, IQ_LEN*sizeof(int16_t),  ACL_MEM_MALLOC_HUGE_FIRST));
    CK(aclrtMalloc((void**)&rdD,  8*sizeof(int32_t),       ACL_MEM_MALLOC_HUGE_FIRST));
    CK(aclrtMalloc((void**)&wsD,  1024,                    ACL_MEM_MALLOC_HUGE_FIRST));
    CK(aclrtMalloc((void**)&tilD, 256,                     ACL_MEM_MALLOC_HUGE_FIRST));

    for(int c=0;c<ncase;++c){
        std::string g = data + "/golden/case_" + std::to_string(c);
        std::string o = data + "/ascend_output/case_" + std::to_string(c);
        std::string mk = "mkdir -p " + o; (void)std::system(mk.c_str());

        auto cap = readbin(g+"/capture.bin");
        auto rd  = readbin(g+"/rd_ptr_in.bin");
        auto dt  = readbin(g+"/delta_T.bin");


        float  dt8[8]={0}; std::memcpy(dt8,dt.data(),sizeof(float));
        int32_t rd8[8]={0}; std::memcpy(rd8,rd.data(),sizeof(int32_t));
        CK(aclrtMemcpy(capD, CAP_LEN*sizeof(int16_t), cap.data(), cap.size(), ACL_MEMCPY_HOST_TO_DEVICE));
        CK(aclrtMemcpy(dtD,  8*sizeof(float),  dt8, 8*sizeof(float),  ACL_MEMCPY_HOST_TO_DEVICE));
        CK(aclrtMemcpy(rdD,  8*sizeof(int32_t),rd8, 8*sizeof(int32_t),ACL_MEMCPY_HOST_TO_DEVICE));

        ACLRT_LAUNCH_KERNEL(win_slice_kernel)(BD, stm, capD, dtD, outD, rdD, wsD, tilD);
        CK(aclrtSynchronizeStream(stm));

        std::vector<int16_t> sout(IQ_LEN); int32_t rdout8[8]={0};
        CK(aclrtMemcpy(sout.data(), IQ_LEN*sizeof(int16_t), outD, IQ_LEN*sizeof(int16_t), ACL_MEMCPY_DEVICE_TO_HOST));
        CK(aclrtMemcpy(rdout8, 8*sizeof(int32_t), rdD, 8*sizeof(int32_t), ACL_MEMCPY_DEVICE_TO_HOST));

        writebin(o+"/slot_iq.bin", sout.data(), IQ_LEN*sizeof(int16_t));
        writebin(o+"/rd_ptr_out.bin", rdout8, sizeof(int32_t));
        std::printf("[case %d] rd_in=%d dt=%+.2f -> rd_out=%d  slot_iq[0:4]=%d %d %d %d\n",
                    c, rd8[0], dt8[0], rdout8[0], sout[0],sout[1],sout[2],sout[3]);
    }

    aclrtFree(capD);aclrtFree(dtD);aclrtFree(outD);aclrtFree(rdD);aclrtFree(wsD);aclrtFree(tilD);
    CK(aclrtDestroyStream(stm)); CK(aclrtResetDevice(0)); CK(aclFinalize());
    return 0;
}
