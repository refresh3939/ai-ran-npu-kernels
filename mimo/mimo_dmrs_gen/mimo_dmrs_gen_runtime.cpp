#include "mimo_dmrs_gen.h"

#include <new>
#include <vector>

#include "acl/acl.h"
#include "aclrtlaunch_mimo_dmrs_gen_kernel.h"

namespace airan::mimo_dmrs_gen {

struct MimoDmrsGenRuntimeV1 {
    int32_t *cinit_host = nullptr;
    KernelMetadata *metadata_host = nullptr;
    void *cinit_device = nullptr;
    void *gmat_device = nullptr;
    void *g1_device = nullptr;
    void *scratch_device = nullptr;
    void *physical_re_device = nullptr;
    void *physical_im_device = nullptr;
    void *debug_device = nullptr;
    void *metadata_device = nullptr;
};

namespace {

constexpr size_t CINIT_BYTES = CINIT_PAD * sizeof(int32_t);
constexpr size_t GMAT_BYTES = MAT_LEN * sizeof(uint16_t);
constexpr size_t G1_BYTES = N_DMRS_PLANE * sizeof(uint16_t);
constexpr size_t PHYSICAL_BYTES = PhysicalOutputElems() * sizeof(uint16_t);
constexpr size_t DEBUG_BYTES = OUT_DBG_LEN * sizeof(float);
constexpr size_t SCRATCH_BYTES = 128;

bool MallocDevice(void **pointer, size_t bytes)
{
    return aclrtMalloc(pointer, bytes, ACL_MEM_MALLOC_HUGE_FIRST) == ACL_ERROR_NONE;
}

}  // namespace

void DestroyRuntime(MimoDmrsGenRuntimeV1 *runtime)
{
    if (runtime == nullptr) return;
    if (runtime->metadata_device != nullptr) aclrtFree(runtime->metadata_device);
    if (runtime->debug_device != nullptr) aclrtFree(runtime->debug_device);
    if (runtime->physical_im_device != nullptr) aclrtFree(runtime->physical_im_device);
    if (runtime->physical_re_device != nullptr) aclrtFree(runtime->physical_re_device);
    if (runtime->scratch_device != nullptr) aclrtFree(runtime->scratch_device);
    if (runtime->g1_device != nullptr) aclrtFree(runtime->g1_device);
    if (runtime->gmat_device != nullptr) aclrtFree(runtime->gmat_device);
    if (runtime->cinit_device != nullptr) aclrtFree(runtime->cinit_device);
    if (runtime->metadata_host != nullptr) aclrtFreeHost(runtime->metadata_host);
    if (runtime->cinit_host != nullptr) aclrtFreeHost(runtime->cinit_host);
    delete runtime;
}

Status CreateRuntime(MimoDmrsGenRuntimeV1 **runtime)
{
    if (runtime == nullptr) return INVALID_ARGUMENT;
    *runtime = nullptr;
    auto *state = new (std::nothrow) MimoDmrsGenRuntimeV1;
    if (state == nullptr) return RESOURCE_ERROR;

    const bool host_ok =
        aclrtMallocHost(reinterpret_cast<void **>(&state->cinit_host), CINIT_BYTES) ==
            ACL_ERROR_NONE &&
        aclrtMallocHost(reinterpret_cast<void **>(&state->metadata_host), TILING_BYTES) ==
            ACL_ERROR_NONE;
    const bool device_ok = host_ok &&
        MallocDevice(&state->cinit_device, CINIT_BYTES) &&
        MallocDevice(&state->gmat_device, GMAT_BYTES) &&
        MallocDevice(&state->g1_device, G1_BYTES) &&
        MallocDevice(&state->scratch_device, SCRATCH_BYTES) &&
        MallocDevice(&state->physical_re_device, PHYSICAL_BYTES) &&
        MallocDevice(&state->physical_im_device, PHYSICAL_BYTES) &&
        MallocDevice(&state->debug_device, DEBUG_BYTES) &&
        MallocDevice(&state->metadata_device, TILING_BYTES);
    if (!device_ok) {
        DestroyRuntime(state);
        return RESOURCE_ERROR;
    }

    std::vector<uint16_t> gmat(MAT_LEN);
    std::vector<uint16_t> g1(N_DMRS_PLANE);
    BuildGoldBasis(gmat.data(), g1.data());
    if (aclrtMemcpy(state->gmat_device, GMAT_BYTES, gmat.data(), GMAT_BYTES,
                    ACL_MEMCPY_HOST_TO_DEVICE) != ACL_ERROR_NONE ||
        aclrtMemcpy(state->g1_device, G1_BYTES, g1.data(), G1_BYTES,
                    ACL_MEMCPY_HOST_TO_DEVICE) != ACL_ERROR_NONE) {
        DestroyRuntime(state);
        return RESOURCE_ERROR;
    }

    *runtime = state;
    return OK;
}

Status Enqueue(MimoDmrsGenRuntimeV1 *runtime, const MimoDmrsGenOpArgsV1 &args)
{
    if (runtime == nullptr) return INVALID_ARGUMENT;
    const Status args_status = ValidateOpArgs(args);
    if (args_status != OK) return args_status;

    PuschMimoLayout derived_layout {};
    const Status profile_status = BuildCurrentProfile(
        *args.config, &derived_layout, runtime->metadata_host, runtime->cinit_host);
    if (profile_status != OK) return profile_status;

    const auto stream = static_cast<aclrtStream>(args.stream);
    if (aclrtMemcpyAsync(runtime->cinit_device, CINIT_BYTES,
                         runtime->cinit_host, CINIT_BYTES,
                         ACL_MEMCPY_HOST_TO_DEVICE, stream) != ACL_ERROR_NONE ||
        aclrtMemcpyAsync(runtime->metadata_device, TILING_BYTES,
                         runtime->metadata_host, TILING_BYTES,
                         ACL_MEMCPY_HOST_TO_DEVICE, stream) != ACL_ERROR_NONE) {
        return RESOURCE_ERROR;
    }

    const uint32_t launch_status = ACLRT_LAUNCH_KERNEL(mimo_dmrs_gen_kernel)(
        BLOCK_DIM, stream, runtime->cinit_device, runtime->gmat_device,
        runtime->g1_device, runtime->scratch_device, runtime->physical_re_device,
        runtime->physical_im_device, runtime->debug_device, runtime->scratch_device,
        runtime->metadata_device);
    if (launch_status != ACL_ERROR_NONE) return LAUNCH_FAILED;

    const size_t logical_bytes =
        LogicalOutputElems(args.config->num_layers, args.layout->num_dmrs_symbols) *
        sizeof(uint16_t);
    if (aclrtMemcpyAsync(args.dmrs_re, logical_bytes, runtime->physical_re_device,
                         logical_bytes, ACL_MEMCPY_DEVICE_TO_DEVICE, stream) != ACL_ERROR_NONE ||
        aclrtMemcpyAsync(args.dmrs_im, logical_bytes, runtime->physical_im_device,
                         logical_bytes, ACL_MEMCPY_DEVICE_TO_DEVICE, stream) != ACL_ERROR_NONE) {
        return RESOURCE_ERROR;
    }
    return OK;
}

}  // namespace airan::mimo_dmrs_gen
