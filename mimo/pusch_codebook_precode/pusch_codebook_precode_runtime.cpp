#include "pusch_codebook_precode_runtime.h"

#include <array>

#include "aclrtlaunch_pusch_codebook_precode_kernel.h"

namespace airan::pusch_precode {
namespace {

bool SameCachedConfig(const RuntimeResource& resource, const PuschMimoConfig& config) {
    return resource.configured &&
           resource.cached_layers == config.num_layers &&
           resource.cached_ports == config.num_tx_ports &&
           resource.cached_tpmi == config.tpmi &&
           resource.cached_prg_size_rb == config.prg_size_rb &&
           resource.cached_codebook_enabled == config.codebook_enabled;
}

Status Configure(RuntimeContext* context, const PuschMimoConfig& config,
                 RuntimeResource** selected, std::string* why) {
    Status status = ValidateConfig(config, why);
    if (status != Status::kSuccess) return status;

    ++context->use_clock;
    for (RuntimeResource& resource : context->resources) {
        if (SameCachedConfig(resource, config)) {
            resource.last_used = context->use_clock;
            ++context->cache_hits;
            *selected = &resource;
            return Status::kSuccess;
        }
    }

    RuntimeResource* target = nullptr;
    for (RuntimeResource& resource : context->resources) {
        if (!resource.configured) {
            target = &resource;
            break;
        }
        if (target == nullptr || resource.last_used < target->last_used) target = &resource;
    }

    CodebookPlan plan;
    status = BuildCodebookPlan(config, &plan, why);
    if (status != Status::kSuccess) return status;
    std::array<aclFloat16, MAX_WEIGHT_ELEMS> weight_re{};
    std::array<aclFloat16, MAX_WEIGHT_ELEMS> weight_im{};
    for (uint32_t i = 0; i < plan.weight_count_padded; ++i) {
        weight_re[i] = aclFloatToFloat16(plan.weight_re[i]);
        weight_im[i] = aclFloatToFloat16(plan.weight_im[i]);
    }
    const size_t weight_bytes = MAX_WEIGHT_ELEMS * sizeof(aclFloat16);
    const size_t prg_bytes = PRG_MAP_PAD * sizeof(uint16_t);
    if (aclrtMemcpy(target->weight_re, weight_bytes, weight_re.data(), weight_bytes,
                    ACL_MEMCPY_HOST_TO_DEVICE) != ACL_SUCCESS ||
        aclrtMemcpy(target->weight_im, weight_bytes, weight_im.data(), weight_bytes,
                    ACL_MEMCPY_HOST_TO_DEVICE) != ACL_SUCCESS ||
        aclrtMemcpy(target->prg_of_rb, prg_bytes, plan.prg_of_rb.data(), prg_bytes,
                    ACL_MEMCPY_HOST_TO_DEVICE) != ACL_SUCCESS ||
        aclrtMemcpy(target->tiling, TILING_BYTES, plan.tiling.data(), TILING_BYTES,
                    ACL_MEMCPY_HOST_TO_DEVICE) != ACL_SUCCESS) {
        if (why != nullptr) *why = "failed to upload precoder resources";
        return Status::kRuntimeError;
    }
    if (!target->configured) ++context->cache_entries;
    target->cached_layers = config.num_layers;
    target->cached_ports = config.num_tx_ports;
    target->cached_tpmi = config.tpmi;
    target->cached_prg_size_rb = config.prg_size_rb;
    target->cached_codebook_enabled = config.codebook_enabled;
    target->configured = true;
    target->last_used = context->use_clock;
    ++context->cache_misses;
    *selected = target;
    return Status::kSuccess;
}

}

Status RuntimeInit(RuntimeContext* context) {
    if (context == nullptr) return Status::kNullArgument;
    *context = RuntimeContext{};
    const size_t weight_bytes = MAX_WEIGHT_ELEMS * sizeof(aclFloat16);
    const size_t prg_bytes = PRG_MAP_PAD * sizeof(uint16_t);
    for (RuntimeResource& resource : context->resources) {
        if (aclrtMalloc(&resource.weight_re, weight_bytes, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS ||
            aclrtMalloc(&resource.weight_im, weight_bytes, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS ||
            aclrtMalloc(&resource.prg_of_rb, prg_bytes, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS ||
            aclrtMalloc(&resource.tiling, TILING_BYTES, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) {
            RuntimeDestroy(context);
            return Status::kRuntimeError;
        }
    }
    context->initialized = true;
    return Status::kSuccess;
}

void RuntimeDestroy(RuntimeContext* context) {
    if (context == nullptr) return;
    for (RuntimeResource& resource : context->resources) {
        if (resource.weight_re != nullptr) aclrtFree(resource.weight_re);
        if (resource.weight_im != nullptr) aclrtFree(resource.weight_im);
        if (resource.prg_of_rb != nullptr) aclrtFree(resource.prg_of_rb);
        if (resource.tiling != nullptr) aclrtFree(resource.tiling);
    }
    *context = RuntimeContext{};
}

Status Launch(RuntimeContext* context,
              const void* layer_grid_re, const void* layer_grid_im,
              const PuschMimoConfig& config,
              void* port_grid_re, void* port_grid_im,
              void* workspace, aclrtStream stream,
              std::string* why) {
    if (context == nullptr || layer_grid_re == nullptr || layer_grid_im == nullptr ||
        port_grid_re == nullptr || port_grid_im == nullptr || workspace == nullptr || stream == nullptr) {
        if (why != nullptr) *why = "null runtime/tensor/workspace/stream argument";
        return Status::kNullArgument;
    }
    if (!context->initialized) {
        if (why != nullptr) *why = "RuntimeInit must be called before Launch";
        return Status::kNotInitialized;
    }
    RuntimeResource* resource = nullptr;
    Status status = Configure(context, config, &resource, why);
    if (status != Status::kSuccess) return status;
    const uint32_t result = ACLRT_LAUNCH_KERNEL(pusch_codebook_precode_kernel)(
        BLOCK_DIM, stream, const_cast<void*>(layer_grid_re), const_cast<void*>(layer_grid_im),
        resource->weight_re, resource->weight_im, resource->prg_of_rb,
        port_grid_re, port_grid_im, workspace, resource->tiling);
    if (result != ACL_SUCCESS) {
        if (why != nullptr) *why = "pusch_codebook_precode kernel launch failed";
        return Status::kRuntimeError;
    }
    return Status::kSuccess;
}

}
