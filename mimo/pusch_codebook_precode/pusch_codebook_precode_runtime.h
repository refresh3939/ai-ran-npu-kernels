



#pragma once

#include <cstdint>

#include "acl/acl.h"
#include "pusch_codebook_precode.h"

namespace airan::pusch_precode {

constexpr uint16_t RUNTIME_CONFIG_CACHE_CAPACITY = 16;

struct RuntimeResource {
    void* weight_re = nullptr;
    void* weight_im = nullptr;
    void* prg_of_rb = nullptr;
    void* tiling = nullptr;
    uint16_t cached_layers = 0;
    uint16_t cached_ports = 0;
    uint16_t cached_tpmi = 0;
    uint16_t cached_prg_size_rb = 0;
    uint8_t cached_codebook_enabled = 0;
    bool configured = false;
    uint64_t last_used = 0;
};

struct RuntimeContext {
    RuntimeResource resources[RUNTIME_CONFIG_CACHE_CAPACITY]{};
    uint64_t use_clock = 0;
    uint32_t cache_hits = 0;
    uint32_t cache_misses = 0;
    uint16_t cache_entries = 0;
    bool initialized = false;
};

Status RuntimeInit(RuntimeContext* context);
void RuntimeDestroy(RuntimeContext* context);








Status Launch(RuntimeContext* context,
              const void* layer_grid_re, const void* layer_grid_im,
              const PuschMimoConfig& config,
              void* port_grid_re, void* port_grid_im,
              void* workspace, aclrtStream stream,
              std::string* why = nullptr);

}
