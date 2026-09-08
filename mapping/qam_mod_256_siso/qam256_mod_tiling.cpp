




#include <cstdlib>
#include <cstring>
#include "qam256_mod.h"

namespace airan {

extern "C" uint8_t* GenerateTiling(const char*  , uint32_t block_dim) {
    uint8_t* buf = (uint8_t*)std::malloc(TILING_TOTAL_SIZE);
    std::memset(buf, 0, TILING_TOTAL_SIZE);
    TilingData* t = reinterpret_cast<TilingData*>(buf);

    const uint32_t n_sc_used = N_SC_USED_DEF;
    const uint32_t n_sc_pad  = N_SC_PAD_DEF;
    const uint32_t n_symbol  = N_SYMBOL_MAX;


    bool is_dmrs[N_SYMBOL_MAX] = {false};
    is_dmrs[DMRS_SYM_0] = true;
    is_dmrs[DMRS_SYM_1] = true;

    uint32_t n_data_sym = 0;
    for (uint32_t s = 0; s < n_symbol; ++s) {
        if (!is_dmrs[s]) { t->phys_sym[n_data_sym] = (int32_t)s; ++n_data_sym; }
    }
    for (uint32_t k = n_data_sym; k < N_SYMBOL_MAX; ++k) t->phys_sym[k] = -1;

    const uint32_t n_re_data = n_data_sym * n_sc_used;
    auto round_up = [](uint32_t v, uint32_t a) { return (v + a - 1) / a * a; };
    const uint32_t n_sym_pad = round_up(n_re_data, 128);

    t->n_sc_used  = (int32_t)n_sc_used;
    t->n_sc_pad   = (int32_t)n_sc_pad;
    t->n_data_sym = (int32_t)n_data_sym;
    t->n_re_data  = (int32_t)n_re_data;
    t->n_sym_pad  = (int32_t)n_sym_pad;
    t->block_dim  = (int32_t)block_dim;

    auto gcd = [](uint32_t a, uint32_t b) { while (b) { uint32_t z=a%b; a=b; b=z; } return a; };
    t->group_sz = (int32_t)(16u / gcd(16u, n_sc_used));
    return buf;
}

extern "C" size_t GetTilingSize()    { return TILING_TOTAL_SIZE; }
extern "C" size_t GetWorkspaceSize() { return WS_TOTAL; }

}
