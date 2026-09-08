#!/bin/bash
set -e
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
KERNEL_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)
OUT="${SCRIPT_DIR}/tiling_shims"
mkdir -p "${OUT}"
rm -f "${OUT}"/*_shim.cpp "${OUT}"/tx_tiling_setup.cpp

DIRS=(fec/ldpc_encode fec/rate_match_siso fec/scramble_siso \
      mapping/qam_mod_256_siso sync/dmrs_gen_siso mapping/re_map_siso \
      ofdm/ofdm_mod_siso)
ENUMS=(ldpc_encode rate_match scramble qam256_mod dmrs_gen re_map ofdm_mod)
SKIP_GT=" ofdm/ofdm_mod_siso "

DECLS=""
BODY=""

for i in "${!DIRS[@]}"; do
  dir="${DIRS[$i]}"; en="${ENUMS[$i]}"

  if [[ "$SKIP_GT" == *" $dir "* ]]; then
    echo "[gen] ${dir}: SKIP GenerateTiling -> zeroed 256B"
    BODY="${BODY}
    a.tiling_bytes[OP_${en}]=256; a.tiling[OP_${en}]=dmalloc_t(256);"
    continue
  fi

  src=$(grep -rlsE 'GenerateTiling[[:space:]]*\(' "${KERNEL_ROOT}/${dir}" --include='*.cpp' 2>/dev/null \
        | grep -vE 'kernel\.cpp$|main\.cpp$' | head -1 || true)

  if [ -z "$src" ]; then
    echo "[gen] ${dir}: no GenerateTiling() def -> zeroed 256B"
    BODY="${BODY}
    a.tiling_bytes[OP_${en}]=256; a.tiling[OP_${en}]=dmalloc_t(256);"
    continue
  fi

  rel=$(realpath --relative-to="${OUT}" "$src")
  if grep -Eq 'uint8_t[[:space:]]*\*[[:space:]]*GenerateTiling' "$src"; then fam="B"; else fam="A"; fi
  echo "[gen] ${dir}: family ${fam}  src=${src}"

  cat > "${OUT}/${en}_shim.cpp" <<SHIM
#define GenerateTiling   gt_${en}
#define GetTilingSize    gts_${en}
#define GetWorkspaceSize gws_${en}
#include "${rel}"
#undef GenerateTiling
#undef GetTilingSize
#undef GetWorkspaceSize
SHIM

  if [ "$fam" = "B" ]; then
    DECLS="${DECLS}
extern \"C\" uint8_t* gt_${en}(const char*, uint32_t);
extern \"C\" size_t   gts_${en}();
extern \"C\" size_t   gws_${en}();"
    BODY="${BODY}
    { uint8_t* h=gt_${en}(SOC, BD); size_t n=gts_${en}();
      if(n>TIL_A) n=TIL_A;
      a.tiling_bytes[OP_${en}]=n; a.tiling[OP_${en}]=dmalloc_t(n);
      CK_T(aclrtMemcpy(a.tiling[OP_${en}],n,h,n,ACL_MEMCPY_HOST_TO_DEVICE));
      size_t w=gws_${en}(); if(w>ws_bytes) ws_bytes=w; /* no free(h): one-time, avoid heap corruption */ }"
  else
    DECLS="${DECLS}
extern \"C\" void gt_${en}(const char*, uint8_t*);"
    BODY="${BODY}
    { size_t n=TIL_A; uint8_t* h=(uint8_t*)std::calloc(1,n); gt_${en}(SOC, h);
      a.tiling_bytes[OP_${en}]=n; a.tiling[OP_${en}]=dmalloc_t(n);
      CK_T(aclrtMemcpy(a.tiling[OP_${en}],n,h,n,ACL_MEMCPY_HOST_TO_DEVICE)); /* no free(h) */ }"
  fi
done

cat > "${OUT}/tx_tiling_setup.cpp" <<SETUP
#include "../tx_chain.h"
#include "acl/acl.h"
#include <cstdlib>
#include <cstdio>
#include <cstring>

#define CK_T(x) do{ aclError _e=(x); if(_e!=ACL_SUCCESS){ \\
  std::fprintf(stderr,"tiling acl fail %d @%d\\n",_e,__LINE__); std::exit(1);} }while(0)

static constexpr size_t TIL_A = 65536;
static const char* SOC = SOC_VERSION;
static constexpr uint32_t BD = 4;
static size_t ws_bytes = 16u*1024*1024;
${DECLS}

static uint8_t* dmalloc_t(size_t b){ void* p=nullptr;
  CK_T(aclrtMalloc(&p,b?b:1,ACL_MEM_MALLOC_HUGE_FIRST)); CK_T(aclrtMemset(p,b?b:1,0,b?b:1));
  return (uint8_t*)p; }

namespace airan_tx {
void setup_tilings(TxArena& a){
${BODY}
    a.ws_bytes = ws_bytes;
    a.ws = dmalloc_t(ws_bytes);
}
} // namespace airan_tx
SETUP

echo
echo "[gen] done -> ${OUT}/"
ls -1 "${OUT}"
