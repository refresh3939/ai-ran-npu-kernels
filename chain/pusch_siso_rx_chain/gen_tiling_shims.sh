#!/bin/bash
# gen_tiling_shims.sh —— 基于 clean 分类目录生成 SISO RX chain 的 tiling shim
# 为每个 kernel 的 tiling.cpp 套 rename shim 绕开 GenerateTiling 符号冲突,
# 并生成 setup_tilings():arena init 时复用各 kernel 自己的 GenerateTiling 填 tiling。
set -e
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
KERNEL_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)
OUT="${SCRIPT_DIR}/tiling_shims"
mkdir -p "${OUT}"
rm -f "${OUT}"/*_shim.cpp "${OUT}"/rx_tiling_setup.cpp   # 清旧产物(避免悬空符号)

# 相对于 kernels/ 的目录；enum 后缀保持现有 runtime ABI。
DIRS=(sync/decimate sync/pss_correlator sync/pss_cfo_estimator \
      sync/sss_correlator sync/ssb_fft sync/pbch_dmrs_correlator \
      sync/cfo_estimate sync/cfo_compensate ofdm/ofdm_demod_siso \
      mapping/re_demap_siso channel_est/channel_est_ls_siso \
      equalization/equalize_zf_siso mapping/qam_demod_256_siso \
      sync/cfo_dmrs sync/timing_tracker \
      fec/descramble_siso fec/rate_dematch_siso fec/ldpc_decode)
ENUMS=(decimate pss_correlator pss_cfo_estimator sss_correlator ssb_fft \
      pbch_dmrs_correlator cfo_estimate cfo_compensate ofdm_demod re_demap \
      channel_est_ls equalize qam256_demod cfo_dmrs timing_tracker \
      descramble rate_dematch ldpc_decode)

DECLS=""
BODY=""

for i in "${!DIRS[@]}"; do
  dir="${DIRS[$i]}"; en="${ENUMS[$i]}"
  # 递归找"定义了 GenerateTiling 函数"的 .cpp:必须是 GenerateTiling( 形式,
  # 排除只在注释里提到的(如 sss "保留 GenerateTiling 入口")、device kernel.cpp、main.cpp
  src=$(grep -rlsE 'GenerateTiling[[:space:]]*\(' "${KERNEL_ROOT}/${dir}" --include='*.cpp' 2>/dev/null \
        | grep -vE 'kernel\.cpp$|main\.cpp$' | head -1 || true)

  if [ -z "$src" ]; then
    echo "[gen] ${dir}: no GenerateTiling() def -> zeroed 256B tiling (需要时 host 填)"
    BODY="${BODY}
    a.tiling_bytes[OP_${en}]=256; a.tiling[OP_${en}]=dmalloc_t(256);"
    continue
  fi

  rel=$(realpath --relative-to="${OUT}" "$src")
  if grep -Eq 'uint8_t[[:space:]]*\*[[:space:]]*GenerateTiling' "$src"; then fam="B"; else fam="A"; fi
  echo "[gen] ${dir}: family ${fam}  src=${src}"

  cat > "${OUT}/${en}_shim.cpp" <<SHIM
// auto-generated. rename ${dir} tiling exports to avoid GenerateTiling collision.
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
      a.tiling_bytes[OP_${en}]=n; a.tiling[OP_${en}]=dmalloc_t(n);
      CK_T(aclrtMemcpy(a.tiling[OP_${en}],n,h,n,ACL_MEMCPY_HOST_TO_DEVICE));
      size_t w=gws_${en}(); if(w>ws_bytes) ws_bytes=w; free(h); }"
  else
    DECLS="${DECLS}
extern \"C\" void gt_${en}(const char*, uint8_t*);"
    BODY="${BODY}
    { size_t n=TIL_A; uint8_t* h=(uint8_t*)std::calloc(1,n); gt_${en}(SOC, h);
      a.tiling_bytes[OP_${en}]=n; a.tiling[OP_${en}]=dmalloc_t(n);
      CK_T(aclrtMemcpy(a.tiling[OP_${en}],n,h,n,ACL_MEMCPY_HOST_TO_DEVICE)); std::free(h); }"
  fi
done

cat > "${OUT}/rx_tiling_setup.cpp" <<SETUP
// auto-generated. fills RxArena::tiling[] + ws by reusing each kernel's real GenerateTiling.
#include "../rx_chain.h"
#include "acl/acl.h"
#include <cstdlib>
#include <cstdio>
#include <cstring>

#define CK_T(x) do{ aclError _e=(x); if(_e!=ACL_SUCCESS){ \\
  std::fprintf(stderr,"tiling acl fail %d @%d\\n",_e,__LINE__); std::exit(1);} }while(0)

static constexpr size_t TIL_A = 4096;          // >= 2*sizeof(TCubeTiling) (ofdm matmul)
static const char* SOC = SOC_VERSION;
static constexpr uint32_t BD = 4;
static size_t ws_bytes = 16u*1024*1024;         // bumped by any B-family GetWorkspaceSize
${DECLS}

static uint8_t* dmalloc_t(size_t b){ void* p=nullptr;
  CK_T(aclrtMalloc(&p,b?b:1,ACL_MEM_MALLOC_HUGE_FIRST)); CK_T(aclrtMemset(p,b?b:1,0,b?b:1));
  return (uint8_t*)p; }

namespace airan_rx {
void setup_tilings(RxArena& a){
${BODY}
    a.ws_bytes = ws_bytes;
    a.ws = dmalloc_t(ws_bytes);
}
} // namespace airan_rx
SETUP

echo
echo "[gen] done -> ${OUT}/"
ls -1 "${OUT}"
