#!/usr/bin/env bash
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root="${PUSCH_MIMO_CLEAN_KERNEL_ROOT:-/home/refresh/AI-RAN-NPU-clean/kernels}"
ascend_home="${ASCEND_HOME_PATH:-/usr/local/Ascend/ascend-toolkit/latest}"

build_one() {
  local name="$1"
  local source="$2"
  local binary="${here}/out/${name}/bin/ascendc_kernels_bbit"
  local library="${here}/out/${name}/lib/libascendc_kernels_npu.so"
  if [[ -x "${binary}" && -r "${library}" && "${PUSCH_MIMO_FORCE_REBUILD:-0}" == 0 ]]; then
    return
  fi
  cmake -S "${root}/${source}" -B "${here}/build/${name}" \
    -DRUN_MODE=npu -DSOC_VERSION=Ascend310P1 \
    -DASCEND_CANN_PACKAGE_PATH="${ascend_home}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="${here}/out/${name}"
  # Build only the production runner. Some categorized trees also declare
  # host-only tests whose warning policy is independent of this device image.
  cmake --build "${here}/build/${name}" --target ascendc_kernels_bbit -j"$(nproc)"
  cmake --install "${here}/build/${name}"
  [[ -x "${binary}" && -r "${library}" ]] || {
    echo "[HARD_FAIL] ${name} build did not install the required NPU artifacts" >&2
    exit 2
  }
}

build_one re_map_batch mapping/re_map_batch
build_one ofdm_mod_batch ofdm/ofdm_mod_batch
build_one ofdm_demod_batch ofdm/ofdm_demod_batch
build_one re_demap_batch mapping/re_demap_batch

echo "[PASS] loopback waveform NPU operators available"
