#!/usr/bin/env bash
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source_root="${PUSCH_MIMO_CLEAN_KERNEL_ROOT:-/home/refresh/AI-RAN-NPU-clean/kernels}"
conda_bin="${PUSCH_MIMO_CONDA_BIN:-/home/refresh/miniconda3/bin/conda}"
ascend_home="${ASCEND_HOME_PATH:-/usr/local/Ascend/ascend-toolkit/latest}"
build_root="${here}/build"
out_root="${here}/out"
work="${here}/work/tx_prefix_rank1"
log_root="${here}/logs/tx_prefix_rank1"

for library in libascendcl.so libplatform.so libregister.so; do
  [[ -r "${ascend_home}/lib64/${library}" ]] || {
    echo "[HARD_FAIL] missing CANN library ${library}" >&2
    exit 2
  }
done
set +e
set +u
if [[ -f "${ascend_home}/bin/setenv.bash" ]]; then
  # shellcheck disable=SC1090
  source "${ascend_home}/bin/setenv.bash"
else
  # shellcheck disable=SC1090
  source "${ascend_home}/set_env.sh"
fi
set -u
set -e

build_operator() {
  local tag="$1"
  local relative="$2"
  local src="${source_root}/${relative}"
  [[ -f "${src}/CMakeLists.txt" ]] || {
    echo "[HARD_FAIL] categorized operator source absent: ${src}" >&2
    exit 2
  }
  cmake -S "${src}" -B "${build_root}/${tag}" \
    -DRUN_MODE=npu -DSOC_VERSION=Ascend310P1 \
    -DASCEND_CANN_PACKAGE_PATH="${ascend_home}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="${out_root}/${tag}" \
    >"${log_root}/build_${tag}.log" 2>&1
  cmake --build "${build_root}/${tag}" -j "$(nproc)" \
    >>"${log_root}/build_${tag}.log" 2>&1
  cmake --install "${build_root}/${tag}" \
    >>"${log_root}/build_${tag}.log" 2>&1
  [[ -x "${out_root}/${tag}/bin/ascendc_kernels_bbit" &&
     -r "${out_root}/${tag}/lib/libascendc_kernels_npu.so" ]] || {
    echo "[HARD_FAIL] ${tag} install artifacts absent" >&2
    exit 2
  }
}

rm -rf "${work}" "${log_root}"
mkdir -p "${work}" "${log_root}"
build_operator ldpc_encode fec/ldpc_encode
build_operator rate_match_mimo fec/rate_match_mimo
build_operator scramble_mimo fec/scramble_mimo
build_operator qam_mod_256_mimo mapping/qam_mod_256_mimo
build_operator layer_map_mimo mapping/layer_map_mimo

[[ -x "${conda_bin}" ]] || {
  echo "[HARD_FAIL] Sionna conda launcher absent: ${conda_bin}" >&2
  exit 2
}
"${conda_bin}" run -n sionna env AIRAN_DATA_DIR="${work}/ldpc_standard" \
  python "${source_root}/fec/ldpc_encode/scripts/ldpc_ref.py" \
  >"${log_root}/generate_ldpc_sionna.log" 2>&1
python3 "${here}/tx_prefix_rank1.py" prepare --root "${work}" \
  --legacy "${work}/ldpc_standard"
mkdir -p \
  "${work}/ldpc/ascend_output" \
  "${work}/rate/ascend_output" \
  "${work}/scramble/ascend_output" \
  "${work}/qam/ascend_output" \
  "${work}/layer/ascend_output"

run_device() {
  local tag="$1"
  local data_dir="$2"
  local operator="$3"
  set +e
  env AIRAN_DATA_DIR="${data_dir}" WARMUP=0 TIMED=1 \
    LD_LIBRARY_PATH="${out_root}/${operator}/lib:${ascend_home}/lib64:${LD_LIBRARY_PATH:-}" \
    "${out_root}/${operator}/bin/ascendc_kernels_bbit" \
    >"${log_root}/${tag}.log" 2>&1
  local status=$?
  set -e
  if [[ ${status} -ne 0 ]]; then
    if grep -Eq '507008|drvRet=4' "${log_root}/${tag}.log"; then
      echo "[DEVICE_BUSY] ${tag}: 507008/drvRet=4; not retried" >&2
      exit 75
    fi
    tail -80 "${log_root}/${tag}.log" >&2
    echo "[HARD_FAIL] ${tag} exited ${status}" >&2
    exit "${status}"
  fi
}

run_device ldpc "${work}/ldpc" ldpc_encode
python3 "${here}/tx_prefix_rank1.py" link-rate --root "${work}"
run_device rate_match "${work}/rate" rate_match_mimo
python3 "${here}/tx_prefix_rank1.py" link-scramble --root "${work}"
run_device scramble "${work}/scramble" scramble_mimo

for slot in $(seq 0 22); do
  slot_tag="$(printf '%02d' "${slot}")"
  python3 "${here}/tx_prefix_rank1.py" select-qam --root "${work}" --slot "${slot}"
  run_device "qam_slot${slot_tag}" "${work}/qam" qam_mod_256_mimo
  python3 "${here}/tx_prefix_rank1.py" link-layer --root "${work}"
  run_device "layer_slot${slot_tag}" "${work}/layer" layer_map_mimo
  python3 "${here}/tx_prefix_rank1.py" collect --root "${work}" --slot "${slot}"
done

python3 "${here}/tx_prefix_rank1.py" verify --root "${work}" | tee "${log_root}/verify.log"
echo "[MILESTONE PASS] Rank1 23-slot real device TX LDPC-to-layer-map prefix"
