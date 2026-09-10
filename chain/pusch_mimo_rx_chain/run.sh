#!/usr/bin/env bash
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
kernel_root="${KERNEL_ROOT:-$(cd "${here}/../.." && pwd)}"
ascend_home="${ASCEND_HOME_PATH:-/usr/local/Ascend/ascend-toolkit/latest}"
staged_root="${PUSCH_MIMO_RX_STAGED_ROOT:-${here}/../pusch_mimo_loopback/work/tx_prefix_rank1}"
work_root="${PUSCH_MIMO_RX_WORK_ROOT:-$(mktemp -d /tmp/pusch_mimo_fused_rx.XXXXXX)}"
profile="${PUSCH_MIMO_RADIO_PROFILE_PATH:-${kernel_root}/common/radio_profiles/fd16x64.json}"
grant="${PUSCH_MIMO_GRANT_PATH:-${kernel_root}/common/pusch_grants/full_band.json}"
runtime_json="${work_root}/fd16x64_rank4.json"
runtime_bin="${work_root}/fd16x64_rank4.bin"
assets="${work_root}/assets"
build="${work_root}/build"
install="${work_root}/install"
receipt="${work_root}/fused_rx_receipt.json"
decoded="${work_root}/decoded_bits.bin"

[[ -f "${staged_root}/artifacts/rx_iq_rank4_23slot.bin" ]] || {
  echo "[HARD_FAIL] missing staged Rank4 waveform: ${staged_root}" >&2
  echo "Run chain/pusch_mimo_loopback/run.sh 4 first, or set PUSCH_MIMO_RX_STAGED_ROOT." >&2
  exit 2
}

mkdir -p "${work_root}"
PYTHONDONTWRITEBYTECODE=1 python3 "${kernel_root}/common/mimo_config_compiler.py" \
  --radio-profile "${profile}" --grant "${grant}" --rank 4 \
  --capabilities "${kernel_root}/common/mimo_operator_capabilities.json" \
  --require-eligible --output "${runtime_json}" --binary-output "${runtime_bin}"
PYTHONDONTWRITEBYTECODE=1 python3 "${here}/scripts/prepare_rx_assets.py" \
  --staged-root "${staged_root}" --output "${assets}"

cmake -S "${here}" -B "${build}" \
  -DRUN_MODE=npu -DSOC_VERSION=Ascend310P1 \
  -DASCEND_CANN_PACKAGE_PATH="${ascend_home}" \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="${install}"
cmake --build "${build}" -j "$(nproc)"
cmake --install "${build}"

set +e
set +u
if [[ -f "${ascend_home}/bin/setenv.bash" ]]; then
  source "${ascend_home}/bin/setenv.bash"
else
  source "${ascend_home}/set_env.sh"
fi
set -u
set -e

set +e
PUSCH_MIMO_RUNTIME_CONFIG="${runtime_bin}" \
LD_LIBRARY_PATH="${install}/lib:${ascend_home}/lib64:${ascend_home}/compiler/lib64:${LD_LIBRARY_PATH:-}" \
  "${install}/bin/pusch_mimo_rx_chain" --input-root "${assets}" \
  --output "${decoded}" --receipt "${receipt}" --rank 4 --slots 23 \
  >"${work_root}/run.log" 2>&1
status=$?
set -e
if [[ ${status} -ne 0 ]]; then
  if grep -Eq '507008|drvRet=4' "${work_root}/run.log"; then
    echo "[DEVICE_BUSY] fused RX could not acquire the NPU; not retried" >&2
    echo "[INFO] work retained at ${work_root}" >&2
    exit 75
  fi
  tail -120 "${work_root}/run.log" >&2
  echo "[HARD_FAIL] fused RX exited ${status}; work=${work_root}" >&2
  exit "${status}"
fi
cat "${work_root}/run.log"
PYTHONDONTWRITEBYTECODE=1 python3 "${here}/scripts/verify_fused_receipt.py" \
  "${receipt}" "${decoded}"
bash "${here}/run_contracts.sh"
echo "[PASS] fused RX artifacts retained at ${work_root}"
