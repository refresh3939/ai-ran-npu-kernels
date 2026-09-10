#!/usr/bin/env bash
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
kernel_root="${KERNEL_ROOT:-$(cd "${here}/../.." && pwd)}"
cann_root="${ASCEND_HOME_PATH:-/usr/local/Ascend/ascend-toolkit/latest}"
rank="${1:-${PUSCH_MIMO_RANK:-4}}"
slots="${2:-${PUSCH_MIMO_SLOTS:-23}}"
radio_profile="${PUSCH_MIMO_RADIO_PROFILE_PATH:-${kernel_root}/common/radio_profiles/fd16x64.json}"
grant="${PUSCH_MIMO_GRANT_PATH:-${kernel_root}/common/pusch_grants/full_band.json}"

case "${rank}" in 1|2|3|4) ;; *) echo "rank must be 1..4" >&2; exit 2 ;; esac
case "${slots}" in
  ''|*[!0-9]*) echo "slots must be 1..23" >&2; exit 2 ;;
  *) if (( slots < 1 || slots > 23 )); then echo "slots must be 1..23" >&2; exit 2; fi ;;
esac

work="$(mktemp -d /tmp/airan-pusch-mimo-tx.XXXXXX)"
keep_work="${PUSCH_MIMO_KEEP_WORK:-0}"
cleanup() {
  if [[ "${keep_work}" != "1" ]]; then rm -rf -- "${work}"; fi
}
trap cleanup EXIT

assets="${PUSCH_MIMO_TX_INPUT_ROOT:-/tmp/airan-pusch-mimo-tx-assets-v1}"
output_dir="${PUSCH_MIMO_TX_OUTPUT_DIR:-/tmp/pusch_mimo_tx_chain_rank${rank}_slots${slots}}"
mkdir -p "${assets}" "${output_dir}"

if [[ ! -s "${assets}/golden/input.bin" || \
      ! -s "${assets}/weights/ldpc_bg1_z384_shifts/shift_A.bin" ]]; then
  echo "[data] generating reviewed LDPC input and shift tables"
  conda_bin="${SIONNA_CONDA_BIN:-conda}"
  env AIRAN_DATA_DIR="${assets}" PYTHONDONTWRITEBYTECODE=1 \
    "${conda_bin}" run -n sionna python \
    "${kernel_root}/fec/ldpc_encode/scripts/ldpc_ref.py"
fi

if [[ ! -s "${assets}/weights/ofdm/iw_dft32_re.bin" || \
      ! -s "${assets}/weights/ofdm/itwiddle_pq_im.bin" ]]; then
  echo "[data] generating reviewed OFDM weights"
  env AIRAN_DATA_DIR="${assets}" OFDM_BATCH_SIZE=4 PYTHONDONTWRITEBYTECODE=1 \
    python3 "${kernel_root}/ofdm/ofdm_mod_batch/scripts/ofdm_mod_ref.py"
fi

echo "[config] compiling radio=$(basename "${radio_profile}") grant=$(basename "${grant}") Rank${rank}"
PYTHONDONTWRITEBYTECODE=1 python3 "${kernel_root}/common/mimo_config_compiler.py" \
  --radio-profile "${radio_profile}" \
  --grant "${grant}" \
  --rank "${rank}" \
  --capabilities "${kernel_root}/common/mimo_operator_capabilities.json" \
  --require-eligible \
  --output "${work}/runtime.json" \
  --binary-output "${work}/runtime.bin"

echo "[build] creating one fused AscendC device image"
cmake -S "${here}" -B "${work}/build" \
  -DASCEND_CANN_PACKAGE_PATH="${cann_root}" \
  -DSOC_VERSION=Ascend310P1 \
  -DRUN_MODE=npu \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="${work}/install"
cmake --build "${work}/build" -j"${PUSCH_MIMO_BUILD_JOBS:-4}"
cmake --install "${work}/build"

iq_file="${output_dir}/tx_iq.bin"
receipt_file="${output_dir}/receipt.json"
echo "[run] Rank${rank}, slots=${slots}, single ACL context and shared device arena"
env PUSCH_MIMO_RUNTIME_CONFIG="${work}/runtime.bin" \
  LD_LIBRARY_PATH="${work}/build/lib:${cann_root}/lib64:${LD_LIBRARY_PATH:-}" \
  "${work}/install/bin/pusch_mimo_tx_chain" \
  --input-root "${assets}" --output "${iq_file}" \
  --receipt "${receipt_file}" --rank "${rank}" --slots "${slots}"

PYTHONDONTWRITEBYTECODE=1 python3 "${here}/scripts/verify_tx_chain.py" \
  --receipt "${receipt_file}" --iq "${iq_file}"
echo "[PASS] IQ: ${iq_file}"
echo "[PASS] receipt: ${receipt_file}"
