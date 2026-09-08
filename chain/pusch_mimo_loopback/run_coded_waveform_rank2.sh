#!/usr/bin/env bash
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
rank="${1:-2}"
[[ "${rank}" =~ ^[234]$ ]] || { echo "usage: $0 [RANK(2|3|4)]" >&2; exit 2; }
source "${here}/build_variant.sh"
load_mimo_build_variant "${rank}"
ports="${rank}"; [[ "${rank}" == 3 ]] && ports=4
default_gain=0.16; [[ "${rank}" == 3 ]] && default_gain=0.22
default_noise=90
if [[ -n "${PUSCH_MIMO_RESOLVED_PLAN:-}" ]]; then
  mapfile -t profile_channel < <(
    python3 -c 'import json,sys; p=json.load(open(sys.argv[1])); print(p["source"]["profile_name"]); print(p["channel"]["gain"]); print(p["channel"]["awgn_std_int16"])' \
      "${PUSCH_MIMO_RESOLVED_PLAN}"
  )
  [[ "${#profile_channel[@]}" -eq 3 ]] || {
    echo "[HARD_FAIL] invalid channel fields in resolved plan" >&2; exit 2; }
  : "${PUSCH_MIMO_RADIO_PROFILE:=${profile_channel[0]}}"
  export PUSCH_MIMO_RADIO_PROFILE
  default_gain="${profile_channel[1]}"
  default_noise="${profile_channel[2]}"
fi
ascend_home="${ASCEND_HOME_PATH:-/usr/local/Ascend/ascend-toolkit/latest}"
work="${here}/work/tx_prefix_rank1"
out_root="${here}/out"
logs="${here}/logs/coded_waveform_rank${rank}"
[[ -f "${work}/artifacts/tx_grid_rank${rank}_result.json" ]] || {
  echo "[HARD_FAIL] Rank${rank} TX grid must pass first" >&2; exit 2; }
for op in re_map_batch ofdm_mod_batch ofdm_demod_batch re_demap_batch; do
  [[ -x "${out_root}/${op}/bin/ascendc_kernels_bbit" && \
     -r "${out_root}/${op}/lib/libascendc_kernels_npu.so" ]] || {
    echo "[HARD_FAIL] missing built device operator ${op}" >&2; exit 2; }
done
for library in libascendcl.so libplatform.so libregister.so; do
  [[ -r "${ascend_home}/lib64/${library}" ]] || {
    echo "[HARD_FAIL] missing CANN library ${library}" >&2; exit 2; }
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
rm -rf "${logs}" "${work}/artifacts/tx_fft_rank${rank}" \
  "${work}/artifacts/rx_grid_rank${rank}" "${work}/ofdm_mod" \
  "${work}/ofdm_demod" "${work}/re_demap"
mkdir -p "${logs}" "${work}/re_map/ascend_output" \
  "${work}/ofdm_mod/data/golden" "${work}/ofdm_mod/data/ascend_output" \
  "${work}/ofdm_demod/data/golden" "${work}/ofdm_demod/data/ascend_output" \
  "${work}/re_demap/data/golden" "${work}/re_demap/data/ascend_output"
python3 "${here}/rank2_23slot.py" prepare-coded --root "${work}" --rank "${rank}"
run_device() {
  local tag="$1"; shift
  set +e
  "$@" >"${logs}/${tag}.log" 2>&1
  local status=$?
  set -e
  if [[ ${status} -ne 0 ]]; then
    if grep -Eq '507008|drvRet=4' "${logs}/${tag}.log"; then
      echo "[DEVICE_BUSY] ${tag}: 507008/drvRet=4; not retried" >&2; exit 75
    fi
    tail -80 "${logs}/${tag}.log" >&2
    echo "[HARD_FAIL] ${tag} exited ${status}" >&2
    exit "${status}"
  fi
}
common_ld="${ascend_home}/lib64:${LD_LIBRARY_PATH:-}"
for slot in $(seq 0 22); do
  tag="$(printf '%02d' "${slot}")"
  python3 "${here}/rank2_23slot.py" select-tx --root "${work}" --slot "${slot}" --rank "${rank}"
  run_device "re_map_slot${tag}" env AIRAN_DATA_DIR="${work}/re_map" WARMUP=0 TIMED=1 \
    LD_LIBRARY_PATH="${out_root}/re_map_batch/lib:${common_ld}" \
    "${out_root}/re_map_batch/bin/ascendc_kernels_bbit"
  python3 "${here}/rank2_23slot.py" collect-tx --root "${work}" --slot "${slot}" --rank "${rank}"
done
python3 "${here}/rank2_23slot.py" assemble-tx --root "${work}" --rank "${rank}"
run_device "ofdm_mod_rank${rank}_23slot" env AIRAN_DATA_DIR="${work}/ofdm_mod" OFDM_BATCH_SIZE="$((23*ports))" \
  LD_LIBRARY_PATH="${out_root}/ofdm_mod_batch/lib:${common_ld}" \
  "${out_root}/ofdm_mod_batch/bin/ascendc_kernels_bbit"
PUSCH_MIMO_CHANNEL_GAIN="${PUSCH_MIMO_CHANNEL_GAIN:-${default_gain}}" \
PUSCH_MIMO_CHANNEL_NOISE_STD="${PUSCH_MIMO_CHANNEL_NOISE_STD:-${default_noise}}" \
  python3 "${here}/rank2_23slot.py" channel --root "${work}" --rank "${rank}"
for slot in $(seq 0 22); do
  tag="$(printf '%02d' "${slot}")"
  python3 "${here}/rank2_23slot.py" select-rx --root "${work}" --slot "${slot}" --rank "${rank}"
  run_device "ofdm_demod_slot${tag}" env AIRAN_DATA_DIR="${work}/ofdm_demod" OFDM_BATCH_SIZE=64 \
    LD_LIBRARY_PATH="${out_root}/ofdm_demod_batch/lib:${common_ld}" \
    "${out_root}/ofdm_demod_batch/bin/ascendc_kernels_bbit"
  cp "${work}/ofdm_demod/data/ascend_output/output_re.bin" \
    "${work}/re_demap/data/golden/input_re.bin"
  cp "${work}/ofdm_demod/data/ascend_output/output_im.bin" \
    "${work}/re_demap/data/golden/input_im.bin"
  run_device "re_demap_slot${tag}" env AIRAN_DATA_DIR="${work}/re_demap" RE_DEMAP_BATCH_SIZE=64 \
    LD_LIBRARY_PATH="${out_root}/re_demap_batch/lib:${common_ld}" \
    "${out_root}/re_demap_batch/bin/ascendc_kernels_bbit"
  python3 "${here}/rank2_23slot.py" collect-rx --root "${work}" --slot "${slot}" --rank "${rank}"
done
PUSCH_MIMO_CHANNEL_GAIN="${PUSCH_MIMO_CHANNEL_GAIN:-${default_gain}}" \
PUSCH_MIMO_CHANNEL_NOISE_STD="${PUSCH_MIMO_CHANNEL_NOISE_STD:-${default_noise}}" \
  python3 "${here}/rank2_23slot.py" verify --root "${work}" --rank "${rank}" | tee "${logs}/verify.log"
echo "[MILESTONE PASS] Rank${rank} coded 23-slot ${ports}-port TX -> full-rank channel -> physical 8Rx / execution-grid ${mimo_detector_rx_capacity}Rx"
