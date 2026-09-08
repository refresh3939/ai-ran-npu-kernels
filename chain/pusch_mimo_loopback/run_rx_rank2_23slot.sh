#!/usr/bin/env bash
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
rank="${1:-2}"
[[ "${rank}" =~ ^[234]$ ]] || { echo "usage: $0 [RANK(2|3|4)]" >&2; exit 2; }
source "${here}/build_variant.sh"
load_mimo_build_variant "${rank}"
ascend_home="${ASCEND_HOME_PATH:-/usr/local/Ascend/ascend-toolkit/latest}"
work="${here}/work/tx_prefix_rank1"
logs="${here}/logs/rx_rank${rank}_23slot"
ce_data="${work}/channel_est_rank${rank}"
ce_case="case_0_m${mimo_ce_nr}_k${rank}_r${mimo_ce_retained_rank}"
det_data="${work}/detect_rank${rank}"
det_case="case_0_m${mimo_detector_rx_capacity}_k${mimo_detector_layer_capacity}_sc1664_r${rank}"
det_in="${det_data}/data/golden/${det_case}"
det_out="${det_data}/data/ascend_output/${det_case}"
result="${work}/artifacts/demod_rank${rank}_23slot"
[[ -f "${work}/artifacts/rank${rank}_slot0_rx_result.json" ]] || {
  echo "[HARD_FAIL] Rank${rank} slot0 RX must pass first" >&2; exit 2; }
required=(
  "${here}/out/channel_est_lmmse_rank${rank}/bin/ascendc_kernels_bbit"
  "${here}/out/mimo_detect_io_pack/bin/pusch_mimo_io_pack_stage"
  "${here}/out/mimo_detect_bri_rank${rank}/bin/pusch_mimo_detector_grouped_stage"
  "${here}/out/qam_demod_256_mimo_batch/bin/pusch_mimo_qam_demod_stage"
  "${here}/out/layer_demap_mimo/bin/pusch_mimo_layer_demap_stage")
for path in "${required[@]}"; do [[ -x "${path}" ]] || { echo "[HARD_FAIL] missing ${path}" >&2; exit 2; }; done
set +e; set +u
if [[ -f "${ascend_home}/bin/setenv.bash" ]]; then source "${ascend_home}/bin/setenv.bash"; else source "${ascend_home}/set_env.sh"; fi
set -u; set -e
rm -rf "${logs}" "${result}"; mkdir -p "${logs}" "${result}" "${det_in}" "${det_out}"
run_device() {
  local tag="$1"; shift
  set +e; "$@" >"${logs}/${tag}.log" 2>&1; local status=$?; set -e
  if [[ ${status} -ne 0 ]]; then
    if grep -Eq '507008|drvRet=4' "${logs}/${tag}.log"; then
      echo "[DEVICE_BUSY] ${tag}; not retried" >&2; exit 75
    fi
    tail -100 "${logs}/${tag}.log" >&2; echo "[HARD_FAIL] ${tag} exited ${status}" >&2; exit "${status}"
  fi
}
for slot in $(seq 0 22); do
  tag="$(printf '%02d' "${slot}")"
  python3 "${here}/ce_rank.py" link --root "${ce_data}" \
    --ls "${work}/artifacts/dmrs_ls_rank${rank}" --slot "${slot}" --rank "${rank}" \
    --nr "${mimo_ce_nr}" --ce-rank "${mimo_ce_retained_rank}" --layer-capacity "${mimo_detector_layer_capacity}"
  rm -f "${ce_data}/ascend_output/${ce_case}/h_cube_time_fused_"{re,im}.bin
  run_device "ce_slot${tag}" env AIRAN_DATA_DIR="${ce_data}" WARMUP=0 TIMED=1 \
    LD_LIBRARY_PATH="${here}/out/channel_est_lmmse_rank${rank}/lib:${ascend_home}/lib64:${LD_LIBRARY_PATH:-}" \
    "${here}/out/channel_est_lmmse_rank${rank}/bin/ascendc_kernels_bbit"
  h_prefix="${ce_data}/ascend_output/${ce_case}/h_cube_time_fused"
  rm -f "${det_in}/"{hrm_re,hrm_im,yvpad_re,yvpad_im,no}.bin
  run_device "io_pack_slot${tag}" env \
    LD_LIBRARY_PATH="${here}/out/mimo_detect_io_pack/lib:${ascend_home}/lib64:${LD_LIBRARY_PATH:-}" \
    "${here}/out/mimo_detect_io_pack/bin/pusch_mimo_io_pack_stage" \
    "${work}/artifacts/rx_grid_rank${rank}/slot${tag}_re.bin" \
    "${work}/artifacts/rx_grid_rank${rank}/slot${tag}_im.bin" "${h_prefix}" \
    "${work}/artifacts/dmrs_ls_rank${rank}/slot${tag}_noise.bin" "${det_in}" "${rank}"
  rm -f "${det_out}/"{xhat_re,xhat_im,no_eff}.bin
  run_device "detector_slot${tag}" env \
    LD_LIBRARY_PATH="${here}/out/mimo_detect_bri_rank${rank}/lib:${ascend_home}/lib64:${ascend_home}/compiler/lib64:${LD_LIBRARY_PATH:-}" \
    "${here}/out/mimo_detect_bri_rank${rank}/bin/pusch_mimo_detector_grouped_stage" \
    "${det_in}" "${det_out}" "${rank}"
  rm -f "${result}/slot${tag}_layer_llr.bin" "${result}/slot${tag}_cw_llr.bin"
  run_device "qam_slot${tag}" env \
    LD_LIBRARY_PATH="${here}/out/qam_demod_256_mimo_batch/lib:${ascend_home}/lib64:${LD_LIBRARY_PATH:-}" \
    "${here}/out/qam_demod_256_mimo_batch/bin/pusch_mimo_qam_demod_stage" \
    "${det_out}" "${result}/slot${tag}_layer_llr.bin" "${rank}"
  run_device "layer_slot${tag}" env \
    LD_LIBRARY_PATH="${here}/out/layer_demap_mimo/lib:${ascend_home}/lib64:${LD_LIBRARY_PATH:-}" \
    "${here}/out/layer_demap_mimo/bin/pusch_mimo_layer_demap_stage" \
    "${result}/slot${tag}_layer_llr.bin" "${result}/slot${tag}_cw_llr.bin" "${rank}"
  rm -f "${result}/slot${tag}_layer_llr.bin"
  echo "[slot ${tag}] Rank${rank} CE -> io_pack -> BRI -> QAM -> layer_demap PASS"
done
python3 "${here}/verify_demod_rank.py" --root "${work}" --rank "${rank}" | tee "${logs}/verify.log"
