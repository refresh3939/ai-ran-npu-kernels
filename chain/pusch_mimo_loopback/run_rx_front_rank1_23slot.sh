#!/usr/bin/env bash
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")"&&pwd)"
source "${here}/build_variant.sh"
load_mimo_build_variant 1
root="${PUSCH_MIMO_CLEAN_KERNEL_ROOT:-/home/refresh/AI-RAN-NPU-clean/kernels}"
ascend_home="${ASCEND_HOME_PATH:-/usr/local/Ascend/ascend-toolkit/latest}"
work="${here}/work/tx_prefix_rank1"
ce_data="${work}/channel_est"
ce_case="case_0_m${mimo_ce_nr}_k1_r${mimo_ce_retained_rank}"
det_case="case_0_m${mimo_detector_rx_capacity}_k${mimo_detector_layer_capacity}_sc1664_r1"
det_data="${work}/detect"
det_in="${det_data}/data/golden/${det_case}"
det_out="${det_data}/data/ascend_output/${det_case}"
result="${work}/artifacts/demod_23slot"
logs="${here}/logs/rx_front_rank1_23slot"

required=(
  "${here}/out/channel_est_lmmse_rank1/bin/ascendc_kernels_bbit"
  "${here}/out/mimo_detect_io_pack/bin/pusch_mimo_io_pack_stage"
  "${here}/out/mimo_detect_bri_rank1/bin/pusch_mimo_detector_grouped_stage"
  "${here}/out/qam_demod_256_mimo_batch/bin/pusch_mimo_qam_demod_stage"
  "${here}/out/layer_demap_mimo/bin/pusch_mimo_layer_demap_stage"
)
for path in "${required[@]}";do [[ -x "$path" ]]||{ echo "[HARD_FAIL] missing built executable $path" >&2;exit 2;};done
[[ -f "${work}/artifacts/dmrs_ls_result.log" ]]||{ echo "[HARD_FAIL] DMRS-LS result absent" >&2;exit 2;}
for library in libascendcl.so libplatform.so libregister.so;do [[ -r "${ascend_home}/lib64/${library}" ]]||{ echo "[HARD_FAIL] missing ${library}" >&2;exit 2;};done
set +e
set +u
if [[ -f "${ascend_home}/bin/setenv.bash" ]];then source "${ascend_home}/bin/setenv.bash";else source "${ascend_home}/set_env.sh";fi
set -u
set -e
rm -rf "$logs" "$result"
mkdir -p "$logs" "$result" "$det_in" "$det_out" "${ce_data}/ascend_output/${ce_case}"

run_device(){
  local tag="$1";shift
  set +e
  "$@" >"${logs}/${tag}.log" 2>&1
  local status=$?
  set -e
  if [[ $status -ne 0 ]];then
    if grep -Eq '507008|drvRet=4' "${logs}/${tag}.log";then echo "[DEVICE_BUSY] ${tag}; not retried" >&2;exit 75;fi
    tail -100 "${logs}/${tag}.log" >&2
    echo "[HARD_FAIL] ${tag} exited ${status}" >&2
    exit "$status"
  fi
}

for slot in $(seq 0 22);do
  tag="$(printf '%02d' "$slot")"
  python3 "${here}/ce_rank.py" link --root "$ce_data" \
    --ls "${work}/artifacts/dmrs_ls" --slot "$slot" --rank 1 \
    --nr "${mimo_ce_nr}" --ce-rank "${mimo_ce_retained_rank}" --layer-capacity "${mimo_detector_layer_capacity}"
  run_device "ce_slot${tag}" env AIRAN_DATA_DIR="$ce_data" WARMUP=0 TIMED=1 \
    LD_LIBRARY_PATH="${here}/out/channel_est_lmmse_rank1/lib:${ascend_home}/lib64:${LD_LIBRARY_PATH:-}" \
    "${here}/out/channel_est_lmmse_rank1/bin/ascendc_kernels_bbit"
  h_prefix="${ce_data}/ascend_output/${ce_case}/h_cube_time_fused"
  run_device "io_pack_slot${tag}" env \
    LD_LIBRARY_PATH="${here}/out/mimo_detect_io_pack/lib:${ascend_home}/lib64:${LD_LIBRARY_PATH:-}" \
    "${here}/out/mimo_detect_io_pack/bin/pusch_mimo_io_pack_stage" \
    "${work}/artifacts/rx_grid/slot${tag}_re.bin" \
    "${work}/artifacts/rx_grid/slot${tag}_im.bin" "$h_prefix" \
    "${work}/artifacts/dmrs_ls/slot${tag}_noise.bin" "$det_in" 1
  run_device "detect_slot${tag}" env \
    LD_LIBRARY_PATH="${here}/out/mimo_detect_bri_rank1/lib:${ascend_home}/lib64:${ascend_home}/compiler/lib64:${LD_LIBRARY_PATH:-}" \
    "${here}/out/mimo_detect_bri_rank1/bin/pusch_mimo_detector_grouped_stage" "$det_in" "$det_out" 1
  run_device "qam_slot${tag}" env \
    LD_LIBRARY_PATH="${here}/out/qam_demod_256_mimo_batch/lib:${ascend_home}/lib64:${LD_LIBRARY_PATH:-}" \
    "${here}/out/qam_demod_256_mimo_batch/bin/pusch_mimo_qam_demod_stage" "$det_out" \
    "${result}/slot${tag}_layer_llr.bin" 1
  run_device "layer_slot${tag}" env \
    LD_LIBRARY_PATH="${here}/out/layer_demap_mimo/lib:${ascend_home}/lib64:${LD_LIBRARY_PATH:-}" \
    "${here}/out/layer_demap_mimo/bin/pusch_mimo_layer_demap_stage" \
    "${result}/slot${tag}_layer_llr.bin" "${result}/slot${tag}_cw_llr.bin" 1
  rm -f "${result}/slot${tag}_layer_llr.bin"
  echo "[slot ${tag}] CE -> io_pack -> grouped BRI -> QAM -> layer_demap PASS"
done
python3 "${here}/verify_demod_23slot.py" --root "$work" | tee "${logs}/verify.log"
