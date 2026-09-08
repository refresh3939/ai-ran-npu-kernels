#!/usr/bin/env bash
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
rank="${1:-2}"
[[ "${rank}" =~ ^[234]$ ]] || { echo "usage: $0 [RANK(2|3|4)]" >&2; exit 2; }
source "${here}/build_variant.sh"
load_mimo_build_variant "${rank}"
root="${PUSCH_MIMO_CLEAN_KERNEL_ROOT:-/home/refresh/AI-RAN-NPU-clean/kernels}"
ascend_home="${ASCEND_HOME_PATH:-/usr/local/Ascend/ascend-toolkit/latest}"
work="${here}/work/tx_prefix_rank1"
logs="${here}/logs/rx_rank${rank}_slot0"
ce_src="${root}/channel_est/channel_est_lmmse_mimo"
ce_data="${work}/channel_est_rank${rank}"
ce_case="case_0_m${mimo_ce_nr}_k${rank}_r${mimo_ce_retained_rank}"
ce_build="${here}/build/channel_est_lmmse_rank${rank}"
ce_out="${here}/out/channel_est_lmmse_rank${rank}"
io_build="${here}/build/mimo_detect_io_pack"
io_out="${here}/out/mimo_detect_io_pack"
det_build="${here}/build/mimo_detect_bri_rank${rank}"
det_out="${here}/out/mimo_detect_bri_rank${rank}"
det_data="${work}/detect_rank${rank}"
det_case="case_0_m${mimo_detector_rx_capacity}_k${mimo_detector_layer_capacity}_sc1664_r${rank}"
det_in="${det_data}/data/golden/${det_case}"
det_result="${det_data}/data/ascend_output/${det_case}"
demod="${work}/artifacts/demod_rank${rank}_slot0"
[[ -f "${work}/artifacts/dmrs_ls_rank${rank}/result.json" ]] || {
  echo "[HARD_FAIL] Rank${rank} DMRS-LS must pass first" >&2; exit 2; }
for library in libascendcl.so libplatform.so libregister.so; do
  [[ -r "${ascend_home}/lib64/${library}" ]] || { echo "[HARD_FAIL] missing ${library}" >&2; exit 2; }
done
set +e; set +u
if [[ -f "${ascend_home}/bin/setenv.bash" ]]; then source "${ascend_home}/bin/setenv.bash"; else source "${ascend_home}/set_env.sh"; fi
set -u; set -e
rm -rf "${logs}" "${ce_data}" "${det_data}" "${det_build}" "${det_out}" "${demod}"
mkdir -p "${logs}" "${det_in}" "${det_result}" "${demod}"

PYTHONDONTWRITEBYTECODE=1 NR="${mimo_ce_nr}" NL="${mimo_ce_nl}" RANK="${mimo_ce_retained_rank}" AIRAN_DATA_DIR="${ce_data}" \
  python3 "${ce_src}/scripts/channel_est_lmmse_ref.py" >"${logs}/generate_ce.log" 2>&1
python3 "${here}/ce_rank.py" prepare --root "${ce_data}" --rank "${rank}" \
  --nr "${mimo_ce_nr}" --ce-rank "${mimo_ce_retained_rank}" --layer-capacity "${mimo_detector_layer_capacity}"
python3 "${here}/ce_rank.py" link --root "${ce_data}" \
  --ls "${work}/artifacts/dmrs_ls_rank${rank}" --slot 0 --rank "${rank}" \
  --nr "${mimo_ce_nr}" --ce-rank "${mimo_ce_retained_rank}" --layer-capacity "${mimo_detector_layer_capacity}"
mkdir -p "${ce_data}/ascend_output/${ce_case}"
cmake -S "${ce_src}" -B "${ce_build}" -DRUN_MODE=npu -DSOC_VERSION=Ascend310P1 \
  -DCE_NR="${mimo_ce_nr}" -DCE_NL="${mimo_ce_nl}" -DCE_RANK="${mimo_ce_retained_rank}" \
  -DCE_BLOCK_DIM="${mimo_ce_block_dim}" -DCE_CUBE_TIME_FUSED="${mimo_ce_cube_time_fused}" \
  -DASCEND_CANN_PACKAGE_PATH="${ascend_home}" -DCMAKE_CXX_FLAGS="-I${here}/shims" \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="${ce_out}" >"${logs}/build_ce.log" 2>&1
cmake --build "${ce_build}" -j "$(nproc)" >>"${logs}/build_ce.log" 2>&1
cmake --install "${ce_build}" >>"${logs}/build_ce.log" 2>&1

cmake -S "${root}/mimo/mimo_detect_io_pack" -B "${io_build}" -DRUN_MODE=npu \
  -DSOC_VERSION=Ascend310P1 -DASCEND_CANN_PACKAGE_PATH="${ascend_home}" \
  -DIO_PACK_RX_CAPACITY="${mimo_detector_rx_capacity}" \
  -DIO_PACK_LAYER_CAPACITY="${mimo_detector_layer_capacity}" \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="${io_out}" >"${logs}/build_io.log" 2>&1
cmake --build "${io_build}" -j "$(nproc)" >>"${logs}/build_io.log" 2>&1
cmake --install "${io_build}" >>"${logs}/build_io.log" 2>&1
g++ -O2 -std=c++17 -D_GLIBCXX_USE_CXX11_ABI=0 -Wall -Wextra -Werror \
  -DMIMO_DETECT_IO_NR="${mimo_detector_rx_capacity}" \
  -DMIMO_DETECT_IO_NL="${mimo_detector_layer_capacity}" \
  -I"${root}/mimo/mimo_detect_io_pack" -I"${root}/common" -I"${io_out}/include/ascendc_kernels_npu" \
  -I"${ascend_home}/include" "${here}/io_pack_stage.cpp" \
  "${root}/common/pusch_mimo_runtime_config.cpp" \
  "${root}/mimo/mimo_detect_io_pack/mimo_detect_io_pack.cpp" \
  "${root}/mimo/mimo_detect_io_pack/mimo_detect_io_pack_runtime.cpp" \
  -L"${io_out}/lib" -lascendc_kernels_npu -L"${ascend_home}/lib64" \
  -lascendcl -lplatform -lregister -ltiling_api -lascendalog -ldl \
  -o "${io_out}/bin/pusch_mimo_io_pack_stage" >>"${logs}/build_io.log" 2>&1

cmake -S "${here}/device_build/mimo_detect_bri_rank1" -B "${det_build}" \
  -DRUN_MODE=npu -DSOC_VERSION=Ascend310P1 -DACTIVE_LAYERS="${rank}" \
  -DDETECTOR_RX_CAPACITY="${mimo_detector_rx_capacity}" \
  -DDETECTOR_LAYER_CAPACITY="${mimo_detector_layer_capacity}" \
  -DDETECTOR_BRI_BLOCK="${mimo_detector_bri_block}" \
  -DCLEAN_KERNEL_ROOT="${root}" -DASCEND_CANN_PACKAGE_PATH="${ascend_home}" \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="${det_out}" >"${logs}/build_detector.log" 2>&1
cmake --build "${det_build}" -j "$(nproc)" >>"${logs}/build_detector.log" 2>&1
cmake --install "${det_build}" >>"${logs}/build_detector.log" 2>&1
g++ -O2 -std=c++17 -D_GLIBCXX_USE_CXX11_ABI=0 -Wall -Wextra -Werror \
  -DMIMO_M="${mimo_detector_rx_capacity}" -DMIMO_K="${mimo_detector_layer_capacity}" \
  -DMIMO_KR="${rank}" -DBRI_B="${mimo_detector_bri_block}" \
  -DMIMO_IO_NR_VALUE="${mimo_detector_rx_capacity}" \
  -DMIMO_IO_LAYER_VALUE="${mimo_detector_layer_capacity}" \
  -DBRI_GROUPED_NR="${mimo_detector_rx_capacity}" \
  -DBRI_GROUPED_NL="${mimo_detector_layer_capacity}" \
  -DGROUP_BATCH=8 -DRE_WINDOW=5824 -DN_WINDOW_PER_CORE=1 \
  -I"${root}/mimo/mimo_detect_bri_batch" -I"${root}/common" -I"${det_out}/include/ascendc_kernels_npu" \
  -I"${ascend_home}/include" "${here}/detector_grouped_stage.cpp" \
  "${root}/common/pusch_mimo_runtime_config.cpp" \
  "${root}/mimo/mimo_detect_bri_batch/mimo_detect_bri_tiling.cpp" \
  -L"${det_out}/lib" -lascendc_kernels_npu -L"${ascend_home}/lib64" \
  -L"${ascend_home}/compiler/lib64" -lascendcl -lplatform -lregister \
  -ltiling_api -lascendalog -lc_sec -ldl \
  -o "${det_out}/bin/pusch_mimo_detector_grouped_stage" >>"${logs}/build_detector.log" 2>&1

build_op() {
  local tag="$1" rel="$2"
  cmake -S "${root}/${rel}" -B "${here}/build/${tag}" -DRUN_MODE=npu \
    -DSOC_VERSION=Ascend310P1 -DASCEND_CANN_PACKAGE_PATH="${ascend_home}" \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="${here}/out/${tag}" \
    >"${logs}/build_${tag}.log" 2>&1
  cmake --build "${here}/build/${tag}" -j "$(nproc)" >>"${logs}/build_${tag}.log" 2>&1
  cmake --install "${here}/build/${tag}" >>"${logs}/build_${tag}.log" 2>&1
}
build_op qam_demod_256_mimo_batch mapping/qam_demod_256_mimo_batch
build_op layer_demap_mimo mapping/layer_demap_mimo
g++ -O2 -std=c++17 -D_GLIBCXX_USE_CXX11_ABI=0 -Wall -Wextra -Werror \
  -I"${root}/mapping/qam_demod_256_mimo_batch" -I"${root}/common" \
  -I"${here}/out/qam_demod_256_mimo_batch/include/ascendc_kernels_npu" -I"${ascend_home}/include" \
  "${here}/qam_demod_stage.cpp" "${root}/common/pusch_mimo_runtime_config.cpp" \
  "${root}/mapping/qam_demod_256_mimo_batch/qam256_demod_batch.cpp" \
  "${root}/mapping/qam_demod_256_mimo_batch/qam256_demod_batch_contract.cpp" \
  -L"${here}/out/qam_demod_256_mimo_batch/lib" -lascendc_kernels_npu \
  -L"${ascend_home}/lib64" -lascendcl -lplatform -lregister -ltiling_api -lascendalog -ldl \
  -o "${here}/out/qam_demod_256_mimo_batch/bin/pusch_mimo_qam_demod_stage" \
  >>"${logs}/build_qam_demod_256_mimo_batch.log" 2>&1
g++ -O2 -std=c++17 -D_GLIBCXX_USE_CXX11_ABI=0 -Wall -Wextra -Werror \
  -I"${root}/mapping/layer_demap_mimo" -I"${root}/common" \
  -I"${here}/out/layer_demap_mimo/include/ascendc_kernels_npu" -I"${ascend_home}/include" \
  "${here}/layer_demap_stage.cpp" "${root}/common/pusch_mimo_runtime_config.cpp" \
  "${root}/mapping/layer_demap_mimo/layer_demap.cpp" \
  -L"${here}/out/layer_demap_mimo/lib" -lascendc_kernels_npu -L"${ascend_home}/lib64" \
  -lascendcl -lplatform -lregister -ltiling_api -lascendalog -ldl \
  -o "${here}/out/layer_demap_mimo/bin/pusch_mimo_layer_demap_stage" \
  >>"${logs}/build_layer_demap_mimo.log" 2>&1

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
run_device ce_slot00 env AIRAN_DATA_DIR="${ce_data}" WARMUP=0 TIMED=1 \
  LD_LIBRARY_PATH="${ce_out}/lib:${ascend_home}/lib64:${LD_LIBRARY_PATH:-}" \
  "${ce_out}/bin/ascendc_kernels_bbit"
python3 "${here}/ce_rank.py" collect --root "${ce_data}" \
  --out "${work}/artifacts/channel_est_rank${rank}" --slot 0 --rank "${rank}" \
  --nr "${mimo_ce_nr}" --ce-rank "${mimo_ce_retained_rank}" --layer-capacity "${mimo_detector_layer_capacity}"
h_prefix="${ce_data}/ascend_output/${ce_case}/h_cube_time_fused"
run_device io_pack_slot00 env LD_LIBRARY_PATH="${io_out}/lib:${ascend_home}/lib64:${LD_LIBRARY_PATH:-}" \
  "${io_out}/bin/pusch_mimo_io_pack_stage" \
  "${work}/artifacts/rx_grid_rank${rank}/slot00_re.bin" "${work}/artifacts/rx_grid_rank${rank}/slot00_im.bin" \
  "${h_prefix}" "${work}/artifacts/dmrs_ls_rank${rank}/slot00_noise.bin" "${det_in}" "${rank}"
run_device detector_slot00 env \
  LD_LIBRARY_PATH="${det_out}/lib:${ascend_home}/lib64:${ascend_home}/compiler/lib64:${LD_LIBRARY_PATH:-}" \
  "${det_out}/bin/pusch_mimo_detector_grouped_stage" "${det_in}" "${det_result}" "${rank}"
run_device qam_slot00 env \
  LD_LIBRARY_PATH="${here}/out/qam_demod_256_mimo_batch/lib:${ascend_home}/lib64:${LD_LIBRARY_PATH:-}" \
  "${here}/out/qam_demod_256_mimo_batch/bin/pusch_mimo_qam_demod_stage" \
  "${det_result}" "${demod}/layer_llr.bin" "${rank}"
run_device layer_slot00 env \
  LD_LIBRARY_PATH="${here}/out/layer_demap_mimo/lib:${ascend_home}/lib64:${LD_LIBRARY_PATH:-}" \
  "${here}/out/layer_demap_mimo/bin/pusch_mimo_layer_demap_stage" \
  "${demod}/layer_llr.bin" "${demod}/cw_llr.bin" "${rank}"
python3 "${here}/verify_rank2_slot0.py" --root "${work}" --rank "${rank}" \
  --detector-rx-capacity "${mimo_detector_rx_capacity}" \
  --detector-layer-capacity "${mimo_detector_layer_capacity}" | tee "${logs}/verify.log"
echo "[MILESTONE PASS] Rank${rank} slot0 actual CE -> io_pack -> BRI -> QAM -> layer demap"
