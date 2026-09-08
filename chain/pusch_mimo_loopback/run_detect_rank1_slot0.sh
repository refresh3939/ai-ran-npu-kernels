#!/usr/bin/env bash
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")"&&pwd)";root="${PUSCH_MIMO_CLEAN_KERNEL_ROOT:-/home/refresh/AI-RAN-NPU-clean/kernels}";ascend_home="${ASCEND_HOME_PATH:-/usr/local/Ascend/ascend-toolkit/latest}";work="${here}/work/tx_prefix_rank1";logs="${here}/logs/detect_rank1"
source "${here}/build_variant.sh";load_mimo_build_variant 1
io_build="${here}/build/mimo_detect_io_pack";io_out="${here}/out/mimo_detect_io_pack";det_build="${here}/build/mimo_detect_bri_rank1";det_out="${here}/out/mimo_detect_bri_rank1";det_data="${work}/detect"
det_case="case_0_m${mimo_detector_rx_capacity}_k${mimo_detector_layer_capacity}_sc1664_r1"
[[ -f "${work}/artifacts/channel_est/slot00_h_re.bin" ]]||{ echo "[HARD_FAIL] CE slot0 absent" >&2;exit 2;}
for x in libascendcl.so libplatform.so libregister.so;do [[ -r "${ascend_home}/lib64/${x}" ]]||exit 2;done
set +e;set +u;if [[ -f "${ascend_home}/bin/setenv.bash" ]];then source "${ascend_home}/bin/setenv.bash";else source "${ascend_home}/set_env.sh";fi;set -u;set -e
rm -rf "${logs}" "${det_data}" "${det_build}" "${det_out}";mkdir -p "${logs}" "${det_data}/data/golden/${det_case}" "${det_data}/data/ascend_output/${det_case}"
cmake -S "${root}/mimo/mimo_detect_io_pack" -B "${io_build}" -DRUN_MODE=npu -DSOC_VERSION=Ascend310P1 -DIO_PACK_RX_CAPACITY="${mimo_detector_rx_capacity}" -DIO_PACK_LAYER_CAPACITY="${mimo_detector_layer_capacity}" -DASCEND_CANN_PACKAGE_PATH="${ascend_home}" -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="${io_out}" >"${logs}/build_io.log" 2>&1
cmake --build "${io_build}" -j "$(nproc)" >>"${logs}/build_io.log" 2>&1;cmake --install "${io_build}" >>"${logs}/build_io.log" 2>&1
g++ -O2 -std=c++17 -D_GLIBCXX_USE_CXX11_ABI=0 -Wall -Wextra -Werror -DMIMO_DETECT_IO_NR="${mimo_detector_rx_capacity}" -DMIMO_DETECT_IO_NL="${mimo_detector_layer_capacity}" -I"${root}/mimo/mimo_detect_io_pack" -I"${root}/common" -I"${io_out}/include/ascendc_kernels_npu" -I"${ascend_home}/include" "${here}/io_pack_stage.cpp" "${root}/common/pusch_mimo_runtime_config.cpp" "${root}/mimo/mimo_detect_io_pack/mimo_detect_io_pack.cpp" "${root}/mimo/mimo_detect_io_pack/mimo_detect_io_pack_runtime.cpp" -L"${io_out}/lib" -lascendc_kernels_npu -L"${ascend_home}/lib64" -lascendcl -lplatform -lregister -ltiling_api -lascendalog -ldl -o "${io_out}/bin/pusch_mimo_io_pack_stage" >>"${logs}/build_io.log" 2>&1
prefix="${work}/artifacts/channel_est/slot00_h";case_dir="${det_data}/data/golden/${det_case}"
LD_LIBRARY_PATH="${io_out}/lib:${ascend_home}/lib64:${LD_LIBRARY_PATH:-}" "${io_out}/bin/pusch_mimo_io_pack_stage" "${work}/artifacts/rx_grid/slot00_re.bin" "${work}/artifacts/rx_grid/slot00_im.bin" "$prefix" "${work}/artifacts/dmrs_ls/slot00_noise.bin" "$case_dir" 1 >"${logs}/io_slot00.log" 2>&1
cmake -S "${here}/device_build/mimo_detect_bri_rank1" -B "${det_build}" -DRUN_MODE=npu -DSOC_VERSION=Ascend310P1 -DACTIVE_LAYERS=1 -DDETECTOR_RX_CAPACITY="${mimo_detector_rx_capacity}" -DDETECTOR_LAYER_CAPACITY="${mimo_detector_layer_capacity}" -DDETECTOR_BRI_BLOCK="${mimo_detector_bri_block}" -DCLEAN_KERNEL_ROOT="${root}" -DASCEND_CANN_PACKAGE_PATH="${ascend_home}" -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="${det_out}" >"${logs}/build_detector.log" 2>&1
cmake --build "${det_build}" -j "$(nproc)" >>"${logs}/build_detector.log" 2>&1;cmake --install "${det_build}" >>"${logs}/build_detector.log" 2>&1
g++ -O2 -std=c++17 -D_GLIBCXX_USE_CXX11_ABI=0 -Wall -Wextra -Werror \
  -DMIMO_M="${mimo_detector_rx_capacity}" -DMIMO_K="${mimo_detector_layer_capacity}" \
  -DMIMO_KR=1 -DBRI_B="${mimo_detector_bri_block}" \
  -DMIMO_IO_NR_VALUE="${mimo_detector_rx_capacity}" \
  -DMIMO_IO_LAYER_VALUE="${mimo_detector_layer_capacity}" \
  -DBRI_GROUPED_NR="${mimo_detector_rx_capacity}" \
  -DBRI_GROUPED_NL="${mimo_detector_layer_capacity}" \
  -DGROUP_BATCH=8 -DRE_WINDOW=5824 -DN_WINDOW_PER_CORE=1 \
  -I"${root}/mimo/mimo_detect_bri_batch" -I"${root}/common" \
  -I"${det_out}/include/ascendc_kernels_npu" -I"${ascend_home}/include" \
  "${here}/detector_grouped_stage.cpp" \
  "${root}/common/pusch_mimo_runtime_config.cpp" \
  "${root}/mimo/mimo_detect_bri_batch/mimo_detect_bri_tiling.cpp" \
  -L"${det_out}/lib" -lascendc_kernels_npu -L"${ascend_home}/lib64" \
  -L"${ascend_home}/compiler/lib64" -lascendcl -lplatform -lregister \
  -ltiling_api -lascendalog -lc_sec -ldl \
  -o "${det_out}/bin/pusch_mimo_detector_grouped_stage" \
  >>"${logs}/build_detector.log" 2>&1
set +e;AIRAN_DATA_DIR="${det_data}" AIRAN_WARMUP=0 AIRAN_TIMED=1 LD_LIBRARY_PATH="${det_out}/lib:${ascend_home}/lib64:${LD_LIBRARY_PATH:-}" "${det_out}/bin/ascendc_kernels_bbit" >"${logs}/detector_slot00.log" 2>&1;status=$?;set -e
if [[ $status -ne 0 ]];then if grep -Eq '507008|drvRet=4' "${logs}/detector_slot00.log";then echo "[DEVICE_BUSY] detector; not retried" >&2;exit 75;fi;tail -100 "${logs}/build_detector.log" >&2;tail -100 "${logs}/detector_slot00.log" >&2;echo "[HARD_FAIL] detector slot0 ${status}" >&2;exit "$status";fi
cat "${logs}/io_slot00.log";cat "${logs}/detector_slot00.log"
set +e
LD_LIBRARY_PATH="${det_out}/lib:${ascend_home}/lib64:${LD_LIBRARY_PATH:-}" \
  "${det_out}/bin/pusch_mimo_detector_grouped_stage" "$case_dir" \
  "${det_data}/data/ascend_output/${det_case}" 1 \
  >"${logs}/detector_grouped_slot00.log" 2>&1
status=$?
set -e
if [[ $status -ne 0 ]];then
  if grep -Eq '507008|drvRet=4' "${logs}/detector_grouped_slot00.log";then echo "[DEVICE_BUSY] grouped detector; not retried" >&2;exit 75;fi
  tail -100 "${logs}/detector_grouped_slot00.log" >&2
  echo "[HARD_FAIL] grouped detector slot0 ${status}" >&2
  exit "$status"
fi
cat "${logs}/detector_grouped_slot00.log"
