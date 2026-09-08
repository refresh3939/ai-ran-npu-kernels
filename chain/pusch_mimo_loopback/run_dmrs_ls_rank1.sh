#!/usr/bin/env bash
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")"&&pwd)";source_root="${PUSCH_MIMO_CLEAN_KERNEL_ROOT:-/home/refresh/AI-RAN-NPU-clean/kernels}";ascend_home="${ASCEND_HOME_PATH:-/usr/local/Ascend/ascend-toolkit/latest}"
source "${here}/build_variant.sh";load_mimo_build_variant 1
build="${here}/build/mimo_dmrs_ls";out="${here}/out/mimo_dmrs_ls";work="${here}/work/tx_prefix_rank1";logs="${here}/logs/dmrs_ls_rank1"
[[ -f "${work}/artifacts/coded_waveform_result.json" ]]||{ echo "[HARD_FAIL] coded waveform must pass first" >&2;exit 2;}
for x in libascendcl.so libplatform.so libregister.so;do [[ -r "${ascend_home}/lib64/${x}" ]]||exit 2;done
set +e;set +u;if [[ -f "${ascend_home}/bin/setenv.bash" ]];then source "${ascend_home}/bin/setenv.bash";else source "${ascend_home}/set_env.sh";fi;set -u;set -e
rm -rf "${logs}" "${work}/artifacts/dmrs_ls";mkdir -p "${logs}" "${work}/artifacts/dmrs_ls"
cmake -S "${source_root}/channel_est/mimo_dmrs_ls" -B "${build}" -DRUN_MODE=npu -DSOC_VERSION=Ascend310P1 -DDMRS_LS_NR="${mimo_detector_rx_capacity}" -DDMRS_LS_BLOCK_DIM="${mimo_dmrs_ls_block_dim}" -DASCEND_CANN_PACKAGE_PATH="${ascend_home}" -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="${out}" >"${logs}/build.log" 2>&1
cmake --build "${build}" -j "$(nproc)" >>"${logs}/build.log" 2>&1;cmake --install "${build}" >>"${logs}/build.log" 2>&1
g++ -O2 -std=c++17 -D_GLIBCXX_USE_CXX11_ABI=0 -Wall -Wextra -Werror -DMIMO_DMRS_LS_NR="${mimo_detector_rx_capacity}" -DMIMO_DMRS_LS_BLOCK_DIM="${mimo_dmrs_ls_block_dim}" -I"${source_root}/channel_est/mimo_dmrs_ls" -I"${source_root}/common" -I"${out}/include/ascendc_kernels_npu" -I"${ascend_home}/include" \
 "${here}/dmrs_ls_stage.cpp" "${source_root}/common/pusch_mimo_runtime_config.cpp" "${source_root}/channel_est/mimo_dmrs_ls/mimo_dmrs_ls.cpp" -L"${out}/lib" -lascendc_kernels_npu -L"${ascend_home}/lib64" -lascendcl -lplatform -lregister -ltiling_api -lascendalog -ldl -o "${out}/bin/pusch_mimo_dmrs_ls_stage" >>"${logs}/build.log" 2>&1
set +e;LD_LIBRARY_PATH="${out}/lib:${ascend_home}/lib64:${LD_LIBRARY_PATH:-}" "${out}/bin/pusch_mimo_dmrs_ls_stage" "${work}/artifacts/rx_grid" "${work}/artifacts/dmrs" "${work}/artifacts/dmrs_ls" 1 >"${logs}/device.log" 2>&1;status=$?;set -e
if [[ $status -ne 0 ]];then if grep -Eq '507008|drvRet=4' "${logs}/device.log";then echo "[DEVICE_BUSY] dmrs-ls; not retried" >&2;exit 75;fi;tail -80 "${logs}/build.log" >&2;tail -80 "${logs}/device.log" >&2;echo "[HARD_FAIL] DMRS-LS ${status}" >&2;exit "$status";fi
cp "${logs}/device.log" "${work}/artifacts/dmrs_ls_result.log";cat "${logs}/device.log"
