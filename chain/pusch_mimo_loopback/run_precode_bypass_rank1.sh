#!/usr/bin/env bash
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; source_root="${PUSCH_MIMO_CLEAN_KERNEL_ROOT:-/home/refresh/AI-RAN-NPU-clean/kernels}"
ascend_home="${ASCEND_HOME_PATH:-/usr/local/Ascend/ascend-toolkit/latest}"; build="${here}/build/pusch_codebook_precode"; out="${here}/out/pusch_codebook_precode"
work="${here}/work/tx_prefix_rank1"; logs="${here}/logs/precode_bypass_rank1"
[[ -f "${work}/artifacts/grid_map_result.log" ]]||{ echo "[HARD_FAIL] grid map must pass first" >&2;exit 2;}
for x in libascendcl.so libplatform.so libregister.so;do [[ -r "${ascend_home}/lib64/${x}" ]]||{ echo "[HARD_FAIL] missing ${x}" >&2;exit 2;};done
set +e;set +u;if [[ -f "${ascend_home}/bin/setenv.bash" ]];then source "${ascend_home}/bin/setenv.bash";else source "${ascend_home}/set_env.sh";fi;set -u;set -e
rm -rf "${logs}" "${work}/artifacts/port_grid";mkdir -p "${logs}" "${work}/artifacts/port_grid"
cmake -S "${source_root}/mimo/pusch_codebook_precode" -B "${build}" -DRUN_MODE=npu -DSOC_VERSION=Ascend310P1 \
 -DASCEND_CANN_PACKAGE_PATH="${ascend_home}" -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="${out}" >"${logs}/build.log" 2>&1
cmake --build "${build}" -j "$(nproc)" >>"${logs}/build.log" 2>&1;cmake --install "${build}" >>"${logs}/build.log" 2>&1
g++ -O2 -std=c++17 -D_GLIBCXX_USE_CXX11_ABI=0 -Wall -Wextra -Werror -I"${source_root}/mimo/pusch_codebook_precode" -I"${source_root}/common" \
 -I"${out}/include/ascendc_kernels_npu" -I"${ascend_home}/include" "${here}/precode_bypass_stage.cpp" "${source_root}/common/pusch_mimo_runtime_config.cpp" \
 "${source_root}/mimo/pusch_codebook_precode/pusch_codebook_precode.cpp" "${source_root}/mimo/pusch_codebook_precode/pusch_codebook_precode_runtime.cpp" \
 -L"${out}/lib" -lascendc_kernels_npu -L"${ascend_home}/lib64" -lascendcl -lplatform -lregister -ltiling_api -lascendalog -ldl \
 -o "${out}/bin/pusch_mimo_precode_bypass_stage" >>"${logs}/build.log" 2>&1
set +e;LD_LIBRARY_PATH="${out}/lib:${ascend_home}/lib64:${LD_LIBRARY_PATH:-}" "${out}/bin/pusch_mimo_precode_bypass_stage" \
 "${work}/artifacts/layer_grid_mapped" "${work}/artifacts/port_grid" 1 >"${logs}/device.log" 2>&1;status=$?;set -e
if [[ ${status} -ne 0 ]];then if grep -Eq '507008|drvRet=4' "${logs}/device.log";then echo "[DEVICE_BUSY] precode; not retried" >&2;exit 75;fi
 tail -80 "${logs}/build.log" >&2;tail -80 "${logs}/device.log" >&2;echo "[HARD_FAIL] precode exited ${status}" >&2;exit "${status}";fi
cp "${logs}/device.log" "${work}/artifacts/precode_result.log";cat "${logs}/device.log"
