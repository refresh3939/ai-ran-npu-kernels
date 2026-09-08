#!/usr/bin/env bash
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source_root="${PUSCH_MIMO_CLEAN_KERNEL_ROOT:-/home/refresh/AI-RAN-NPU-clean/kernels}"
ascend_home="${ASCEND_HOME_PATH:-/usr/local/Ascend/ascend-toolkit/latest}"
build="${here}/build/mimo_dmrs_gen"
out="${here}/out/mimo_dmrs_gen"
work="${here}/work/tx_prefix_rank1"
logs="${here}/logs/dmrs_rank1"

[[ -f "${work}/artifacts/tx_prefix_result.json" ]] || {
  echo "[HARD_FAIL] run_tx_prefix_rank1.sh must pass first" >&2; exit 2; }
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

rm -rf "${logs}" "${work}/artifacts/dmrs"
mkdir -p "${logs}" "${work}/artifacts/dmrs"
cmake -S "${source_root}/mimo/mimo_dmrs_gen" -B "${build}" \
  -DRUN_MODE=npu -DSOC_VERSION=Ascend310P1 \
  -DASCEND_CANN_PACKAGE_PATH="${ascend_home}" -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="${out}" >"${logs}/build.log" 2>&1
cmake --build "${build}" -j "$(nproc)" >>"${logs}/build.log" 2>&1
cmake --install "${build}" >>"${logs}/build.log" 2>&1

g++ -O2 -std=c++17 -D_GLIBCXX_USE_CXX11_ABI=0 -Wall -Wextra -Werror \
  -I"${source_root}/mimo/mimo_dmrs_gen" -I"${source_root}/common" \
  -I"${out}/include/ascendc_kernels_npu" -I"${ascend_home}/include" \
  "${here}/dmrs_stage.cpp" \
  "${source_root}/common/pusch_mimo_runtime_config.cpp" \
  "${source_root}/mimo/mimo_dmrs_gen/mimo_dmrs_gen.cpp" \
  "${source_root}/mimo/mimo_dmrs_gen/mimo_dmrs_gen_runtime.cpp" \
  -L"${out}/lib" -lascendc_kernels_npu -L"${ascend_home}/lib64" \
  -lascendcl -lplatform -lregister -ltiling_api -lascendalog -ldl \
  -o "${out}/bin/pusch_mimo_dmrs_stage" >>"${logs}/build.log" 2>&1

set +e
LD_LIBRARY_PATH="${out}/lib:${ascend_home}/lib64:${LD_LIBRARY_PATH:-}" \
  "${out}/bin/pusch_mimo_dmrs_stage" "${work}/artifacts/dmrs" 1 \
  >"${logs}/device.log" 2>&1
status=$?
set -e
if [[ ${status} -ne 0 ]]; then
  if grep -Eq '507008|drvRet=4' "${logs}/device.log"; then
    echo "[DEVICE_BUSY] mimo_dmrs_gen: 507008/drvRet=4; not retried" >&2
    exit 75
  fi
  tail -80 "${logs}/build.log" >&2
  tail -80 "${logs}/device.log" >&2
  echo "[HARD_FAIL] Rank1 DMRS stage exited ${status}" >&2
  exit "${status}"
fi
python3 "${here}/verify_dmrs_rank1.py" --root "${work}/artifacts/dmrs" \
  --result "${work}/artifacts/dmrs_result.json" | tee "${logs}/verify.log"
echo "[MILESTONE PASS] Rank1 23-slot real device DMRS generation"
