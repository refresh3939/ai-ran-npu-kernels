#!/usr/bin/env bash
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source_root="${PUSCH_MIMO_CLEAN_KERNEL_ROOT:-/home/refresh/AI-RAN-NPU-clean/kernels}"
ascend_home="${ASCEND_HOME_PATH:-/usr/local/Ascend/ascend-toolkit/latest}"
work="${here}/work/tx_prefix_rank1"
logs="${here}/logs/tx_grid_rank2"
dmrs_build="${here}/build/mimo_dmrs_gen"
dmrs_out="${here}/out/mimo_dmrs_gen"
grid_build="${here}/build/mimo_resource_grid_map"
grid_out="${here}/out/mimo_resource_grid_map"
precode_build="${here}/build/pusch_codebook_precode"
precode_out="${here}/out/pusch_codebook_precode"

[[ -f "${work}/artifacts/tx_prefix_rank2_result.json" ]] || {
  echo "[HARD_FAIL] run_tx_rank2.sh must pass first" >&2
  exit 2
}
for library in libascendcl.so libplatform.so libregister.so; do
  [[ -r "${ascend_home}/lib64/${library}" ]] || {
    echo "[HARD_FAIL] missing CANN library ${library}" >&2
    exit 2
  }
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

rm -rf "${logs}" \
  "${work}/artifacts/dmrs_rank2" \
  "${work}/artifacts/layer_grid_mapped_rank2" \
  "${work}/artifacts/port_grid_rank2"
mkdir -p "${logs}" \
  "${work}/artifacts/dmrs_rank2" \
  "${work}/artifacts/layer_grid_mapped_rank2" \
  "${work}/artifacts/port_grid_rank2"

cmake -S "${source_root}/mimo/mimo_dmrs_gen" -B "${dmrs_build}" \
  -DRUN_MODE=npu -DSOC_VERSION=Ascend310P1 \
  -DASCEND_CANN_PACKAGE_PATH="${ascend_home}" -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="${dmrs_out}" >"${logs}/build_dmrs.log" 2>&1
cmake --build "${dmrs_build}" -j "$(nproc)" >>"${logs}/build_dmrs.log" 2>&1
cmake --install "${dmrs_build}" >>"${logs}/build_dmrs.log" 2>&1
g++ -O2 -std=c++17 -D_GLIBCXX_USE_CXX11_ABI=0 -Wall -Wextra -Werror \
  -I"${source_root}/mimo/mimo_dmrs_gen" -I"${source_root}/common" \
  -I"${dmrs_out}/include/ascendc_kernels_npu" -I"${ascend_home}/include" \
  "${here}/dmrs_stage.cpp" \
  "${source_root}/common/pusch_mimo_runtime_config.cpp" \
  "${source_root}/mimo/mimo_dmrs_gen/mimo_dmrs_gen.cpp" \
  "${source_root}/mimo/mimo_dmrs_gen/mimo_dmrs_gen_runtime.cpp" \
  -L"${dmrs_out}/lib" -lascendc_kernels_npu -L"${ascend_home}/lib64" \
  -lascendcl -lplatform -lregister -ltiling_api -lascendalog -ldl \
  -o "${dmrs_out}/bin/pusch_mimo_dmrs_stage" >>"${logs}/build_dmrs.log" 2>&1

cmake -S "${source_root}/mapping/mimo_resource_grid_map" -B "${grid_build}" \
  -DRUN_MODE=npu -DSOC_VERSION=Ascend310P1 \
  -DASCEND_CANN_PACKAGE_PATH="${ascend_home}" -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="${grid_out}" >"${logs}/build_grid.log" 2>&1
cmake --build "${grid_build}" -j "$(nproc)" >>"${logs}/build_grid.log" 2>&1
cmake --install "${grid_build}" >>"${logs}/build_grid.log" 2>&1
g++ -O2 -std=c++17 -D_GLIBCXX_USE_CXX11_ABI=0 -Wall -Wextra -Werror \
  -I"${source_root}/mapping/mimo_resource_grid_map" -I"${source_root}/common" \
  -I"${grid_out}/include/ascendc_kernels_npu" -I"${ascend_home}/include" \
  "${here}/grid_map_stage.cpp" \
  "${source_root}/common/pusch_mimo_runtime_config.cpp" \
  "${source_root}/mapping/mimo_resource_grid_map/mimo_resource_grid_map.cpp" \
  -L"${grid_out}/lib" -lascendc_kernels_npu -L"${ascend_home}/lib64" \
  -lascendcl -lplatform -lregister -ltiling_api -lascendalog -ldl \
  -o "${grid_out}/bin/pusch_mimo_grid_map_stage" >>"${logs}/build_grid.log" 2>&1

cmake -S "${source_root}/mimo/pusch_codebook_precode" -B "${precode_build}" \
  -DRUN_MODE=npu -DSOC_VERSION=Ascend310P1 \
  -DASCEND_CANN_PACKAGE_PATH="${ascend_home}" -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="${precode_out}" >"${logs}/build_precode.log" 2>&1
cmake --build "${precode_build}" -j "$(nproc)" >>"${logs}/build_precode.log" 2>&1
cmake --install "${precode_build}" >>"${logs}/build_precode.log" 2>&1
g++ -O2 -std=c++17 -D_GLIBCXX_USE_CXX11_ABI=0 -Wall -Wextra -Werror \
  -I"${source_root}/mimo/pusch_codebook_precode" -I"${source_root}/common" \
  -I"${precode_out}/include/ascendc_kernels_npu" -I"${ascend_home}/include" \
  "${here}/precode_bypass_stage.cpp" \
  "${source_root}/common/pusch_mimo_runtime_config.cpp" \
  "${source_root}/mimo/pusch_codebook_precode/pusch_codebook_precode.cpp" \
  "${source_root}/mimo/pusch_codebook_precode/pusch_codebook_precode_runtime.cpp" \
  -L"${precode_out}/lib" -lascendc_kernels_npu -L"${ascend_home}/lib64" \
  -lascendcl -lplatform -lregister -ltiling_api -lascendalog -ldl \
  -o "${precode_out}/bin/pusch_mimo_precode_bypass_stage" >>"${logs}/build_precode.log" 2>&1

run_device() {
  local label="$1"
  local log="$2"
  local op_lib="$3"
  shift 3
  set +e
  LD_LIBRARY_PATH="${op_lib}:${ascend_home}/lib64:${LD_LIBRARY_PATH:-}" \
    "$@" >"${log}" 2>&1
  local status=$?
  set -e
  if [[ ${status} -ne 0 ]]; then
    if grep -Eq '507008|drvRet=4' "${log}"; then
      echo "[DEVICE_BUSY] ${label}: 507008/drvRet=4; not retried" >&2
      exit 75
    fi
    tail -80 "${log}" >&2
    echo "[HARD_FAIL] ${label} exited ${status}" >&2
    exit "${status}"
  fi
}

run_device dmrs "${logs}/dmrs.log" "${dmrs_out}/lib" \
  "${dmrs_out}/bin/pusch_mimo_dmrs_stage" \
  "${work}/artifacts/dmrs_rank2" 2
run_device grid_map "${logs}/grid.log" "${grid_out}/lib" \
  "${grid_out}/bin/pusch_mimo_grid_map_stage" \
  "${work}/artifacts/layer_grid_rank2" \
  "${work}/artifacts/dmrs_rank2" \
  "${work}/artifacts/layer_grid_mapped_rank2" 2
run_device precode_bypass "${logs}/precode.log" "${precode_out}/lib" \
  "${precode_out}/bin/pusch_mimo_precode_bypass_stage" \
  "${work}/artifacts/layer_grid_mapped_rank2" \
  "${work}/artifacts/port_grid_rank2" 2

python3 - "${work}" <<'PY'
import hashlib
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1]) / "artifacts"
expected = {
    "dmrs_rank2": 2 * 2 * 896 * 2,
    "layer_grid_mapped_rank2": 2 * 14 * 1664 * 2,
    "port_grid_rank2": 2 * 14 * 1664 * 2,
}
hashes = {}
for directory, size in expected.items():
    for slot in range(23):
        for component in ("re", "im"):
            path = root / directory / f"slot{slot:02d}_{component}.bin"
            if not path.is_file() or path.stat().st_size != size:
                raise SystemExit(f"[HARD_FAIL] {path} expected exactly {size} bytes")
            data = path.read_bytes()
            if not any(data):
                raise SystemExit(f"[HARD_FAIL] {path} is all-zero")
            hashes[str(path.relative_to(root))] = hashlib.sha256(data).hexdigest()
result = {
    "schema": "airan.pusch_mimo.rank2.tx_grid.v1",
    "rank": 2,
    "slots": 23,
    "actual_npu_stages": ["mimo_dmrs_gen", "mimo_resource_grid_map", "pusch_codebook_precode_bypass"],
    "ports": [1000, 1002],
    "files_checked": len(hashes),
    "sha256": hashes,
    "status": "PASS",
}
(root / "tx_grid_rank2_result.json").write_text(json.dumps(result, indent=2) + "\n")
print("[PASS] Rank2 DMRS/grid/bypass artifacts exact-size, nonzero, hashed")
PY

echo "[MILESTONE PASS] Rank2 actual DMRS -> grid map -> bypass, 23 slots"
