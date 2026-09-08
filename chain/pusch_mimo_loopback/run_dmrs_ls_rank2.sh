#!/usr/bin/env bash
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
rank="${1:-2}"
[[ "${rank}" =~ ^[234]$ ]] || { echo "usage: $0 [RANK(2|3|4)]" >&2; exit 2; }
source "${here}/build_variant.sh"
load_mimo_build_variant "${rank}"
source_root="${PUSCH_MIMO_CLEAN_KERNEL_ROOT:-/home/refresh/AI-RAN-NPU-clean/kernels}"
ascend_home="${ASCEND_HOME_PATH:-/usr/local/Ascend/ascend-toolkit/latest}"
build="${here}/build/mimo_dmrs_ls"
out="${here}/out/mimo_dmrs_ls"
work="${here}/work/tx_prefix_rank1"
logs="${here}/logs/dmrs_ls_rank${rank}"
result="${work}/artifacts/dmrs_ls_rank${rank}"
[[ -f "${work}/artifacts/coded_waveform_rank${rank}_result.json" ]] || {
  echo "[HARD_FAIL] Rank${rank} coded waveform must pass first" >&2; exit 2; }
for library in libascendcl.so libplatform.so libregister.so; do
  [[ -r "${ascend_home}/lib64/${library}" ]] || { echo "[HARD_FAIL] missing ${library}" >&2; exit 2; }
done
set +e; set +u
if [[ -f "${ascend_home}/bin/setenv.bash" ]]; then source "${ascend_home}/bin/setenv.bash"; else source "${ascend_home}/set_env.sh"; fi
set -u; set -e
rm -rf "${logs}" "${result}"; mkdir -p "${logs}" "${result}"
cmake -S "${source_root}/channel_est/mimo_dmrs_ls" -B "${build}" \
  -DRUN_MODE=npu -DSOC_VERSION=Ascend310P1 -DASCEND_CANN_PACKAGE_PATH="${ascend_home}" \
  -DDMRS_LS_NR="${mimo_detector_rx_capacity}" \
  -DDMRS_LS_BLOCK_DIM="${mimo_dmrs_ls_block_dim}" \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="${out}" >"${logs}/build.log" 2>&1
cmake --build "${build}" -j "$(nproc)" >>"${logs}/build.log" 2>&1
cmake --install "${build}" >>"${logs}/build.log" 2>&1
g++ -O2 -std=c++17 -D_GLIBCXX_USE_CXX11_ABI=0 -Wall -Wextra -Werror \
  -DMIMO_DMRS_LS_NR="${mimo_detector_rx_capacity}" \
  -DMIMO_DMRS_LS_BLOCK_DIM="${mimo_dmrs_ls_block_dim}" \
  -I"${source_root}/channel_est/mimo_dmrs_ls" -I"${source_root}/common" \
  -I"${out}/include/ascendc_kernels_npu" \
  -I"${ascend_home}/include" "${here}/dmrs_ls_stage.cpp" \
  "${source_root}/common/pusch_mimo_runtime_config.cpp" \
  "${source_root}/channel_est/mimo_dmrs_ls/mimo_dmrs_ls.cpp" \
  -L"${out}/lib" -lascendc_kernels_npu -L"${ascend_home}/lib64" \
  -lascendcl -lplatform -lregister -ltiling_api -lascendalog -ldl \
  -o "${out}/bin/pusch_mimo_dmrs_ls_stage" >>"${logs}/build.log" 2>&1
set +e
LD_LIBRARY_PATH="${out}/lib:${ascend_home}/lib64:${LD_LIBRARY_PATH:-}" \
  "${out}/bin/pusch_mimo_dmrs_ls_stage" \
  "${work}/artifacts/rx_grid_rank${rank}" "${work}/artifacts/dmrs_rank${rank}" "${result}" "${rank}" \
  >"${logs}/device.log" 2>&1
status=$?; set -e
if [[ ${status} -ne 0 ]]; then
  if grep -Eq '507008|drvRet=4' "${logs}/device.log"; then
    echo "[DEVICE_BUSY] Rank2 DMRS-LS; not retried" >&2; exit 75
  fi
  tail -100 "${logs}/build.log" >&2; tail -100 "${logs}/device.log" >&2
  echo "[HARD_FAIL] Rank2 DMRS-LS exited ${status}" >&2; exit "${status}"
fi
python3 - "${result}" "${rank}" "${mimo_detector_rx_capacity}" <<'PY'
import json, pathlib, sys
import numpy as np
root=pathlib.Path(sys.argv[1])
rank=int(sys.argv[2])
nr=int(sys.argv[3])
count=np.fromfile(root/'pilot_count.bin',np.uint16)
expected={2:[798,798,798,798],3:[399,399,399,399,798,798],4:[399]*8}[rank]
if count.size != 16 or count[:2*rank].tolist() != expected:
    raise SystemExit(f'[HARD_FAIL] Rank{rank} pilot counts {count.tolist()}')
for slot in range(23):
    for plane in ('re','im'):
        p=root/f'slot{slot:02d}_h_{plane}.bin'
        if not p.is_file() or p.stat().st_size != nr*rank*2*832*2:
            raise SystemExit(f'[HARD_FAIL] bad Rank{rank} LS output {p}')
models={2:['COMB2_798']*2,3:['FD_OCC2_399','FD_OCC2_399','COMB2_798'],4:['FD_OCC2_399']*4}[rank]
result={'schema':'airan.pusch_mimo.dmrs_ls.v1','rank':rank,'slots':23,
        'observation_models':models,'actual_npu_stage':'mimo_dmrs_ls','status':'PASS'}
(root/'result.json').write_text(json.dumps(result,indent=2)+'\n')
print(f'[PASS] Rank{rank} DMRS-LS outputs observation={models}')
PY
cat "${logs}/device.log"
