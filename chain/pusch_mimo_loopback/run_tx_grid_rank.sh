#!/usr/bin/env bash
set -euo pipefail
rank="${1:-}"; [[ "${rank}" =~ ^[234]$ ]] || { echo "usage: $0 RANK(2|3|4)" >&2; exit 2; }
ports="${rank}"; [[ "${rank}" == 3 ]] && ports=4
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root="${PUSCH_MIMO_CLEAN_KERNEL_ROOT:-/home/refresh/AI-RAN-NPU-clean/kernels}"
ascend_home="${ASCEND_HOME_PATH:-/usr/local/Ascend/ascend-toolkit/latest}"
work="${here}/work/tx_prefix_rank1"; logs="${here}/logs/tx_grid_rank${rank}"
dmrs_out="${here}/out/mimo_dmrs_gen"; grid_out="${here}/out/mimo_resource_grid_map"; precode_out="${here}/out/pusch_codebook_precode"
[[ -f "${work}/artifacts/tx_prefix_rank${rank}_result.json" ]] || { echo "[HARD_FAIL] Rank${rank} prefix absent" >&2; exit 2; }
set +e; set +u
if [[ -f "${ascend_home}/bin/setenv.bash" ]]; then source "${ascend_home}/bin/setenv.bash"; else source "${ascend_home}/set_env.sh"; fi
set -u; set -e
rm -rf "${logs}" "${work}/artifacts/dmrs_rank${rank}" "${work}/artifacts/layer_grid_mapped_rank${rank}" "${work}/artifacts/port_grid_rank${rank}"
mkdir -p "${logs}" "${work}/artifacts/dmrs_rank${rank}" "${work}/artifacts/layer_grid_mapped_rank${rank}" "${work}/artifacts/port_grid_rank${rank}"
g++ -O2 -std=c++17 -D_GLIBCXX_USE_CXX11_ABI=0 -Wall -Wextra -Werror \
  -I"${root}/mimo/mimo_dmrs_gen" -I"${root}/common" -I"${dmrs_out}/include/ascendc_kernels_npu" -I"${ascend_home}/include" \
  "${here}/dmrs_stage.cpp" "${root}/common/pusch_mimo_runtime_config.cpp" "${root}/mimo/mimo_dmrs_gen/mimo_dmrs_gen.cpp" "${root}/mimo/mimo_dmrs_gen/mimo_dmrs_gen_runtime.cpp" \
  -L"${dmrs_out}/lib" -lascendc_kernels_npu -L"${ascend_home}/lib64" -lascendcl -lplatform -lregister -ltiling_api -lascendalog -ldl \
  -o "${dmrs_out}/bin/pusch_mimo_dmrs_stage" >"${logs}/build.log" 2>&1
g++ -O2 -std=c++17 -D_GLIBCXX_USE_CXX11_ABI=0 -Wall -Wextra -Werror \
  -I"${root}/mapping/mimo_resource_grid_map" -I"${root}/common" -I"${grid_out}/include/ascendc_kernels_npu" -I"${ascend_home}/include" \
  "${here}/grid_map_stage.cpp" "${root}/common/pusch_mimo_runtime_config.cpp" "${root}/mapping/mimo_resource_grid_map/mimo_resource_grid_map.cpp" \
  -L"${grid_out}/lib" -lascendc_kernels_npu -L"${ascend_home}/lib64" -lascendcl -lplatform -lregister -ltiling_api -lascendalog -ldl \
  -o "${grid_out}/bin/pusch_mimo_grid_map_stage" >>"${logs}/build.log" 2>&1
g++ -O2 -std=c++17 -D_GLIBCXX_USE_CXX11_ABI=0 -Wall -Wextra -Werror \
  -I"${root}/mimo/pusch_codebook_precode" -I"${root}/common" -I"${precode_out}/include/ascendc_kernels_npu" -I"${ascend_home}/include" \
  "${here}/precode_bypass_stage.cpp" "${root}/common/pusch_mimo_runtime_config.cpp" "${root}/mimo/pusch_codebook_precode/pusch_codebook_precode.cpp" \
  "${root}/mimo/pusch_codebook_precode/pusch_codebook_precode_runtime.cpp" \
  -L"${precode_out}/lib" -lascendc_kernels_npu -L"${ascend_home}/lib64" -lascendcl -lplatform -lregister -ltiling_api -lascendalog -ldl \
  -o "${precode_out}/bin/pusch_mimo_precode_bypass_stage" >>"${logs}/build.log" 2>&1
run_device(){ local tag="$1" lib="$2"; shift 2; set +e; env LD_LIBRARY_PATH="${lib}:${ascend_home}/lib64:${LD_LIBRARY_PATH:-}" "$@" >"${logs}/${tag}.log" 2>&1; local status=$?; set -e
 if [[ ${status} -ne 0 ]];then if grep -Eq '507008|drvRet=4' "${logs}/${tag}.log";then echo "[DEVICE_BUSY] ${tag}; not retried" >&2;exit 75;fi;tail -80 "${logs}/${tag}.log" >&2;exit "${status}";fi;}
run_device dmrs "${dmrs_out}/lib" "${dmrs_out}/bin/pusch_mimo_dmrs_stage" "${work}/artifacts/dmrs_rank${rank}" "${rank}"
run_device grid "${grid_out}/lib" "${grid_out}/bin/pusch_mimo_grid_map_stage" "${work}/artifacts/layer_grid_rank${rank}" "${work}/artifacts/dmrs_rank${rank}" "${work}/artifacts/layer_grid_mapped_rank${rank}" "${rank}"
run_device precode "${precode_out}/lib" "${precode_out}/bin/pusch_mimo_precode_bypass_stage" "${work}/artifacts/layer_grid_mapped_rank${rank}" "${work}/artifacts/port_grid_rank${rank}" "${rank}"
python3 - "${work}" "${rank}" "${ports}" <<'PY'
import json,pathlib,sys
root=pathlib.Path(sys.argv[1]);rank=int(sys.argv[2]);ports=int(sys.argv[3])
for directory,size in ((f'dmrs_rank{rank}',rank*2*896*2),(f'layer_grid_mapped_rank{rank}',rank*14*1664*2),(f'port_grid_rank{rank}',ports*14*1664*2)):
 files=sorted((root/'artifacts'/directory).glob('slot*_*.bin'))
 if len(files)!=46 or any(p.stat().st_size!=size or not any(p.read_bytes()) for p in files):raise SystemExit(f'[HARD_FAIL] {directory}')
result={'schema':'airan.pusch_mimo.tx_grid.v1','rank':rank,'tx_ports':ports,'slots':23,
 'precode':'P4_L3_TPMI6' if rank==3 else 'bypass','actual_npu_stages':['mimo_dmrs_gen','mimo_resource_grid_map','pusch_codebook_precode'],'status':'PASS'}
(root/f'artifacts/tx_grid_rank{rank}_result.json').write_text(json.dumps(result,indent=2)+'\n')
print(f'[PASS] Rank{rank} DMRS/grid/precode exact-size nonzero')
PY
