#!/usr/bin/env bash
set -euo pipefail
rank="${1:-}"
[[ "${rank}" =~ ^[234]$ ]] || { echo "usage: $0 RANK(2|3|4)" >&2; exit 2; }
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root="${PUSCH_MIMO_CLEAN_KERNEL_ROOT:-/home/refresh/AI-RAN-NPU-clean/kernels}"
ascend_home="${ASCEND_HOME_PATH:-/usr/local/Ascend/ascend-toolkit/latest}"
work="${here}/work/tx_prefix_rank1"; logs="${here}/logs/tx_rank${rank}"
[[ -f "${work}/artifacts/tx_prefix_result.json" ]] || { echo '[HARD_FAIL] Rank1 base LDPC source absent' >&2; exit 2; }
set +e; set +u
if [[ -f "${ascend_home}/bin/setenv.bash" ]]; then source "${ascend_home}/bin/setenv.bash"; else source "${ascend_home}/set_env.sh"; fi
set -u; set -e
rm -rf "${logs}" "${work}/artifacts/layer_grid_rank${rank}"
mkdir -p "${logs}" "${work}/scramble/ascend_output/rank${rank}"
g++ -O2 -std=c++17 -D_GLIBCXX_USE_CXX11_ABI=0 -Wall -Wextra -Werror \
  -I"${root}/fec/scramble_mimo" -I"${here}/out/scramble_mimo/include/ascendc_kernels_npu" \
  -I"${ascend_home}/include" "${here}/scramble_stage.cpp" "${root}/fec/scramble_mimo/scramble_mimo.cpp" \
  -L"${here}/out/scramble_mimo/lib" -lascendc_kernels_npu -L"${ascend_home}/lib64" \
  -lascendcl -lplatform -lregister -ltiling_api -lascendalog -ldl \
  -o "${here}/out/scramble_mimo/bin/pusch_mimo_scramble_stage" >"${logs}/build.log" 2>&1
run_device(){ local tag="$1"; shift; set +e; "$@" >"${logs}/${tag}.log" 2>&1; local status=$?; set -e
  if [[ ${status} -ne 0 ]]; then if grep -Eq '507008|drvRet=4' "${logs}/${tag}.log"; then echo "[DEVICE_BUSY] ${tag}; not retried" >&2; exit 75; fi; tail -80 "${logs}/${tag}.log" >&2; exit "${status}"; fi; }
python3 "${here}/tx_prefix_rank1.py" link-rate --root "${work}" --rank "${rank}"
rm -f "${work}/rate/ascend_output/rank${rank}/bits_nr.bin"
run_device rate_match env AIRAN_DATA_DIR="${work}/rate" WARMUP=0 TIMED=1 \
  LD_LIBRARY_PATH="${here}/out/rate_match_mimo/lib:${ascend_home}/lib64:${LD_LIBRARY_PATH:-}" \
  "${here}/out/rate_match_mimo/bin/ascendc_kernels_bbit"
python3 "${here}/tx_prefix_rank1.py" link-scramble --root "${work}" --rank "${rank}"
rm -f "${work}/scramble/ascend_output/rank${rank}/bits_qam.bin"
run_device scramble env LD_LIBRARY_PATH="${here}/out/scramble_mimo/lib:${ascend_home}/lib64:${LD_LIBRARY_PATH:-}" \
  "${here}/out/scramble_mimo/bin/pusch_mimo_scramble_stage" \
  "${work}/rate/ascend_output/rank${rank}/bits_nr.bin" \
  "${work}/scramble/ascend_output/rank${rank}/bits_qam.bin" "${rank}"
for slot in $(seq 0 22); do tag="$(printf '%02d' "${slot}")"
  python3 "${here}/tx_prefix_rank1.py" select-qam --root "${work}" --slot "${slot}" --rank "${rank}"
  run_device "qam_${tag}" env AIRAN_DATA_DIR="${work}/qam" WARMUP=0 TIMED=1 \
    LD_LIBRARY_PATH="${here}/out/qam_mod_256_mimo/lib:${ascend_home}/lib64:${LD_LIBRARY_PATH:-}" \
    "${here}/out/qam_mod_256_mimo/bin/ascendc_kernels_bbit"
  python3 "${here}/tx_prefix_rank1.py" link-layer --root "${work}" --rank "${rank}"
  run_device "layer_${tag}" env AIRAN_DATA_DIR="${work}/layer" WARMUP=0 TIMED=1 \
    LD_LIBRARY_PATH="${here}/out/layer_map_mimo/lib:${ascend_home}/lib64:${LD_LIBRARY_PATH:-}" \
    "${here}/out/layer_map_mimo/bin/ascendc_kernels_bbit"
  python3 "${here}/tx_prefix_rank1.py" collect --root "${work}" --slot "${slot}" --rank "${rank}"
  echo "[slot ${tag}] Rank${rank} rate/QAM/layer PASS"
done
python3 - "${work}" "${rank}" <<'PY'
import hashlib,json,pathlib,sys
root=pathlib.Path(sys.argv[1]); rank=int(sys.argv[2]); bits=root/f'scramble/ascend_output/rank{rank}/bits_qam.bin'
expected=23*8*rank*19200*2
if not bits.is_file() or bits.stat().st_size!=expected or not any(bits.read_bytes()):raise SystemExit('[HARD_FAIL] scrambled output')
files=sorted((root/f'artifacts/layer_grid_rank{rank}').glob('slot*_*.bin'))
if len(files)!=46 or any(p.stat().st_size!=rank*19200*2 for p in files):raise SystemExit('[HARD_FAIL] layer outputs')
result={'schema':'airan.pusch_mimo.tx_prefix.v1','rank':rank,'slots':23,'source':'actual shared LDPC code blocks',
 'actual_npu_stages':['rate_match_mimo','scramble_mimo','qam_mod_256_mimo','layer_map_mimo'],
 'scrambled_bits_sha256':hashlib.sha256(bits.read_bytes()).hexdigest(),'status':'PASS'}
(root/f'artifacts/tx_prefix_rank{rank}_result.json').write_text(json.dumps(result,indent=2)+'\n')
PY
