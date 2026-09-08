#!/usr/bin/env bash
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")"&&pwd)";root="${PUSCH_MIMO_CLEAN_KERNEL_ROOT:-/home/refresh/AI-RAN-NPU-clean/kernels}";ascend_home="${ASCEND_HOME_PATH:-/usr/local/Ascend/ascend-toolkit/latest}";work="${here}/work/tx_prefix_rank1";logs="${here}/logs/tx_rank2";rm -rf "$logs" "${work}/artifacts/layer_grid_rank2";mkdir -p "$logs" "${work}/scramble/ascend_output/rank2"
set +e;set +u;if [[ -f "${ascend_home}/bin/setenv.bash" ]];then source "${ascend_home}/bin/setenv.bash";else source "${ascend_home}/set_env.sh";fi;set -u;set -e
g++ -O2 -std=c++17 -D_GLIBCXX_USE_CXX11_ABI=0 -Wall -Wextra -Werror -I"${root}/fec/scramble_mimo" -I"${here}/out/scramble_mimo/include/ascendc_kernels_npu" -I"${ascend_home}/include" "${here}/scramble_stage.cpp" "${root}/fec/scramble_mimo/scramble_mimo.cpp" -L"${here}/out/scramble_mimo/lib" -lascendc_kernels_npu -L"${ascend_home}/lib64" -lascendcl -lplatform -lregister -ltiling_api -lascendalog -ldl -o "${here}/out/scramble_mimo/bin/pusch_mimo_scramble_stage" >"${logs}/build.log" 2>&1
run(){ local tag="$1";shift;set +e;"$@" >"${logs}/${tag}.log" 2>&1;s=$?;set -e;if [[ $s -ne 0 ]];then if grep -Eq '507008|drvRet=4' "${logs}/${tag}.log";then echo "[DEVICE_BUSY] $tag; not retried" >&2;exit 75;fi;tail -80 "${logs}/${tag}.log" >&2;exit "$s";fi;}
[[ -x "${here}/out/rate_match_mimo/bin/ascendc_kernels_bbit" ]] || { echo '[HARD_FAIL] rate_match_mimo binary absent' >&2; exit 2; }
python3 "${here}/tx_prefix_rank1.py" link-rate --root "$work" --rank 2
rm -f "${work}/rate/ascend_output/rank2/bits_nr.bin"
run rate_match env AIRAN_DATA_DIR="${work}/rate" WARMUP=0 TIMED=1 \
  LD_LIBRARY_PATH="${here}/out/rate_match_mimo/lib:${ascend_home}/lib64:${LD_LIBRARY_PATH:-}" \
  "${here}/out/rate_match_mimo/bin/ascendc_kernels_bbit"
[[ -f "${work}/rate/ascend_output/rank2/bits_nr.bin" && \
   $(stat -c %s "${work}/rate/ascend_output/rank2/bits_nr.bin") -eq 14131200 ]] || {
  echo '[HARD_FAIL] actual Rank2 rate-match output missing/exact-size mismatch' >&2; exit 1; }
run scramble env LD_LIBRARY_PATH="${here}/out/scramble_mimo/lib:${ascend_home}/lib64:${LD_LIBRARY_PATH:-}" "${here}/out/scramble_mimo/bin/pusch_mimo_scramble_stage" "${work}/rate/ascend_output/rank2/bits_nr.bin" "${work}/scramble/ascend_output/rank2/bits_qam.bin" 2
for slot in $(seq 0 22);do t=$(printf '%02d' "$slot");python3 "${here}/tx_prefix_rank1.py" select-qam --root "$work" --slot "$slot" --rank 2;run "qam_${t}" env AIRAN_DATA_DIR="${work}/qam" WARMUP=0 TIMED=1 LD_LIBRARY_PATH="${here}/out/qam_mod_256_mimo/lib:${ascend_home}/lib64:${LD_LIBRARY_PATH:-}" "${here}/out/qam_mod_256_mimo/bin/ascendc_kernels_bbit";python3 "${here}/tx_prefix_rank1.py" link-layer --root "$work" --rank 2;run "layer_${t}" env AIRAN_DATA_DIR="${work}/layer" WARMUP=0 TIMED=1 LD_LIBRARY_PATH="${here}/out/layer_map_mimo/lib:${ascend_home}/lib64:${LD_LIBRARY_PATH:-}" "${here}/out/layer_map_mimo/bin/ascendc_kernels_bbit";python3 "${here}/tx_prefix_rank1.py" collect --root "$work" --slot "$slot" --rank 2;echo "[slot $t] Rank2 QAM/layer PASS";done
python3 - "${work}" <<'PY'
import hashlib
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
bits = root / "scramble/ascend_output/rank2/bits_qam.bin"
if not bits.is_file() or bits.stat().st_size != 14_131_200:
    raise SystemExit("[HARD_FAIL] Rank2 scrambled uint16 bits must be exactly 14131200 bytes")
if not any(bits.read_bytes()):
    raise SystemExit("[HARD_FAIL] Rank2 scrambled bits are all-zero")
layer = root / "artifacts/layer_grid_rank2"
files = sorted(layer.glob("slot*_*.bin"))
if len(files) != 46 or any(p.stat().st_size != 76_800 for p in files):
    raise SystemExit("[HARD_FAIL] Rank2 layer artifacts require 46 exact 76800-byte files")
result = {
    "schema": "airan.pusch_mimo.rank2.tx_prefix.v1",
    "rank": 2,
    "slots": 23,
    "actual_npu_stages": ["rate_match_mimo", "scramble_mimo", "qam_mod_256_mimo", "layer_map_mimo"],
    "scrambled_bits_sha256": hashlib.sha256(bits.read_bytes()).hexdigest(),
    "status": "PASS",
}
(root / "artifacts/tx_prefix_rank2_result.json").write_text(json.dumps(result, indent=2) + "\n")
PY
echo '[MILESTONE PASS] Rank2 actual rate output -> scramble -> QAM -> layer map, 23 slots'
