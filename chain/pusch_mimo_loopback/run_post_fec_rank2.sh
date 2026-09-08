#!/usr/bin/env bash
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
rank="${1:-2}"
[[ "${rank}" =~ ^[234]$ ]] || { echo "usage: $0 [RANK(2|3|4)]" >&2; exit 2; }
root="${PUSCH_MIMO_CLEAN_KERNEL_ROOT:-/home/refresh/AI-RAN-NPU-clean/kernels}"
ascend_home="${ASCEND_HOME_PATH:-/usr/local/Ascend/ascend-toolkit/latest}"
conda_bin="${PUSCH_MIMO_CONDA_BIN:-/home/refresh/miniconda3/bin/conda}"
work="${here}/work/tx_prefix_rank1"
logs="${here}/logs/post_fec_rank${rank}"
post="${work}/artifacts/post_fec_rank${rank}"
decoder="${work}/decoder_rank${rank}"
[[ -f "${work}/artifacts/demod_rank${rank}_23slot/result.json" ]] || {
  echo "[HARD_FAIL] Rank${rank} 23-slot RX front result absent" >&2; exit 2; }
for library in libascendcl.so libplatform.so libregister.so; do
  [[ -r "${ascend_home}/lib64/${library}" ]] || { echo "[HARD_FAIL] missing ${library}" >&2; exit 2; }
done
[[ -x "${conda_bin}" ]] || { echo "[HARD_FAIL] conda missing" >&2; exit 2; }
set +e; set +u
if [[ -f "${ascend_home}/bin/setenv.bash" ]]; then source "${ascend_home}/bin/setenv.bash"; else source "${ascend_home}/set_env.sh"; fi
set -u; set -e
rm -rf "${logs}" "${post}" "${decoder}"
mkdir -p "${logs}" "${post}"
build() {
  local tag="$1" rel="$2"
  cmake -S "${root}/${rel}" -B "${here}/build/${tag}" -DRUN_MODE=npu \
    -DSOC_VERSION=Ascend310P1 -DASCEND_CANN_PACKAGE_PATH="${ascend_home}" \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="${here}/out/${tag}" \
    >"${logs}/build_${tag}.log" 2>&1
  cmake --build "${here}/build/${tag}" -j "$(nproc)" >>"${logs}/build_${tag}.log" 2>&1
  cmake --install "${here}/build/${tag}" >>"${logs}/build_${tag}.log" 2>&1
}
build descramble_mimo fec/descramble_mimo
build rate_dematch_mimo fec/rate_dematch_mimo
build ldpc_decode fec/ldpc_decode
g++ -O2 -std=c++17 -D_GLIBCXX_USE_CXX11_ABI=0 -Wall -Wextra -Werror \
  -I"${root}/fec/descramble_mimo" -I"${here}/out/descramble_mimo/include/ascendc_kernels_npu" \
  -I"${ascend_home}/include" "${here}/descramble_stage.cpp" "${root}/fec/descramble_mimo/descramble_mimo.cpp" \
  -L"${here}/out/descramble_mimo/lib" -lascendc_kernels_npu -L"${ascend_home}/lib64" \
  -lascendcl -lplatform -lregister -ltiling_api -lascendalog -ldl \
  -o "${here}/out/descramble_mimo/bin/pusch_mimo_descramble_stage" >>"${logs}/build_descramble_mimo.log" 2>&1
g++ -O2 -std=c++17 -D_GLIBCXX_USE_CXX11_ABI=0 -Wall -Wextra -Werror \
  -I"${root}/fec/rate_dematch_mimo" -I"${here}/out/rate_dematch_mimo/include/ascendc_kernels_npu" \
  -I"${ascend_home}/include" "${here}/rate_dematch_stage.cpp" "${root}/fec/rate_dematch_mimo/rate_dematch_mimo.cpp" \
  -L"${here}/out/rate_dematch_mimo/lib" -lascendc_kernels_npu -L"${ascend_home}/lib64" \
  -lascendcl -lplatform -lregister -ltiling_api -lascendalog -ldl \
  -o "${here}/out/rate_dematch_mimo/bin/pusch_mimo_rate_dematch_stage" >>"${logs}/build_rate_dematch_mimo.log" 2>&1
g++ -O2 -std=c++17 -D_GLIBCXX_USE_CXX11_ABI=0 -Wall -Wextra -Werror \
  -I"${root}/fec/ldpc_decode" -I"${here}/out/ldpc_decode/include/ascendc_kernels_npu" \
  -I"${ascend_home}/include" "${here}/ldpc_decode_stage.cpp" \
  -L"${here}/out/ldpc_decode/lib" -lascendc_kernels_npu -L"${ascend_home}/lib64" \
  -lascendcl -lplatform -lregister -ltiling_api -lascendalog -ldl \
  -o "${here}/out/ldpc_decode/bin/pusch_mimo_ldpc_decode_stage" >>"${logs}/build_ldpc_decode.log" 2>&1
run_device() {
  local tag="$1"; shift
  set +e; "$@" >"${logs}/${tag}.log" 2>&1; local status=$?; set -e
  if [[ ${status} -ne 0 ]]; then
    if grep -Eq '507008|drvRet=4' "${logs}/${tag}.log"; then
      echo "[DEVICE_BUSY] ${tag}; not retried" >&2; exit 75
    fi
    tail -100 "${logs}/${tag}.log" >&2; echo "[HARD_FAIL] ${tag} exited ${status}" >&2; exit "${status}"
  fi
  cat "${logs}/${tag}.log"
}
cw="${work}/artifacts/demod_rank${rank}_23slot/cw_llr_23slot.bin"
for pair in matched:321 wrong:322; do
  name="${pair%%:*}"; cell="${pair##*:}"
  rm -f "${post}/llr_nr_${name}.bin" "${post}/ldpc_llr_${name}.bin"
  run_device "descramble_${name}" env \
    LD_LIBRARY_PATH="${here}/out/descramble_mimo/lib:${ascend_home}/lib64:${LD_LIBRARY_PATH:-}" \
    "${here}/out/descramble_mimo/bin/pusch_mimo_descramble_stage" \
    "${cw}" "${post}/llr_nr_${name}.bin" "${cell}" "${rank}"
  run_device "rate_${name}" env \
    LD_LIBRARY_PATH="${here}/out/rate_dematch_mimo/lib:${ascend_home}/lib64:${LD_LIBRARY_PATH:-}" \
    "${here}/out/rate_dematch_mimo/bin/pusch_mimo_rate_dematch_stage" \
    "${post}/llr_nr_${name}.bin" "${post}/ldpc_llr_${name}.bin" "${rank}"
done
"${conda_bin}" run -n sionna python "${here}/prepare_decoder_standard.py" --work "${work}" \
  >"${logs}/decoder_standard.log" 2>&1
for name in matched wrong; do
  data="${decoder}/data_${name}"; output="${decoder}/output_${name}"
  rm -rf "${data}" "${output}"; mkdir -p "${data}" "${output}"
  cp "${post}/ldpc_llr_${name}.bin" "${data}/lam_in.bin"
  cp "${work}/decoder/data/degrees.bin" "${work}/decoder/data/edge_offsets.bin" "${data}/"
  rm -f "${output}/decoded_bits.bin"
  run_device "decode_${name}" env \
    LD_LIBRARY_PATH="${here}/out/ldpc_decode/lib:${ascend_home}/lib64:${LD_LIBRARY_PATH:-}" \
    "${here}/out/ldpc_decode/bin/pusch_mimo_ldpc_decode_stage" "${data}" \
    "${work}/decoder/weights/ldpc_bg1_z384_shifts" "${output}/decoded_bits.bin"
  [[ -f "${output}/decoded_bits.bin" && $(stat -c %s "${output}/decoded_bits.bin") -eq 1208064 ]] || {
    echo "[HARD_FAIL] Rank${rank} decoder ${name} output exact size mismatch" >&2; exit 1; }
done
python3 "${here}/verify_rank_triangle.py" --root "${work}" --rank "${rank}" | tee "${logs}/triangle.log"
