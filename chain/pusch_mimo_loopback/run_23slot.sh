#!/usr/bin/env bash
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ascend_home="${ASCEND_HOME_PATH:-/usr/local/Ascend/ascend-toolkit/latest}"
work="${here}/work/rank1"
out_root="${here}/out"
log_root="${here}/logs/rank1_23slot"

PUSCH_MIMO_BUILD_ONLY=1 bash "${here}/run.sh"

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

rm -rf "${work}" "${log_root}"
mkdir -p "${work}" "${log_root}"
python3 "${here}/rank1_23slot.py" prepare --root "${work}"
mkdir -p \
  "${work}/re_map/ascend_output" \
  "${work}/ofdm_mod/data/golden" "${work}/ofdm_mod/data/ascend_output" \
  "${work}/ofdm_demod/data/golden" "${work}/ofdm_demod/data/ascend_output" \
  "${work}/re_demap/data/golden" "${work}/re_demap/data/ascend_output"

run_device() {
  local tag="$1"
  shift
  set +e
  "$@" >"${log_root}/${tag}.log" 2>&1
  local status=$?
  set -e
  if [[ ${status} -ne 0 ]]; then
    if grep -Eq '507008|drvRet=4' "${log_root}/${tag}.log"; then
      echo "[DEVICE_BUSY] ${tag}: 507008/drvRet=4; not retried" >&2
      exit 75
    fi
    tail -80 "${log_root}/${tag}.log" >&2
    echo "[HARD_FAIL] ${tag} exited ${status}" >&2
    exit "${status}"
  fi
}

common_ld="${ascend_home}/lib64:${LD_LIBRARY_PATH:-}"
for slot in $(seq 0 22); do
  python3 "${here}/rank1_23slot.py" select-tx --root "${work}" --slot "${slot}"
  run_device "re_map_slot$(printf '%02d' "${slot}")" env \
    AIRAN_DATA_DIR="${work}/re_map" WARMUP=0 TIMED=1 \
    LD_LIBRARY_PATH="${out_root}/re_map_batch/lib:${common_ld}" \
    "${out_root}/re_map_batch/bin/ascendc_kernels_bbit"
  python3 "${here}/rank1_23slot.py" collect-tx --root "${work}" --slot "${slot}"
done

python3 "${here}/rank1_23slot.py" assemble-tx --root "${work}"
run_device ofdm_mod_23slot env \
  AIRAN_DATA_DIR="${work}/ofdm_mod" OFDM_BATCH_SIZE=23 \
  LD_LIBRARY_PATH="${out_root}/ofdm_mod_batch/lib:${common_ld}" \
  "${out_root}/ofdm_mod_batch/bin/ascendc_kernels_bbit"
python3 "${here}/rank1_23slot.py" channel --root "${work}"

for slot in $(seq 0 22); do
  python3 "${here}/rank1_23slot.py" select-rx --root "${work}" --slot "${slot}"
  run_device "ofdm_demod_slot$(printf '%02d' "${slot}")" env \
    AIRAN_DATA_DIR="${work}/ofdm_demod" OFDM_BATCH_SIZE=64 \
    LD_LIBRARY_PATH="${out_root}/ofdm_demod_batch/lib:${common_ld}" \
    "${out_root}/ofdm_demod_batch/bin/ascendc_kernels_bbit"
  cp "${work}/ofdm_demod/data/ascend_output/output_re.bin" \
     "${work}/re_demap/data/golden/input_re.bin"
  cp "${work}/ofdm_demod/data/ascend_output/output_im.bin" \
     "${work}/re_demap/data/golden/input_im.bin"
  run_device "re_demap_slot$(printf '%02d' "${slot}")" env \
    AIRAN_DATA_DIR="${work}/re_demap" RE_DEMAP_BATCH_SIZE=64 \
    LD_LIBRARY_PATH="${out_root}/re_demap_batch/lib:${common_ld}" \
    "${out_root}/re_demap_batch/bin/ascendc_kernels_bbit"
  python3 "${here}/rank1_23slot.py" collect-rx --root "${work}" --slot "${slot}"
done

python3 "${here}/rank1_23slot.py" verify --root "${work}" | tee "${log_root}/verify.log"
echo "[SUBCHAIN PASS] Rank1 23-slot real device OFDM subchain"
echo "[INFO] run ./run.sh 1 for the complete coded acceptance path"
