#!/usr/bin/env bash
set -euo pipefail

CURRENT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "${CURRENT_DIR}"
export ASCEND_HOME_DIR=${ASCEND_HOME_DIR:-/usr/local/Ascend/ascend-toolkit/latest}
set +eu
if [[ -f "${ASCEND_HOME_DIR}/bin/setenv.bash" ]]; then
    source "${ASCEND_HOME_DIR}/bin/setenv.bash"
else
    source "${ASCEND_HOME_DIR}/../set_env.sh"
fi
set -eu

export AIRAN_DATA_DIR=${AIRAN_DATA_DIR:-${CURRENT_DIR}/data}
SOC_VERSION=${SOC_VERSION:-Ascend310P1}
DMRS_LS_NR=${DMRS_LS_NR:-64}
[[ "${DMRS_LS_NR}" =~ ^(16|32|64)$ ]] || {
    echo "[HARD_FAIL] DMRS_LS_NR must be 16, 32, or 64" >&2; exit 2; }
DMRS_LS_BLOCK_DIM=${DMRS_LS_BLOCK_DIM:-$((DMRS_LS_NR / 16))}
[[ "${DMRS_LS_BLOCK_DIM}" -eq $((DMRS_LS_NR / 16)) ]] || {
    echo "[HARD_FAIL] DMRS_LS_BLOCK_DIM must equal DMRS_LS_NR/16" >&2; exit 2; }
DMRS_LS_NR="${DMRS_LS_NR}" python3 "${CURRENT_DIR}/scripts/gen_data.py"
mkdir -p "${AIRAN_DATA_DIR}/ascend_output"

rm -rf "${CURRENT_DIR}/build" "${CURRENT_DIR}/out"
cmake -S "${CURRENT_DIR}" -B "${CURRENT_DIR}/build" \
    -DRUN_MODE=npu -DSOC_VERSION="${SOC_VERSION}" \
    -DDMRS_LS_NR="${DMRS_LS_NR}" \
    -DDMRS_LS_BLOCK_DIM="${DMRS_LS_BLOCK_DIM}" \
    -DASCEND_CANN_PACKAGE_PATH="${ASCEND_HOME_DIR}" \
    -DCMAKE_INSTALL_PREFIX="${CURRENT_DIR}/out"
cmake --build "${CURRENT_DIR}/build" -j"$(nproc)"
"${CURRENT_DIR}/build/mimo_dmrs_ls_contract_test"
cmake --install "${CURRENT_DIR}/build"
export LD_LIBRARY_PATH="${CURRENT_DIR}/out/lib:${ASCEND_HOME_DIR}/lib64:${LD_LIBRARY_PATH:-}"
"${CURRENT_DIR}/out/bin/ascendc_kernels_bbit"
DMRS_LS_NR="${DMRS_LS_NR}" python3 "${CURRENT_DIR}/scripts/verify_result.py"
