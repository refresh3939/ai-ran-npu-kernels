#!/usr/bin/env bash
set -euo pipefail

CURRENT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "${CURRENT_DIR}"

cleanup() {
    if [[ "${KEEP_ARTIFACTS:-0}" != "1" ]]; then
        rm -rf "${CURRENT_DIR}/build" "${CURRENT_DIR}/out" \
               "${CURRENT_DIR}/data" "${CURRENT_DIR}/scripts/__pycache__"
    fi
}
trap cleanup EXIT
cleanup

RUN_MODE=${RUN_MODE:-npu}
SOC_VERSION=${SOC_VERSION:-Ascend310P1}
ASCEND_HOME_DIR=${ASCEND_HOME_DIR:-/usr/local/Ascend/ascend-toolkit/latest}
if [[ ! -d "${ASCEND_HOME_DIR}" ]]; then
    echo "[error] ASCEND_HOME_DIR=${ASCEND_HOME_DIR} does not exist"
    exit 1
fi
set +eu
if [[ -f "${ASCEND_HOME_DIR}/bin/setenv.bash" ]]; then
    source "${ASCEND_HOME_DIR}/bin/setenv.bash"
else
    source "${ASCEND_HOME_DIR}/set_env.sh"
fi
set -eu

export AIRAN_DATA_DIR=${AIRAN_DATA_DIR:-${CURRENT_DIR}/data}
export ASCEND_GLOBAL_LOG_LEVEL=${ASCEND_GLOBAL_LOG_LEVEL:-3}
python3 "${CURRENT_DIR}/scripts/gen_data.py"
mkdir -p "${AIRAN_DATA_DIR}/ascend_output"

cmake -S "${CURRENT_DIR}" -B "${CURRENT_DIR}/build" \
    -DRUN_MODE="${RUN_MODE}" -DSOC_VERSION="${SOC_VERSION}" \
    -DASCEND_CANN_PACKAGE_PATH="${ASCEND_HOME_DIR}" \
    -DCMAKE_INSTALL_PREFIX="${CURRENT_DIR}/out"
cmake --build "${CURRENT_DIR}/build" -j"$(nproc)"
cmake --install "${CURRENT_DIR}/build"
"${CURRENT_DIR}/out/bin/qam_mod_256_mimo_contract_test"

export LD_LIBRARY_PATH="${CURRENT_DIR}/out/lib:${ASCEND_HOME_DIR}/lib64:${LD_LIBRARY_PATH:-}"
"${CURRENT_DIR}/out/bin/ascendc_kernels_bbit"
python3 "${CURRENT_DIR}/scripts/verify_result.py"
