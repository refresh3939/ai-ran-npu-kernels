#!/usr/bin/env bash
set -euo pipefail

CURRENT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
RUN_MODE=npu
SOC_VERSION=${SOC_VERSION:-Ascend310P1}
KEEP_ARTIFACTS=0
while getopts ":r:v:k" opt; do
    case "${opt}" in
        r) RUN_MODE=${OPTARG} ;;
        v) SOC_VERSION=${OPTARG} ;;
        k) KEEP_ARTIFACTS=1 ;;
        *) echo "usage: bash run.sh [-r npu] [-v Ascend310P1|Ascend310P3] [-k]"; exit 2 ;;
    esac
done
if [[ "${RUN_MODE}" != "npu" ]]; then
    echo "[error] the validation launcher currently supports RUN_MODE=npu only"
    exit 2
fi

ASCEND_HOME_DIR=${ASCEND_HOME_DIR:-/usr/local/Ascend/ascend-toolkit/latest}
if [[ ! -d "${ASCEND_HOME_DIR}" ]]; then
    echo "[error] ASCEND_HOME_DIR=${ASCEND_HOME_DIR} does not exist"
    exit 1
fi
set +eu
if [[ -f "${ASCEND_HOME_DIR}/bin/setenv.bash" ]]; then
    source "${ASCEND_HOME_DIR}/bin/setenv.bash"
else
    source "${ASCEND_HOME_DIR}/../set_env.sh"
fi
set -eu

WORK_DIR=$(mktemp -d "${TMPDIR:-/tmp}/rate_dematch_mimo.XXXXXX")
if [[ "${KEEP_ARTIFACTS}" -eq 0 ]]; then
    trap 'rm -rf "${WORK_DIR}"' EXIT
else
    echo "[run] preserving artifacts at ${WORK_DIR}"
fi
DATA_DIR=${WORK_DIR}/data
BUILD_DIR=${WORK_DIR}/build
OUT_DIR=${WORK_DIR}/out

PYTHONDONTWRITEBYTECODE=1 python3 "${CURRENT_DIR}/scripts/rate_dematch_ref.py" \
    --generate --root "${DATA_DIR}"
cmake -S "${CURRENT_DIR}" -B "${BUILD_DIR}" \
    -DRUN_MODE=npu -DSOC_VERSION="${SOC_VERSION}" \
    -DASCEND_CANN_PACKAGE_PATH="${ASCEND_HOME_DIR}" \
    -DCMAKE_INSTALL_PREFIX="${OUT_DIR}"
cmake --build "${BUILD_DIR}" -j"$(nproc)"
cmake --install "${BUILD_DIR}"
export LD_LIBRARY_PATH="${OUT_DIR}/lib:${ASCEND_HOME_DIR}/lib64:${LD_LIBRARY_PATH:-}"
export AIRAN_DATA_DIR="${DATA_DIR}"
mkdir -p "${DATA_DIR}/ascend_output"
"${OUT_DIR}/bin/ascendc_kernels_bbit"
PYTHONDONTWRITEBYTECODE=1 python3 "${CURRENT_DIR}/scripts/verify_result.py" \
    --root "${DATA_DIR}"
