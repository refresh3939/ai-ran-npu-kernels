#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")"; pwd)
cd "${SCRIPT_DIR}"

RUN_MODE="npu"
SOC_VERSION="Ascend310P3"
while getopts ":r:v:" opt; do
    case ${opt} in
        r) RUN_MODE=${OPTARG} ;;
        v) SOC_VERSION=${OPTARG} ;;
        *) echo "Usage: bash run.sh [-r cpu|sim|npu] [-v SOC_VERSION]"; exit 1 ;;
    esac
done

ASCEND_HOME_PATH=${ASCEND_HOME_PATH:-/usr/local/Ascend/ascend-toolkit/latest}
if [ ! -d "${ASCEND_HOME_PATH}" ]; then
    echo "[error] ASCEND_HOME_PATH=${ASCEND_HOME_PATH} does not exist"
    exit 1
fi
source "${ASCEND_HOME_PATH}/bin/setenv.bash" 2>/dev/null || source "${ASCEND_HOME_PATH}/set_env.sh"

export AIRAN_DATA_DIR=${SCRIPT_DIR}/data
export ASCEND_GLOBAL_LOG_LEVEL=3
python3 scripts/gen_data.py
mkdir -p data/ascend_output build out

cmake -S . -B build \
    -DRUN_MODE="${RUN_MODE}" \
    -DSOC_VERSION="${SOC_VERSION}" \
    -DASCEND_CANN_PACKAGE_PATH="${ASCEND_HOME_PATH}" \
    -DCMAKE_INSTALL_PREFIX="${SCRIPT_DIR}/out"
cmake --build build -j
cmake --install build

export LD_LIBRARY_PATH="${SCRIPT_DIR}/out/lib:${ASCEND_HOME_PATH}/lib64:${LD_LIBRARY_PATH:-}"
"${SCRIPT_DIR}/out/bin/ascendc_kernels_bbit"
python3 scripts/verify_result.py
