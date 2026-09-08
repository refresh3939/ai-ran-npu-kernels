#!/bin/bash
# re_demap_batch — generate + build + run + verify
set -e

SCRIPT_DIR=$(cd "$(dirname "$0")"; pwd)
cd "${SCRIPT_DIR}"

RUN_MODE="npu"
SOC_VERSION="Ascend310P1"
BATCH_SIZE="${RE_DEMAP_BATCH_SIZE:-8}"

while getopts ":r:v:b:" opt; do
    case ${opt} in
        r) RUN_MODE=${OPTARG} ;;
        v) SOC_VERSION=${OPTARG} ;;
        b) BATCH_SIZE=${OPTARG} ;;
        ?) echo "Usage: bash run.sh [-r cpu|sim|npu] [-v SOC_VERSION] [-b rx_antennas]"; exit 1 ;;
    esac
done

if [ -z "${ASCEND_HOME_PATH}" ]; then
    export ASCEND_HOME_PATH="/usr/local/Ascend/ascend-toolkit/latest"
fi
if [ ! -d "${ASCEND_HOME_PATH}" ]; then
    echo "[error] ASCEND_HOME_PATH=${ASCEND_HOME_PATH} does not exist"
    exit 1
fi
source "${ASCEND_HOME_PATH}/bin/setenv.bash" 2>/dev/null || \
    source "${ASCEND_HOME_PATH}/set_env.sh"

export AIRAN_DATA_DIR=${SCRIPT_DIR}
export RE_DEMAP_BATCH_SIZE=${BATCH_SIZE}
export ASCEND_GLOBAL_LOG_LEVEL=3
export ASCEND_SLOG_PRINT_TO_STDOUT=0

echo "[run.sh] re_demap_batch mode=${RUN_MODE} soc=${SOC_VERSION} NR=${BATCH_SIZE}"
python3 "${SCRIPT_DIR}/scripts/re_demap_batch_ref.py"
mkdir -p "${AIRAN_DATA_DIR}/data/ascend_output"

rm -rf "${SCRIPT_DIR}/build" "${SCRIPT_DIR}/out"
mkdir -p "${SCRIPT_DIR}/build"
cd "${SCRIPT_DIR}/build"
cmake .. -DRUN_MODE=${RUN_MODE} -DSOC_VERSION=${SOC_VERSION} \
    -DASCEND_CANN_PACKAGE_PATH=${ASCEND_HOME_PATH}
make -j
make install

cd "${SCRIPT_DIR}/out/bin"
export LD_LIBRARY_PATH=${SCRIPT_DIR}/out/lib:${ASCEND_HOME_PATH}/lib64:${LD_LIBRARY_PATH}
./ascendc_kernels_bbit
python3 "${SCRIPT_DIR}/scripts/verify_result.py"
