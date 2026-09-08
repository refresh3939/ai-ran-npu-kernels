#!/bin/bash
# run.sh — build + run rate_match (mirrors ldpc_decode run.sh)
#   bash run.sh -r npu -v Ascend310P1
#   bash run.sh -r sim -v Ascend310P3
set -e

SCRIPT_DIR=$(cd $(dirname $0); pwd)
cd ${SCRIPT_DIR}

RUN_MODE="npu"
SOC_VERSION="Ascend310P3"

while getopts ":r:v:" opt; do
    case $opt in
        r) RUN_MODE=${OPTARG} ;;
        v) SOC_VERSION=${OPTARG} ;;
        ?) echo "Usage: bash run.sh [-r cpu|sim|npu] [-v SOC_VERSION]"; exit 1 ;;
    esac
done

if [ -z "${ASCEND_HOME_PATH}" ]; then
    export ASCEND_HOME_PATH="/usr/local/Ascend/ascend-toolkit/latest"
fi
if [ ! -d "${ASCEND_HOME_PATH}" ]; then
    echo "[error] ASCEND_HOME_PATH=${ASCEND_HOME_PATH} 不存在"; exit 1
fi
source ${ASCEND_HOME_PATH}/bin/setenv.bash 2>/dev/null || \
source ${ASCEND_HOME_PATH}/set_env.sh

echo "[run.sh] RUN_MODE=${RUN_MODE} SOC=${SOC_VERSION} ASCEND_HOME_PATH=${ASCEND_HOME_PATH}"

rm -rf build out
mkdir -p build
cd build
cmake .. \
    -DRUN_MODE=${RUN_MODE} \
    -DSOC_VERSION=${SOC_VERSION} \
    -DASCEND_CANN_PACKAGE_PATH=${ASCEND_HOME_PATH}
make -j
make install

cd ${SCRIPT_DIR}/out/bin
export LD_LIBRARY_PATH=${SCRIPT_DIR}/out/lib:${ASCEND_HOME_PATH}/lib64:${LD_LIBRARY_PATH}
export AIRAN_DATA_DIR=$(cd ${SCRIPT_DIR}/../../..; pwd)

echo "[run.sh] AIRAN_DATA_DIR=${AIRAN_DATA_DIR}"

# Generate the complete golden set when any file is absent or has stale dtype/shape.
GOLD_DIR=${AIRAN_DATA_DIR}/data/golden/tx/rate_match
CW_BYTES=$((143 * 25344))
LAYOUT_BYTES=$((23 * 8 * 19200 * 2))
CW_SIZE=0
LAYOUT_SIZE=0
[ -f "${GOLD_DIR}/codeword_in.bin" ] && CW_SIZE=$(stat -c %s "${GOLD_DIR}/codeword_in.bin")
[ -f "${GOLD_DIR}/layout_ref.bin" ] && LAYOUT_SIZE=$(stat -c %s "${GOLD_DIR}/layout_ref.bin")
if [ "${CW_SIZE}" -ne "${CW_BYTES}" ] || [ "${LAYOUT_SIZE}" -ne "${LAYOUT_BYTES}" ]; then
    echo "[run.sh] golden missing/stale -> generating via scripts/build_kernel_artifacts.py"
    ( cd ${SCRIPT_DIR}/scripts && AIRAN_DATA_DIR=${AIRAN_DATA_DIR} \
        PYTHONDONTWRITEBYTECODE=1 python3 build_kernel_artifacts.py )
fi

echo "[run.sh] executing..."
./ascendc_kernels_bbit
