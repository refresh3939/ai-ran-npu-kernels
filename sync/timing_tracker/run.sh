#!/bin/bash
# timing_tracker — build + run + verify
set -e

SCRIPT_DIR=$(cd $(dirname $0); pwd)
cd ${SCRIPT_DIR}

# 检查 cmake/npu_lib.cmake (跟 ofdm_demod 共用)
if [ ! -f "${SCRIPT_DIR}/cmake/npu_lib.cmake" ]; then
    echo "[run.sh] cmake/npu_lib.cmake 不存在, 从 ofdm_demod 复制..."
    OFDM_DIR="${SCRIPT_DIR}/../ofdm_matmul_micro"
    if [ ! -d "${OFDM_DIR}/cmake" ]; then
        OFDM_DIR="${SCRIPT_DIR}/../../rx/ofdm_matmul_micro"
    fi
    if [ -d "${OFDM_DIR}/cmake" ]; then
        cp -r "${OFDM_DIR}/cmake" "${SCRIPT_DIR}/"
        echo "[run.sh] copied from ${OFDM_DIR}/cmake/"
    else
        echo "[error] cmake/npu_lib.cmake 找不到, 请手动从 ofdm_demod 复制 cmake/ 目录过来"
        exit 1
    fi
fi

RUN_MODE="npu"
SOC_VERSION="Ascend310P3"

while getopts ":r:v:" opt
do
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
    echo "[error] ASCEND_HOME_PATH=${ASCEND_HOME_PATH} not found"
    exit 1
fi
source ${ASCEND_HOME_PATH}/bin/setenv.bash 2>/dev/null || \
source ${ASCEND_HOME_PATH}/set_env.sh

echo "================================================="
echo "[run.sh]  timing_tracker"
echo "[run.sh]  RUN_MODE=${RUN_MODE}  SOC=${SOC_VERSION}"
echo "================================================="

export AIRAN_DATA_DIR=${SCRIPT_DIR}
echo "[run.sh]  AIRAN_DATA_DIR=${AIRAN_DATA_DIR}"
export ASCEND_GLOBAL_LOG_LEVEL=3

mkdir -p ${AIRAN_DATA_DIR}/data/ascend_output

echo ""
echo "[run.sh] generating ref + weights + golden ..."
python3 ${SCRIPT_DIR}/scripts/timing_tracker_ref.py

echo ""
echo "[run.sh] cmake build ..."
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

echo ""
echo "[run.sh] executing kernel ..."
./ascendc_kernels_bbit

echo ""
echo "[run.sh] verifying ..."
python3 ${SCRIPT_DIR}/scripts/verify_result.py
