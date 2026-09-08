#!/bin/bash
# sss_correlator — build + run
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
    echo "[error] ASCEND_HOME_PATH=${ASCEND_HOME_PATH} not found"
    exit 1
fi
source ${ASCEND_HOME_PATH}/bin/setenv.bash 2>/dev/null || \
source ${ASCEND_HOME_PATH}/set_env.sh

echo "================================================="
echo "[run.sh]  sss_correlator"
echo "[run.sh]  RUN_MODE=${RUN_MODE}  SOC=${SOC_VERSION}"
echo "[run.sh]  CASE=${SSS_CASE_DIR:-case_0_clean_min_n_id_1}"
echo "================================================="

# AIRAN_DATA_DIR 默认 = SCRIPT_DIR (case 在 SCRIPT_DIR/data/golden/rx/sss_correlator/)
# 但 ref 落 golden 到 ~/AI-RAN-NPU/data, 所以这里要指向那里
export AIRAN_DATA_DIR=${AIRAN_DATA_DIR:-${SCRIPT_DIR}/../../..}
echo "[run.sh]  AIRAN_DATA_DIR=${AIRAN_DATA_DIR}"

mkdir -p ${AIRAN_DATA_DIR}/data/ascend_output

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
export ASCEND_GLOBAL_LOG_LEVEL=3

echo ""
echo "[run.sh] executing kernel ..."
./ascendc_kernels_bbit
