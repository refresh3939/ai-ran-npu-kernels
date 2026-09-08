#!/bin/bash
# interpolate ×8 — gen golden + build + run + verify  (对标 decimate run.sh)
set -e

SCRIPT_DIR=$(cd $(dirname $0); pwd)
cd ${SCRIPT_DIR}

RUN_MODE="npu"
SOC_VERSION="Ascend310P3"
PROFILE=0

while getopts ":r:v:p" opt
do
    case $opt in
        r) RUN_MODE=${OPTARG} ;;
        v) SOC_VERSION=${OPTARG} ;;
        p) PROFILE=1 ;;
        ?) echo "Usage: bash run.sh [-r cpu|sim|npu] [-v SOC_VERSION] [-p]"; exit 1 ;;
    esac
done

if [ -z "${ASCEND_HOME_PATH}" ]; then
    export ASCEND_HOME_PATH="/usr/local/Ascend/ascend-toolkit/latest"
fi
if [ ! -d "${ASCEND_HOME_PATH}" ]; then
    echo "[error] ASCEND_HOME_PATH=${ASCEND_HOME_PATH} 不存在"
    exit 1
fi
source ${ASCEND_HOME_PATH}/bin/setenv.bash 2>/dev/null || \
source ${ASCEND_HOME_PATH}/set_env.sh

echo "================================================="
echo "[run.sh]  interpolate ×8"
echo "[run.sh]  RUN_MODE=${RUN_MODE}  SOC=${SOC_VERSION}  PROFILE=${PROFILE}"
echo "================================================="

export AIRAN_DATA_DIR=${SCRIPT_DIR}
echo "[run.sh]  AIRAN_DATA_DIR=${AIRAN_DATA_DIR}"

# golden + FIR 系数 (idempotent)
echo "[run.sh]  生成 golden + FIR 系数 ..."
python3 ${SCRIPT_DIR}/scripts/interpolate_ref.py

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

echo ""
if [ "${PROFILE}" = "1" ]; then
    echo "[run.sh] msprof ..."
    msprof --application="./ascendc_kernels_bbit" \
           --aic-metrics=PipeUtilization --ai-core=on --task-time=on \
           --output=${SCRIPT_DIR}/prof
else
    echo "[run.sh] executing kernel ..."
    ./ascendc_kernels_bbit
fi

echo ""
echo "[run.sh] verifying ..."
python3 ${SCRIPT_DIR}/scripts/verify_result.py
