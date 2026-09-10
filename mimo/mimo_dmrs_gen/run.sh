#!/bin/bash
# dmrs_gen — PUSCH DMRS generator (TX): build + run + verify
set -e

SCRIPT_DIR=$(cd $(dirname $0); pwd)
cd ${SCRIPT_DIR}
RUN_ROOT=${AIRAN_RUN_ROOT:-${SCRIPT_DIR}}
BUILD_DIR=${AIRAN_BUILD_DIR:-${RUN_ROOT}/build}
OUT_DIR=${AIRAN_OUT_DIR:-${RUN_ROOT}/out}

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
    echo "[error] ASCEND_HOME_PATH=${ASCEND_HOME_PATH} 不存在"
    exit 1
fi
source ${ASCEND_HOME_PATH}/bin/setenv.bash 2>/dev/null || \
source ${ASCEND_HOME_PATH}/set_env.sh

echo "================================================="
echo "[run.sh]  mimo_dmrs_gen (TX K-layer DMRS)"
echo "[run.sh]  RUN_MODE=${RUN_MODE}  SOC=${SOC_VERSION}"
echo "================================================="

export AIRAN_DATA_DIR=${AIRAN_DATA_DIR:-${RUN_ROOT}}
echo "[run.sh]  AIRAN_DATA_DIR=${AIRAN_DATA_DIR}"

# Always regenerate: the cases intentionally cover different ranks/port orders.
echo "[run.sh] generating golden via mimo_dmrs_gen_ref.py ..."
python3 ${SCRIPT_DIR}/scripts/mimo_dmrs_gen_ref.py
mkdir -p ${AIRAN_DATA_DIR}/data/ascend_output

export ASCEND_GLOBAL_LOG_LEVEL=3

rm -rf "${BUILD_DIR}" "${OUT_DIR}"
mkdir -p "${BUILD_DIR}"
cmake -S "${SCRIPT_DIR}" -B "${BUILD_DIR}" \
    -DRUN_MODE=${RUN_MODE} \
    -DSOC_VERSION=${SOC_VERSION} \
    -DASCEND_CANN_PACKAGE_PATH=${ASCEND_HOME_PATH} \
    -DCMAKE_INSTALL_PREFIX="${OUT_DIR}"
cmake --build "${BUILD_DIR}" -j
cmake --install "${BUILD_DIR}"

cd "${OUT_DIR}/bin"
export LD_LIBRARY_PATH=${OUT_DIR}/lib:${ASCEND_HOME_PATH}/lib64:${LD_LIBRARY_PATH}

echo ""
echo "[run.sh] executing kernel ..."
./ascendc_kernels_bbit

echo ""
echo "[run.sh] verifying ..."
python3 ${SCRIPT_DIR}/scripts/verify_result.py
