#!/bin/bash
set -e

SCRIPT_DIR=$(cd $(dirname $0); pwd)
cd ${SCRIPT_DIR}

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
echo "[run.sh]  RUN_MODE=${RUN_MODE}  SOC=${SOC_VERSION}"
echo "[run.sh]  ASCEND_HOME_PATH=${ASCEND_HOME_PATH}"
echo "================================================="

# ── Prepare test data (算子自包含: data/ 在当前算子目录下) ─────────────────
AIRAN_DATA_DIR=${SCRIPT_DIR}/data
mkdir -p ${AIRAN_DATA_DIR}/golden ${AIRAN_DATA_DIR}/ascend_output

echo "[run.sh] Generating grouped/padded test vectors..."
cd ${SCRIPT_DIR}
python3 scripts/scramble_ref.py
cp scripts/test_data/scramble_input.bin     ${AIRAN_DATA_DIR}/golden/input.bin
cp scripts/test_data/scramble_output.bin    ${AIRAN_DATA_DIR}/golden/golden.bin
cp scripts/test_data/scramble_gold_flat.bin ${AIRAN_DATA_DIR}/golden/gold_flat.bin
echo "[run.sh] Test vectors ready."

# ── Build ─────────────────────────────────────────────────────────────────
rm -rf build out
mkdir -p build
cd build

cmake .. \
    -DRUN_MODE=${RUN_MODE} \
    -DSOC_VERSION=${SOC_VERSION} \
    -DASCEND_CANN_PACKAGE_PATH=${ASCEND_HOME_PATH}
make -j
make install

# ── Run ───────────────────────────────────────────────────────────────────
cd ${SCRIPT_DIR}/out/bin
export LD_LIBRARY_PATH=${SCRIPT_DIR}/out/lib:${ASCEND_HOME_PATH}/lib64:${LD_LIBRARY_PATH}
export AIRAN_DATA_DIR=${AIRAN_DATA_DIR}

mkdir -p ${AIRAN_DATA_DIR}/ascend_output

echo ""
echo "[run.sh] executing..."
./ascendc_kernels_bbit
