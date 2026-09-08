#!/bin/bash
# re_map — build + run + verify
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
source ${ASCEND_HOME_PATH}/bin/setenv.bash 2>/dev/null || \
source ${ASCEND_HOME_PATH}/set_env.sh

# 静音 INFO 日志
export ASCEND_GLOBAL_LOG_LEVEL=3
export ASCEND_SLOG_PRINT_TO_STDOUT=0

echo "================================================="
echo "[run.sh]  re_map   RUN_MODE=${RUN_MODE}  SOC=${SOC_VERSION}"
echo "================================================="

export AIRAN_DATA_DIR=${SCRIPT_DIR}
mkdir -p ${AIRAN_DATA_DIR}/data/ascend_output

# 没有 golden/weights 就先生成
if [ ! -f "${AIRAN_DATA_DIR}/weights/scatter_idx.bin" ]; then
    echo "[run.sh] 生成 golden + scatter 表 ..."
    python3 ${SCRIPT_DIR}/scripts/re_map_ref.py
fi

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
