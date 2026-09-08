#!/bin/bash
# llr_assemble 编译 + 运行 + 校验脚本(数据本地化)
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

export ASCEND_GLOBAL_LOG_LEVEL=3

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

# 数据本地化:根 = 算子目录本身(golden + 输出都在 ${SCRIPT_DIR}/data/ 下)
export AIRAN_DATA_DIR=${SCRIPT_DIR}
mkdir -p ${AIRAN_DATA_DIR}/data/ascend_output/rx/llr_assemble

echo ""
echo "[run.sh] executing..."
./ascendc_kernels_bbit || true     # 不因 binary 自检 FAIL 中断,交由 Python verify 终判

cd ${SCRIPT_DIR}
echo ""
python scripts/verify_result.py