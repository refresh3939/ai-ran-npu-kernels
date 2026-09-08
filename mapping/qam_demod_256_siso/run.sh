#!/bin/bash
# qam256_demod 编译 + 运行脚本(自包含多 case:ref 生成 → build → kernel → 逐 case 校验)
set -e

SCRIPT_DIR=$(cd $(dirname $0); pwd)
cd ${SCRIPT_DIR}

RUN_MODE="npu"
SOC_VERSION="Ascend310P1"
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

# ---- kernel 自包含数据目录(绝对路径,ref.py 与 binary 共用)----
export AIRAN_DATA_DIR=${SCRIPT_DIR}/data

echo "================================================="
echo "[run.sh]  RUN_MODE=${RUN_MODE}  SOC=${SOC_VERSION}"
echo "[run.sh]  ASCEND_HOME_PATH=${ASCEND_HOME_PATH}"
echo "[run.sh]  AIRAN_DATA_DIR=${AIRAN_DATA_DIR}"
echo "================================================="

# ---- 1) 生成多 case golden(写 data/golden/case_N_xxx/ + 建空 ascend_output/case_N_xxx/)----
echo "[run.sh] generating golden (multi-case)..."
python3 ${SCRIPT_DIR}/scripts/qam256_demod_ref.py

export ASCEND_GLOBAL_LOG_LEVEL=3

# ---- 2) 编译 ----
rm -rf build out
mkdir -p build
cd build
cmake .. \
    -DRUN_MODE=${RUN_MODE} \
    -DSOC_VERSION=${SOC_VERSION} \
    -DASCEND_CANN_PACKAGE_PATH=${ASCEND_HOME_PATH}
make -j
make install

# ---- 3) 运行(从 out/bin;AIRAN_DATA_DIR 绝对路径,cwd 无关)----
cd ${SCRIPT_DIR}/out/bin
export LD_LIBRARY_PATH=${SCRIPT_DIR}/out/lib:${ASCEND_HOME_PATH}/lib64:${LD_LIBRARY_PATH}

echo ""
echo "[run.sh] executing..."
./ascendc_kernels_bbit
