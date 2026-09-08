#!/bin/bash
# ldpc_encode 编译 + 运行脚本（self-contained，对称 ldpc_decode/run.sh）
# 用法:
#   bash run.sh -r npu -v Ascend310P1
#   bash run.sh -r npu -v Ascend310P3 -p     # msprof PipeUtilization
#   bash run.sh -r sim -v Ascend310P3
#   bash run.sh -r cpu -v Ascend310P3
#
# 前置: python scripts/ldpc_ref.py   # 生成 data/golden + data/weights（一次即可）

set -e

SCRIPT_DIR=$(cd $(dirname $0); pwd)
cd ${SCRIPT_DIR}

RUN_MODE="npu"
SOC_VERSION="Ascend310P1"
DO_PROF=0

while getopts ":r:v:p" opt
do
    case $opt in
        r) RUN_MODE=${OPTARG} ;;
        v) SOC_VERSION=${OPTARG} ;;
        p) DO_PROF=1 ;;
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

# self-contained: 数据全部在 kernel 本地 data/ 下
export AIRAN_DATA_DIR=${SCRIPT_DIR}/data
mkdir -p ${AIRAN_DATA_DIR}/ascend_output

echo "================================================="
echo "[run.sh]  RUN_MODE=${RUN_MODE}  SOC=${SOC_VERSION}  PROF=${DO_PROF}"
echo "[run.sh]  ASCEND_HOME_PATH=${ASCEND_HOME_PATH}"
echo "[run.sh]  AIRAN_DATA_DIR=${AIRAN_DATA_DIR}"
echo "================================================="

# 前置检查：golden / weights 缺失则自动生成（需 sionna 环境）
if [ ! -f "${AIRAN_DATA_DIR}/golden/output.bin" ] || \
   [ ! -f "${AIRAN_DATA_DIR}/weights/ldpc_bg1_z384_shifts/shift_A.bin" ]; then
    echo "[run.sh] golden/weights 缺失，自动生成: python scripts/ldpc_ref.py"
    python scripts/ldpc_ref.py
    if [ ! -f "${AIRAN_DATA_DIR}/golden/output.bin" ] || \
       [ ! -f "${AIRAN_DATA_DIR}/weights/ldpc_bg1_z384_shifts/shift_A.bin" ]; then
        echo "[error] 自动生成失败（检查 sionna/tensorflow 环境）"
        exit 1
    fi
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
echo "[run.sh] executing..."

if [ ${DO_PROF} -eq 1 ]; then
    PROF_OUT=${SCRIPT_DIR}/prof_out
    rm -rf ${PROF_OUT}; mkdir -p ${PROF_OUT}
    echo "[run.sh] msprof output → ${PROF_OUT}"
    msprof \
        --output=${PROF_OUT} \
        --aic-metrics=PipeUtilization \
        --ai-core=on \
        --task-time=on \
        ./ascendc_kernels_bbit
    REPORT_DIR=$(find ${PROF_OUT} -type d -name "mindstudio_profiler_output" 2>/dev/null | head -1)
    echo "[run.sh] msprof reports: ${REPORT_DIR}"
    ls ${REPORT_DIR}/*.csv 2>/dev/null
else
    ./ascendc_kernels_bbit
fi

# 二次校验（main.cpp 已内部 bit-exact；这里独立复核 dump 的 ascend_output）
echo ""
echo "[run.sh] verify_result..."
cd ${SCRIPT_DIR}
python scripts/verify_result.py
