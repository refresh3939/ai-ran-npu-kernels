#!/bin/bash
# pss_acquisition — build + run
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
echo "[run.sh]  pss_acquisition"
echo "[run.sh]  RUN_MODE=${RUN_MODE}  SOC=${SOC_VERSION}"
echo "================================================="

# AIRAN_DATA_DIR 指向 project root (kernels/rx/pss_acquisition 上去 3 层)
export AIRAN_DATA_DIR=$(cd ${SCRIPT_DIR}/../../..; pwd)
echo "[run.sh]  AIRAN_DATA_DIR=${AIRAN_DATA_DIR}"

# 确认 golden 数据存在,否则提示用户先跑 ref
GOLDEN_DIR=${AIRAN_DATA_DIR}/data/golden/rx/pss_acquisition
if [ ! -d "${GOLDEN_DIR}" ] || [ ! -f "${GOLDEN_DIR}/case_0_pss0_cfo+0_t+0/input.bin" ]; then
    echo "[run.sh] golden 数据不存在,先跑:"
    echo "         python3 scripts/pss_acquisition_ref.py"
    exit 1
fi

# 确认 PSS 模板文件存在
if [ ! -f "${GOLDEN_DIR}/pss_templates.bin" ]; then
    echo "[run.sh] pss_templates.bin 不存在,生成:"
    python3 ${SCRIPT_DIR}/scripts/gen_pss_templates.py
fi

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
# 抑制 INFO 日志洪水
export ASCEND_GLOBAL_LOG_LEVEL=3

echo ""
echo "[run.sh] executing kernel ..."
./ascendc_kernels_bbit

echo ""
echo "[run.sh] verifying ..."
python3 ${SCRIPT_DIR}/scripts/verify_result.py
