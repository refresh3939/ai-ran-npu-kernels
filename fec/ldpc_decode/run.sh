#!/bin/bash
# ldpc_decode 编译 + 运行脚本（self-contained，对称 ldpc_encode/run.sh）
# 用法:
#   bash run.sh -r npu -v Ascend310P1            # 默认 5 dB
#   bash run.sh -r npu -v Ascend310P1 -s 10      # 选择 SNR 数据集
#   bash run.sh -r npu -v Ascend310P1 -p         # msprof PipeUtilization
#   bash run.sh -r sim -v Ascend310P3
#   bash run.sh -r cpu -v Ascend310P3

set -e

SCRIPT_DIR=$(cd $(dirname $0); pwd)
cd ${SCRIPT_DIR}

RUN_MODE="npu"
SOC_VERSION="Ascend310P1"
SNR_DB=5
DO_PROF=0

while getopts ":r:v:s:p" opt
do
    case $opt in
        r) RUN_MODE=${OPTARG} ;;
        v) SOC_VERSION=${OPTARG} ;;
        s) SNR_DB=${OPTARG} ;;
        p) DO_PROF=1 ;;
        ?) echo "Usage: bash run.sh [-r cpu|sim|npu] [-v SOC_VERSION] [-s SNR_DB] [-p]"; exit 1 ;;
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

# 与 encoder 一样，运行时只使用算子目录内的 data/ 和 weights/。
export AIRAN_DATA_DIR=${SCRIPT_DIR}/data/snr_${SNR_DB}db
export AIRAN_WEIGHT_DIR=${SCRIPT_DIR}/data/weights/ldpc_bg1_z384_shifts
export AIRAN_OUTPUT_DIR=${SCRIPT_DIR}/data/ascend_output
mkdir -p ${AIRAN_OUTPUT_DIR}

if [ ! -f "${AIRAN_DATA_DIR}/lam_in.bin" ] || \
   [ ! -f "${AIRAN_DATA_DIR}/info_bits.bin" ]; then
    echo "[error] SNR ${SNR_DB} dB 数据集不完整: ${AIRAN_DATA_DIR}"
    echo "        请先运行: conda run -n sionna python scripts/gen_data.py --snr-list ${SNR_DB}"
    exit 1
fi
if [ ! -f "${AIRAN_WEIGHT_DIR}/shift_table.bin" ]; then
    echo "[error] LDPC shift table 不存在: ${AIRAN_WEIGHT_DIR}/shift_table.bin"
    exit 1
fi

echo "================================================="
echo "[run.sh]  RUN_MODE=${RUN_MODE}  SOC=${SOC_VERSION}  SNR=${SNR_DB}dB  PROF=${DO_PROF}"
echo "[run.sh]  ASCEND_HOME_PATH=${ASCEND_HOME_PATH}"
echo "[run.sh]  AIRAN_DATA_DIR=${AIRAN_DATA_DIR}"
echo "[run.sh]  AIRAN_WEIGHT_DIR=${AIRAN_WEIGHT_DIR}"
echo "================================================="

rm -rf build out
mkdir -p build
cd build

cmake .. \
    -DRUN_MODE=${RUN_MODE} \
    -DSOC_VERSION=${SOC_VERSION} \
    -DASCEND_CANN_PACKAGE_PATH=${ASCEND_HOME_PATH}
make -j
make install

# 跑测试
cd ${SCRIPT_DIR}/out/bin
export LD_LIBRARY_PATH=${SCRIPT_DIR}/out/lib:${ASCEND_HOME_PATH}/lib64:${LD_LIBRARY_PATH}

echo ""
echo "[run.sh] executing..."

if [ ${DO_PROF} -eq 1 ]; then
    # msprof 方式一(官方推荐): app 直接放末尾,不用 --application=
    PROF_OUT=${SCRIPT_DIR}/prof_out
    rm -rf ${PROF_OUT}
    mkdir -p ${PROF_OUT}

    echo "[run.sh] msprof output → ${PROF_OUT}"
    msprof \
        --output=${PROF_OUT} \
        --aic-metrics=PipeUtilization \
        --ai-core=on \
        --task-time=on \
        ./ascendc_kernels_bbit

    echo ""
    echo "================================================="
    echo "[run.sh] msprof 完成,reports 位置:"
    REPORT_DIR=$(find ${PROF_OUT} -type d -name "mindstudio_profiler_output" 2>/dev/null | head -1)
    if [ -n "${REPORT_DIR}" ]; then
        echo "  ${REPORT_DIR}"
        echo ""
        echo "[run.sh] 关键 CSV 文件:"
        ls ${REPORT_DIR}/*.csv 2>/dev/null
    else
        echo "  未找到 mindstudio_profiler_output,所有 prof_out 子目录:"
        find ${PROF_OUT} -maxdepth 3 -type d
    fi
    echo "================================================="
else
    ./ascendc_kernels_bbit
fi

echo ""
echo "[run.sh] verify_result..."
cd ${SCRIPT_DIR}
python3 scripts/verify_result.py
