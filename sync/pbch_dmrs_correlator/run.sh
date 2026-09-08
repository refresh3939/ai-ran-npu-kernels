#!/bin/bash
# pbch_dmrs_correlator — build + run (operator-self-contained)
#
# Usage:
#   bash run.sh                  # build + 跑默认 case (case_0_clean_L4_i0)
#   bash run.sh -c CASE_NAME     # build + 跑指定 case
#   bash run.sh -a               # build + 跑全部 8 个 case (调 scripts/run_all_cases.py)
#   bash run.sh -r cpu|sim|npu   # 切换 run mode (默认 npu)
#   bash run.sh -v SOC_VERSION   # 切换 SoC (默认 Ascend310P3)
#
# Layout: 工程自包含,所有数据都在 ./data/ 下,不依赖全局 AIRAN_DATA_DIR.
set -e

SCRIPT_DIR=$(cd $(dirname $0); pwd)
cd ${SCRIPT_DIR}

RUN_MODE="npu"
SOC_VERSION="Ascend310P3"
RUN_ALL=0

while getopts ":r:v:c:a" opt; do
    case $opt in
        r) RUN_MODE=${OPTARG} ;;
        v) SOC_VERSION=${OPTARG} ;;
        c) export PBCH_DMRS_CASE_DIR=${OPTARG} ;;
        a) RUN_ALL=1 ;;
        ?) echo "Usage: bash run.sh [-r cpu|sim|npu] [-v SOC] [-c CASE] [-a]"; exit 1 ;;
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

# 算子本地 data 根 = SCRIPT_DIR
export PBCH_DMRS_DATA_ROOT=${SCRIPT_DIR}

echo "================================================="
echo "[run.sh]  pbch_dmrs_correlator"
echo "[run.sh]  RUN_MODE=${RUN_MODE}  SOC=${SOC_VERSION}"
if [ ${RUN_ALL} -eq 1 ]; then
    echo "[run.sh]  MODE: run all cases"
else
    echo "[run.sh]  CASE: ${PBCH_DMRS_CASE_DIR:-case_0_clean_L4_i0}"
fi
echo "[run.sh]  DATA_ROOT=${PBCH_DMRS_DATA_ROOT}"
echo "================================================="

mkdir -p ${PBCH_DMRS_DATA_ROOT}/data/ascend_output

# ── 借用 SSS Correlator 的 cmake/ (cpu_lib.cmake + npu_lib.cmake) ──
# 这两个文件来自 CANN sample, 在 SSS 已 PASS, 我们复用同一份.
if [ ! -d "cmake" ]; then
    SSS_CMAKE=${SCRIPT_DIR}/../sss_correlator/cmake
    if [ -d "${SSS_CMAKE}" ]; then
        echo "[run.sh] borrowing cmake/ from ${SSS_CMAKE}"
        cp -r ${SSS_CMAKE} cmake
    else
        echo "[error] cmake/ not found at ${SSS_CMAKE}"
        echo "[error] expected sibling sss_correlator/cmake/ to exist"
        echo "[error] please manually copy cpu_lib.cmake / npu_lib.cmake into ${SCRIPT_DIR}/cmake/"
        exit 1
    fi
fi

# ── 生成 test cases (如果还没生成) ──
CASE_ROOT=${PBCH_DMRS_DATA_ROOT}/data/golden
HAS_CASES=0
if [ -d "${CASE_ROOT}" ]; then
    if [ -n "$(ls -A ${CASE_ROOT} 2>/dev/null | grep '^case_')" ]; then
        HAS_CASES=1
    fi
fi
if [ ${HAS_CASES} -eq 0 ]; then
    echo "[run.sh] generating test cases ..."
    python3 ${SCRIPT_DIR}/scripts/gen_test_cases.py
fi

# ── build ──
rm -rf build out
mkdir -p build
cd build
cmake .. \
    -DRUN_MODE=${RUN_MODE} \
    -DSOC_VERSION=${SOC_VERSION} \
    -DASCEND_CANN_PACKAGE_PATH=${ASCEND_HOME_PATH}
make -j
make install

cd ${SCRIPT_DIR}

# ── run ──
export LD_LIBRARY_PATH=${SCRIPT_DIR}/out/lib:${ASCEND_HOME_PATH}/lib64:${LD_LIBRARY_PATH}
export ASCEND_GLOBAL_LOG_LEVEL=3

if [ ${RUN_ALL} -eq 1 ]; then
    echo ""
    echo "[run.sh] running all cases via scripts/run_all_cases.py ..."
    python3 ${SCRIPT_DIR}/scripts/run_all_cases.py
else
    echo ""
    echo "[run.sh] executing kernel (single case) ..."
    cd ${SCRIPT_DIR}/out/bin
    ./ascendc_kernels_bbit
fi
