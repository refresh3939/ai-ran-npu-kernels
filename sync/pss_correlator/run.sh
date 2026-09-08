#!/bin/bash
# pss_correlator — build + run + verify
#
# 用法:
#   bash run.sh                          # 跑 case_0 (默认)
#   bash run.sh -c case_3_pss0_cfop25000_t1000_snr30   # 跑指定 case
#   bash run.sh -a                       # 跑全部 8 cases (自动遍历 data/golden/)
#   bash run.sh -r cpu                   # CPU 模式
#   bash run.sh -v Ascend910B            # 指定 SOC
set -e

SCRIPT_DIR=$(cd $(dirname $0); pwd)
cd ${SCRIPT_DIR}

# ───── 解析参数 ─────────────────────────────────────────────────
RUN_MODE="npu"
SOC_VERSION="Ascend310P3"
CASE_DIR=""
RUN_ALL=false
SKIP_VERIFY=false

while getopts ":r:v:c:as" opt; do
    case $opt in
        r) RUN_MODE=${OPTARG} ;;
        v) SOC_VERSION=${OPTARG} ;;
        c) CASE_DIR=${OPTARG} ;;
        a) RUN_ALL=true ;;
        s) SKIP_VERIFY=true ;;
        ?) echo "Usage: bash run.sh [-r cpu|npu] [-v SOC] [-c CASE_DIR] [-a] [-s]"
           echo "  -a: 跑全部 8 cases"
           echo "  -c: 指定单个 case (默认 case_0)"
           echo "  -s: 跳过 verify"
           exit 1 ;;
    esac
done

# ───── CANN env ─────────────────────────────────────────────────
if [ -z "${ASCEND_HOME_PATH}" ]; then
    export ASCEND_HOME_PATH="/usr/local/Ascend/ascend-toolkit/latest"
fi
if [ ! -d "${ASCEND_HOME_PATH}" ]; then
    echo "[error] ASCEND_HOME_PATH=${ASCEND_HOME_PATH} not found"
    exit 1
fi
source ${ASCEND_HOME_PATH}/bin/setenv.bash 2>/dev/null || \
source ${ASCEND_HOME_PATH}/set_env.sh

export AIRAN_DATA_DIR=${SCRIPT_DIR}
export ASCEND_GLOBAL_LOG_LEVEL=3

# 数据目录 (本地)
GOLDEN_ROOT="${SCRIPT_DIR}/data/golden/rx/pss_correlator"
OUTPUT_ROOT="${SCRIPT_DIR}/data/ascend_output"
mkdir -p ${OUTPUT_ROOT}

# ───── 自动生成 golden (如缺) ──────────────────────────────────
if [ ! -f "${GOLDEN_ROOT}/pss_ref.bin" ] || [ ! -f "${GOLDEN_ROOT}/twiddle.bin" ]; then
    echo "[run.sh] golden missing, running Python ref to generate..."
    python3 ${SCRIPT_DIR}/scripts/pss_correlator_ref.py
fi

echo "================================================="
echo "[run.sh]  pss_correlator"
echo "[run.sh]  RUN_MODE=${RUN_MODE}  SOC=${SOC_VERSION}"
echo "[run.sh]  DATA_DIR=${SCRIPT_DIR}/data/"
echo "================================================="

# ───── Build (只一次) ───────────────────────────────────────────
echo ""
echo "[run.sh] === Build ==="
rm -rf build out
mkdir -p build
cd build
cmake .. \
    -DRUN_MODE=${RUN_MODE} \
    -DSOC_VERSION=${SOC_VERSION} \
    -DASCEND_CANN_PACKAGE_PATH=${ASCEND_HOME_PATH} > /dev/null
make -j > /dev/null
make install > /dev/null
echo "[run.sh] build OK"

cd ${SCRIPT_DIR}/out/bin
export LD_LIBRARY_PATH=${SCRIPT_DIR}/out/lib:${ASCEND_HOME_PATH}/lib64:${LD_LIBRARY_PATH}

# ───── 决定跑哪些 case ──────────────────────────────────────────
if [ "$RUN_ALL" = true ]; then
    # 自动列出所有 case
    mapfile -t CASES < <(ls "${GOLDEN_ROOT}" | grep '^case_' | sort)
    if [ "${#CASES[@]}" -eq 0 ]; then
        echo "[run.sh] ERROR: no cases found under ${GOLDEN_ROOT}"
        exit 1
    fi
    echo "[run.sh] running ALL ${#CASES[@]} cases"
elif [ -n "$CASE_DIR" ]; then
    CASES=("$CASE_DIR")
    echo "[run.sh] running 1 case: ${CASE_DIR}"
else
    # 默认 case_0
    CASES=("case_0_pss0_cfop0_t12345_snr30")
    echo "[run.sh] running default case: ${CASES[0]}"
fi

# ───── 跑 + verify ──────────────────────────────────────────────
PASS_COUNT=0
FAIL_COUNT=0
FAIL_CASES=()
LATENCIES=()

for case_name in "${CASES[@]}"; do
    echo ""
    echo "═══════════════════════════════════════════════════════════════════"
    echo "[run.sh] Running: $case_name"
    echo "═══════════════════════════════════════════════════════════════════"

    export PSS_CASE_DIR="$case_name"
    LOG=$(./ascendc_kernels_bbit 2>&1)
    echo "$LOG"

    LAT=$(echo "$LOG" | grep -oP 'p50 \K[0-9.]+' | head -1)
    LATENCIES+=("$LAT")

    # 自动跑 verify (除非 -s)
    if [ "$SKIP_VERIFY" = true ]; then
        continue
    fi

    echo ""
    echo "[run.sh] --- Verify ---"
    if python3 ${SCRIPT_DIR}/scripts/verify_result.py "$case_name"; then
        PASS_COUNT=$((PASS_COUNT + 1))
    else
        FAIL_COUNT=$((FAIL_COUNT + 1))
        FAIL_CASES+=("$case_name")
    fi
done

# ───── 总结 ─────────────────────────────────────────────────────
if [ "$SKIP_VERIFY" = true ]; then
    echo ""
    echo "[run.sh] (verify skipped)"
elif [ "${#CASES[@]}" -gt 1 ]; then
    echo ""
    echo "╔══════════════════════════════════════════════════════════════════════╗"
    echo "║  PSS Correlator summary (${#CASES[@]} cases)"
    echo "╠══════════════════════════════════════════════════════════════════════╣"
    for i in "${!CASES[@]}"; do
        printf "║  %-50s  p50=%s us\n" "${CASES[$i]}" "${LATENCIES[$i]}"
    done
    echo "║"
    echo "║  PASS: $PASS_COUNT / ${#CASES[@]}"
    echo "║  FAIL: $FAIL_COUNT / ${#CASES[@]}"
    if [ "$FAIL_COUNT" -gt 0 ]; then
        echo "║  Failed:"
        for c in "${FAIL_CASES[@]}"; do
            echo "║    - $c"
        done
    fi
    echo "╚══════════════════════════════════════════════════════════════════════╝"
    [ "$FAIL_COUNT" -eq 0 ]
fi