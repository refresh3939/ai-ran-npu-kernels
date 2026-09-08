#!/usr/bin/env bash
# 跑全部 8 个 SSS Correlator case, 对比 truth.json 验证 PASS
set -e
cd "$(dirname "$0")"

# AIRAN_DATA_DIR 推断: 当前在 ~/AI-RAN-NPU/kernels/rx/sss_correlator/,
# 向上 3 级 = ~/AI-RAN-NPU/
export AIRAN_DATA_DIR="$(cd ../../..; pwd)"
echo ">>> AIRAN_DATA_DIR = $AIRAN_DATA_DIR"

CASES=(
    "case_0_clean_min_n_id_1"
    "case_1_mid_pos_pss1"
    "case_2_max_n_id_1_pss2"
    "case_3_small_pos_cfo_pcid_126"
    "case_4_small_neg_cfo_pcid_505"
    "case_5_int_cfo_p1_pcid_999"
    "case_6_low_snr_0db"
    "case_7_low_snr_n3db"
)

# 编译一次
echo ">>> Building kernel..."
source ${ASCEND_HOME_PATH}/set_env.sh 2>/dev/null || true
rm -rf build out
mkdir -p build && cd build
cmake .. >/dev/null
make -j8 install >/dev/null 2>&1
cd ..

GOLDEN_ROOT="$AIRAN_DATA_DIR/data/golden/rx/sss_correlator"

PASS_COUNT=0
FAIL_COUNT=0
RESULTS=()
LATENCIES=()

for CASE in "${CASES[@]}"; do
    echo ""
    echo "═══════════════════════════════════════════════════════════════"
    echo ">>> Running $CASE"
    echo "═══════════════════════════════════════════════════════════════"

    # 解析 expected.json 拿期望值
    EXP_FILE="$GOLDEN_ROOT/$CASE/expected.json"

    if [ ! -f "$EXP_FILE" ]; then
        echo "  ⚠️  no expected.json, skip verification"
        continue
    fi

    EXP_N_ID_1=$(grep -oP '"n_id_1"\s*:\s*\K[0-9]+' "$EXP_FILE" 2>/dev/null || echo "?")
    EXP_TAU=$(grep -oP '"tau_star"\s*:\s*\K-?[0-9]+' "$EXP_FILE" 2>/dev/null || echo "?")

    # 跑 kernel
    SSS_CASE_DIR="$CASE" \
        ASCEND_GLOBAL_LOG_LEVEL=3 \
        LD_LIBRARY_PATH="$(pwd)/out/lib:${ASCEND_HOME_PATH}/lib64:${LD_LIBRARY_PATH:-}" \
        ./out/bin/ascendc_kernels_bbit 2>&1 | tee /tmp/sss_out.txt

    # 解析 NPU 输出
    NPU_N_ID_1=$(grep -oP "\[0\] n_id_1\s+=\s+\K[0-9.-]+" /tmp/sss_out.txt | head -1 | cut -d. -f1)
    NPU_TAU=$(grep -oP "tau_star\s+=\s+\K-?[0-9.]+" /tmp/sss_out.txt | head -1 | cut -d. -f1)
    NPU_P50=$(grep -oP "p50\s+\K[0-9.]+" /tmp/sss_out.txt | head -1)

    # 比较
    if [ "$NPU_N_ID_1" = "$EXP_N_ID_1" ] && [ "$NPU_TAU" = "$EXP_TAU" ]; then
        echo "  ✓ PASS: n_id_1=$NPU_N_ID_1 tau=$NPU_TAU p50=$NPU_P50µs"
        PASS_COUNT=$((PASS_COUNT + 1))
        RESULTS+=("$CASE: ✓ PASS (n_id_1=$NPU_N_ID_1, tau=$NPU_TAU, p50=${NPU_P50}µs)")
    else
        echo "  ✗ FAIL: NPU n_id_1=$NPU_N_ID_1 tau=$NPU_TAU vs expected n_id_1=$EXP_N_ID_1 tau=$EXP_TAU"
        FAIL_COUNT=$((FAIL_COUNT + 1))
        RESULTS+=("$CASE: ✗ FAIL (NPU n_id_1=$NPU_N_ID_1 tau=$NPU_TAU vs expected n_id_1=$EXP_N_ID_1 tau=$EXP_TAU)")
    fi
done

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo ">>> SUMMARY: $PASS_COUNT PASS / $FAIL_COUNT FAIL / ${#CASES[@]} total"
echo "═══════════════════════════════════════════════════════════════"
for r in "${RESULTS[@]}"; do
    echo "  $r"
done