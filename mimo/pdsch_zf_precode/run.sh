#!/bin/bash
# precode_zf_mimo_detect_opt — 从 mimo_detect 移植配置、PACK、数据闭环和稳定计时方式
#
# 固定: BLOCK_DIM=4。默认: NT=64 NL=NLR=16 PACK=0 BRI_B=8 N_SC_PAD=1664
# 全量 P=1 case 默认每核用一个 RE_BATCH=5824 调度窗，输出仍每 64 RE 写回。
# 小层数默认自动 PACK；也可 PACK=0 做补零对照。
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")"; pwd)
cd "$SCRIPT_DIR"
RUN_MODE=npu
SOC_VERSION=Ascend310P1
while getopts ":r:v:" opt; do
    case "$opt" in
        r) RUN_MODE=$OPTARG ;;
        v) SOC_VERSION=$OPTARG ;;
        *) echo "Usage: bash run.sh [-r cpu|sim|npu] [-v SOC]"; exit 1 ;;
    esac
done

if [ -z "${ASCEND_HOME_PATH:-}" ]; then
    export ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit/latest
fi
source "$ASCEND_HOME_PATH/bin/setenv.bash" 2>/dev/null || source "$ASCEND_HOME_PATH/set_env.sh"

_NT=${NT:-64}
_NL=${NL:-16}
_NLR=${NLR:-${NL_REAL:-$_NL}}
_B=${BRI_B:-8}
_BD=4
if [ -n "${BLOCK_DIM+x}" ] && [ "$BLOCK_DIM" != 4 ]; then
    echo "[FAIL] 本优化版本按要求只使用 4 个核，BLOCK_DIM 必须为 4"
    exit 2
fi
if [ -n "${PACK+x}" ]; then
    _PK=$PACK
elif [ "$_NLR" -lt "$_NL" ]; then
    _PK=1
else
    _PK=0
fi

for value in "$_NT" "$_NL" "$_NLR" "$_B" "$_BD"; do
    if ! [[ "$value" =~ ^[1-9][0-9]*$ ]]; then
        echo "[FAIL] 维度、BRI_B 和 BLOCK_DIM 必须为正整数: $value"
        exit 2
    fi
done
if [ "$_PK" != 0 ] && [ "$_PK" != 1 ]; then
    echo "[FAIL] PACK 只能是 0 或 1"
    exit 2
fi
if [ "$_PK" = 1 ] && [ $((_NL % _NLR)) -ne 0 ]; then
    echo "[FAIL] PACK=1 要求 NLR=$_NLR 整除 NL=$_NL"
    exit 2
fi
_P=1
[ "$_PK" = 1 ] && _P=$((_NL / _NLR))
if [ -n "${RE_BATCH+x}" ]; then
    _RB=$RE_BATCH
elif [ "$_P" -eq 1 ]; then
    _RB=5824
else
    # PACK 全量 case 的每核 unit 数通常不能被 64 整除，保守保留 16。
    _RB=16
fi
if ! [[ "$_RB" =~ ^[1-9][0-9]*$ ]] || [ $((_RB % 16)) -ne 0 ]; then
    echo "[FAIL] RE_BATCH 必须是 16 的正整数倍: $_RB"
    exit 2
fi

SCPAD=${AIRAN_SCPAD_SET:-1664}
if ! [[ "$SCPAD" =~ ^[1-9][0-9]*$ ]]; then
    echo "[FAIL] N_SC_PAD 必须为正整数: $SCPAD"
    exit 2
fi
_NRE=$((14*SCPAD))
if [ $((_NRE % _P)) -ne 0 ]; then
    echo "[FAIL] N_RE=$_NRE 不能被 P=$_P 整除"
    exit 2
fi
_NUNIT=$((_NRE/_P))
if [ $((_NUNIT % (_BD*_RB))) -ne 0 ]; then
    echo "[FAIL] N_UNIT=$_NUNIT 必须被 BLOCK_DIM*RE_BATCH=$((_BD*_RB)) 整除"
    exit 2
fi

export AIRAN_DATA_DIR=$SCRIPT_DIR
export AIRAN_SCPAD=$SCPAD

# ExternalProject kernel 不继承普通 cmake -D；与 mimo_detect 一样把编译期配置写进副本 header。
sed -i "s/^#define MIMO_M .*/#define MIMO_M $_NT/" precode_zf.h
sed -i "s/^#define MIMO_K .*/#define MIMO_K $_NL/" precode_zf.h
sed -i "s/^#define MIMO_KR .*/#define MIMO_KR $_NLR/" precode_zf.h
sed -i "s/^#define AIRAN_PACK .*/#define AIRAN_PACK $_PK/" precode_zf.h
sed -i "s/^#define AIRAN_BLOCK_DIM .*/#define AIRAN_BLOCK_DIM 4/" precode_zf.h
sed -i "s/^#define AIRAN_RE_BATCH .*/#define AIRAN_RE_BATCH $_RB/" precode_zf.h
sed -i "s/^#define BRI_B .*/#define BRI_B  $_B/" precode_zf.h
sed -i "s/^#define N_SC_PAD_VAL .*/#define N_SC_PAD_VAL $SCPAD/" precode_zf.h

CASE="case_0_m${_NT}_k${_NL}_sc${SCPAD}"
[ "$_NLR" != "$_NL" ] && CASE="${CASE}_r${_NLR}"
[ "$_P" != 1 ] && CASE="${CASE}_pk"
echo "==== precode_zf_opt mode=$RUN_MODE soc=$SOC_VERSION NT=$_NT NL=$_NL NLR=$_NLR PACK=$_PK P=$_P B=$_B blockDim=$_BD RE_BATCH=$_RB sc=$SCPAD case=$CASE ===="

GOLD="$SCRIPT_DIR/data/golden/$CASE"
EXPECTED_W_BYTES=$((_NUNIT*_NT*_NL*2))
EXPECTED_S_BYTES=$((_NUNIT*_NL*2))
NEED_GOLDEN=0
[ ! -f "$GOLD/wrm_re.bin" ] && NEED_GOLDEN=1
[ -f "$GOLD/wrm_re.bin" ] && [ "$(stat -c %s "$GOLD/wrm_re.bin")" -ne "$EXPECTED_W_BYTES" ] && NEED_GOLDEN=1
[ -f "$GOLD/srm_re.bin" ] && [ "$(stat -c %s "$GOLD/srm_re.bin")" -ne "$EXPECTED_S_BYTES" ] && NEED_GOLDEN=1
[ "${FORCE_GOLDEN:-0}" = 1 ] && NEED_GOLDEN=1
# BRI_B 影响 PACK BRI 对照和 stage 数据；显式调 B 时强制重建，避免漂亮但陈旧的结果。
[ -n "${BRI_B+x}" ] && NEED_GOLDEN=1
if [ "$NEED_GOLDEN" -eq 1 ]; then
    NT=$_NT NL=$_NL NLR=$_NLR PACK=$_PK BRI_B=$_B AIRAN_SCPAD=$SCPAD \
        AIRAN_DATA_DIR="$SCRIPT_DIR/data" python3 "$SCRIPT_DIR/scripts/precode_zf_ref.py"
fi

# 只清理这个新副本、这个确定 case 的输出，禁止 verifier 读取上一轮陈旧结果。
rm -rf "$SCRIPT_DIR/data/ascend_output/$CASE"
mkdir -p "$SCRIPT_DIR/data/ascend_output/$CASE"
rm -rf "$SCRIPT_DIR/build" "$SCRIPT_DIR/out"
mkdir -p "$SCRIPT_DIR/build"
cd "$SCRIPT_DIR/build"
cmake .. -DRUN_MODE="$RUN_MODE" -DSOC_VERSION="$SOC_VERSION" -DASCEND_CANN_PACKAGE_PATH="$ASCEND_HOME_PATH"
make -j
make install

cd "$SCRIPT_DIR/out/bin"
export LD_LIBRARY_PATH="$SCRIPT_DIR/out/lib:$ASCEND_HOME_PATH/lib64:${LD_LIBRARY_PATH:-}"
./ascendc_kernels_bbit

NT=$_NT NL=$_NL NLR=$_NLR PACK=$_PK AIRAN_SCPAD=$SCPAD \
    AIRAN_DATA_DIR="$SCRIPT_DIR" python3 "$SCRIPT_DIR/scripts/verify_result.py"
