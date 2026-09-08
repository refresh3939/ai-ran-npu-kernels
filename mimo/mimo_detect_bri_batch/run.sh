#!/bin/bash
# mimo_detect_bri_batch — massive MIMO BRI. build + run + verify
# Production chain ABI selects NR=16/32/64 to match mimo_detect_io_pack:
#   physical NL=16, 14x1664 RE, PACK=0.
# NL_REAL selects active L; H columns [L,16) must be zero. Output storage is
# always xhat/no_eff [16,23296], and downstream consumes only [0,L).
# BRI convergence is required at both shell and C++ host boundaries:
#   NR >= 8*min(BRI_B,L), 2*BRI_B >= L, and BRI_B divides 16.
set -e
SCRIPT_DIR=$(cd $(dirname $0); pwd); cd ${SCRIPT_DIR}
RUN_MODE="npu"; SOC_VERSION="Ascend310P1"
while getopts ":r:v:" opt; do case $opt in
    r) RUN_MODE=${OPTARG} ;; v) SOC_VERSION=${OPTARG} ;;
    ?) echo "Usage: bash run.sh [-r cpu|sim|npu] [-v SOC]"; exit 1 ;;
esac; done

if [ -z "${ASCEND_HOME_PATH}" ]; then export ASCEND_HOME_PATH="/usr/local/Ascend/ascend-toolkit/latest"; fi
source ${ASCEND_HOME_PATH}/bin/setenv.bash 2>/dev/null || source ${ASCEND_HOME_PATH}/set_env.sh

_NR=${NR:-64}; _NL=${NL:-16}; _NLR=${NL_REAL:-${_NL}}; _RC=${RULE_CHECK:-1}
if [ -n "${BRI_B:-}" ]; then
    _B=${BRI_B}
elif [ "${_NLR}" -le 2 ]; then _B=1
elif [ "${_NLR}" -le 4 ]; then _B=2
elif [ "${_NLR}" -le 8 ]; then _B=4
else _B=8
fi
_PK=${PACK:-0}; _P=1
echo "==== mimo_detect_bri_batch  RUN_MODE=${RUN_MODE}  SOC=${SOC_VERSION}  (NR=${_NR} NL=${_NL} activeL=${_NLR} B=${_B} PACK=${_PK}) ===="
export AIRAN_DATA_DIR=${SCRIPT_DIR}

if { [ "${_NR}" -ne 16 ] && [ "${_NR}" -ne 32 ] && [ "${_NR}" -ne 64 ]; } ||
   [ "${_NL}" -ne 16 ] || [ "${_PK}" -ne 0 ]; then
    echo "[run.sh][contract] ERROR: io_pack ABI requires NR=16/32/64 NL=16 PACK=0" >&2
    exit 2
fi
if [ "${_NLR}" -lt 1 ] || [ "${_NLR}" -gt 16 ]; then
    echo "[run.sh][contract] ERROR: active NL_REAL must be in [1,16]" >&2
    exit 2
fi
if [ "${_B}" -lt 1 ] || [ $((16 % _B)) -ne 0 ]; then
    echo "[run.sh][contract] ERROR: BRI_B must be a positive divisor of 16" >&2
    exit 2
fi
_MIN_B_L=${_B}; [ "${_NLR}" -lt "${_B}" ] && _MIN_B_L=${_NLR}
if [ "${_NR}" -lt $((8 * _MIN_B_L)) ] || [ $((2 * _B)) -lt "${_NLR}" ]; then
    echo "[run.sh][contract] ERROR: require NR>=8*min(BRI_B,L) and 2*BRI_B>=L" >&2
    exit 2
fi
if [ "${_RC}" -ne 1 ]; then
    echo "[run.sh][contract] ERROR: RULE_CHECK cannot be disabled for the production ABI" >&2
    exit 2
fi
if [ -n "${AIRAN_SCPAD_SET}" ] && [ "${AIRAN_SCPAD_SET}" -ne 1664 ]; then
    echo "[run.sh][contract] ERROR: AIRAN_SCPAD_SET must be 1664 for io_pack ABI" >&2
    exit 2
fi
if [ "${AIRAN_DEBUG:-0}" -ne 0 ]; then
    echo "[run.sh][contract] ERROR: AIRAN_DEBUG changes the physical ABI and is not supported here" >&2
    exit 2
fi

echo "[run.sh] 维度: NR=${_NR} NL=${_NL} NL_REAL=${_NLR} BRI_B=${_B} PACK=${_PK}(P=${_P}) RULE_CHECK=${_RC} (CMake build variant)"
SCPAD=1664
export AIRAN_SCPAD=${SCPAD}

# CASE: NL_REAL < NL 时加 _r 后缀 (否则 32×8 会和 32×16 共用同一个 golden 目录)
CASE="case_0_m${_NR}_k${_NL}_sc${SCPAD}"
[ "${_NLR}" != "${_NL}" ] && CASE="${CASE}_r${_NLR}"

# BRI_B is not encoded in the historical case name, so regenerate the golden
# on every standalone run to prevent a stale preconditioner reference.
echo "[run.sh] 生成 golden ${CASE} ..."
NR=${_NR} NL=${_NL} NL_REAL=${_NLR} BRI_B=${_B} PACK=${_PK} AIRAN_SCPAD=${SCPAD} \
    AIRAN_DATA_DIR=${AIRAN_DATA_DIR}/data python3 ${SCRIPT_DIR}/scripts/mimo_detect_bri_ref.py

# 每次跑之前清空本 case 的输出目录 —— 让"读到上一轮别的配置留下的陈旧输出"
# 退化成缺文件报错, 而不是静默给出漂亮的错答案. (2026-07-26 的教训)
rm -rf ${AIRAN_DATA_DIR}/data/ascend_output/${CASE}
mkdir -p ${AIRAN_DATA_DIR}/data/ascend_output/${CASE}

rm -rf build out; mkdir -p build; cd build
cmake .. -DRUN_MODE=${RUN_MODE} -DSOC_VERSION=${SOC_VERSION} \
  -DASCEND_CANN_PACKAGE_PATH=${ASCEND_HOME_PATH} -DN_SC_PAD_VAL=${SCPAD} \
  -DDETECTOR_RX_CAPACITY=${_NR} -DDETECTOR_LAYER_CAPACITY=${_NL} \
  -DACTIVE_LAYERS=${_NLR} -DDETECTOR_BRI_BLOCK=${_B} -DDETECTOR_PACK=${_PK}
make -j; make install

cd ${SCRIPT_DIR}/out/bin
export LD_LIBRARY_PATH=${SCRIPT_DIR}/out/lib:${ASCEND_HOME_PATH}/lib64:${LD_LIBRARY_PATH}
echo "[run.sh] executing ..."; ./ascendc_kernels_bbit

echo "[run.sh] verifying ..."
NR=${_NR} NL=${_NL} NL_REAL=${_NLR} PACK=${_PK} AIRAN_SCPAD=${SCPAD} python3 ${SCRIPT_DIR}/scripts/verify_result.py
