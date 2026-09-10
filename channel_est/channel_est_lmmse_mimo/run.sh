#!/usr/bin/env bash
set -euo pipefail

CURRENT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "${CURRENT_DIR}"
RUN_ROOT=${AIRAN_RUN_ROOT:-${CURRENT_DIR}}

export ASCEND_HOME_DIR=${ASCEND_HOME_DIR:-/usr/local/Ascend/ascend-toolkit/latest}
# CANN's environment scripts contain optional-variable reads and condition
# probes that are not safe under the caller's errexit/nounset settings.
set +eu
if [[ -f "${ASCEND_HOME_DIR}/bin/setenv.bash" ]]; then
    # shellcheck disable=SC1090
    source "${ASCEND_HOME_DIR}/bin/setenv.bash"
else
    # shellcheck disable=SC1090
    source "${ASCEND_HOME_DIR}/../set_env.sh"
fi
set -eu

export NR=${NR:-64}
export NL=${NL:-2}
export RANK=${RANK:-96}
export BLOCK_DIM=${BLOCK_DIM:-4}
export AIRAN_DATA_DIR=${AIRAN_DATA_DIR:-${RUN_ROOT}/data}
SOC_VERSION=${SOC_VERSION:-Ascend310P1}
# CE64_* aliases keep old experiment commands usable after promotion.
CE_CUBE_TIME_FUSED=${CE_CUBE_TIME_FUSED:-${CE64_CUBE_TIME_FUSED:-OFF}}
CE_CUBE_TIME_POST_GEMM=${CE_CUBE_TIME_POST_GEMM:-${CE64_CUBE_TIME_POST_GEMM:-OFF}}
case "${NR}" in 8|16|32|64) ;; *) echo "NR must be one of 8,16,32,64" >&2; exit 2 ;; esac
case "${NL}" in 1|2|3|4) ;; *) echo "NL must be one of 1,2,3,4" >&2; exit 2 ;; esac
if (( RANK < 16 || RANK > 128 || RANK % 16 != 0 )); then
    echo "RANK must be 16-aligned in [16,128]" >&2
    exit 2
fi
case "${BLOCK_DIM}" in 1|2|4) ;; *) echo "BLOCK_DIM must be 1, 2, or 4" >&2; exit 2 ;; esac
CASE="case_0_m${NR}_k${NL}_r${RANK}"
GOLD="${AIRAN_DATA_DIR}/golden/${CASE}"
ASC="${AIRAN_DATA_DIR}/ascend_output/${CASE}"

if [[ "${CE_CUBE_TIME_FUSED}" == "ON" && "${CE_CUBE_TIME_POST_GEMM}" == "ON" ]]; then
    echo "CE_CUBE_TIME_FUSED and CE_CUBE_TIME_POST_GEMM are mutually exclusive" >&2
    exit 2
elif [[ "${CE_CUBE_TIME_POST_GEMM}" == "ON" ]]; then
    if [[ "${NR}" == "8" ]]; then
        echo "post-frequency Cube requires NR=16,32,64; use the default path for NR=8" >&2
        exit 2
    fi
    if [[ "${NL}" == "1" ]]; then
        echo "post-frequency Cube uses a two-layer batch; Rank1 must use the default Vector path" >&2
        exit 2
    fi
    if [[ "${NL}" == "3" ]]; then
        echo "post-frequency Cube uses a two-layer batch; Rank3 must use the default Vector path" >&2
        exit 2
    fi
    BUILD_DIR="${AIRAN_BUILD_DIR:-${RUN_ROOT}/build_${CASE}_post}"
    OUT_DIR="${AIRAN_OUT_DIR:-${RUN_ROOT}/out_${CASE}_post}"
    REQUIRED_INPUT="${GOLD}/cube_time_post_matrix.bin"
    VERIFY_ARGS=(--cube-time-post)
elif [[ "${CE_CUBE_TIME_FUSED}" == "ON" ]]; then
    BUILD_DIR="${AIRAN_BUILD_DIR:-${RUN_ROOT}/build_${CASE}_fused}"
    OUT_DIR="${AIRAN_OUT_DIR:-${RUN_ROOT}/out_${CASE}_fused}"
    REQUIRED_INPUT="${GOLD}/cube_time_fused_d1_im.bin"
    VERIFY_ARGS=(--cube-time-fused)
else
    BUILD_DIR="${AIRAN_BUILD_DIR:-${RUN_ROOT}/build_${CASE}_vector}"
    OUT_DIR="${AIRAN_OUT_DIR:-${RUN_ROOT}/out_${CASE}_vector}"
    REQUIRED_INPUT="${GOLD}/gold_h_re.bin"
    VERIFY_ARGS=()
fi

printf '[config] NR=%s NL=%s RANK=%s blockDim=%s case=%s\n' \
    "${NR}" "${NL}" "${RANK}" "${BLOCK_DIM}" "${CASE}"

PYTHONDONTWRITEBYTECODE=1 python3 "${CURRENT_DIR}/scripts/observation_model_ref.py" --self-test
PYTHONDONTWRITEBYTECODE=1 python3 "${CURRENT_DIR}/scripts/channel_est_lmmse_ref.py"
mkdir -p "${ASC}"

rm -rf "${BUILD_DIR}" "${OUT_DIR}"
cmake -S "${CURRENT_DIR}" -B "${BUILD_DIR}" \
    -DRUN_MODE=npu -DSOC_VERSION="${SOC_VERSION}" \
    -DCE_NR="${NR}" -DCE_NL="${NL}" -DCE_RANK="${RANK}" \
    -DCE_BLOCK_DIM="${BLOCK_DIM}" \
    -DCE_CUBE_TIME_FUSED="${CE_CUBE_TIME_FUSED}" \
    -DCE_CUBE_TIME_POST_GEMM="${CE_CUBE_TIME_POST_GEMM}" \
    -DASCEND_CANN_PACKAGE_PATH="${ASCEND_HOME_DIR}" \
    -DCMAKE_INSTALL_PREFIX="${OUT_DIR}"
cmake --build "${BUILD_DIR}" -j"$(nproc)"
cmake --install "${BUILD_DIR}"
ctest --test-dir "${BUILD_DIR}" --output-on-failure

export LD_LIBRARY_PATH="${OUT_DIR}/lib:${ASCEND_HOME_DIR}/lib64:${LD_LIBRARY_PATH:-}"
"${OUT_DIR}/bin/ascendc_kernels_bbit"
PYTHONDONTWRITEBYTECODE=1 python3 "${CURRENT_DIR}/scripts/channel_est_lmmse_verify.py" "${VERIFY_ARGS[@]}"
