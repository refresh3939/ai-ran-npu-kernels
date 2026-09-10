#!/usr/bin/env bash
set -euo pipefail

CURRENT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "${CURRENT_DIR}"
RUN_ROOT=${AIRAN_RUN_ROOT:-${CURRENT_DIR}}
BUILD_DIR=${AIRAN_BUILD_DIR:-${RUN_ROOT}/build}
OUT_DIR=${AIRAN_OUT_DIR:-${RUN_ROOT}/out}
export ASCEND_HOME_DIR=${ASCEND_HOME_DIR:-/usr/local/Ascend/ascend-toolkit/latest}
set +eu
if [[ -f "${ASCEND_HOME_DIR}/bin/setenv.bash" ]]; then
    source "${ASCEND_HOME_DIR}/bin/setenv.bash"
else
    source "${ASCEND_HOME_DIR}/../set_env.sh"
fi
set -eu

export AIRAN_DATA_DIR=${AIRAN_DATA_DIR:-${RUN_ROOT}/data}
SOC_VERSION=${SOC_VERSION:-Ascend310P1}
IO_PACK_RX_CAPACITY=${IO_PACK_RX_CAPACITY:-64}
IO_PACK_LAYER_CAPACITY=${IO_PACK_LAYER_CAPACITY:-16}
[[ "${IO_PACK_RX_CAPACITY}" =~ ^(16|32|64)$ ]] || {
    echo "[HARD_FAIL] IO_PACK_RX_CAPACITY must be 16, 32, or 64" >&2; exit 2; }
[[ "${IO_PACK_LAYER_CAPACITY}" == 16 ]] || {
    echo "[HARD_FAIL] IO_PACK_LAYER_CAPACITY must be 16" >&2; exit 2; }
GEN_ARGS=()
if [[ -n "${AIRAN_UPSTREAM_H_DIR:-}" ]]; then
    GEN_ARGS+=(--upstream-h-dir "${AIRAN_UPSTREAM_H_DIR}")
fi
IO_PACK_RX_CAPACITY="${IO_PACK_RX_CAPACITY}" \
IO_PACK_LAYER_CAPACITY="${IO_PACK_LAYER_CAPACITY}" \
python3 "${CURRENT_DIR}/scripts/gen_data.py" "${GEN_ARGS[@]}"
mkdir -p "${AIRAN_DATA_DIR}/ascend_output"

rm -rf "${BUILD_DIR}" "${OUT_DIR}"
cmake -S "${CURRENT_DIR}" -B "${BUILD_DIR}" \
    -DRUN_MODE=npu -DSOC_VERSION="${SOC_VERSION}" \
    -DASCEND_CANN_PACKAGE_PATH="${ASCEND_HOME_DIR}" \
    -DIO_PACK_RX_CAPACITY="${IO_PACK_RX_CAPACITY}" \
    -DIO_PACK_LAYER_CAPACITY="${IO_PACK_LAYER_CAPACITY}" \
    -DCMAKE_INSTALL_PREFIX="${OUT_DIR}"
cmake --build "${BUILD_DIR}" -j"$(nproc)"
cmake --install "${BUILD_DIR}"
export LD_LIBRARY_PATH="${OUT_DIR}/lib:${ASCEND_HOME_DIR}/lib64:${LD_LIBRARY_PATH:-}"
"${OUT_DIR}/bin/ascendc_kernels_bbit"
IO_PACK_RX_CAPACITY="${IO_PACK_RX_CAPACITY}" \
IO_PACK_LAYER_CAPACITY="${IO_PACK_LAYER_CAPACITY}" \
python3 "${CURRENT_DIR}/scripts/verify_result.py"
