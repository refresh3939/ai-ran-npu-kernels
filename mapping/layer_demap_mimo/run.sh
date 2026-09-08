#!/usr/bin/env bash
set -euo pipefail

CURRENT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "${CURRENT_DIR}"
export ASCEND_HOME_DIR=${ASCEND_HOME_DIR:-/usr/local/Ascend/ascend-toolkit/latest}
set +eu
if [[ -f "${ASCEND_HOME_DIR}/bin/setenv.bash" ]]; then
    source "${ASCEND_HOME_DIR}/bin/setenv.bash"
else
    source "${ASCEND_HOME_DIR}/../set_env.sh"
fi
set -eu

export AIRAN_DATA_DIR=${AIRAN_DATA_DIR:-${CURRENT_DIR}/data}
SOC_VERSION=${SOC_VERSION:-Ascend310P1}
python3 "${CURRENT_DIR}/scripts/gen_data.py"
mkdir -p "${AIRAN_DATA_DIR}/ascend_output"

rm -rf "${CURRENT_DIR}/build" "${CURRENT_DIR}/out"
cmake -S "${CURRENT_DIR}" -B "${CURRENT_DIR}/build" \
    -DRUN_MODE=npu -DSOC_VERSION="${SOC_VERSION}" \
    -DASCEND_CANN_PACKAGE_PATH="${ASCEND_HOME_DIR}" \
    -DCMAKE_INSTALL_PREFIX="${CURRENT_DIR}/out"
cmake --build "${CURRENT_DIR}/build" -j"$(nproc)"
cmake --install "${CURRENT_DIR}/build"
export LD_LIBRARY_PATH="${CURRENT_DIR}/out/lib:${ASCEND_HOME_DIR}/lib64:${LD_LIBRARY_PATH:-}"
"${CURRENT_DIR}/out/bin/layer_demap_contract_test"
if [[ "${HOST_ONLY:-0}" == "1" ]]; then
    echo "HOST_ONLY=1: AscendC build and host contract complete; skipping device run"
    exit 0
fi
"${CURRENT_DIR}/out/bin/ascendc_kernels_bbit"
python3 "${CURRENT_DIR}/scripts/verify_result.py"
