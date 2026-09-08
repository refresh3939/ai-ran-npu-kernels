#!/usr/bin/env bash
set -euo pipefail

SOURCE_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
RUN_MODE=npu
SOC_VERSION=Ascend310P1
while getopts ":r:v:" opt; do
    case "${opt}" in
        r) RUN_MODE=${OPTARG} ;;
        v) SOC_VERSION=${OPTARG} ;;
        *) echo "Usage: bash run.sh [-r npu|sim] [-v Ascend310P1|Ascend310P3]"; exit 2 ;;
    esac
done

ASCEND_ROOT=${ASCEND_HOME_DIR:-${ASCEND_HOME_PATH:-/usr/local/Ascend/ascend-toolkit/latest}}
if [[ ! -d "${ASCEND_ROOT}" ]]; then
    echo "[FAIL] Ascend toolkit not found: ${ASCEND_ROOT}" >&2
    exit 1
fi
set +eu
if [[ -f "${ASCEND_ROOT}/bin/setenv.bash" ]]; then
    source "${ASCEND_ROOT}/bin/setenv.bash"
else
    source "${ASCEND_ROOT}/set_env.sh"
fi
set -eu

RUN_ROOT=$(mktemp -d "${TMPDIR:-/tmp}/rate_match_mimo.XXXXXX")
cleanup() {
    case "${RUN_ROOT}" in
        /tmp/rate_match_mimo.*) rm -rf -- "${RUN_ROOT}" ;;
        *) echo "[WARN] refusing unexpected cleanup target: ${RUN_ROOT}" >&2 ;;
    esac
}
trap cleanup EXIT

export AIRAN_DATA_DIR="${RUN_ROOT}/data"
export PYTHONDONTWRITEBYTECODE=1
python3 "${SOURCE_DIR}/scripts/gen_data.py"

g++ -std=c++17 -O2 -Wall -Wextra -Werror \
    -DRATE_MATCH_MIMO_HOST_ONLY -I"${SOURCE_DIR}" \
    "${SOURCE_DIR}/tests/host_contract_test.cpp" \
    "${SOURCE_DIR}/rate_match_mimo.cpp" \
    -o "${RUN_ROOT}/host_contract_test"
"${RUN_ROOT}/host_contract_test"
echo "[PASS] host ABI/reference contract"

cmake -S "${SOURCE_DIR}" -B "${RUN_ROOT}/build" \
    -DRUN_MODE="${RUN_MODE}" \
    -DSOC_VERSION="${SOC_VERSION}" \
    -DASCEND_CANN_PACKAGE_PATH="${ASCEND_ROOT}" \
    -DCMAKE_INSTALL_PREFIX="${RUN_ROOT}/out"
cmake --build "${RUN_ROOT}/build" -j"$(nproc)"
cmake --install "${RUN_ROOT}/build"

export LD_LIBRARY_PATH="${RUN_ROOT}/out/lib:${ASCEND_ROOT}/lib64:${LD_LIBRARY_PATH:-}"
"${RUN_ROOT}/out/bin/ascendc_kernels_bbit"
python3 "${SOURCE_DIR}/scripts/verify_result.py"
