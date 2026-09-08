#!/bin/bash
set -e

SCRIPT_DIR=$(cd "$(dirname "$0")"; pwd)
cd "${SCRIPT_DIR}"

RUN_MODE="npu"
SOC_VERSION="Ascend310P3"
while getopts ":r:v:" opt; do
    case ${opt} in
        r) RUN_MODE=${OPTARG} ;;
        v) SOC_VERSION=${OPTARG} ;;
        ?) echo "Usage: bash run.sh [-r cpu|sim|npu] [-v SOC_VERSION]"; exit 1 ;;
    esac
done

g++ -O2 -std=c++17 -Wall -Wextra -Werror \
    tests/test_codebook.cpp pusch_codebook_precode.cpp -o /tmp/pusch_codebook_precode_host_test
/tmp/pusch_codebook_precode_host_test
g++ -O2 -std=c++17 -Wall -Wextra -Werror -DASCENDC_CPU_DEBUG \
    tests/test_adjacent_contract.cpp pusch_codebook_precode.cpp \
    ../../mapping/mimo_resource_grid_map/mimo_resource_grid_map.cpp \
    ../../mapping/re_map_batch/re_map_batch.cpp \
    -o /tmp/pusch_codebook_precode_adjacent_contract_test
/tmp/pusch_codebook_precode_adjacent_contract_test
python3 scripts/test_mimo_detect_chain.py

if [ -z "${ASCEND_HOME_PATH}" ]; then
    export ASCEND_HOME_PATH="/usr/local/Ascend/ascend-toolkit/latest"
fi
source "${ASCEND_HOME_PATH}/bin/setenv.bash" 2>/dev/null || \
source "${ASCEND_HOME_PATH}/set_env.sh"
export ASCEND_GLOBAL_LOG_LEVEL=3
export ASCEND_SLOG_PRINT_TO_STDOUT=0

rm -rf build out
mkdir -p build
cd build
cmake .. -DRUN_MODE="${RUN_MODE}" -DSOC_VERSION="${SOC_VERSION}" \
    -DASCEND_CANN_PACKAGE_PATH="${ASCEND_HOME_PATH}"
make -j
make install

cd "${SCRIPT_DIR}/out/bin"
export LD_LIBRARY_PATH="${SCRIPT_DIR}/out/lib:${ASCEND_HOME_PATH}/lib64:${LD_LIBRARY_PATH}"
./ascendc_kernels_bbit
