#!/bin/bash
# merge_dmrs — TX DMRS comb-2 insertion (host op): golden + build + run + verify
set -e
SCRIPT_DIR=$(cd $(dirname $0); pwd); cd ${SCRIPT_DIR}

if [ -z "${ASCEND_HOME_PATH}" ]; then
    export ASCEND_HOME_PATH="/usr/local/Ascend/ascend-toolkit/latest"
fi
source ${ASCEND_HOME_PATH}/bin/setenv.bash 2>/dev/null || \
source ${ASCEND_HOME_PATH}/set_env.sh

export AIRAN_DATA_DIR=${SCRIPT_DIR}
echo "[run.sh] merge_dmrs  AIRAN_DATA_DIR=${AIRAN_DATA_DIR}"

# §0 golden(可用 DMRS_GEN_GOLDEN=<dmrs_gen case dir> 注入真导频)
if [ ! -f "${AIRAN_DATA_DIR}/data/golden/case_0/gu_merged_re.bin" ]; then
    echo "[run.sh] generating golden via merge_dmrs_ref.py ..."
    python3 ${SCRIPT_DIR}/scripts/merge_dmrs_ref.py
fi

rm -rf build && mkdir build && cd build
cmake .. -DASCEND_HOME_PATH=${ASCEND_HOME_PATH}
make -j
cd ${SCRIPT_DIR}

export LD_LIBRARY_PATH=${ASCEND_HOME_PATH}/lib64:${LD_LIBRARY_PATH}
echo ""; echo "[run.sh] executing ..."
./build/merge_dmrs_bbit
