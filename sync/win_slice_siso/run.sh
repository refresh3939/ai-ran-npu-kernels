#!/bin/bash
# win_slice 独立测试:gen golden -> build -> run -> verify
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

: "${ASCEND_HOME_PATH:=/usr/local/Ascend/ascend-toolkit/latest}"
for c in "${ASCEND_HOME_PATH}/set_env.sh" /usr/local/Ascend/ascend-toolkit/set_env.sh \
         "${ASCEND_TOOLKIT_HOME}/set_env.sh" "$HOME/Ascend/ascend-toolkit/set_env.sh"; do
    if [ -f "$c" ]; then echo "[env] source $c"; source "$c"; break; fi
done
: "${ASCEND_HOME_PATH:=/usr/local/Ascend/ascend-toolkit/latest}"
export AIRAN_DATA_DIR="${SCRIPT_DIR}/data"

# (1) golden
python3 scripts/ref_win_slice.py

# (2) build
rm -rf build out
mkdir build && cd build
cmake .. -DASCEND_CANN_PACKAGE_PATH="${ASCEND_HOME_PATH}" -DSOC_VERSION=Ascend310P3 \
         -DRUN_MODE=npu -DCMAKE_INSTALL_PREFIX="${SCRIPT_DIR}/out"
make -j && make install
cd "${SCRIPT_DIR}"

# (3) run
cd out/bin
LD_LIBRARY_PATH="${SCRIPT_DIR}/out/lib:${ASCEND_HOME_PATH}/lib64:${LD_LIBRARY_PATH}" ./win_slice 5
cd "${SCRIPT_DIR}"

# (4) verify 5/5
python3 scripts/verify_win_slice.py 5
