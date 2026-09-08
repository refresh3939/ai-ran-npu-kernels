#!/bin/bash
# rx_chain —— 整链 host 编排 (Ascend 310P1, blockDim=4)
# 默认跑 profile(计时);正常解码跑 RX_PROFILE=0 bash run.sh
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# (1) env —— 找得到 set_env.sh 就 source,找不到就假设 shell 里已 source(conda base 常已 source)
: "${ASCEND_HOME_PATH:=/usr/local/Ascend/ascend-toolkit/latest}"
for c in "${ASCEND_HOME_PATH}/set_env.sh" \
         /usr/local/Ascend/ascend-toolkit/set_env.sh \
         "${ASCEND_TOOLKIT_HOME}/set_env.sh" \
         "$HOME/Ascend/ascend-toolkit/set_env.sh"; do
    if [ -f "$c" ]; then echo "[env] source $c"; source "$c"; break; fi
done
: "${ASCEND_HOME_PATH:=/usr/local/Ascend/ascend-toolkit/latest}"

export AIRAN_DATA_DIR="${SCRIPT_DIR}/data"   # data/weights/*.bin, data/capture.bin ...

# (1.5) 强制重新生成 tiling shim(先彻底清,再 gen;杜绝旧产物残留)
rm -rf "${SCRIPT_DIR}/tiling_shims"
bash "${SCRIPT_DIR}/gen_tiling_shims.sh"
echo "[run] tiling_shims regenerated; sss should be absent:"
ls "${SCRIPT_DIR}/tiling_shims" | grep -i sss && { echo "!! 还有 sss shim -> gen 是旧版,从 outputs 重新下载 gen_tiling_shims.sh"; exit 1; } || echo "  (no sss shim, OK)"

# (2) clean rebuild + install
rm -rf build out
mkdir build && cd build
cmake .. \
    -DASCEND_CANN_PACKAGE_PATH="${ASCEND_HOME_PATH}" \
    -DSOC_VERSION=Ascend310P1 \
    -DRUN_MODE=npu \
    -DCMAKE_INSTALL_PREFIX="${SCRIPT_DIR}/out"
make -j
make install
cd "${SCRIPT_DIR}"

# (3) run —— 默认开 profile;RX_PROFILE=0 bash run.sh 跑正常解码
: "${RX_PROFILE:=1}"; export RX_PROFILE
cd out/bin
LD_LIBRARY_PATH="${SCRIPT_DIR}/out/lib:${ASCEND_HOME_PATH}/lib64:${LD_LIBRARY_PATH}" ./rx_chain "$@"
