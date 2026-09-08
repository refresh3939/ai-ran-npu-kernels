#!/usr/bin/env bash
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")"&&pwd)";src="${PUSCH_MIMO_CLEAN_KERNEL_ROOT:-/home/refresh/AI-RAN-NPU-clean/kernels}/channel_est/channel_est_lmmse_mimo";ascend_home="${ASCEND_HOME_PATH:-/usr/local/Ascend/ascend-toolkit/latest}"
source "${here}/build_variant.sh";load_mimo_build_variant 1
build="${here}/build/channel_est_lmmse_rank1";out="${here}/out/channel_est_lmmse_rank1";work="${here}/work/tx_prefix_rank1";data="${work}/channel_est";logs="${here}/logs/channel_est_rank1"
[[ -f "${work}/artifacts/dmrs_ls_result.log" ]]||{ echo "[HARD_FAIL] DMRS-LS must pass first" >&2;exit 2;}
for x in libascendcl.so libplatform.so libregister.so;do [[ -r "${ascend_home}/lib64/${x}" ]]||exit 2;done
set +e;set +u;if [[ -f "${ascend_home}/bin/setenv.bash" ]];then source "${ascend_home}/bin/setenv.bash";else source "${ascend_home}/set_env.sh";fi;set -u;set -e
rm -rf "${logs}" "${data}" "${work}/artifacts/channel_est";mkdir -p "${logs}" "${work}/artifacts/channel_est"
PYTHONDONTWRITEBYTECODE=1 NR="${mimo_ce_nr}" NL="${mimo_ce_nl}" RANK="${mimo_ce_retained_rank}" AIRAN_DATA_DIR="${data}" python3 "${src}/scripts/channel_est_lmmse_ref.py" >"${logs}/generate_weights.log" 2>&1
python3 "${here}/ce_rank.py" prepare --root "${data}" --rank 1 --nr "${mimo_ce_nr}" --ce-rank "${mimo_ce_retained_rank}" --layer-capacity "${mimo_detector_layer_capacity}"
python3 "${here}/ce_rank.py" link --root "${data}" --ls "${work}/artifacts/dmrs_ls" --slot 0 --rank 1 --nr "${mimo_ce_nr}" --ce-rank "${mimo_ce_retained_rank}" --layer-capacity "${mimo_detector_layer_capacity}"
mkdir -p "${data}/ascend_output/case_0_m${mimo_ce_nr}_k1_r${mimo_ce_retained_rank}"
cmake -S "${src}" -B "${build}" -DRUN_MODE=npu -DSOC_VERSION=Ascend310P1 -DCE_NR="${mimo_ce_nr}" -DCE_NL="${mimo_ce_nl}" -DCE_RANK="${mimo_ce_retained_rank}" -DCE_BLOCK_DIM="${mimo_ce_block_dim}" -DCE_CUBE_TIME_FUSED="${mimo_ce_cube_time_fused}" -DASCEND_CANN_PACKAGE_PATH="${ascend_home}" -DCMAKE_CXX_FLAGS="-I${here}/shims" -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="${out}" >"${logs}/build.log" 2>&1
cmake --build "${build}" -j "$(nproc)" >>"${logs}/build.log" 2>&1;cmake --install "${build}" >>"${logs}/build.log" 2>&1
set +e;AIRAN_DATA_DIR="${data}" WARMUP=0 TIMED=1 LD_LIBRARY_PATH="${out}/lib:${ascend_home}/lib64:${LD_LIBRARY_PATH:-}" "${out}/bin/ascendc_kernels_bbit" >"${logs}/slot00.log" 2>&1;status=$?;set -e
if [[ $status -ne 0 ]];then if grep -Eq '507008|drvRet=4' "${logs}/slot00.log";then echo "[DEVICE_BUSY] CE; not retried" >&2;exit 75;fi;tail -100 "${logs}/build.log" >&2;tail -100 "${logs}/slot00.log" >&2;echo "[HARD_FAIL] CE slot0 ${status}" >&2;exit "$status";fi
python3 "${here}/ce_rank.py" collect --root "${data}" --out "${work}/artifacts/channel_est" --slot 0 --rank 1 --nr "${mimo_ce_nr}" --ce-rank "${mimo_ce_retained_rank}" --layer-capacity "${mimo_detector_layer_capacity}"
cp "${data}/weight_manifest.json" "${work}/artifacts/channel_est_weight_manifest.json";cat "${logs}/slot00.log"
