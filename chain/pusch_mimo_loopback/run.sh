#!/usr/bin/env bash
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
kernel_root="${PUSCH_MIMO_CLEAN_KERNEL_ROOT:-/home/refresh/AI-RAN-NPU-clean/kernels}"
selection="${1:-all}"
[[ "${selection}" =~ ^(all|[1-4])$ ]] || { echo "usage: $0 [all|1|2|3|4]" >&2; exit 2; }
radio_profile="${PUSCH_MIMO_RADIO_PROFILE:-legacy}"
channel_gain_overridden=0
[[ -v PUSCH_MIMO_CHANNEL_GAIN ]] && channel_gain_overridden=1
channel_noise_overridden=0
[[ -v PUSCH_MIMO_CHANNEL_NOISE_STD ]] && channel_noise_overridden=1
# A run must never inherit a plan compiled for a previous profile or Rank.
unset PUSCH_MIMO_RESOLVED_PLAN PUSCH_MIMO_RUNTIME_CONFIG
evidence_suffix=""
if [[ "${radio_profile}" == "fd8x8_rank1_4" ]]; then
  python3 "${here}/tests/test_radio_profile.py"
  evidence_suffix="_fd8x8"
elif [[ "${radio_profile}" != "legacy" ]]; then
  echo "[HARD_FAIL] unsupported PUSCH_MIMO_RADIO_PROFILE=${radio_profile}" >&2
  exit 2
fi

resolve_rank_profile() {
  local rank="$1"
  if [[ "${radio_profile}" == "legacy" ]]; then
    return
  fi
  local profile_path="${PUSCH_MIMO_PROFILE_PATH:-${kernel_root}/common/profiles/${radio_profile}.json}"
  local compiler="${kernel_root}/common/mimo_profile_compiler.py"
  local capabilities="${kernel_root}/common/mimo_operator_capabilities.json"
  local plan_dir="${here}/work/resolved_plans"
  local plan_path="${plan_dir}/${radio_profile}_rank${rank}.json"
  local runtime_path="${plan_dir}/${radio_profile}_rank${rank}.bin"
  [[ -f "${profile_path}" && -f "${compiler}" && -f "${capabilities}" ]] || {
    echo "[HARD_FAIL] common profile/compiler/capabilities missing" >&2
    exit 2
  }
  mkdir -p "${plan_dir}"
  python3 "${compiler}" --profile "${profile_path}" --rank "${rank}" \
    --capabilities "${capabilities}" --require-eligible \
    --output "${plan_path}" --binary-output "${runtime_path}"
  export PUSCH_MIMO_RESOLVED_PLAN="${plan_path}"
  export PUSCH_MIMO_RUNTIME_CONFIG="${runtime_path}"
  if [[ "${channel_gain_overridden}" == "0" ]]; then
    PUSCH_MIMO_CHANNEL_GAIN="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["channel"]["gain"])' "${plan_path}")"
    export PUSCH_MIMO_CHANNEL_GAIN
  fi
  if [[ "${channel_noise_overridden}" == "0" ]]; then
    PUSCH_MIMO_CHANNEL_NOISE_STD="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["channel"]["awgn_std_int16"])' "${plan_path}")"
    export PUSCH_MIMO_CHANNEL_NOISE_STD
  fi
}
[[ -f "${kernel_root}/common/pusch_mimo_types.h" ]] || {
  echo "[HARD_FAIL] canonical ABI missing: ${kernel_root}/common/pusch_mimo_types.h" >&2; exit 2; }

bash "${here}/build_waveform_ops.sh"

# Every rank consumes the same actual Sionna-validated LDPC code blocks.
bash "${here}/run_tx_prefix_rank1.sh"

if [[ "${selection}" == "all" || "${selection}" == "1" ]]; then
  resolve_rank_profile 1
  bash "${here}/run_dmrs_rank1.sh"
  bash "${here}/run_grid_map_rank1.sh"
  bash "${here}/run_precode_bypass_rank1.sh"
  bash "${here}/run_coded_waveform_rank1.sh"
  bash "${here}/run_dmrs_ls_rank1.sh"
  bash "${here}/run_channel_est_rank1_slot0.sh"
  bash "${here}/run_detect_rank1_slot0.sh"
  bash "${here}/run_demod_rank1_slot0.sh"
  bash "${here}/run_rx_front_rank1_23slot.sh"
  bash "${here}/run_post_fec_rank1.sh"
  python3 "${here}/publish_evidence.py" --root "${here}/work/tx_prefix_rank1" --rank 1 \
    --output "${here}/evidence/rank1_23slot_e2e${evidence_suffix}.json"
fi

for rank in 2 3 4; do
  if [[ "${selection}" != "all" && "${selection}" != "${rank}" ]]; then continue; fi
  resolve_rank_profile "${rank}"
  bash "${here}/run_tx_rank.sh" "${rank}"
  bash "${here}/run_tx_grid_rank.sh" "${rank}"
  bash "${here}/run_coded_waveform_rank2.sh" "${rank}"
  bash "${here}/run_dmrs_ls_rank2.sh" "${rank}"
  bash "${here}/run_rx_rank2_slot0.sh" "${rank}"
  bash "${here}/run_rx_rank2_23slot.sh" "${rank}"
  bash "${here}/run_post_fec_rank2.sh" "${rank}"
  python3 "${here}/publish_evidence.py" --root "${here}/work/tx_prefix_rank1" --rank "${rank}" \
    --output "${here}/evidence/rank${rank}_23slot_e2e${evidence_suffix}.json"
done

KERNEL_ROOT="${kernel_root}" bash "${here}/../pusch_mimo_tx_chain/run.sh"
KERNEL_ROOT="${kernel_root}" bash "${here}/../pusch_mimo_rx_chain/run.sh"
echo "[E2E PASS] PUSCH MIMO ${selection}: radio=${radio_profile} 23-slot actual-device coded loopback"
