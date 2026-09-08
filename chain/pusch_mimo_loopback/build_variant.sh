#!/usr/bin/env bash
# Sourced helper: expose one validated resolved-plan build variant to run scripts.
load_mimo_build_variant() {
  local requested_rank="$1"
  mimo_variant_name="legacy_rx64_layer16"
  mimo_ce_nr=64
  mimo_ce_nl="${requested_rank}"
  mimo_ce_retained_rank=96
  mimo_ce_block_dim=4
  mimo_ce_cube_time_fused=ON
  mimo_detector_rx_capacity=64
  mimo_detector_layer_capacity=16
  mimo_detector_bri_block=8
  if [[ -n "${PUSCH_MIMO_RESOLVED_PLAN:-}" ]]; then
    local here variant_values
    here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    mapfile -t variant_values < <(
      python3 "${here}/resolve_build_variant.py" \
        --plan "${PUSCH_MIMO_RESOLVED_PLAN}" --rank "${requested_rank}" --format lines
    )
    [[ "${#variant_values[@]}" -eq 9 ]] || {
      echo "[HARD_FAIL] invalid build-variant resolver output" >&2
      return 2
    }
    mimo_variant_name="${variant_values[0]}"
    mimo_ce_nr="${variant_values[1]}"
    mimo_ce_nl="${variant_values[2]}"
    mimo_ce_retained_rank="${variant_values[3]}"
    mimo_ce_block_dim="${variant_values[4]}"
    mimo_ce_cube_time_fused="${variant_values[5]}"
    mimo_detector_rx_capacity="${variant_values[6]}"
    mimo_detector_layer_capacity="${variant_values[7]}"
    mimo_detector_bri_block="${variant_values[8]}"
  fi
  [[ "${mimo_detector_rx_capacity}" -ge 16 && $((mimo_detector_rx_capacity % 16)) -eq 0 ]] || {
    echo "[HARD_FAIL] detector RX capacity must be a positive multiple of 16" >&2
    return 2
  }
  mimo_dmrs_ls_block_dim=$((mimo_detector_rx_capacity / 16))
  echo "[dispatch] ${mimo_variant_name}: CE=${mimo_ce_nr}x${mimo_ce_nl}/r${mimo_ce_retained_rank}, detector=${mimo_detector_rx_capacity}x${mimo_detector_layer_capacity}/b${mimo_detector_bri_block}"
}
