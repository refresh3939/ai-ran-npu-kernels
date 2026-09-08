#!/usr/bin/env bash
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
selection="${1:-all}"
[[ "${selection}" =~ ^(all|[1-4])$ ]] || {
  echo "usage: $0 [all|1|2|3|4]" >&2
  exit 2
}

export PUSCH_MIMO_RADIO_PROFILE=fd8x8_rank1_4
# run.sh compiles the common profile into one ResolvedPlan per active Rank.
# Explicit caller gain/noise environment variables remain supported overrides.
python3 "${here}/tests/test_radio_profile.py"
exec bash "${here}/run.sh" "${selection}"
