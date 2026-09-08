#!/usr/bin/env bash
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
kernel_root="${KERNEL_ROOT:-$(cd "${here}/../.." && pwd)}"

PYTHONDONTWRITEBYTECODE=1 python3 "${here}/../pusch_mimo_contract_harness.py" \
  "${here}/contract_matrix.json" --kernel-root "${kernel_root}" --mode check
PYTHONDONTWRITEBYTECODE=1 python3 "${here}/tests/test_contract_harness.py"
