#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
chains=(
  "${root}/chain/pusch_siso_tx_chain/CMakeLists.txt"
  "${root}/chain/pusch_siso_rx_chain/CMakeLists.txt"
)

if find "${root}" -type d \( -iname experiment -o -iname experiments -o -iname '*override*' \) -print -quit | grep -q .; then
  echo "clean-tree check failed: experimental directory exists under ${root}" >&2
  exit 1
fi

if find "${root}" -type d \( -name build -o -name out -o -name __pycache__ \) \
    -print -quit | grep -q . ||
   find "${root}" -type f \( -name '*.pyc' -o -name '*.o' -o -name '*.so' \
       -o -name '*.a' -o -name '*.bin' -o -name '*.npy' -o -name '*.npz' \
       -o -name '*.zip' -o -name 'CMakeCache.txt' \) \
    -print -quit | grep -q .; then
    echo "clean-tree check failed: generated build/cache artifact exists under ${root}" >&2
    exit 1
fi

if find "${root}" -mindepth 2 -type d \
    \( -name data -o -name weights -o -name auto_gen -o -name profiling \
       -o -name prof -o -name prof_out -o -name CMakeFiles \) \
    -print -quit | grep -q .; then
  echo "clean-tree check failed: generated data/weight/build directory exists under ${root}" >&2
  exit 1
fi

if grep -RIE --include='CMakeLists.txt' --include='*.cmake' \
    'AI-RAN-NPU/kernels/|experiments/|experiment_overrides|/overrides/' \
    "${root}" >/dev/null; then
  echo "clean-tree check failed: build metadata references a non-clean/experimental source" >&2
  exit 1
fi

if grep -E 'file\([[:space:]]*GLOB(_RECURSE)?' "${chains[@]}" >/dev/null; then
  echo "clean-tree check failed: production chain CMake uses a source glob" >&2
  exit 1
fi

echo "CLEAN_TREE_CHECK PASS"
