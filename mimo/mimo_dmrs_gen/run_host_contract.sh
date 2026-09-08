#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")"; pwd)
TEST_BIN=$(mktemp /tmp/mimo_dmrs_gen_host_contract.XXXXXX)
trap 'rm -f "$TEST_BIN"' EXIT

g++ -std=c++17 -O2 -Wall -Wextra -Werror \
    -I"$SCRIPT_DIR" \
    "$SCRIPT_DIR/scripts/host_contract_test.cpp" \
    "$SCRIPT_DIR/mimo_dmrs_gen.cpp" \
    -o "$TEST_BIN"
"$TEST_BIN"
