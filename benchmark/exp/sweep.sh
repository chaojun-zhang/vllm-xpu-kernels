#!/bin/bash
# Usage: PY=/path/python ZE_AFFINITY_MASK=0,1,2,3 bash sweep.sh
# Needs vllm (branch xpu/direct-dcp-a2a) importable by $PY.
set -u
PY=${PY:-python}
DIR=$(cd "$(dirname "$0")" && pwd)
export TOKENS=${TOKENS:-1,2,4,8,16,32,64,128}
for world in 2 4; do
  for sig in 0 1; do
    for groups in 128 64 32 16; do
      echo "### world=$world SIGNAL_IN_DISPATCH=$sig TARGET_GROUPS=$groups"
      DCP_A2A_SIGNAL_IN_DISPATCH=$sig DCP_A2A_TARGET_GROUPS=$groups \
        timeout 600 "$PY" "$DIR/bench_a2a.py" "$world" graph 2>&1 | grep -E '^\|'
    done
  done
done
