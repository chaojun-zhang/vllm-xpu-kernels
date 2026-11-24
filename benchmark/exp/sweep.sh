#!/bin/bash
# Usage: PY=/path/python ZE_AFFINITY_MASK=0,1,2,3 bash sweep.sh
# TARGET_GROUPS=128 reproduces 2e87fce exactly.
set -u
PY=${PY:-python}
DIR=$(cd "$(dirname "$0")" && pwd)
export TOKENS=${TOKENS:-1,2,4,8,16,32,64,128}
for world in 2 4; do
  for groups in 128 32 16 8; do
    echo "### world=$world TARGET_GROUPS=$groups"
    DCP_A2A_TARGET_GROUPS=$groups \
      timeout 600 "$PY" "$DIR/bench_a2a.py" "$world" graph 2>&1 | grep -E '^\| [0-9]'
  done
done
