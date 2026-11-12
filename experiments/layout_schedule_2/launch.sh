#!/usr/bin/env bash
# Launch one worker per GPU over the shared ratio-sweep queue.
# Usage: launch.sh [num_gpus]
set -uo pipefail
N="${1:-4}"
EXP=/home/susanket/gpt-sandbox/experiments/layout_schedule_2
mkdir -p "$EXP/logs"
# clean any stale state from a previous (aborted) night
rm -rf /tmp/gpt2_state
mkdir -p /tmp/gpt2_state/locks /tmp/gpt2_state/fails /tmp/gpt2_state/att
pids=()
for i in $(seq 0 $((N - 1))); do
  setsid bash "$EXP/worker.sh" "$i" > "$EXP/logs/worker_gpu${i}.out" 2>&1 < /dev/null &
  pids+=($!)
  echo "launched worker gpu$i (pid ${pids[-1]})"
done
echo "all $N workers launched. watch: $EXP/status.sh"
