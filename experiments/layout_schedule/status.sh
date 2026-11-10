#!/usr/bin/env bash
# Quick status: how many runs done per sweep, current step of the active run, GPU util.
set -uo pipefail
RUNS=/data/susanket/text/gpt/runs
echo "=== $(date '+%F %T') ==="
for sweep in layout_synth_local_rule layout_synth_recall layout_synth_hop2 layout_f1_jumbo layout_f2_jumbo layout_f1_small layout_f1_med; do
  d="$RUNS/$sweep"
  [ -d "$d" ] || continue
  done=$(find "$d" -name eval.json 2>/dev/null | wc -l)
  total=$(find "$d" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l)
  printf "%-26s %3s/%-3s runs\n" "$sweep" "$done" "$total"
done
echo "--- queue A (last line) ---"; tail -1 experiments/layout_schedule/queue_A.log 2>/dev/null
echo "--- queue B (last line) ---"; tail -1 experiments/layout_schedule/queue_B.log 2>/dev/null
echo "--- GPU ---"; nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader
