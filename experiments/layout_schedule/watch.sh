#!/usr/bin/env bash
# Append a status snapshot every invocation. Run in a loop or on demand.
cd /home/susanket/gpt-sandbox
LOG=experiments/layout_schedule/queue_watch.log
{
  echo "=== $(date '+%F %T') ==="
  for sweep in layout_synth_local_rule layout_synth_recall layout_synth_hop2 layout_f1_jumbo layout_f2_jumbo layout_f1_small layout_f1_med; do
    d=/data/susanket/text/gpt/runs/$sweep
    [ -d "$d" ] || continue
    printf "  %-26s %s runs done\n" "$sweep" "$(find "$d" -name eval.json 2>/dev/null | wc -l)"
  done
  echo "  A: $(tail -1 experiments/layout_schedule/queue_A.log)"
  echo "  B: $(tail -1 experiments/layout_schedule/queue_B.log)"
  nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader | sed 's/^/  gpu /'
} >> "$LOG"
