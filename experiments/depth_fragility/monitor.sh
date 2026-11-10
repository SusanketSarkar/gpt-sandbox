#!/usr/bin/env bash
# One-shot progress snapshot for the depth-fragility study.
set -u
RUNS=/data/susanket/text/gpt/runs
STATE=/home/susanket/gpt-sandbox/experiments/depth_fragility/state
echo "=== status.log tail ==="
tail -4 "$STATE/status.log" 2>/dev/null
echo "=== cells ==="
for c in depth_m_core depth_d_core depth_m_profile depth_d_profile_a depth_d_profile_b depth_m_global depth_d_global; do
  d="$RUNS/$c"
  [ -d "$d" ] || continue
  total=$(ls "$d" | wc -l)
  done=$(grep -l '"status": "done"' "$d"/*/run.json 2>/dev/null | wc -l)
  cur=$(ls -t "$d" 2>/dev/null | head -1)
  step=$(python - "$d/$cur/metrics.jsonl" <<'EOF' 2>/dev/null
import json, sys
try:
    rows=[json.loads(l) for l in open(sys.argv[1]) if l.strip()]
    tr=[r for r in rows if r.get('split')=='train']
    print(f"step {tr[-1]['step']}/{tr[-1].get('max_steps','?')}" if tr else "no steps")
except Exception as e:
    print("done?" if not rows else f"err {e!r}")
EOF
)
  echo "  $c: $done/$total done   current: $cur ($step)"
done
echo "=== gpus ==="
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader 2>/dev/null
echo "=== live tails ==="
for f in "$STATE"/logs/*.log; do
  [ -s "$f" ] || continue
  echo "-- $(basename $f)"
  tail -1 "$f"
done
