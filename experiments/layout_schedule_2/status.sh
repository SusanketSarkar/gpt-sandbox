#!/usr/bin/env bash
# Progress dashboard for the ratio-sweep night.
RUNS=/data/susanket/text/gpt/runs
EXP=/home/susanket/gpt-sandbox/experiments/layout_schedule_2
STATE=/tmp/gpt2_state
total=$(python3 -c "import json;print(len(json.load(open('$EXP/runlist.json'))))")
done=$(python3 - <<'EOF'
import json, os
RUNS = "/data/susanket/text/gpt/runs"
rl = json.load(open("/home/susanket/gpt-sandbox/experiments/layout_schedule_2/runlist.json"))
n = sum(1 for r in rl if os.path.exists(f'{RUNS}/{r["sweep"]}/{r["variant"]}-s{r["seed"]}/eval.json'))
print(n)
EOF
)
fails=$(ls "$STATE/fails" 2>/dev/null | wc -l)
locks=$(ls "$STATE/locks" 2>/dev/null | wc -l)
echo "done $done / $total   failed-markers $fails   active-locks $locks"
echo "=== by sweep ==="
python3 - <<'EOF'
import json, os
from collections import defaultdict
RUNS = "/data/susanket/text/gpt/runs"
rl = json.load(open("/home/susanket/gpt-sandbox/experiments/layout_schedule_2/runlist.json"))
agg = defaultdict(lambda: [0, 0])
for r in rl:
    agg[r["sweep"]][1] += 1
    if os.path.exists(f'{RUNS}/{r["sweep"]}/{r["variant"]}-s{r["seed"]}/eval.json'):
        agg[r["sweep"]][0] += 1
for s in sorted(agg):
    d, t = agg[s]
    bar = "#" * d + "-" * (t - d)
    print(f"  {s:20s} {d:3d}/{t:3d}  {bar}")
EOF
echo "=== active runs ==="
for l in "$STATE"/locks/*; do [ -d "$l" ] && echo "  ${l##*/} (age $(( $(date +%s) - $(stat -c %Y "$l") ))s)"; done 2>/dev/null
echo "=== recent worker log lines ==="
tail -n 2 "$EXP"/logs/gpu*.log 2>/dev/null | grep -E "^\S+ \S+|START|DONE|FAIL" | tail -12
