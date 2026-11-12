#!/usr/bin/env bash
# layout_schedule_2 — watchdog: every CHECK_S seconds, run monitor.sh (which
# reports health + auto-relaunches any dead worker). Self-locks so only one
# watchdog runs. Survives the launching shell (setsid).
set -uo pipefail
EXP=/home/susanket/gpt-sandbox/experiments/layout_schedule_2
STATE=/tmp/gpt2_state
mkdir -p "$STATE"
LOCK="$STATE/watchdog.lock"

# self-lock: if a watchdog is already running, exit.
if [ -e "$LOCK" ]; then
  old_pid=$(cat "$LOCK" 2>/dev/null || echo "")
  if [ -n "$old_pid" ] && kill -0 "$old_pid" 2>/dev/null; then
    echo "watchdog already running (pid $old_pid); exiting."
    exit 0
  fi
fi
echo $$ > "$LOCK"
trap 'rm -f "$LOCK"' EXIT

CHECK_S="${CHECK_S:-900}"   # 15 min
RUNS=/data/susanket/text/gpt/runs
echo "[$(date +%FT%T)] watchdog started (pid $$, check every ${CHECK_S}s)" | tee -a "$EXP/logs/watchdog.log"
while :; do
  bash "$EXP/monitor.sh" >> "$EXP/logs/watchdog.log" 2>&1
  # pending = runs that are neither done (eval.json) nor failed (fail marker)
  # NOTE: fail-marker name mirrors pick.py exactly: name.replace("/", "__"), name="sweep/variant-seed"
  pending=$(python3 -c "
import json, os
rl=json.load(open('$EXP/runlist.json'))
fails=set(os.listdir('$STATE/fails')) if os.path.isdir('$STATE/fails') else set()
def flat(r):  # mirror pick.py: (sweep/variant-seed).replace('/', '__')
    return (f\"{r['sweep']}/{r['variant']}-s{r['seed']}\".replace('/','__'))
p=0
for r in rl:
    done=os.path.exists(f\"$RUNS/{r['sweep']}/{r['variant']}-s{r['seed']}/eval.json\")
    if not done and flat(r) not in fails:
        p+=1
print(p)" 2>/dev/null)
  echo "[$(date +%FT%T)] watchdog: pending_runs=${pending:-?}" | tee -a "$EXP/logs/watchdog.log"
  if [ "${pending:-999}" -eq 0 ]; then
    echo "[$(date +%FT%T)] watchdog: queue exhausted (0 pending); stopping." | tee -a "$EXP/logs/watchdog.log"
    break
  fi
  sleep "$CHECK_S"
done
echo "[$(date +%FT%T)] watchdog stopped." | tee -a "$EXP/logs/watchdog.log"
