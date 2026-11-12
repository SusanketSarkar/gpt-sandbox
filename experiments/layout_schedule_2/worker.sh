#!/usr/bin/env bash
# GPU worker: steal-pick runs from the shared queue and train them on this GPU.
# Usage: worker.sh <gpu_id>
set -uo pipefail
GPU="${1:?gpu id}"
EXP=/home/susanket/gpt-sandbox/experiments/layout_schedule_2
STATE=/tmp/gpt2_state
RUNS=/data/susanket/text/gpt/runs
mkdir -p "$STATE/locks" "$STATE/fails" "$STATE/att" "$EXP/logs"
LOG="$EXP/logs/gpu${GPU}.log"

while :; do
  pick="$(python3 "$EXP/pick.py")"
  [ -z "$pick" ] && { echo "[$(date +%FT%T)] gpu$GPU: queue empty, exiting" | tee -a "$LOG"; break; }
  IFS='|' read -r cfg name seed <<< "$pick"
  flat="${name//\//__}"
  lock="$STATE/locks/$flat"
  # atomic claim
  mkdir "$lock" 2>/dev/null || continue
  echo "$(date +%FT%T)" > "$lock/ts"
  echo "[$(date +%FT%T)] gpu$GPU START $name (seed $seed)" | tee -a "$LOG"
  cd /home/susanket/gpt-sandbox
  gpt train -c "$cfg" -n "$name" --set seed="$seed" --device "cuda:$GPU" --overwrite >> "$LOG" 2>&1
  rc=$?
  if [ $rc -eq 0 ] && [ -f "$RUNS/$name/eval.json" ]; then
    echo "[$(date +%FT%T)] gpu$GPU DONE  $name" | tee -a "$LOG"
    rm -f "$STATE/att/$flat"
  else
    att="$(cat "$STATE/att/$flat" 2>/dev/null || echo 0)"
    att=$((att + 1))
    echo "$att" > "$STATE/att/$flat"
    echo "[$(date +%FT%T)] gpu$GPU FAIL  $name (rc=$rc, attempt $att)" | tee -a "$LOG"
    if [ "$att" -ge 3 ]; then touch "$STATE/fails/$flat"; fi
  fi
  rm -rf "$lock" 2>/dev/null
done
echo "[$(date +%FT%T)] gpu$GPU: finished" | tee -a "$LOG"
