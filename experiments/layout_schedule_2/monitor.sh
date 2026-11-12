#!/usr/bin/env bash
# layout_schedule_2 — health check + self-heal.
#   - reports done/failed/active per sweep
#   - relaunches any worker whose GPU has no live worker (auto-recovery)
#   - safe to run repeatedly
set -uo pipefail
EXP=/home/susanket/gpt-sandbox/experiments/layout_schedule_2
# N = number of GPUs we manage. GPU3 is NOT reserved for this job (user's own
# GPU, e.g. LM Studio), so we only manage GPUs 0..N-1 = 0,1,2. Do not set N=4.
N=3
echo "=== progress ==="
bash "$EXP/status.sh" 2>/dev/null | sed -n '1p;3,11p'
echo "=== failures (permanent, 3 attempts) ==="
fl=$(ls /tmp/gpt2_state/fails/ 2>/dev/null | wc -l); echo "  permanent-fail markers: $fl"
ls /tmp/gpt2_state/fails/ 2>/dev/null | sed 's/^/    FAIL: /'
echo "=== worker liveness per GPU (and auto-relaunch) ==="
# collect the set of gpu-ids that have a live worker (robust: read last arg)
live_gpus=""
for p in $(pgrep -f "worker\.sh" 2>/dev/null); do
  [ "$p" = "$$" ] && continue
  last=$(ps -o args= -p "$p" 2>/dev/null | awk '{print $NF}')
  case "$last" in 0|1|2|3) live_gpus="$live_gpus $last ";; esac
done
for i in $(seq 0 $((N-1))); do
  if echo "$live_gpus" | grep -qE "(^| )$i( |$)"; then
    echo "  gpu$i: worker ALIVE"
  else
    echo "  gpu$i: worker DEAD -> relaunching"
    setsid bash "$EXP/worker.sh" "$i" > "$EXP/logs/worker_gpu${i}.out" 2>&1 < /dev/null &
    echo "    relaunched (pid $!)"
  fi
done
echo "=== OOM / error scan in worker logs (last 40 lines each) ==="
for i in $(seq 0 $((N-1))); do
  hits=$(tail -40 "$EXP/logs/gpu${i}.log" 2>/dev/null | grep -ciE "out of memory|OOM|CUDA error|Traceback|RuntimeError|Killed")
  [ "$hits" -gt 0 ] && { echo "  gpu$i: $hits error line(s):"; tail -40 "$EXP/logs/gpu${i}.log" | grep -iE "out of memory|OOM|CUDA error|Traceback|RuntimeError|Killed" | tail -3 | sed 's/^/    /'; }
done
echo "=== done. ==="
