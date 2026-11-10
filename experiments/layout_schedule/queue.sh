#!/usr/bin/env bash
# Overnight orchestration: run a sequence of `gpt` invocations sequentially on one GPU.
# Usage: queue.sh <tag> <cuda:N> <gpt-args...> -- <gpt-args...> -- ...
# The sweep runner is resume-safe (skips runs that already have eval.json).
set -u
cd /home/susanket/gpt-sandbox
tag=$1; shift
GPU=$1; shift
LOG=experiments/layout_schedule/queue_$tag.log
mkdir -p experiments/layout_schedule

run_one() {
  local GPU=$1; shift
  echo "==== [$(date '+%F %T')] START: gpt $* ====" >>"$LOG"
  gpt "$@" >>"$LOG" 2>&1
  local rc=$?
  echo "==== [$(date '+%F %T')] END rc=$rc: gpt $* ====" >>"$LOG"
}

echo "==== queue_$tag on $GPU started $(date '+%F %T') ====" >>"$LOG"
job=""
for arg in "$@"; do
  if [ "$arg" = "--" ]; then
    [ -n "$job" ] && run_one "$GPU" $job
    job=""
    continue
  fi
  job="$job $arg"
done
[ -n "$job" ] && run_one "$GPU" $job
echo "==== queue_$tag finished $(date '+%F %T') ====" >>"$LOG"
