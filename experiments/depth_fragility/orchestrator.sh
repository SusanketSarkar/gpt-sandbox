#!/usr/bin/env bash
# Depth-fragility study orchestrator: 3 GPU workers, each owns a queue of sweep files.
#   gpu0: D core (16L: plain + first/mid/last x 3 sigmas, 3 seeds)                 ~315 GPU-min
#   gpu1: M core (8L: same, 3 seeds) + D profile a (layers 0-7, sigmas .3/1.0)     ~318 GPU-min
#   gpu2: D profile b (layers 8-15) + M profile + both global-noise controls       ~295 GPU-min
# After each sweep file: `gpt eval <run> --deep` for the runs that need ablation/lens profiles
# (all plain runs, the sig10 anchor runs, and the global sig10 control).
# Resumable: `gpt sweep` skips runs that already have eval.json.
set -u
export PYTHONUNBUFFERED=1
REPO=/home/susanket/gpt-sandbox
RUNS=/data/susanket/text/gpt/runs
STATE=$REPO/experiments/depth_fragility/state
mkdir -p "$STATE/logs"

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$STATE/status.log"; }

deep_targets() { # $1 = cell prefix (depth_m or depth_d)
  local p=$1 d
  # plain runs (all seeds) + sig10 anchor runs (seed 0) + global sig10 control
  for d in "$RUNS/${p}_core/plain-s0" "$RUNS/${p}_core/plain-s1" "$RUNS/${p}_core/plain-s2" \
           "$RUNS/${p}_core"/*_sig10-s0 "$RUNS/${p}_global/all_sig10-s0"; do
    [ -f "$d/run.json" ] || continue
    grep -q '"status": "done"' "$d/run.json" 2>/dev/null || continue
    if ! grep -q 'ablate_dloss/L0' "$d/eval.json" 2>/dev/null; then
      echo "$d"
    fi
  done
}

worker() {
  local gpu=$1; shift
  local f name rc d
  for f in "$@"; do
    name=$(basename "$f" .yaml)
    log "gpu$gpu START $f"
    (cd "$REPO" && gpt sweep "configs/sweeps/$f" --device "cuda:$gpu") > "$STATE/logs/$name.log" 2>&1
    rc=$?
    log "gpu$gpu sweep $f rc=$rc"
    for d in $(deep_targets "$(echo $name | sed 's/_core//;s/_profile_[ab]//;s/_global//')"); do
      # mkdir-based lock so two workers never deep-eval the same run concurrently
      if mkdir "$d/.deepeval.lock" 2>/dev/null; then
        log "gpu$gpu deep-eval $d"
        (cd "$REPO" && gpt eval "$d" --deep --device "cuda:$gpu") > "$STATE/logs/$(basename "$d").eval.log" 2>&1
        log "gpu$gpu deep-eval $d rc=$?"
        rm -rf "$d/.deepeval.lock"
      else
        log "gpu$gpu deep-eval $d skipped (locked by another worker)"
      fi
    done
    log "gpu$gpu DONE $f"
  done
}

worker 0 depth_d_core.yaml &
w0=$!
worker 1 depth_m_core.yaml depth_d_profile_a.yaml &
w1=$!
worker 2 depth_d_profile_b.yaml depth_m_profile.yaml depth_m_global.yaml depth_d_global.yaml &
w2=$!
wait "$w0" "$w1" "$w2"
log "ALL WORKERS DONE"
