#!/usr/bin/env bash
# Gates study orchestrator: 3 GPU workers, each owns a queue of sweep files (seeds 0/1/2 each).
#   gpu0: size S (8L d256, tinystories) then size L (8L d768, edu_mix)
#   gpu1: size M (8L d512, tinystories) then depth D (16L d512, edu_mix)
#   gpu2: size M on edu_mix
# After each sweep file: `gpt eval <run> --deep` for every run in the cell lacking ablation.
set -u
REPO=/home/susanket/gpt-sandbox
RUNS=/data/susanket/text/gpt/runs
STATE=$REPO/experiments/gates/state
mkdir -p "$STATE/logs"

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$STATE/status.log"; }

worker() {
  local gpu=$1; shift
  local f name rc cell d
  for f in "$@"; do
    name=$(basename "$f" .yaml)
    log "gpu$gpu START $f"
    (cd "$REPO" && gpt sweep "configs/sweeps/$f" --device "cuda:$gpu") > "$STATE/logs/$name.log" 2>&1
    rc=$?
    log "gpu$gpu sweep $f rc=$rc"
    cell=$(sed -n 's/^name:[[:space:]]*//p' "$REPO/configs/sweeps/$f" | head -1)
    for d in "$RUNS/$cell"/*/; do
      [ -f "$d/run.json" ] || continue
      if ! grep -q 'ablate_dloss/L0' "$d/eval.json" 2>/dev/null; then
        log "gpu$gpu deep-eval $d"
        (cd "$REPO" && gpt eval "$d" --deep --device "cuda:$gpu") > "$STATE/logs/$(basename "$d").eval.log" 2>&1
        log "gpu$gpu deep-eval $d rc=$?"
      fi
    done
    log "gpu$gpu DONE $f"
  done
}

worker 0 gates_s.yaml gates_l_edu.yaml &
w0=$!
worker 1 gates_m.yaml gates_d_edu.yaml &
w1=$!
worker 2 gates_m_edu.yaml &
w2=$!
wait "$w0" "$w1" "$w2"
log "ALL WORKERS DONE"
