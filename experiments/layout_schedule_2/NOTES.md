# layout_schedule_2 — session notes

## STATUS (as of 03:49, 2026-11-11) — 235/327 runs done

**Core question answered.** The hypothesis ("attention-heavy is better than the conventional 2:1")
is **disproven**, with a crucial nuance:

1. **Core ratio sweep (ratio_jumbo, 84 runs, 48 sublayers):** MLP-heavy (~2:1) beats attention-heavy
   (1:4) by 0.074 val-loss (8.8× above the 0.008 noise floor). Flat MLP-heavy plateau + monotone
   attention-heavy degradation. Attention-heavy (38A) is also the least stable (sd 0.022).
2. **Parameter-matched control (ratio_pm_jumbo, 24/36, the crux):** at fixed ~80M params the ratio
   effect nearly vanishes (spread 0.004, within noise). → **The core sweep's "ratio effect" is a
   SIZE effect** (MLP-heavy = more params). For LM, total parameters matter, not the ratio.
3. **Retrieval (ratio_synth_recall, 42 runs):** 1:1 (12A) is the **only** ratio that reliably solves
   recall (all seeds ≥0.78, sd 0.083); every other ratio is bimodal (sd 0.28–0.42). Best ratio (12A)
   is NOT the largest (5A is) → **a genuine, size-independent ratio effect** for retrieval.
4. 1-hop & local probes: saturated (no signal).

**Bottom line:** no single universal optimal ratio. For LM the "optimal ratio" is really a size
effect (bigger is better; ratio barely matters at fixed size). For retrieval, 1:1 is a genuine
optimum. The "attention-heavy is better" hypothesis is disproved; the "task-dependent" branch is
supported (with the LM-size-effect twist).

**Still running:** ratio_depth24 (1/42), ratio_fine_jumbo (0/24), ratio_jumbo_long (0/15),
ratio_pm_jumbo (24/36). Workers + watchdog running detached; will auto-complete.

**Artifacts:** `BLOG.md` (findings), `tables.md`, `summary.json`, `all_runs.json`,
`figs/fig1..fig7.png`, `DESIGN.md`. Runs under `/data/susanket/text/gpt/runs/ratio_*/`.
Regenerate: `bash finalize.sh`.

## Timeline (UTC, 2026-11-10)

- 16:1x — read layout_schedule (DESIGN, BLOG, COMPLETIONS), audited the gpt-sandbox
  framework (model, metrics, trainer, CLI). Confirmed: sublayer layout strings are fully
  general (`A`/`M` sequences), so ratio sweeps need zero model code changes.
- 16:2x — measured H100 throughput: jumbo ~170–250k tok/s (smoke 250 steps: 39 s).
  Confirmed best-checkpoint protocol; synth best val lands at step 5k–9k of 20k, so
  16k steps is ample.
- 16:3x — built: `gen_ratio_sweeps.py` (327-run config matrix + runlist), `pick.py`,
  `worker.sh`, `launch.sh`, `status.sh`, `analyze.py`, `make_figs.py`, `DESIGN.md`.
- 16:41 — GPU0 worker up (setsid detached); 16:43 GPU1; 16:46 GPU2/3. All four training.
  (Incident: first launch attempt used a relative log path from the wrong cwd — workers
  2/3 didn't start; relaunched with absolute paths. Lesson: `cd X && A & B &` only cds A.)
- 16:55 — analyzer tested on partial data; tables render correctly; legacy baselines fold
  into the 1:1 anchor group.

## Ops

- Queue: `/home/susanket/gpt-sandbox/experiments/layout_schedule_2/runlist.json` (327 runs,
  priority 0 = ratio_jumbo … 5 = ratio_jumbo_long).
- State: `/tmp/gpt2_state/{locks,fails,att}` — atomic mkdir locks, 90-min stale-steal,
  3 attempts per run.
- Logs: `logs/gpu{0..3}.log` (worker lines), `logs/worker_gpu{0..3}.out` (raw train output).
- Dashboard: `bash status.sh`.
- Rerun a single config: `gpt train -c configs/sweeps/ratio2/<cfg>.yaml -n <sweep>/<variant>-s<seed> --set seed=<s> --gpu <N> --overwrite`.
- Reanalyze any time: `python3 analyze.py` → `tables.md`, `summary.json`, `all_runs.json`.
- Figures: `python3 make_figs.py` → `figs/*.png`.

## Watch items

1. RESOLVED (17:10): "leaked locks" were just runs that were mid-training when checked;
   their workers (incl. one orphan `gpt train` reparented to the opencode shell from the
   first launch attempt) didn't rmdir on completion. Harmless: pick.py skips done runs
   (eval.json) BEFORE checking locks. Cleaned with `rm -rf`. System verified healthy:
   exactly 4 workers (gpu0-3), 4 live runs, one per GPU, GPU mem 34-42 GB (< 80 GB), no
   duplicate workers, no strays.
2. OOM risk at r41 (10A+38M, 90.8M params) — 40 GB used at r11 (75.9M); r41 should fit in
   80 GB. Watch memory as the MLP-heavy configs run.
3. Param-matched configs: verify d_ff lands within ±0.2% (gen script asserts the formula).
4. If ratio_jumbo finishes early, the queue auto-moves to synth (prio 1). Nothing to do.
5. Morning: run `bash finalize.sh` (analyze + figs), deep-dive layer stats
   (delta_rms/L*, out_stream_ratio/L* per sublayer type), write BLOG.md.

## 18:40 — overnight run confirmed healthy + ETA

- Disk: 5.7T free on /data (runs ~100-200GB total, fine). RAM: 937GiB avail. No OOM/disk risk.
- ETA ~15h -> finish ~11:30 AM. Core ratio_jumbo sweep (84 runs) first, ~by midnight.
- System: 4 workers + 1 watchdog (auto-relaunch every 15min), 0 fails, 21/327 done.
- Monitoring plan: periodic checks of logs/watchdog.log + progress; finalize.sh for tables/figs;
  BLOG_DRAFT.md -> fill FINDINGS after results; write final BLOG.md in the morning.

## 18:32 — worker restart + watchdog (and a near-miss bug scare, resolved)

- I briefly suspected a fail-name collision bug in worker.sh, killed the workers, and
  re-examined. RESOLVED: there is NO bug. pick.py/worker.sh key runs by
  `flat = (sweep/variant-seed).replace('/', '__')` which is unique per (sweep,variant,seed);
  pick.py also has 90-min stale-lock stealing + per-run attempt counter (max 3). Design is sound.
- Restarted 4 workers cleanly (cleaned stale locks first). 4 workers + 4 gpt train (cuda0-3),
  1 watchdog. No duplicates.
- watchdog.sh runs monitor.sh every 15 min: auto-relaunches dead workers (accurate gpu-id
  detection), reports pending_runs, stops when 0 pending. Self-locked (one instance).
- IMPORTANT for any pgrep/ps checks: the search pattern self-matches the checking shell;
  filter with `grep -vE 'bash -c|grep'` or read the pid list, don't trust raw grep -c.

## Verified (17:50)

- System healthy: 4 workers ALIVE (gpu0-3), 0 failures, 0 OOM, locks clean (fix works).
- Design matches user spec exactly: jumbo=48 sublayers d=512 (per jumbo_base.yaml), ratio grid
  4:1..1:4 (nA in {10,12,16,24,32,36,38}), 4 arrangements/ratio (alt+3 random), 3 seeds, depth
  ablation (24 sublayers d=256), param-matched control, refinement, synth retrieval tasks, long.
- All 6 required metrics captured: val_loss, recall_acc (one/multi/long-range), flops_per_token,
  params_total, convergence (loss+gnorm/step), wall-clock (tok_per_s throughput).
- analyze.py intentionally maps legacy layout_f1_jumbo/baseline-s* (3 runs, identical config,
  val 2.710) into ratio_jumbo/r11/arr=legacy as extra anchors. Other legacy (med/small, A-D)
  correctly excluded (name pattern). "loaded 16" = 13 mine + 3 legacy anchors. Valid.
- r11 anchor val_loss stable (2.693-2.710, sd~0.005) -> LM comparison reliable. recall_acc
  noisier (rec/64 0.16-0.35 across seeds) -> use val_loss primary, recall secondary/trend.
- finalize.sh (analyze+7 figs) tested on partial data: works. monitor.sh self-heal tested:
  correctly detects all 4 workers, no false relaunch.
- Wall-clock per jumbo run ~15.5 min.

## Schedule (measured ~15.5 min/jumbo run, 4 GPUs)

- ratio_jumbo (core, prio 0, 84 runs):  ~5.4 h -> done ~22:45  <-- PRIMARY ANSWER
- ratio_synth (retrieval, prio 1, 126): ~2.6 h -> ~01:15
- ratio_pm_jumbo (control, prio 2, 36): ~2.3 h -> ~03:45
- ratio_depth24 (ablation, prio 3, 42): ~1.4 h -> ~05:15
- ratio_fine_jumbo (refine, prio 4, 24): ~1.5 h -> ~06:45
- ratio_jumbo_long (stretch, prio 5, 15): ~2 h -> ~08:45
Total ~61 GPU-h = ~15.2 wall-h. Core + retrieval + control + ablation all done by ~05:15.

## Early numbers (partial — do not over-read)

- ratio_jumbo 1:1 anchor (3 legacy seeds, pre-existing): val 2.7028/2.7061/2.7163 (last),
  best 1.8117–1.8165. These are the reference point every ratio must beat/lose to.
- ratio_synth_recall 5A19M (MLP-heavy) s0: task_em 0.89, long_recall acc ≈0.03–0.10 at
  d=8..128 → short-range recall ok, long-range retrieval weak. (Matches hypothesis direction;
  need the attention-heavy side + other seeds before drawing conclusions.)
- ratio_depth24 5A19M s0: val 2.864 (last), best 1.957 — need 12A12M anchor for comparison.

## Design decisions (why)

- **48 sublayers jumbo, 24 depth-ablation**: user asked for "jumbo 24-layer" models and
  depth ablations; 48 is the natural jumbo scale in this repo (d=512), 24 = half, d=256.
- **d_ff fixed (raw) vs param-matched**: the raw sweep is the primary hypothesis test as
  specified (width + expansion fixed, ratio varied → params vary ±20%); the pm family
  isolates the ratio effect from the parameter-count effect.
- **alt + 3 random arrangements**: layout_schedule showed order matters; random arrangements
  bound the arrangement effect so the ratio effect can be attributed cleanly.
- **3 seeds**: layout_schedule measured seed noise ≈ ±0.0026; effects must clear that.
- **best-checkpoint eval**: established protocol; fixed eval windows + cosine schedule make
  last-step eval biased (loss climbs after LR minimum).
- **seeds 0/1/2 + arrangement seeds 1000+nA*7+i**: reproducible, documented, no collisions.

## Access notes

- Code: /home/susanket/gpt-sandbox (this repo). Data/runs: /data/susanket/text/gpt.
- GPUs 0–3 (H100 80 GB) were free; llama-server holds ~5–10 GB each (inference, not a
  contention issue).
- No other repos needed; all experiment code lives here.
