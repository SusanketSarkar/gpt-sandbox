#!/usr/bin/env bash
# layout_schedule_2 — regenerate all derived artifacts from whatever runs exist.
# Safe to run at any point (even mid-experiment); reads eval.json/metrics.jsonl.
set -uo pipefail
cd "$(dirname "$0")"
echo "=== analyze (tables + summary + all_runs) ==="
python3 analyze.py
echo "=== figures ==="
python3 make_figs.py
echo "=== done. artifacts: tables.md summary.json all_runs.json figs/*.png ==="
ls -la tables.md summary.json figs/ 2>/dev/null | sed -n '1,20p'
