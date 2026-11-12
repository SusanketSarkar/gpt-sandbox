"""Pick the next actionable run (shared across GPU workers). Prints 'cfg|name|seed' or nothing."""
import json
import os
import time

STATE = os.environ.get("GPT2_STATE", "/tmp/gpt2_state")
RUNS = os.environ.get("GPT2_RUNS", "/data/susanket/text/gpt/runs")
LIST = os.environ.get("GPT2_LIST", "/home/susanket/gpt-sandbox/experiments/layout_schedule_2/runlist.json")
MAX_ATT = 3
STALE_LOCK_S = 90 * 60  # a jumbo run takes <35 min; steal locks older than this

runlist = json.load(open(LIST))
runlist.sort(key=lambda r: (r["prio"], r["sweep"], r["variant"], r["seed"]))
now = time.time()
for r in runlist:
    name = f'{r["sweep"]}/{r["variant"]}-s{r["seed"]}'
    flat = name.replace("/", "__")
    if os.path.exists(f"{RUNS}/{name}/eval.json"):
        continue
    if os.path.exists(f"{STATE}/fails/{flat}"):
        continue
    try:
        att = int(open(f"{STATE}/att/{flat}").read().strip())
    except (FileNotFoundError, ValueError):
        att = 0
    if att >= MAX_ATT:
        continue
    lock = f"{STATE}/locks/{flat}"
    if os.path.exists(lock):
        try:
            if now - os.path.getmtime(lock) < STALE_LOCK_S:
                continue
        except OSError:
            continue
    print(f'{r["cfg"]}|{name}|{r["seed"]}')
    break
