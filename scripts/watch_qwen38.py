#!/usr/bin/env python3
"""Poll HuggingFace for new Qwen3.8 models (esp. a dense ~27B).

Runs forever, writes a heartbeat + alerts to stdout, and appends any NEW
model to results/QWEN38_NEW.flag. Designed to run under nohup on the remote
machine so it keeps watching even if the laptop is switched off.

Baseline (already published at setup time, 2026-08-12):
  Qwen/Qwen3.8-2.4T-A95B        (MoE, arch tag qwen3_5_moe_text)
  Qwen/Qwen3.8-2.4T-A95B-FP8

Usage:
  nohup python3 scripts/watch_qwen38.py > results/watch_qwen38.log 2>&1 &
"""
import json
import os
import time
import urllib.request
from datetime import datetime

API = "https://huggingface.co/api/models?author=Qwen&search={q}&limit=100"
POLL_S = 300  # 5 minutes
HERE = os.path.dirname(os.path.abspath(__file__))
FLAG = os.path.join(HERE, "..", "results", "QWEN38_NEW.flag")

KNOWN = {
    "Qwen/Qwen3.8-2.4T-A95B",
    "Qwen/Qwen3.8-2.4T-A95B-FP8",
}


def now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def fetch(q):
    try:
        with urllib.request.urlopen(API.format(q=q), timeout=30) as r:
            return {m["modelId"]: m for m in json.load(r)}
    except Exception as e:  # noqa: BLE001 - keep the watcher alive
        print(f"[{now()}] fetch error ({q}): {e}", flush=True)
        return None


def looks_dense_27b(model_id):
    name = model_id.split("/")[-1].upper()
    # dense 27B has "27B" and no MoE active-param marker like "A3B"/"A95B"
    return "27B" in name and not any(
        tok.startswith("A") and tok[1:].replace("B", "").isdigit()
        for tok in name.replace("-", " ").split()
    )


def main():
    print(f"[{now()}] watcher started; baseline={sorted(KNOWN)}; poll={POLL_S}s", flush=True)
    seen = set(KNOWN)
    while True:
        ids = fetch("Qwen3.8")
        if ids is not None:
            for m in sorted(set(ids) - seen):
                tags = ids[m].get("tags", [])
                arch = next((t for t in tags if t.startswith("qwen3_5")), "?")
                pipe = ids[m].get("pipeline_tag", "?")
                star = "  <== DENSE 27B CANDIDATE" if looks_dense_27b(m) else ""
                msg = f"NEW MODEL: {m}  arch={arch}  pipeline={pipe}{star}"
                print(f"[{now()}] {msg}", flush=True)
                with open(FLAG, "a") as f:
                    f.write(f"[{now()}] {msg}\n")
                seen.add(m)
            print(f"[{now()}] heartbeat; {len(ids)} qwen3.8 repos on HF", flush=True)
        time.sleep(POLL_S)


if __name__ == "__main__":
    main()
