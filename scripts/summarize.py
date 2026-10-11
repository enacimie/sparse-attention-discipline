"""Summarize experiment JSONL logs into a comparison table (CSV + stdout)."""
import argparse
import json
import os

import pandas as pd

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


def load(name):
    path = os.path.join(RESULTS_DIR, name)
    rows = []
    if os.path.exists(path):
        for line in open(path):
            rows.append(json.loads(line))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default="local_window.jsonl")
    args = ap.parse_args()
    df = load(args.log)
    if df.empty:
        print("no results in", args.log)
        return
    cols = [c for c in ["model", "context", "window", "method", "pair_ratio",
                        "kl_div", "logit_cosine", "needle_recall", "peak_vram_mb"]
            if c in df.columns]
    df = df[cols].sort_values([c for c in ["model", "context", "window"] if c in df.columns])
    out = os.path.join(RESULTS_DIR, args.log.replace(".jsonl", "_summary.csv"))
    df.to_csv(out, index=False)
    print(df.to_string(index=False))
    print("\n[saved]", out)


if __name__ == "__main__":
    main()
