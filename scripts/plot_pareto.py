"""Plot the quality/efficiency Pareto front from the optimization + approx results."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")


def load(name):
    rows = []
    p = os.path.join(RESULTS, name)
    if os.path.exists(p):
        for line in open(p):
            rows.append(json.loads(line))
    return rows


fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# --- Left: optimization Pareto (cosine vs pair_ratio) ---
opt = load("optimize.jsonl")
seen = {}
for r in opt:
    key = (r["window"], r["n_global"], r["topk"])
    seen[key] = r
xs = [r["pair_ratio"] for r in seen.values()]
ys = [r["logit_cosine"] for r in seen.values()]
labels = [(r["window"], r["n_global"], r["topk"]) for r in seen.values()]
axes[0].scatter(xs, ys, c="steelblue", s=60, alpha=0.7, edgecolors="k")
best = min(seen.values(), key=lambda r: r["pair_ratio"] - r["logit_cosine"])
for r in seen.values():
    if r["logit_cosine"] > 0.97 and r["pair_ratio"] < 0.2:
        axes[0].annotate(f"w{r['window']}g{r['n_global']}k{r['topk']}",
                         (r["pair_ratio"], r["logit_cosine"]), fontsize=7,
                         textcoords="offset points", xytext=(4, 4))
axes[0].set_xlabel("pair ratio (fraction of attention pairs computed)  → less is better")
axes[0].set_ylabel("logit cosine vs dense  → higher is better")
axes[0].set_title("Pareto: fidelity vs compute (local_global_topk, ctx=2048)")
axes[0].grid(alpha=0.3)
axes[0].axvline(1.0, color="gray", ls="--", lw=0.8)

# --- Right: method comparison (approx.jsonl) ---
approx = [r for r in load("approx.jsonl") if r["model"].endswith("350M-Base") and r["context"] == 2048]
for r in approx:
    if r["method"] == "dense":
        continue
    axes[1].scatter(r["pair_ratio"], r["logit_cosine"], s=70, alpha=0.8)
    axes[1].annotate(r["method"].replace("_", " "), (r["pair_ratio"], r["logit_cosine"]),
                     fontsize=7, textcoords="offset points", xytext=(4, 4))
axes[1].set_xlabel("pair ratio")
axes[1].set_ylabel("logit cosine vs dense")
axes[1].set_title("Method comparison (LFM2.5-350M-Base, ctx=2048)")
axes[1].grid(alpha=0.3)

plt.tight_layout()
out = os.path.join(RESULTS, "pareto.png")
plt.savefig(out, dpi=130)
print("[saved]", out)
