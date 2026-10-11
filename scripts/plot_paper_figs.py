"""Paper figures (round-7 reviewer response): generated from the archived JSONL.

Produces PDF figures in results/figs/:
  fig_budget_curve.pdf   recall vs pair_ratio (retrieval budget curve, 0.8B + 4B)
  fig_fidelity_ctx.pdf   logit-cos vs context, lg_topk vs budget-matched random (4 models)
  fig_decomposition.pdf  structure-vs-content decomposition (grouped bars, 4 models)
  fig_efficiency.pdf     kernel-only speedup + end-to-end speedup (negative result)

No data is edited or interpolated; every point comes from the archived JSONL files
listed in each loader. Run: python scripts/plot_paper_figs.py
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")
FIGS = os.path.join(RESULTS, "figs")
os.makedirs(FIGS, exist_ok=True)

plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.3,
                     "figure.dpi": 150, "savefig.bbox": "tight"})

C_LG = "#1f77b4"     # lg_topk
C_RND = "#d62728"    # random controls
C_OR = "#7f7f7f"     # dense reference


def load(name):
    rows = []
    p = os.path.join(RESULTS, name)
    if os.path.exists(p):
        for line in open(p):
            rows.append(json.loads(line))
    return rows


def short(model):
    return model.split("/")[-1]


# ---------------------------------------------------------------- Fig 1
def fig_budget_curve():
    rows = load("m5_budget_curve.jsonl") + load("m5_budget_curve_4b.jsonl")
    seen = {}
    for r in rows:
        seen[(r["model"], r["method"], r["topk"])] = r  # dedup (files overlap on 0.8B)
    rows = list(seen.values())
    models = ["Qwen/Qwen3.5-0.8B", "Qwen/Qwen3.5-4B-Base"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
    for ax, mo in zip(axes, models):
        for method, lbl, c, mk in [("local_global_topk", "lg_topk (content)", C_LG, "o"),
                                   ("local_global_randtopk", "lg_randtopk (struct.-matched random)",
                                    C_RND, "s")]:
            pts = sorted([r for r in rows if r["model"] == mo and r["method"] == method],
                         key=lambda r: r["pair_ratio"])
            ax.plot([r["pair_ratio"] for r in pts], [r["recall"] for r in pts],
                    marker=mk, color=c, label=lbl, linewidth=1.5, markersize=5)
        ax.axhline(1.0, color=C_OR, linestyle=":", linewidth=1, label="dense (1.0)")
        ax.set_xscale("log")
        ax.set_xlabel("measured pair_ratio")
        ax.set_title(short(mo))
        ax.set_ylim(-0.05, 1.08)
    axes[0].set_ylabel("needle recall")
    axes[0].legend(fontsize=7.5, loc="center right")
    fig.suptitle("Retrieval budget curve: recall vs pair budget (ctx 2048, n=9/point, Wilson CIs in text)",
                 fontsize=10)
    fig.savefig(os.path.join(FIGS, "fig_budget_curve.pdf"))
    plt.close(fig)


# ---------------------------------------------------------------- Fig 2
def fig_fidelity_ctx():
    ci = load("ci_repeat.jsonl")
    rob = load("robustness.jsonl")

    def agg_ci(model, ctx, method):
        vals = [v for r in ci if r["model"] == model and r["context"] == ctx
                and r["method"] == method for v in r["cos_list"]]
        if not vals:
            return None
        m = sum(vals) / len(vals)
        s = (sum((v - m) ** 2 for v in vals) / max(1, len(vals) - 1)) ** 0.5
        return m, s

    def agg_rob(model, ctx, method):
        vals = [r for r in rob if r["model"] == model and r["context"] == ctx
                and r["method"] == method]
        if not vals:
            return None
        r = vals[-1]
        return r["cos_mean"], r.get("cos_std", 0.0)

    panels = [
        ("Qwen/Qwen3.5-4B-Base", "4B hybrid (4 seeds x 5 chunks)", agg_ci,
         [("lg_topk", "uniform_random")]),
        ("Qwen/Qwen3.5-9B-Base", "9B hybrid (n=5; 1024 single-run not shown)", agg_rob,
         [("local_global_topk", "uniform_random_m")]),
        ("Qwen/Qwen3-0.6B", "Qwen3-0.6B dense (4 seeds x 5 chunks)", agg_ci,
         [("lg_topk", "uniform_random")]),
        ("Qwen/Qwen3-1.7B", "Qwen3-1.7B dense (4 seeds x 5 chunks)", agg_ci,
         [("lg_topk", "uniform_random")]),
    ]
    ctxs = [1024, 2048, 4096]
    fig, axes = plt.subplots(2, 2, figsize=(9, 6.2), sharex=True)
    for ax, (mo, title, agg, pairs) in zip(axes.flat, panels):
        for lg_m, rnd_m in pairs:
            for method, lbl, c, mk in [(lg_m, "lg_topk", C_LG, "o"),
                                       (rnd_m, "uniform_random (budget-matched)", C_RND, "s")]:
                xs, ys, errs = [], [], []
                for ctx in ctxs:
                    v = agg(mo, ctx, method)
                    if v is None:
                        continue
                    xs.append(ctx)
                    ys.append(v[0])
                    errs.append(v[1])
                ax.errorbar(xs, ys, yerr=errs, marker=mk, color=c, label=lbl,
                            linewidth=1.5, markersize=5, capsize=3)
        ax.set_xscale("log")
        ax.set_title(title, fontsize=9)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("logit cosine vs dense")
    axes[1][0].set_xlabel("context length")
    axes[1][1].set_xlabel("context length")
    axes[0][0].legend(fontsize=7.5, loc="lower left")
    fig.suptitle("Fidelity vs context under a fixed recipe (w=64, sinks=4, top-k=32): "
                 "lg_topk ~context-invariant, random collapses", fontsize=10)
    fig.savefig(os.path.join(FIGS, "fig_fidelity_ctx.pdf"))
    plt.close(fig)


# ---------------------------------------------------------------- Fig 3
def fig_decomposition():
    m1 = load("m1_control.jsonl")
    models = ["Qwen/Qwen3.5-0.8B", "Qwen/Qwen3.5-4B-Base", "Qwen/Qwen3-0.6B",
              "Qwen/Qwen3-1.7B"]
    labels = ["uniform_random\n(no structure)", "lg_randtopk\n(+local+sinks)",
              "lg_topk\n(+content top-k)"]
    keys = ["uniform_random_m", "local_global_randtopk", "local_global_topk"]
    colors = [C_RND, "#ff9f43", C_LG]
    fig, ax = plt.subplots(figsize=(9, 3.6))
    import numpy as np
    x = np.arange(len(models))
    w = 0.26
    for i, (key, lbl, c) in enumerate(zip(keys, labels, colors)):
        ys, errs = [], []
        for mo in models:
            rows = [r for r in m1 if r["model"] == mo and r["method"] == key]
            if rows:
                ys.append(rows[-1]["cos_mean"])
                errs.append(rows[-1].get("cos_std", 0.0))
            else:
                ys.append(float("nan"))
                errs.append(0.0)
        ax.bar(x + (i - 1) * w, ys, width=w, yerr=errs, capsize=3,
               color=c, label=lbl)
    ax.set_xticks(x)
    ax.set_xticklabels([short(m) + ("\n(hybrid)" if "3.5" in m else "\n(dense)")
                        for m in models])
    ax.set_ylabel("logit cosine vs dense (ctx 2048)")
    ax.set_ylim(0, 1.08)
    ax.legend(fontsize=8, loc="lower right")
    ax.set_title("Decomposition: structure (local+sinks) dominates fidelity; "
                 "content top-k adds +0.02-0.04", fontsize=10)
    fig.savefig(os.path.join(FIGS, "fig_decomposition.pdf"))
    plt.close(fig)


# ---------------------------------------------------------------- Fig 4
def fig_efficiency():
    fb = load("flex_bench.jsonl")
    fm = load("flex_model.jsonl")
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6))

    ax = axes[0]
    for hd, c in [(128, C_LG), (256, "#2ca02c")]:
        pts = sorted([r for r in fb if r["head_dim"] == hd], key=lambda r: r["seq"])
        ax.plot([r["seq"] for r in pts], [r["speedup"] for r in pts], marker="o",
                color=c, label=f"kernel-only, head_dim={hd}", linewidth=1.5)
    ax.axhline(1.0, color=C_OR, linestyle=":", linewidth=1)
    ax.set_xscale("log")
    ax.set_xlabel("sequence length")
    ax.set_ylabel("speedup vs dense SDPA")
    ax.set_title("(a) Attention kernel only (FlexAttention block-sparse)")
    ax.legend(fontsize=8)

    ax = axes[1]
    models = ["Qwen/Qwen3-0.6B", "Qwen/Qwen3-1.7B", "Qwen/Qwen3.5-4B-Base"]
    colors = [C_RND, "#ff9f43", C_LG]
    for mo, c in zip(models, colors):
        pts = sorted([r for r in fm if r["model"] == mo and r["method"] == "lg_topk_block"],
                     key=lambda r: r["context"])
        ax.plot([r["context"] for r in pts], [r["speedup"] for r in pts], marker="o",
                color=c, label=short(mo), linewidth=1.5)
        ooms = sorted({r["context"] for r in fm if r["model"] == mo and r["method"] == "OOM"})
        if ooms:
            ax.scatter(ooms, [0.55] * len(ooms), marker="x", color=c, s=40, zorder=5)
    ax.axhline(1.0, color=C_OR, linestyle=":", linewidth=1, label="break-even")
    ax.set_xscale("log")
    ax.set_xlabel("context length")
    ax.set_ylabel("end-to-end speedup")
    ax.set_ylim(0.5, 1.15)
    ax.set_title("(b) Full-model forward (block-sparse; x = flex OOM)")
    ax.legend(fontsize=8)
    fig.suptitle("Efficiency: the kernel speeds up, the model does not "
                 "(scoring ~96% of dense attention + unfused masked SDPA)", fontsize=10)
    fig.savefig(os.path.join(FIGS, "fig_efficiency.pdf"))
    plt.close(fig)


if __name__ == "__main__":
    fig_budget_curve()
    fig_fidelity_ctx()
    fig_decomposition()
    fig_efficiency()
    print("wrote:", ", ".join(sorted(os.listdir(FIGS))))
