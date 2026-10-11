"""Bayesian optimization (Optuna) of approximate-attention hyperparameters.

Searches the (window, n_global, topk) space of the local_global_topk hybrid to
trace the quality/efficiency Pareto front. Objective: maximize fidelity (cosine)
while minimizing pair_ratio. We use a scalarized objective and report all trials
so the full Pareto front can be reconstructed.

Each evaluation is cheap: one forward pass at a fixed context + fidelity vs a
precomputed dense reference.

Usage:
  python scripts/run_optimize.py --model lfm2.5-350m-base --context 2048 --n-trials 24
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from attention_interceptor import apply_method, reset_pair_counter, restore, PAIR_COUNTER  # noqa: E402
from attention_methods import AttentionMethod  # noqa: E402
from model_loader import MODELS, inspect_model, load_model  # noqa: E402
from utils import append_jsonl, fidelity_metrics, set_seed  # noqa: E402

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results", "optimize.jsonl")


@torch.no_grad()
def get_logits(model, ids):
    return model(ids).logits[0].float().cpu()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODELS.keys()))
    ap.add_argument("--context", type=int, default=2048)
    ap.add_argument("--n-trials", type=int, default=24)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    model_id = MODELS[args.model]
    info = inspect_model(model_id)
    model, tok = load_model(model_id, dtype=args.dtype)
    device = next(model.parameters()).device
    full_layers = info.full_attention_layers
    print(f"[load] {model_id} full_attn={full_layers}")

    ids = tok("The history of the region spans many centuries. " * (args.context // 12),
              return_tensors="pt").input_ids[:, :args.context].to(device)
    L = ids.shape[1]
    dense_pairs = (L * (L + 1) // 2) * info.n_full_attention

    set_seed(args.seed)
    ref_logits = get_logits(model, ids)

    def evaluate(method):
        originals = apply_method(model, method, full_layers)
        set_seed(args.seed)
        try:
            approx = get_logits(model, ids)
            fid = fidelity_metrics(approx, ref_logits)
            reset_pair_counter()
            get_logits(model, ids)
            computed = PAIR_COUNTER["computed"]
        finally:
            restore(model, originals, full_layers)
        return fid, computed / dense_pairs

    def objective(trial):
        window = trial.suggest_categorical("window", [32, 64, 128, 256])
        n_global = trial.suggest_categorical("n_global", [0, 1, 4, 16])
        topk = trial.suggest_categorical("topk", [0, 16, 32, 64])
        method = AttentionMethod(name="local_global_topk", window=window,
                                 n_global=n_global, topk=topk, seed=args.seed)
        fid, ratio = evaluate(method)
        trial.set_user_attr("pair_ratio", ratio)
        trial.set_user_attr("kl_div", fid["kl_div"])
        trial.set_user_attr("logit_cosine", fid["logit_cosine"])
        # scalarized: reward fidelity, penalize compute. lambda trades quality vs cost.
        lam = 0.5
        score = fid["logit_cosine"] - lam * ratio
        append_jsonl(RESULTS, {"experiment": "optimize", "model": model_id,
                               "context": args.context, "window": window, "n_global": n_global,
                               "topk": topk, "pair_ratio": round(ratio, 4),
                               "logit_cosine": fid["logit_cosine"], "kl_div": fid["kl_div"],
                               "score": round(score, 5)})
        return score

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=args.seed))
    study.optimize(objective, n_trials=args.n_trials)

    print("\n=== Pareto-relevant trials (sorted by cosine) ===")
    trials = sorted(study.trials, key=lambda t: -t.user_attrs.get("pair_ratio", 1))
    for t in study.trials:
        p = t.params
        print(f"  w={p['window']:>4} g={p['n_global']:>2} k={p['topk']:>3} | "
              f"cos={t.user_attrs.get('logit_cosine', 0):.4f} "
              f"ratio={t.user_attrs.get('pair_ratio', 0):.3f} kl={t.user_attrs.get('kl_div', 0):.5f}")
    print("\n[best]", study.best_trial.params,
          "| cosine=", round(study.best_trial.user_attrs.get("logit_cosine", 0), 4),
          "| ratio=", round(study.best_trial.user_attrs.get("pair_ratio", 0), 3))


if __name__ == "__main__":
    main()
