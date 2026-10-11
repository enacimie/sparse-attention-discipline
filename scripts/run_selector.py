"""Experiment 2+10: extract query-key features and train classical ML selectors.

Step 1 (extract): from the dense model, for sampled query positions in each
full_attention layer, compute cheap features for candidate keys and the label
(whether the key is in the true top-k by attention contribution).

Step 2 (train): fit LightGBM / logistic regression to predict top-k membership.
Evaluate precision@k / recall@k, then use the selector as an attention mask and
measure fidelity vs dense.

This is the core 'classical ML selector' experiment of the study.

Usage:
  python scripts/run_selector.py --model lfm2.5-350m-base --context 1024 --topk 64
"""

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from model_loader import MODELS, inspect_model, load_model  # noqa: E402
from attention_interceptor import apply_method, restore, reset_pair_counter, PAIR_COUNTER  # noqa: E402
from attention_methods import AttentionMethod  # noqa: E402
from utils import GpuMeter, append_jsonl, fidelity_metrics, set_seed  # noqa: E402

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results", "selector.jsonl")


@torch.no_grad()
def capture_layer_qkv(model, tok, ids, layer_idx):
    """Capture post-RoPE Q, K, V and the true attention output for one layer."""
    layer_list = None
    for n, mod in model.named_modules():
        if n.endswith("layers") and isinstance(mod, torch.nn.ModuleList):
            layer_list = mod
            break
    attn = getattr(layer_list[layer_idx], "self_attn", None) or getattr(layer_list[layer_idx], "attention")
    captured = {}
    orig = attn.forward

    def spy(hidden_states, position_embeddings, attention_mask=None, past_key_values=None, **kw):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, attn.head_dim)
        qp = attn.q_proj(hidden_states)
        if hasattr(attn, "q_norm"):
            q, _g = torch.chunk(qp.view(*input_shape, -1, attn.head_dim * 2), 2, dim=-1)
            q = attn.q_norm(q.view(hidden_shape)).transpose(1, 2)
            k = attn.k_norm(attn.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        else:
            qn = getattr(attn, "q_layernorm", None); kn = getattr(attn, "k_layernorm", None)
            q = attn.q_proj(hidden_states).view(hidden_shape)
            k = attn.k_proj(hidden_states).view(hidden_shape)
            if qn is not None:
                q = qn(q); k = kn(k)
            q = q.transpose(1, 2); k = k.transpose(1, 2)
        v = attn.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        cos, sin = position_embeddings
        mod = sys.modules.get(type(attn).__module__)
        rope = getattr(mod, "apply_rotary_pos_emb", None)
        if rope is not None:
            q, k = rope(q, k, cos, sin)
        captured["q"] = q.detach(); captured["k"] = k.detach(); captured["v"] = v.detach()
        return orig(hidden_states, position_embeddings, attention_mask, past_key_values, **kw)
    attn.forward = spy
    try:
        model(ids)
    finally:
        attn.forward = orig
    return captured["q"], captured["k"], captured["v"]


@torch.no_grad()
def build_dataset(q, k, v, topk, max_queries=128):
    """Build (features, label) for sampled queries. Features per (query_i, key_j):
    [qk_dot, rel_dist, abs_dist, q_norm, k_norm, recency]. Label: key_j in true top-k
    by contribution alpha_ij*||v_j||. Returns numpy arrays.
    """
    # q,k,v: (1, n_kv, L, d) -> mean over heads -> (L, d)
    qm = q[0].mean(0).float(); km = k[0].mean(0).float(); vm = v[0].mean(0).float()
    L, d = qm.shape
    # true attention scores (softmax over causal)
    scores = (qm @ km.T) / (d ** 0.5)
    causal = torch.tril(torch.ones(L, L, dtype=torch.bool, device=qm.device))
    scores = scores.masked_fill(~causal, float("-inf"))
    attn = torch.softmax(scores, dim=-1)                      # (L, L)
    contrib = attn * vm.norm(dim=-1).unsqueeze(0)             # alpha_ij * ||v_j||
    topk_idx = contrib.topk(min(topk, L), dim=-1).indices     # (L, topk)

    q_norm = qm.norm(dim=-1)
    k_norm = km.norm(dim=-1)
    feats, labels = [], []
    qi = torch.linspace(topk, L - 1, steps=min(max_queries, L - topk)).long()
    for i in qi.tolist():
        cand = torch.arange(0, i + 1, device=qm.device)       # causal candidates
        dots = (qm[i] * km[cand]).sum(-1)
        rel = (i - cand).float()
        f = torch.stack([
            dots, rel, rel.abs(), q_norm[i].expand_as(rel), k_norm[cand],
            1.0 / (1.0 + rel),
        ], dim=-1).cpu().numpy()
        lab = np.isin(cand.cpu().numpy(), topk_idx[i].cpu().numpy()).astype(np.int64)
        feats.append(f); labels.append(lab)
    return np.concatenate(feats), np.concatenate(labels)


def train_selector(X, y):
    """Train LightGBM to predict top-k membership. Returns the model."""
    import lightgbm as lgb
    pos = max(1, int(y.sum())); neg = max(1, int((y == 0).sum()))
    clf = lgb.LGBMClassifier(n_estimators=200, num_leaves=31, learning_rate=0.1,
                             scale_pos_weight=neg / pos, verbose=-1)
    clf.fit(X, y)
    return clf


@torch.no_grad()
def get_logits(model, ids):
    return model(ids).logits[0].float().cpu()


def fidelity_phase(model, tok, info, clf, ctx, budget, device, seed=42):
    """Close the loop: use the trained selector as an attention mask and compare its
    fidelity vs dense AND vs a uniform-random mask at the SAME pair budget.

    Evaluated on a DIFFERENT text than the selector's training data (held-out).
    """
    from attention_methods import AttentionMethod, set_selector
    set_selector(clf)
    full_layers = info.full_attention_layers
    ids = tok("Coastal winds shape the dunes and the tide charts of the harbor. "
              * (ctx // 12), return_tensors="pt").input_ids[:, :ctx].to(device)
    L = ids.shape[1]
    dense_pairs = (L * (L + 1) // 2) * info.n_full_attention

    set_seed(seed)
    ref = get_logits(model, ids)

    out = {}
    candidates = [
        ("selector", AttentionMethod(name="selector", topk=budget)),
        ("uniform_random", AttentionMethod(name="uniform_random", topk=budget, random_k=0)),
        ("topk", AttentionMethod(name="topk", topk=budget)),
    ]
    for name, m in candidates:
        originals = apply_method(model, m, full_layers)
        set_seed(seed)
        approx = get_logits(model, ids)
        fid = fidelity_metrics(approx, ref)
        reset_pair_counter()
        with torch.no_grad():
            get_logits(model, ids)
        computed = PAIR_COUNTER["computed"]
        restore(model, originals, full_layers)
        out[name] = {"pair_ratio": round(computed / dense_pairs, 4), **fid}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODELS.keys()))
    ap.add_argument("--context", type=int, default=1024)
    ap.add_argument("--topk", type=int, default=64)
    ap.add_argument("--fidelity-context", type=int, default=1024)
    ap.add_argument("--budgets", type=int, nargs="+", default=[32, 64])
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    model_id = MODELS[args.model]
    info = inspect_model(model_id)
    model, tok = load_model(model_id, dtype=args.dtype)
    device = next(model.parameters()).device
    set_seed(args.seed)
    ids = tok("The history of the region spans many centuries. " * (args.context // 12),
              return_tensors="pt").input_ids[:, :args.context].to(device)
    print(f"[load] {model_id} full_attn={info.full_attention_layers}")

    # build dataset from the LAST full-attention layer (most semantic)
    layer_idx = info.full_attention_layers[-1]
    q, k, v = capture_layer_qkv(model, tok, ids, layer_idx)
    X, y = build_dataset(q, k, v, args.topk)
    print(f"  dataset: X={X.shape} positives={int(y.sum())} ({100*y.mean():.1f}%)")

    # split by query (first 70% queries train, rest test)
    n = len(y)
    split = int(n * 0.7)
    clf = train_selector(X[:split], y[:split])
    proba = clf.predict_proba(X[split:])[:, 1]
    ytest = y[split:]
    # precision/recall @ topk predicted
    order = np.argsort(-proba)[: args.topk * 4]
    tp = ytest[order].sum()
    prec = tp / max(1, len(order))
    rec = tp / max(1, ytest.sum())
    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(ytest, proba)
    rec_d = {"experiment": "selector", "model": model_id, "layer": layer_idx,
             "context": args.context, "topk": args.topk, "precision": round(float(prec), 4),
             "recall": round(float(rec), 4), "roc_auc": round(float(auc), 4),
             "n_train": int(split), "n_test": int(n - split)}
    append_jsonl(RESULTS, rec_d)
    print(f"  LightGBM selector: AUC={auc:.3f} precision={prec:.3f} recall={rec:.3f}")
    print(f"  feature importances: {dict(zip(['qk_dot','rel','abs_rel','q_norm','k_norm','recency'], clf.feature_importances_))}")

    # --- Close the loop: selector AS an attention method vs random control ---
    for budget in args.budgets:
        fid = fidelity_phase(model, tok, info, clf, args.fidelity_context, budget,
                             device, seed=args.seed)
        print(f"  [fidelity ctx={args.fidelity_context} budget={budget}]")
        for name, r in fid.items():
            print(f"    {name:15s} pair_ratio={r['pair_ratio']:.4f} "
                  f"cos={r['logit_cosine']:.4f} kl={r['kl_div']:.4f}")
            append_jsonl(RESULTS, {"experiment": "selector_as_method", "method": name,
                                   "model": model_id, "context": args.fidelity_context,
                                   "budget": budget, **r})


if __name__ == "__main__":
    main()
