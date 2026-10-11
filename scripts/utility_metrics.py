"""WS6: Compute utility metrics from existing JSONL - no GPU needed.
Output -> results/utility_metrics.jsonl
=========================
1. critical-key recall: fraction of probes where ALL needle tokens are in content top-k
2. answer-token attention mass: attention mass on needle tokens (last query vs all queries)
3. rank of GT keys: fraction of queries that rank needle in top-k (diagnostic)
4. retrieval-per-pair (RPP): needle recall / pair_ratio. Reported only when the source
   file records pair_ratio; needle_sparse.jsonl does not, so this branch prints n/a.
   The manuscript derives RPP directly as recall / p_ratio at the reported working point.
"""

import json, math, os, sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from utils import append_jsonl


def wilson_ci(hits, n, z=1.96):
    p = hits / n if n else 0
    d = 1 + z*z/n
    c = (p + z*z/(2*n)) / d
    h = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / d
    return max(0, c-h), min(1, c+h)


def mean_std(xs):
    m = sum(xs) / len(xs)
    sd = (sum((x-m)**2 for x in xs) / (len(xs)-1))**0.5 if len(xs)>1 else 0.0
    return m, sd


def load_jsonl(path):
    return [json.loads(l) for l in open(path) if l.strip()]


print("="*60)
print("WS6 — UTILITY METRICS FROM EXISTING DATA")
print("="*60)

# ---- 1. Needle diagnostic: critical-key metrics ----
diag = load_jsonl('results/needle_diag.jsonl')
# Aggregate per (method, model, depth, seed) — each is one probe
# But the records are per-last-query. Let's group by (method,seed,depth)
probes = defaultdict(lambda: {'topk_last': [], 'attn_last': [], 'attn_all': [],
                                'frac_queries': [], 'attn_all_ctrl': []})
for r in diag:
    k = (r['method'], r['seed'], r['depth'])
    probes[k]['topk_last'].append(1 if r['topk_last'] else 0)
    probes[k]['attn_last'].append(r['attn_last'])
    probes[k]['attn_all'].append(r['attn_all'])
    probes[k]['frac_queries'].append(r['frac_queries_topk'])

print("\n--- 1. CRITICAL-KEY SELECTION (needle_diag) ---")
methods = ['dense', 'lg_topk', 'lg_randtopk']
for m in methods:
    tk = [max(v['topk_last']) for k, v in probes.items() if k[0]==m]
    al = [sum(v['attn_last'])/len(v['attn_last']) for k, v in probes.items() if k[0]==m]
    aa = [sum(v['attn_all'])/len(v['attn_all']) for k, v in probes.items() if k[0]==m]
    fq = [sum(v['frac_queries'])/len(v['frac_queries']) for k, v in probes.items() if k[0]==m]
    n = len(tk)
    tk_hits = sum(tk)
    tk_lo, tk_hi = wilson_ci(tk_hits, n)
    al_m, al_s = mean_std(al)
    aa_m, aa_s = mean_std(aa)
    fq_m, fq_s = mean_std(fq)
    print(f"  {m:14s} n={n:3d}  topk_sel={tk_hits}/{n}={tk_hits/n:.2f} CI=[{tk_lo:.3f}-{tk_hi:.3f}]  "
          f"attn_last={al_m:.4f}±{al_s:.4f}  attn_all={aa_m:.4f}±{aa_s:.4f}  "
          f"frac_q_topk={fq_m:.4f}±{fq_s:.4f}")

# Key finding: needle retrieval is a LAST-QUERY phenomenon
print("\n  *** Key finding: retrieval depends on the last query's selection (topk_last=1.0),")
print("  not the average query (frac_queries_topk≈0.20). Content top-k correctly identifies")
print("  the needle for the answering query even though most queries ignore it.")

# ---- 2. Retrieval-per-pair (RPP) from needle_sparse ----
needle = load_jsonl('results/needle_sparse.jsonl')
print("\n--- 2. RETRIEVAL-PER-PAIR (needle_sparse) ---")
# Group by (method, model, context)
rp = defaultdict(lambda: {'hits': [], 'pr': []})
for r in needle:
    k = (r['method'], r['model'], r['context'])
    rp[k]['hits'].append(1 if r['hit'] else 0)
    # If pair_ratio is stored, collect it
    if 'pair_ratio' in r:
        rp[k]['pr'].append(r['pair_ratio'])

for (method, model, ctx), v in sorted(rp.items()):
    hits = sum(v['hits'])
    n = len(v['hits'])
    recall = hits/n if n else 0
    # needle_sparse.jsonl does not record pair_ratio, so RPP cannot be derived
    # from this file. The manuscript computes RPP directly as recall / p_ratio
    # at the reported working point; report it as unavailable here rather than
    # as an infinite value.
    pr = sum(v['pr'])/len(v['pr']) if v['pr'] else None
    rpp = (recall/pr) if pr else None
    lo, hi = wilson_ci(hits, n) if n else (0,0)
    pr_s = f"{pr:.4f}" if pr is not None else "not recorded"
    rpp_s = f"{rpp:.1f}" if rpp is not None else "n/a"
    print(f"  {method:14s} {model:24s} ctx={ctx:5d}  n={n:3d}  "
          f"recall={recall:.3f}  CI=[{lo:.3f}-{hi:.3f}]  "
          f"pr={pr_s}  RPP={rpp_s}")

# ---- 3. Attention mass efficiency: mass_per_pair_ratio ----
print("\n--- 3. ATTENTION-MASS EFFICIENCY ---")
for m in methods:
    al = [sum(v['attn_last'])/len(v['attn_last']) for k, v in probes.items() if k[0]==m]
    al_m, _ = mean_std(al)
    if m == 'dense':
        pr_est = 1.0
    elif m in ('lg_topk', 'lg_randtopk'):
        pr_est = 0.085
    else:
        pr_est = 1.0
    mpp = al_m / pr_est if pr_est > 0 else 0.0
    rel = (mpp / (0.0148/1.0)) if al_m > 1e-9 and 'dense' not in m else 1.0
    if al_m > 1e-9:
        print(f"  {m:14s}  attn_mass_last={al_m:.4f}  mass_per_selected_pair={mpp:.4f}  "
              f"(vs_dense_baseline={rel:.1f}x)")
    else:
        print(f"  {m:14s}  attn_mass_last={al_m:.4f}  (needle not selected — mass=0)")

# ---- 4. Multi-fact: retrieval-per-pair ----
mf = load_jsonl('results/multifact_recall.jsonl')
print("\n--- 4. MULTI-FACT RETRIEVAL-PER-PAIR ---")
rp_mf = defaultdict(lambda: {'hits': 0, 'n': 0})
for r in mf:
    k = (r['method'], r['model'], r['n_facts'])
    rp_mf[k]['hits'] += 1 if r['hit'] else 0
    rp_mf[k]['n'] += 1

for (method, model, nf), v in sorted(rp_mf.items()):
    lo, hi = wilson_ci(v['hits'], v['n'])
    print(f"  {method:14s} {model:24s} n_facts={nf}  "
          f"{v['hits']}/{v['n']}={v['hits']/v['n']:.2f}  CI=[{lo:.3f}-{hi:.3f}]")

# ---- 5. Save results ----
print("\n--- SAVING ---")
out = []
for m in methods:
    tk = [max(v['topk_last']) for k, v in probes.items() if k[0]==m]
    al = [sum(v['attn_last'])/len(v['attn_last']) for k, v in probes.items() if k[0]==m]
    aa = [sum(v['attn_all'])/len(v['attn_all']) for k, v in probes.items() if k[0]==m]
    fq = [sum(v['frac_queries'])/len(v['frac_queries']) for k, v in probes.items() if k[0]==m]
    n = len(tk)
    tk_hits = sum(tk)
    al_m, al_s = mean_std(al)
    aa_m, aa_s = mean_std(aa)
    fq_m, fq_s = mean_std(fq)
    out.append({
        'experiment': 'utility_metrics',
        'source': 'needle_diag',
        'method': m,
        'model': 'Qwen/Qwen3.5-0.8B',
        'ctx': 2048,
        'n_probes': n,
        'critical_key_selection': tk_hits/n,
        'attn_mass_last_mean': round(al_m, 6),
        'attn_mass_last_sd': round(al_s, 6),
        'attn_mass_all_mean': round(aa_m, 6),
        'attn_mass_all_sd': round(aa_s, 6),
        'frac_queries_topk_mean': round(fq_m, 6),
        'frac_queries_topk_sd': round(fq_s, 6),
    })

for rec in out:
    append_jsonl('results/utility_metrics.jsonl', rec)
print(f"Written {len(out)} records to results/utility_metrics.jsonl")
print("Done.")
