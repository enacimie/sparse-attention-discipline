"""Approximate attention methods for full_attention layers.

Each method decides WHICH keys each query attends to (a selection mask), then we
compute attention ONLY over the selected pairs. This is where the real complexity
reduction happens: we avoid computing the full QK^T for position-based methods,
and for content-based methods we select a candidate subset before attending.

Methods are selected via AttentionMethod and applied by wrapping the attention
forward of the model's full_attention layers. The dense baseline is name="dense".
"""

from dataclasses import dataclass
import math

import torch


# Holder for a trained classical-ML selector (set via set_selector). Used by the
# "selector" content method to score query-key pairs offline-learned importance.
_SELECTOR = {"clf": None}


def set_selector(clf):
    """Install a trained selector (e.g. LightGBM) for the 'selector' attention method."""
    _SELECTOR["clf"] = clf


@dataclass
class AttentionMethod:
    """Configuration for an attention approximation method."""
    name: str = "dense"          # dense|local_window|local_global|topk|random_topk|uniform_random|selector|cluster|ann|local_global_topk|local_global_randtopk|lsh|bandit|nystrom
    window: int = 256            # sliding window size (local_window / local_global)
    n_global: int = 0            # always-attended global tokens (local_global)
    n_landmarks: int = 64        # landmark keys for nystrom
    topk: int = 64               # keys kept per query (topk / random_topk)
    random_k: int = 0            # extra uniformly-sampled keys per query (random_topk)
    n_clusters: int = 32         # clusters of keys (cluster)
    top_clusters: int = 4        # clusters selected per query (cluster)
    ann_k: int = 64              # neighbors retrieved per query (ann)
    mass_tau: float = 0.9        # cumulative softmax-mass target (local_global_adaptive)
    dilation: int = 1            # Longformer: atiende a 1 de cada `dilation` dentro de la ventana
    n_rand_blocks: int = 0       # BigBird: bloques de claves aleatorios por bloque de queries
    rand_block_size: int = 32    # BigBird: tamano de esos bloques
    seed: int = 42


def build_attention_mask(method: AttentionMethod, seq_len: int, device,
                         causal: bool = True) -> torch.Tensor:
    """Position-based keep-mask (seq_len, seq_len): True = pair is computed.

    Used for dense / local_window / local_global. Content-based methods
    (topk / cluster / ann) build their mask from q/k at runtime instead.
    """
    i = torch.arange(seq_len, device=device).view(-1, 1)
    j = torch.arange(seq_len, device=device).view(1, -1)
    causal_mask = (j <= i) if causal else torch.ones(seq_len, seq_len, dtype=torch.bool, device=device)

    if method.name == "dense":
        keep = causal_mask
    elif method.name in ("local_window", "local_global"):
        local = (i - j).abs() <= method.window
        if causal:
            local = local & (j <= i)
        keep = causal_mask & local
        if method.name == "local_global" and method.n_global > 0:
            g = min(method.n_global, seq_len)
            glob = torch.zeros(seq_len, seq_len, dtype=torch.bool, device=device)
            glob[:, :g] = True          # first g tokens are global
            glob[:, -g:] = True         # last g tokens are global
            keep = keep | (glob & causal_mask)

    elif method.name == "longformer":
        # Ventana deslizante dilatada + globales, en variante causal.
        # Con dilatacion d y ventana w, atiende a las claves con 0 <= (i-j) < w*d y (i-j) % d == 0:
        # w claves repartidas en un campo receptivo w*d. d=1 degenera en la ventana local normal.
        d = max(1, int(method.dilation))
        w = max(1, int(method.window))
        dist = i - j
        local = (dist >= 0) & (dist < w * d) & (dist % d == 0)
        keep = causal_mask & local
        if method.n_global > 0:
            g = min(method.n_global, seq_len)
            glob = torch.zeros(seq_len, seq_len, dtype=torch.bool, device=device)
            glob[:, :g] = True
            keep = keep | (glob & causal_mask)

    elif method.name == "bigbird":
        # Ventana deslizante + globales + bloques aleatorios de claves, en variante causal.
        # Los bloques aleatorios se eligen por BLOQUE DE QUERIES (no por query individual), que es
        # como lo define el paper: todas las queries de un bloque comparten los mismos bloques de
        # claves aleatorios. La semilla hace la realizacion reproducible.
        w = max(1, int(method.window))
        local = ((i - j) >= 0) & ((i - j) < w)
        keep = causal_mask & local
        if method.n_global > 0:
            g = min(method.n_global, seq_len)
            glob = torch.zeros(seq_len, seq_len, dtype=torch.bool, device=device)
            glob[:, :g] = True
            keep = keep | (glob & causal_mask)
        nb = max(1, int(method.rand_block_size))
        nrb = max(0, int(method.n_rand_blocks))
        if nrb > 0:
            n_qb = (seq_len + nb - 1) // nb
            gen = torch.Generator().manual_seed(int(method.seed))
            qb_of_query = torch.arange(seq_len, device=device) // nb
            rand_part = torch.zeros(seq_len, seq_len, dtype=torch.bool, device=device)
            for qb in range(n_qb):
                # solo bloques de claves causalmente alcanzables (kb <= qb)
                permitidos = list(range(0, qb + 1))
                if not permitidos:
                    continue
                k = min(nrb, len(permitidos))
                idx = torch.randperm(len(permitidos), generator=gen)[:k].tolist()
                elegidos = [permitidos[t] for t in idx]
                filas = (qb_of_query == qb)
                for kb in elegidos:
                    lo, hi = kb * nb, min((kb + 1) * nb, seq_len)
                    rand_part[filas, lo:hi] = True
            keep = keep | (rand_part & causal_mask)

    else:
        # Sin este aviso, un nombre mal escrito cae aqui y devuelve la mascara causal COMPLETA:
        # se mediria dense creyendo medir otra cosa, y la comparacion por pair_ratio quedaria
        # invalidada sin dejar rastro.
        import warnings
        warnings.warn(
            f"build_attention_mask: metodo desconocido {method.name!r}; "
            "se devuelve la mascara causal completa (equivalente a dense).",
            stacklevel=2)
        keep = causal_mask
    return keep


@torch.no_grad()
def build_content_mask(method: AttentionMethod, query: torch.Tensor, key: torch.Tensor,
                       causal: bool = True, value: torch.Tensor = None) -> torch.Tensor:
    """Content-based selection. query: (n_heads, q_len, d), key: (n_heads, k_len, d)
    (NOT yet GQA-repeated), value: (n_kv_heads, k_len, d) optional (value-aware
    oracles). Returns a boolean keep-mask (q_len, k_len) shared across
    heads: we take the union of important keys across heads so one sparse set is used.
    Only causally-allowed pairs are kept. We compute scores on head-mean q/k to avoid
    forming per-head full score matrices; the mask itself is (q_len, k_len).
    """
    if query.dim() == 4:
        query = query[0]  # drop batch (b=1 in local experiments)
    if key.dim() == 4:
        key = key[0]
    if value is not None and value.dim() == 4:
        value = value[0]
    n_h, q_len, d = query.shape
    k_len = key.shape[1]
    device = query.device
    # Queries occupy the LAST q_len positions of the kv_len cache. For prefill
    # offset=0; for autoregressive decode (q_len=1) offset=kv_len-1, so the single
    # query may attend to every past key.
    offset = k_len - q_len
    i = torch.arange(q_len, device=device).view(-1, 1)
    j = torch.arange(k_len, device=device).view(1, -1)
    q_pos = i + offset
    causal_mask = (j <= q_pos) if causal else torch.ones(q_len, k_len, dtype=torch.bool, device=device)

    qm = query.mean(dim=0)          # (q_len, d)
    km = key.mean(dim=0)            # (k_len, d)

    if method.name == "topk":
        # exact top-k by true dot product = upper bound for content methods
        scores = (qm @ km.T).masked_fill(~causal_mask, float("-inf"))
        k = min(method.topk, k_len)
        idx = scores.topk(k, dim=-1).indices
        keep = torch.zeros(q_len, k_len, dtype=torch.bool, device=device)
        keep.scatter_(1, idx, True)
        keep = keep & causal_mask
    elif method.name == "random_topk":
        scores = (qm @ km.T).masked_fill(~causal_mask, float("-inf"))
        k = min(method.topk, k_len)
        idx = scores.topk(k, dim=-1).indices
        keep = torch.zeros(q_len, k_len, dtype=torch.bool, device=device)
        keep.scatter_(1, idx, True)
        if method.random_k > 0:
            g = torch.Generator(device=device).manual_seed(method.seed)
            rand = torch.randint(0, k_len, (q_len, method.random_k), generator=g, device=device)
            keep.scatter_(1, rand, True)
        keep = keep & causal_mask
    elif method.name == "uniform_random":
        # CONTROL: pure uniform-random selection with the SAME budget as random_topk
        # (topk + random_k keys per query), ignoring content entirely. If a smart method
        # does not beat this at equal pair count, its apparent gain is an artifact.
        # Vectorized: sample random scores, rank within each causal row, keep top-(total_k).
        total_k = min(method.topk + method.random_k, k_len)
        g = torch.Generator(device=device).manual_seed(method.seed)
        rand_scores = torch.rand(q_len, k_len, generator=g, device=device)
        rand_scores = rand_scores.masked_fill(~causal_mask, float("-inf"))
        # number of valid candidates per row (causal, with decode offset)
        n_cand = (torch.arange(1, q_len + 1, device=device) + offset).clamp(max=k_len)
        kth = torch.minimum(torch.tensor(total_k, device=device), n_cand)  # per-row k
        # threshold: the kth largest random score per row
        sorted_scores, _ = rand_scores.sort(dim=-1, descending=True)
        thresh = sorted_scores.gather(1, (kth - 1).clamp(min=0).unsqueeze(1))  # (q_len,1)
        keep = (rand_scores >= thresh) & causal_mask
        # ensure at least the current query's own position is kept
        self_pos = (j == q_pos)
        keep = keep | (self_pos & causal_mask)
    elif method.name == "selector":
        # Classical-ML selector: keep the top-k pairs by trained importance score.
        keep = _selector_mask(qm, km, method, causal_mask)
    elif method.name == "cluster":
        # cluster keys, then per query pick top_clusters nearest centroids
        keep = _cluster_mask(qm, km, method, causal_mask)
    elif method.name == "ann":
        # approximate nearest keys per query via FAISS (content ANN)
        keep = _ann_mask(qm, km, method, causal_mask)
    elif method.name == "lsh":
        # random-projection (sign) hashing: bucket q and k, attend within same buckets
        keep = _lsh_mask(qm, km, method, causal_mask)
    elif method.name == "bandit":
        # UCB1 over key blocks: prefer high-scoring blocks, explore with UCB bonus
        keep = _bandit_mask(qm, km, method, causal_mask)
    elif method.name == "nystrom":
        # Nyström/landmarks: attend to a few landmark keys (low-rank proxy)
        keep = _nystrom_mask(qm, km, method, causal_mask)
    elif method.name == "local_global_topk":
        # hybrid: causal local window (look back `window`) + global tokens + content top-k
        dist = q_pos - j  # >=0 for causal past; we keep 0 <= dist <= window
        local = (dist >= 0) & (dist <= method.window) & causal_mask
        g = min(method.n_global, k_len)
        glob = torch.zeros(q_len, k_len, dtype=torch.bool, device=device)
        if g > 0:
            glob[:, :g] = True
            glob[:, -g:] = True
        base = local | (glob & causal_mask)
        if method.topk and method.topk > 0:
            scores = (qm @ km.T).masked_fill(~causal_mask, float("-inf"))
            k = min(method.topk, k_len)
            idx = scores.topk(k, dim=-1).indices
            tk = torch.zeros(q_len, k_len, dtype=torch.bool, device=device)
            tk.scatter_(1, idx, True)
            base = base | (tk & causal_mask)
        keep = base & causal_mask
    elif method.name == "local_global_randtopk":
        # STRUCTURALLY-MATCHED CONTROL (reviewer M1): identical to local_global_topk — same causal
        # local window + same sinks — but the `topk` free keys are chosen UNIFORMLY AT RANDOM
        # instead of by content score. Isolates the content-awareness contribution: lg_topk beats
        # the claim's null only if it beats THIS at equal pair budget (not just uniform_random,
        # which lacks the local/sinks structure).
        dist = q_pos - j  # >=0 for causal past; we keep 0 <= dist <= window
        local = (dist >= 0) & (dist <= method.window) & causal_mask
        g = min(method.n_global, k_len)
        glob = torch.zeros(q_len, k_len, dtype=torch.bool, device=device)
        if g > 0:
            glob[:, :g] = True
            glob[:, -g:] = True
        base = local | (glob & causal_mask)
        if method.topk and method.topk > 0:
            k = min(method.topk, k_len)
            gen = torch.Generator(device=device).manual_seed(method.seed)
            rand_scores = torch.rand(q_len, k_len, generator=gen, device=device)
            rand_scores = rand_scores.masked_fill(~causal_mask, float("-inf"))
            idx = rand_scores.topk(k, dim=-1).indices
            tk = torch.zeros(q_len, k_len, dtype=torch.bool, device=device)
            tk.scatter_(1, idx, True)
            base = base | (tk & causal_mask)
        keep = base & causal_mask
    elif method.name in ("local_global_oracle", "local_global_voracle"):
        # DENSE-POSTERIOR ORACLE (diagnostic upper bound, NOT a practical method):
        # identical structure to local_global_topk (same causal local window + sinks),
        # but the free top-k keys are chosen from the DENSE attention's own posterior:
        # per-query-head softmax weights (GQA-mapped) averaged over query heads
        # (oracle), or those weights times the value norm (voracle = value-aware
        # selection). Selection requires the full dense score matrix, which is why
        # this is an upper bound on what any selection at this budget could achieve,
        # not a deployable mask. Bounds the headroom of q.k-based scoring (reviewers
        # round-7 W5/Q4: oracle top-k and value-aware selection).
        dist = q_pos - j  # >=0 for causal past; we keep 0 <= dist <= window
        local = (dist >= 0) & (dist <= method.window) & causal_mask
        g = min(method.n_global, k_len)
        glob = torch.zeros(q_len, k_len, dtype=torch.bool, device=device)
        if g > 0:
            glob[:, :g] = True
            glob[:, -g:] = True
        base = local | (glob & causal_mask)
        if method.topk and method.topk > 0:
            n_q = query.shape[0]
            n_kv = key.shape[0]
            groups = max(1, n_q // n_kv)
            key_rep = key.repeat_interleave(groups, dim=0)[:n_q]
            scores = torch.bmm(query.float(), key_rep.float().transpose(1, 2)) / math.sqrt(d)
            scores = scores.masked_fill(~causal_mask.unsqueeze(0), float("-inf"))
            w = torch.softmax(scores, dim=-1).mean(dim=0)          # (q_len, k_len)
            if method.name == "local_global_voracle" and value is not None:
                vm = value.mean(dim=0).float().norm(dim=-1)        # (k_len,)
                w = w * vm.unsqueeze(0)
            k = min(method.topk, k_len)
            idx = w.topk(k, dim=-1).indices
            tk = torch.zeros(q_len, k_len, dtype=torch.bool, device=device)
            tk.scatter_(1, idx, True)
            base = base | (tk & causal_mask)
        keep = base & causal_mask
    elif method.name == "local_global_adaptive":
        # ADAPTIVE content top-k (reviewer W7.3/Q6): keep the minimal set of highest-
        # scoring keys whose cumulative softmax mass >= mass_tau, capped at topk. Peaked
        # attention spends fewer keys; diffuse attention spends up to topk. Same
        # local+sinks structure as local_global_topk so budgets stay comparable.
        dist = q_pos - j
        local = (dist >= 0) & (dist <= method.window) & causal_mask
        g = min(method.n_global, k_len)
        glob = torch.zeros(q_len, k_len, dtype=torch.bool, device=device)
        if g > 0:
            glob[:, :g] = True
            glob[:, -g:] = True
        base = local | (glob & causal_mask)
        if method.topk and method.topk > 0:
            scores = (qm @ km.T).masked_fill(~causal_mask, float("-inf"))
            probs = torch.softmax(scores, dim=-1)
            sorted_probs, sorted_idx = probs.sort(dim=-1, descending=True)
            cum = sorted_probs.cumsum(dim=-1)
            n_keep = (cum < method.mass_tau).sum(dim=-1) + 1
            n_keep = n_keep.clamp(min=1, max=min(method.topk, k_len))
            arange_k = torch.arange(k_len, device=device).unsqueeze(0)
            rank_mask = arange_k < n_keep.unsqueeze(1)
            tk = torch.zeros(q_len, k_len, dtype=torch.bool, device=device)
            tk.scatter_(1, sorted_idx, rank_mask)
            base = base | (tk & causal_mask)
        keep = base & causal_mask
    elif method.name == "local_global_entropy":
        # ENTROPY-MODULATED top-k (reviewer W5/W7.3): the per-query budget k scales with
        # the normalized entropy of the query's attention distribution. Low entropy (peaked)
        # -> few keys; high entropy (diffuse) -> up to topk. Same local+sinks structure.
        dist = q_pos - j
        local = (dist >= 0) & (dist <= method.window) & causal_mask
        g = min(method.n_global, k_len)
        glob = torch.zeros(q_len, k_len, dtype=torch.bool, device=device)
        if g > 0:
            glob[:, :g] = True
            glob[:, -g:] = True
        base = local | (glob & causal_mask)
        if method.topk and method.topk > 0:
            scores = (qm @ km.T).masked_fill(~causal_mask, float("-inf"))
            probs = torch.softmax(scores, dim=-1)
            entropy = -(probs * torch.log(probs.clamp(min=1e-12))).sum(dim=-1)
            n_cand = (torch.arange(1, q_len + 1, device=device) + offset).clamp(max=k_len).float()
            max_ent = torch.log(n_cand.clamp(min=2.0))
            norm_ent = (entropy / max_ent).clamp(0.0, 1.0)
            k_per_query = (1.0 + (method.topk - 1) * norm_ent).round().long()
            k_per_query = k_per_query.clamp(min=1, max=min(method.topk, k_len))
            sorted_scores, sorted_idx = scores.sort(dim=-1, descending=True)
            arange_k = torch.arange(k_len, device=device).unsqueeze(0)
            rank_mask = arange_k < k_per_query.unsqueeze(1)
            tk = torch.zeros(q_len, k_len, dtype=torch.bool, device=device)
            tk.scatter_(1, sorted_idx, rank_mask)
            base = base | (tk & causal_mask)
        keep = base & causal_mask
    elif method.name == "local_global_perhead":
        # UNION OF PER-HEAD top-k (reviewer W5/W7.3): each query head votes its own top-k
        # content keys and we keep the union (one shared sparse set). Captures head-specific
        # critical keys that head-averaged q.k scoring washes out. Budget = measured union
        # size (>= single top-k), reported by the harness. Same local+sinks structure.
        dist = q_pos - j
        local = (dist >= 0) & (dist <= method.window) & causal_mask
        g = min(method.n_global, k_len)
        glob = torch.zeros(q_len, k_len, dtype=torch.bool, device=device)
        if g > 0:
            glob[:, :g] = True
            glob[:, -g:] = True
        base = local | (glob & causal_mask)
        if method.topk and method.topk > 0:
            n_q = query.shape[0]
            n_kv = key.shape[0]
            groups = max(1, n_q // n_kv)
            key_rep = key.repeat_interleave(groups, dim=0)[:n_q]
            scores_h = torch.bmm(query.float(), key_rep.float().transpose(1, 2)) / math.sqrt(d)
            scores_h = scores_h.masked_fill(~causal_mask.unsqueeze(0), float("-inf"))
            k = min(method.topk, k_len)
            idx_h = scores_h.topk(k, dim=-1).indices
            tk = torch.zeros(q_len, k_len, dtype=torch.bool, device=device)
            for h in range(n_q):
                tk.scatter_(1, idx_h[h], True)
            base = base | (tk & causal_mask)
        keep = base & causal_mask
    elif method.name in ("lg_topk_block", "lgr_block"):
        # BLOCK-ALIGNED local+global+topk (for flex_attention real speedup). Whole blocks are
        # kept/masked so the block-sparse kernel can skip them. window = block size, topk = number
        # of top key-blocks per query-block, n_global>0 keeps block 0 (sinks). lg_topk_block picks
        # the top blocks by block-mean q.k; lgr_block is the RANDOM control at the same block budget.
        BLOCK = max(16, method.window)
        B_top = max(0, method.topk)
        n_kb = (k_len + BLOCK - 1) // BLOCK
        kblock = (torch.arange(k_len, device=device) // BLOCK).clamp(max=n_kb - 1)
        q_idx = torch.arange(q_len, device=device) + offset
        qblock = (q_idx // BLOCK).clamp(max=n_kb - 1)
        ones_k = torch.ones(k_len, device=device)
        km_b = torch.zeros(n_kb, d, device=device, dtype=km.dtype).index_add_(0, kblock, km)
        kc = torch.zeros(n_kb, device=device).index_add_(0, kblock, ones_k).clamp(min=1)
        km_b = km_b / kc.unsqueeze(1)
        ones_q = torch.ones(q_len, device=device)
        qm_b = torch.zeros(n_kb, d, device=device, dtype=qm.dtype).index_add_(0, qblock, qm)
        qc = torch.zeros(n_kb, device=device).index_add_(0, qblock, ones_q).clamp(min=1)
        qm_b = qm_b / qc.unsqueeze(1)
        qb = torch.arange(n_kb, device=device).view(-1, 1)
        kb = torch.arange(n_kb, device=device).view(1, -1)
        block_causal = kb <= qb
        keep_b = torch.eye(n_kb, dtype=torch.bool, device=device)          # diagonal = local
        if method.n_global > 0:
            keep_b[:, 0] = True                                            # sinks block
        if B_top > 0:
            strict = kb < qb
            B_sel = min(B_top, max(0, n_kb - 1))
            if B_sel > 0:
                if method.name == "lg_topk_block":
                    cand = (qm_b @ km_b.T).masked_fill(~strict, float("-inf"))
                else:
                    g = torch.Generator(device=device).manual_seed(method.seed)
                    cand = torch.rand(n_kb, n_kb, generator=g, device=device).masked_fill(
                        ~strict, float("-inf"))
                _, topb = cand.topk(B_sel, dim=-1)
                keep_b.scatter_(1, topb, True)
        keep_b = keep_b & block_causal
        keep = keep_b[qblock][:, kblock] & causal_mask
    else:
        keep = causal_mask
    return keep


def _selector_mask(qm, km, method, causal_mask):
    """Classical-ML selector: score every (query,key) pair with a trained model
    (LightGBM) on cheap features and keep the top-k predicted-important pairs per
    query (budget = method.topk). Features match build_dataset in run_selector.py:
    [qk_dot, rel, abs_rel, q_norm, k_norm, recency].
    """
    clf = _SELECTOR["clf"]
    if clf is None:
        raise RuntimeError("selector not installed; call set_selector(clf) first")
    q_len, d = qm.shape
    k_len = km.shape[0]
    device = qm.device
    i = torch.arange(q_len, device=device).view(-1, 1)
    j = torch.arange(k_len, device=device).view(1, -1)
    rel = (i - j).float()
    dots = qm.float() @ km.float().T                                   # unscaled, as trained
    qn = qm.float().norm(dim=-1).unsqueeze(1).expand(q_len, k_len)
    kn = km.float().norm(dim=-1).unsqueeze(0).expand(q_len, k_len)
    rec = 1.0 / (1.0 + rel.clamp(min=0))
    feats = torch.stack([dots, rel, rel.abs(), qn, kn, rec], dim=-1)   # (q,k,6)
    proba = clf.predict_proba(feats.cpu().numpy().reshape(-1, 6))[:, 1]
    scores = torch.from_numpy(proba.reshape(q_len, k_len)).to(device).float()
    scores = scores.masked_fill(~causal_mask, float("-inf"))
    k = min(method.topk, k_len)
    idx = scores.topk(k, dim=-1).indices
    keep = torch.zeros(q_len, k_len, dtype=torch.bool, device=device)
    keep.scatter_(1, idx, True)
    return keep & causal_mask


def _cluster_mask(qm, km, method, causal_mask):
    """Cluster keys with FAISS KMeans; attend only to top_clusters nearest clusters."""
    import numpy as np
    import faiss
    q_len, d = qm.shape
    k_len = km.shape[0]
    device = qm.device
    nc = min(method.n_clusters, max(2, k_len // 8))
    k_np = km.float().cpu().numpy()
    q_np = qm.float().cpu().numpy()
    kmeans = faiss.Kmeans(d, nc, niter=20, seed=method.seed, gpu=False)
    kmeans.train(k_np)
    cent = torch.from_numpy(kmeans.centroids).to(device)         # (nc, d)
    assign = torch.from_numpy(kmeans.assign(k_np)[1]).to(device)  # (k_len,)
    # per query: distance to centroids -> pick top_clusters
    qd = torch.cdist(qm.float(), cent.float())                    # (q_len, nc)
    tc = min(method.top_clusters, nc)
    topc = qd.topk(tc, dim=-1, largest=False).indices            # (q_len, tc)
    # build mask: pair (i,j) kept if assign[j] in topc[i]
    keep = torch.zeros(q_len, k_len, dtype=torch.bool, device=device)
    in_cluster = (assign.view(1, -1) == topc.view(q_len, tc, 1)).any(dim=1)  # (q_len, k_len)
    keep = in_cluster & causal_mask
    return keep


def _ann_mask(qm, km, method, causal_mask):
    """Approximate top-ann_k keys per query via FAISS inner product."""
    import numpy as np
    import faiss
    q_len, d = qm.shape
    k_len = km.shape[0]
    device = qm.device
    k_np = km.float().cpu().numpy()
    q_np = qm.float().cpu().numpy()
    index = faiss.IndexFlatIP(d)
    index.add(k_np)
    k = min(method.ann_k, k_len)
    _, idx = index.search(q_np, k)                               # (q_len, k)
    keep = torch.zeros(q_len, k_len, dtype=torch.bool, device=device)
    keep.scatter_(1, torch.from_numpy(idx).to(device), True)
    keep = keep & causal_mask
    return keep


def _lsh_mask(qm, km, method, causal_mask):
    """Locality-sensitive hashing via random projections (sign hashing).

    Project q and k onto n_hash random hyperplanes; two tokens share a bucket if
    their sign vectors match in enough bits. Each query attends to keys in its bucket.
    """
    q_len, d = qm.shape
    k_len = km.shape[0]
    device = qm.device
    n_bits = max(4, int(math.log2(max(k_len, 2))))  # ~log2(k_len) hash bits
    g = torch.Generator(device=device).manual_seed(method.seed)
    R = torch.randn(d, n_bits, generator=g, device=device, dtype=torch.float32)
    q_bits = torch.sign(qm.float() @ R)                          # (q_len, n_bits)
    k_bits = torch.sign(km.float() @ R)                          # (k_len, n_bits)
    # similarity of hash codes = number of matching bits
    matches = (q_bits.unsqueeze(1) == k_bits.unsqueeze(0)).sum(-1)  # (q_len, k_len)
    # keep keys whose hash matches in >= threshold bits (tunable via topk as proxy)
    threshold = max(1, int(n_bits * 0.6))
    keep = (matches >= threshold) & causal_mask
    # always allow self/local to avoid empty rows
    eye = torch.eye(q_len, k_len, dtype=torch.bool, device=device)
    keep = keep | (eye & causal_mask)
    return keep


def _bandit_mask(qm, km, method, causal_mask):
    """UCB1 contextual bandit over key blocks.

    Split keys into blocks; score each block by mean max-similarity (exploitation)
    plus a UCB exploration bonus inversely proportional to how often it was chosen.
    Keep keys in the top-scoring blocks. Stochastic, balances explore/exploit.
    """
    q_len, d = qm.shape
    k_len = km.shape[0]
    device = qm.device
    block = max(16, method.topk)                 # block size
    n_blocks = (k_len + block - 1) // block
    # exploitation: per-block mean of max query-key score
    scores = qm.float() @ km.float().T             # (q_len, k_len)
    block_score = scores.mean(dim=0)             # (k_len,) importance per key
    # pad to full blocks
    pad = n_blocks * block - k_len
    if pad > 0:
        block_score = torch.cat([block_score, torch.full((pad,), float("-inf"), device=device)])
    block_score = block_score.view(n_blocks, block).mean(dim=1)   # (n_blocks,)
    # UCB bonus: uniform prior on pulls -> add exploration term
    pulls = torch.ones(n_blocks, device=device)
    ucb = block_score + torch.sqrt(torch.log(torch.tensor(float(q_len), device=device)) / pulls)
    n_keep = max(1, min(n_blocks, (method.topk // block) + 1))
    top_blocks = ucb.topk(n_keep).indices
    keep_blocks = torch.zeros(n_blocks, dtype=torch.bool, device=device)
    keep_blocks[top_blocks] = True
    keep = keep_blocks.repeat_interleave(block)[:k_len]           # (k_len,)
    keep = keep.unsqueeze(0).expand(q_len, -1) & causal_mask
    eye = torch.eye(q_len, k_len, dtype=torch.bool, device=device)
    keep = keep | (eye & causal_mask)
    return keep


def _nystrom_mask(qm, km, method, causal_mask):
    """Nyström/landmark selection: pick a few landmark keys (uniform + high-norm) and
    attend only to them. Effective given the observed low effective rank (2-6)."""
    q_len, d = qm.shape
    k_len = km.shape[0]
    device = qm.device
    m = max(1, min(method.n_landmarks, k_len))
    # landmark selection: high-norm keys (stable) + recency, causal-respecting
    knorm = km.float().norm(dim=-1)                              # (k_len,)
    _, idx = knorm.topk(m)                                        # top-m norm keys
    keep = torch.zeros(q_len, k_len, dtype=torch.bool, device=device)
    keep[:, idx] = True
    # also keep a small local window for fluency
    i = torch.arange(q_len, device=device).view(-1, 1)
    j = torch.arange(k_len, device=device).view(1, -1)
    local = ((i - j) >= 0) & ((i - j) <= 16)
    keep = keep | local
    keep = keep & causal_mask
    return keep


def count_selected_pairs(mask: torch.Tensor) -> int:
    """Number of query-key pairs actually computed (efficiency metric)."""
    return int(mask.sum().item())
