"""KV cache eviction policies for efficient generation (Experiment 15).

During decode we keep only a bounded set of KV entries per full-attention layer:
  - recency: keep the last `keep` tokens (sliding cache).
  - sink+recency: keep the first `n_sink` tokens (attention sink) + last `keep`.
  - We evict from the model's KV cache after each decode step.

This reduces KV cache memory and per-step attention cost during long generation.
We measure quality (needle recall on long generation) and VRAM saved vs the dense
full cache.

Eviction is applied post-hoc to the DynamicCache: after each forward we crop the
cache to the allowed index set, so subsequent steps attend only to kept tokens.
"""

import torch


class EvictionPolicy:
    """Base eviction policy: returns indices to KEEP given current cache length."""

    def __init__(self, keep: int = 256, n_sink: int = 4):
        self.keep = keep
        self.n_sink = n_sink

    def keep_indices(self, cache_len: int, device):
        raise NotImplementedError


class RecencyEviction(EvictionPolicy):
    """Keep only the most recent `keep` tokens (plus sinks if configured)."""

    def keep_indices(self, cache_len: int, device):
        if cache_len <= self.keep + self.n_sink:
            return torch.arange(cache_len, device=device)
        sink = torch.arange(self.n_sink, device=device)
        recent = torch.arange(cache_len - self.keep, cache_len, device=device)
        return torch.cat([sink, recent])


class KNormEviction(EvictionPolicy):
    """Keep sinks + the `keep` tokens with the largest key norm (||k||).

    Motivated by two findings: (1) our LightGBM selector found k_norm the strongest
    importance signal; (2) Q2model found attention projection weights highly sensitive.
    Key norm is a near-free importance signal (no dot products needed)."""

    def keep_indices_from_keys(self, keys: torch.Tensor):
        # keys: (b, n_kv, cache_len, head_dim). Score = L2 norm over head_dim.
        cache_len = keys.shape[2]
        device = keys.device
        if cache_len <= self.keep + self.n_sink:
            return torch.arange(cache_len, device=device)
        knorm = keys[0].float().norm(dim=-1).mean(dim=0)  # mean over kv heads -> (cache_len,)
        sink = torch.arange(self.n_sink, device=device)
        # exclude sink positions from the top-k competition, then add them back
        scores = knorm.clone()
        scores[: self.n_sink] = float("-inf")
        top = scores.topk(self.keep).indices
        idx = torch.cat([sink, top]).unique().sort().values
        return idx


def evict_cache(past, policy: EvictionPolicy):
    """Crop a DynamicCache in-place according to the policy. Only full-attention
    layers (those with keys/values tensors) are evicted; linear/conv layers keep
    their recurrent state untouched. Returns nothing (in-place)."""
    if past is None or not hasattr(past, "layers"):
        return
    for ly in past.layers:
        k = getattr(ly, "keys", None)
        v = getattr(ly, "values", None)
        if not (torch.is_tensor(k) and torch.is_tensor(v)):
            continue  # linear/conv layer: skip
        cache_len = k.shape[2]
        if isinstance(policy, KNormEviction):
            idx = policy.keep_indices_from_keys(k)
        else:
            idx = policy.keep_indices(cache_len, k.device)
        if idx.numel() < cache_len:
            ly.keys = k.index_select(2, idx)
            ly.values = v.index_select(2, idx)


def cache_kv_bytes(past) -> int:
    """Total bytes stored in the KV cache (full-attention layers only)."""
    total = 0
    if past is None or not hasattr(past, "layers"):
        return 0
    for ly in past.layers:
        for attr in ("keys", "values"):
            t = getattr(ly, attr, None)
            if torch.is_tensor(t):
                total += t.numel() * t.element_size()
    return total
