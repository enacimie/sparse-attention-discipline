"""Experiment 16: external retrieval for long context (replaces full attention).

Instead of attending to the whole context, we chunk it, embed each chunk with the
model's own input embeddings (mean-pooled, cheap), retrieve the top-k chunks most
relevant to the question, and give the model ONLY those chunks + question.

This tests whether retrieval substitutes long attention for needle-type tasks.

Methods: BM25 (rank_bm25) and dense chunk embeddings (model input embeddings + FAISS).
"""

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from needle import FILLER, make_needle  # noqa: E402


def chunk_text_by_tokens(tokenizer, text: str, chunk_tokens: int):
    """Split text into chunks of ~chunk_tokens tokens. Returns list[str]."""
    ids = tokenizer(text, add_special_tokens=False).input_ids
    chunks = []
    for i in range(0, len(ids), chunk_tokens):
        piece = ids[i:i + chunk_tokens]
        chunks.append(tokenizer.decode(piece))
    return chunks


def bm25_retrieve(question: str, chunks, top_k: int):
    """Retrieve top-k chunk indices by BM25."""
    from rank_bm25 import BM25Okapi
    tokenized = [c.lower().split() for c in chunks]
    bm25 = BM25Okapi(tokenized)
    scores = bm25.get_scores(question.lower().split())
    idx = np.argsort(scores)[::-1][:top_k]
    return sorted(idx.tolist())


def dense_retrieve(question: str, chunks, top_k: int, embed_fn):
    """Retrieve top-k chunk indices by dense embedding cosine similarity."""
    import faiss
    q = embed_fn([question])                       # (1, d)
    c = embed_fn(chunks)                           # (n, d)
    index = faiss.IndexFlatIP(c.shape[1])
    faiss.normalize_L2(c)
    faiss.normalize_L2(q)
    index.add(c)
    _, idx = index.search(q, top_k)
    return sorted(idx[0].tolist())


def random_retrieve(chunks, top_k: int, rng):
    """CONTROL: pick top-k chunks uniformly at random (no content signal)."""
    k = min(top_k, len(chunks))
    return sorted(rng.sample(range(len(chunks)), k))


def mean_pool_embed_fn(model, tok, device):
    """Mean-pool the model's input embeddings over chunk tokens (cheap, no full forward)."""
    emb = model.get_input_embeddings().weight  # (vocab, d)

    @torch.no_grad()
    def embed(texts):
        out = []
        for t in texts:
            ids = tok(t, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
            e = emb[ids].mean(dim=1)  # (1, d)
            out.append(e.float().cpu())
        return torch.cat(out, dim=0).numpy()
    return embed


def contextual_embed_fn(model, tok, device, max_len=192, batch=16):
    """Dense retriever using the model's last hidden state (mean-pool) as encoder.

    Measured result: POOR for retrieval (decoder hidden states are anisotropic, not trained
    for sentence similarity) — worse than BM25 and degrades with context length. Kept as a
    documented negative result / baseline.
    """
    @torch.no_grad()
    def embed(texts):
        out = []
        for i in range(0, len(texts), batch):
            sub = texts[i:i + batch]
            enc = tok(sub, add_special_tokens=False, return_tensors="pt", padding=True,
                      truncation=True, max_length=max_len).to(device)
            kwargs = {k: v for k, v in enc.items() if k in ("input_ids", "attention_mask")}
            res = model(output_hidden_states=True, use_cache=False, return_dict=True, **kwargs)
            h = res.hidden_states[-1]                      # (B, L, d) last-layer states
            mask = enc["attention_mask"].unsqueeze(-1).float()
            pooled = (h * mask).sum(1) / mask.sum(1).clamp(min=1)
            out.append(pooled.float().cpu())
        return torch.cat(out, dim=0).numpy()
    return embed


def sentence_transformer_embed_fn(model_name="all-MiniLM-L6-v2", device="cuda"):
    """Trained sentence encoder (contrastive) as the dense retriever. Loads lazily."""
    from sentence_transformers import SentenceTransformer
    st_model = SentenceTransformer(model_name, device=device)

    @torch.no_grad()
    def embed(texts):
        return st_model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return embed
