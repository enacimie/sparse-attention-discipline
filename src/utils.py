"""Shared utilities for the efficient-attention study.

All code, comments and documentation are in English as required by prompt.md.
Provides: reproducible seeding, GPU memory/temperature metrics, JSONL run logging.
"""

import json
import os
import random
import time
from contextlib import contextmanager
from dataclasses import dataclass, field

import numpy as np
import torch

try:
    import pynvml  # provided by nvidia-ml-py

    pynvml.nvmlInit()
    _NVML_OK = True
except Exception:  # pragma: no cover - NVML may be unavailable
    _NVML_OK = False


def set_seed(seed: int = 42) -> None:
    """Seed all RNGs for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def _gpu_temp_c() -> float:
    """Return GPU core temperature in Celsius, or -1 if unavailable."""
    if not _NVML_OK:
        return -1.0
    try:
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        return float(pynvml.nvmlDeviceGetTemperature(h, 0))
    except Exception:
        return -1.0


@dataclass
class GpuMeter:
    """Track peak VRAM (MB) and GPU temperature around a block of work."""

    peak_vram_mb: float = 0.0
    start_vram_mb: float = 0.0
    temp_c: float = -1.0

    def __enter__(self):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            self.start_vram_mb = torch.cuda.memory_allocated() / 2**20
        return self

    def __exit__(self, *exc):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            self.peak_vram_mb = torch.cuda.max_memory_allocated() / 2**20
        self.temp_c = _gpu_temp_c()
        return False

    def to_dict(self) -> dict:
        return {
            "peak_vram_mb": round(self.peak_vram_mb, 1),
            "temp_c": round(self.temp_c, 1),
        }


def append_jsonl(path: str, record: dict) -> None:
    """Append a run record to a JSONL log file (one JSON object per line)."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


@contextmanager
def timer():
    """Context manager yielding a dict that gets 'seconds' filled in."""
    out = {"seconds": 0.0}
    t0 = time.perf_counter()
    try:
        yield out
    finally:
        out["seconds"] = time.perf_counter() - t0


def kl_divergence(approx_logits: torch.Tensor, ref_logits: torch.Tensor) -> float:
    """Mean KL(ref || approx) over sequence, both (seq_len, vocab)."""
    log_q = torch.log_softmax(approx_logits.float(), dim=-1)
    log_p = torch.log_softmax(ref_logits.float(), dim=-1)
    kl = (log_p.exp() * (log_p - log_q)).sum(-1).mean()
    return float(kl)


def mse(a: torch.Tensor, b: torch.Tensor) -> float:
    """Mean squared error between two tensors."""
    return float(torch.mean((a.float() - b.float()) ** 2))


def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    """Mean per-row cosine similarity between two (seq_len, d) tensors."""
    a = a.float().flatten(1) if a.dim() > 2 else a.float()
    b = b.float().flatten(1) if b.dim() > 2 else b.float()
    return float(torch.nn.functional.cosine_similarity(a, b, dim=-1).mean())


def fidelity_metrics(approx_logits: torch.Tensor, ref_logits: torch.Tensor) -> dict:
    """Bundle of fidelity metrics vs the dense baseline."""
    return {
        "kl_div": round(kl_divergence(approx_logits, ref_logits), 5),
        "logit_mse": round(mse(approx_logits, ref_logits), 5),
        "logit_cosine": round(cosine_sim(approx_logits, ref_logits), 5),
    }
