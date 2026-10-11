"""Qwen3.8-27B: benchmark comparativo 16-bit vs Q4 vs Q4+lg_topk vs Q4+H2O (A100 80GB)."""
import sys, os, time, torch, random
sys.path.insert(0, "/root/mlattention/src")
from attention_interceptor import apply_method, reset_pair_counter, restore, PAIR_COUNTER
from attention_methods import AttentionMethod
from model_loader import inspect_model, load_model
from kv_eviction import evict_cache, RecencyEviction, cache_kv_bytes
from needle import build_needle_prompt, check_needle, make_needle
from utils import fidelity_metrics, set_seed, append_jsonl

MODEL_PATH = "/root/qwen38"
OUT = "/root/mlattention/results/qwen38_bench.jsonl"

def get_logits(model, inp):
    with torch.no_grad():
        return model(inp).logits[0].cpu().float()

def measure_decode_tps(model, ids, n_tokens=48):
    cur = ids[:, -1:]
    past = None
    t0 = time.time()
    with torch.no_grad():
        for _ in range(n_tokens):
            out = model(cur, past_key_values=past, use_cache=True)
            past = out.past_key_values
            cur = out.logits[:, -1].argmax(dim=-1, keepdim=True)
    return n_tokens / (time.time() - t0)

def measure_needle(model, tok, dev, ctx, method=None, fl=None):
    hits = tot = 0
    for d in [0.1, 0.5, 0.9]:
        for s in [42, 43]:
            rng = random.Random(s)
            needle = make_needle(rng)
            prompt, ndl = build_needle_prompt(tok, ctx, needle, d)
            ids = tok(prompt, return_tensors="pt").input_ids.to(dev)
            with torch.no_grad():
                if method is not None:
                    orig = apply_method(model, method, fl)
                out = model.generate(ids, max_new_tokens=40, do_sample=False, pad_token_id=tok.eos_token_id)
                if method is not None:
                    restore(model, orig, fl)
            gen = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
            hits += int(check_needle(gen, ndl)); tot += 1
    return hits, tot

def measure_kv(model, tok, dev, ctx, kind, keep=128):
    tot_kv = 0.0; hits = tot = 0
    for d in [0.1, 0.5, 0.9]:
        for s in [42, 43]:
            rng = random.Random(s)
            needle = make_needle(rng)
            prompt, ndl = build_needle_prompt(tok, ctx, needle, d)
            ids = tok(prompt, return_tensors="pt").input_ids.to(dev)
            with torch.no_grad():
                out = model(ids, use_cache=True)
                past = out.past_key_values
                cur = ids[:, -1:]
                gen_ids = []
                for _ in range(32):
                    out = model(cur, past_key_values=past, use_cache=True)
                    past = out.past_key_values
                    if kind == "h2o":
                        evict_cache(past, RecencyEviction(keep=keep, n_sink=4))
                    cur = out.logits[:, -1].argmax(dim=-1, keepdim=True)
                    gen_ids.append(cur.item())
            gen = tok.decode(gen_ids, skip_special_tokens=True)
            hits += int(check_needle(gen, ndl)); tot += 1
            tot_kv += cache_kv_bytes(past) / 2**20
    return tot_kv / tot, hits, tot

def main():
    ctx = int(os.environ.get("CTX", "2048"))
    keep = int(os.environ.get("KEEP", "128"))
    set_seed(42)
    info = inspect_model(MODEL_PATH)
    fl = info.full_attention_layers

    paragraph = ("Machine learning has transformed computation and data analysis. "
                 "Neural networks learn patterns from examples. The transformer uses attention. "
                 "Large language models exhibit emergent capabilities as they scale. ")
    text = (paragraph * 90).strip()

    # ---------- 1. 16-bit (BF16) ----------
    print("[1/5] 16-bit ...", flush=True)
    torch.cuda.reset_peak_memory_stats()
    model, tok = load_model(MODEL_PATH, dtype="bfloat16")
    dev = next(model.parameters()).device
    ids = tok(text, return_tensors="pt").input_ids[0][:ctx].unsqueeze(0).to(dev)
    vram16 = torch.cuda.max_memory_allocated() / 1e9
    ref16 = get_logits(model, ids)  # referencia de calidad (BF16 denso)
    h16, t16 = measure_needle(model, tok, dev, ctx)
    tps16 = measure_decode_tps(model, ids)
    print(f"  VRAM={vram16:.1f}GB needle={h16}/{t16} tok/s={tps16:.1f}", flush=True)
    append_jsonl(OUT, {"config":"16bit","vram_gb":round(vram16,2),"needle_recall":round(h16/t16,3),
                       "tok_s":round(tps16,1)})
    del model; torch.cuda.empty_cache()

    # ---------- 2. Q4 (dense) ----------
    print("[2/5] Q4 ...", flush=True)
    torch.cuda.reset_peak_memory_stats()
    model, tok = load_model(MODEL_PATH, load_in_4bit=True)
    dev = next(model.parameters()).device
    ids = tok(text, return_tensors="pt").input_ids[0][:ctx].unsqueeze(0).to(dev)
    vram_q4 = torch.cuda.max_memory_allocated() / 1e9
    ref_q4 = get_logits(model, ids)
    q4_fid = fidelity_metrics(ref_q4, ref16)["logit_cosine"]  # quantization fidelity
    h_q4, t_q4 = measure_needle(model, tok, dev, ctx)
    tps_q4 = measure_decode_tps(model, ids)
    print(f"  VRAM={vram_q4:.1f}GB cos_vs_bf16={q4_fid:.4f} needle={h_q4}/{t_q4} tok/s={tps_q4:.1f}", flush=True)
    append_jsonl(OUT, {"config":"q4","vram_gb":round(vram_q4,2),"cos_vs_bf16":round(q4_fid,4),
                       "needle_recall":round(h_q4/t_q4,3),"tok_s":round(tps_q4,1)})

    # ---------- 3. Q4 + sparse ----------
    print("[3/5] Q4 + sparse ...", flush=True)
    methods = [
        ("lg_topk", AttentionMethod(name="local_global_topk", window=64, n_global=4, topk=32)),
        ("lg_perhead", AttentionMethod(name="local_global_perhead", window=64, n_global=4, topk=32)),
        ("lg_randtopk", AttentionMethod(name="local_global_randtopk", window=64, n_global=4, topk=32, seed=42)),
        ("uniform_random", AttentionMethod(name="uniform_random", topk=32, seed=42)),
    ]
    for name, m in methods:
        reset_pair_counter()
        orig = apply_method(model, m, fl)
        lg = get_logits(model, ids)
        restore(model, orig, fl)
        fm = fidelity_metrics(lg, ref_q4)     # sparse vs Q4-dense
        fm16 = fidelity_metrics(lg, ref16)    # sparse vs 16-bit (total)
        pr = PAIR_COUNTER["computed"] / PAIR_COUNTER["full"]
        h, t = measure_needle(model, tok, dev, ctx, method=m, fl=fl)
        print(f"  {name:14s} cos_q4={fm['logit_cosine']:.4f} cos_16bit={fm16['logit_cosine']:.4f} pr={pr:.4f} needle={h}/{t}", flush=True)
        append_jsonl(OUT, {"config":"q4_sparse","method":name,"cos_vs_q4":round(fm["logit_cosine"],4),
                           "cos_vs_16bit":round(fm16["logit_cosine"],4),"pair_ratio":round(pr,4),
                           "needle_recall":round(h/t,3)})

    # ---------- 4. KV eviction (dense vs H2O) ----------
    print("[4/5] KV eviction ...", flush=True)
    kvd, hd, td = measure_kv(model, tok, dev, ctx, "dense")
    kvh, hh, th = measure_kv(model, tok, dev, ctx, "h2o", keep=keep)
    print(f"  dense kv={kvd:.1f}MB needle={hd}/{td}", flush=True)
    print(f"  h2o   kv={kvh:.1f}MB needle={hh}/{th} (keep={keep})", flush=True)
    append_jsonl(OUT, {"config":"kv_dense","kv_mb":round(kvd,2),"needle_recall":round(hd/td,3)})
    append_jsonl(OUT, {"config":"kv_h2o","keep":keep,"kv_mb":round(kvh,2),"needle_recall":round(hh/th,3)})

    print("[done]", flush=True)

if __name__ == "__main__":
    main()
