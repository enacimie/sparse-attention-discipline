"""Qwen3.8-27B 4-bit: fidelidad ctx 2048 + needle retrieval."""
import sys, os, time, torch, random
sys.path.insert(0, "/root/mlattention/src")
from attention_interceptor import apply_method, reset_pair_counter, restore, PAIR_COUNTER
from attention_methods import AttentionMethod
from model_loader import inspect_model, load_model
from utils import fidelity_metrics, set_seed, append_jsonl

MODEL_PATH = "/root/qwen38"
OUT = "/root/mlattention/results/qwen38_run2.jsonl"

def main():
    torch.cuda.reset_peak_memory_stats()
    set_seed(42)
    model, tok = load_model(MODEL_PATH, load_in_4bit=True)
    info = inspect_model(MODEL_PATH)
    fl = info.full_attention_layers
    dev = next(model.parameters()).device

    # --- Fidelidad ctx 2048 (texto variado) ---
    paragraph = ("Machine learning has transformed the way we approach computation and data analysis. "
                 "Neural networks learn patterns from examples rather than explicit rules. "
                 "The transformer architecture relies on attention mechanisms to weigh token importance. "
                 "Large language models exhibit emergent capabilities as they scale in size. ")
    text = (paragraph * 90).strip()
    ids = tok(text, return_tensors="pt").input_ids[0][:2048]
    input_ids = ids.unsqueeze(0).to(dev)
    print(f"[ctx] {input_ids.shape[1]} tokens", flush=True)

    @torch.no_grad()
    def get_logits(m, inp):
        return m(inp).logits[0].cpu().float()

    methods = [
        ("dense", None),
        ("lg_topk", AttentionMethod(name="local_global_topk", window=64, n_global=4, topk=32)),
        ("lg_perhead", AttentionMethod(name="local_global_perhead", window=64, n_global=4, topk=32)),
        ("lg_randtopk", AttentionMethod(name="local_global_randtopk", window=64, n_global=4, topk=32, seed=42)),
        ("uniform_random", AttentionMethod(name="uniform_random", topk=32, seed=42)),
        ("streamingllm", AttentionMethod(name="local_global", window=89, n_global=4)),
    ]
    ref = get_logits(model, input_ids)
    for name, m in methods:
        reset_pair_counter()
        if m is None:
            lg, pr = ref, 1.0
        else:
            orig = apply_method(model, m, fl)
            lg = get_logits(model, input_ids)
            restore(model, orig, fl)
            pr = PAIR_COUNTER["computed"] / PAIR_COUNTER["full"]
        fm = fidelity_metrics(lg, ref)
        print(f"[fid] {name:14s} cos={fm['logit_cosine']:.4f} kl={fm['kl_div']:.5f} pr={pr:.4f}", flush=True)
        append_jsonl(OUT, {"metric":"fidelity","method":name,"ctx":2048,
                           "cos":round(fm["logit_cosine"],5),"kl":round(fm["kl_div"],6),
                           "pair_ratio":round(pr,5)})

    peak = torch.cuda.max_memory_allocated()/1e9
    print(f"[vram] peak {peak:.2f} GB", flush=True)
    append_jsonl(OUT, {"metric":"vram_peak_gb","value":round(peak,3)})

    # --- Retrieval: needle (passkey) con generación ---
    print("[needle] iniciando ...", flush=True)
    for depth in [0.1, 0.5, 0.9]:
        rng = random.Random(42 + int(depth*10))
        needle = str(rng.randint(1000000, 9999999))
        filler_tok = 1800
        filler = "The grass is green and the sky is blue today. " * (filler_tok // 10)
        fids = tok(filler, return_tensors="pt").input_ids[0]
        nids = tok(needle, return_tensors="pt").input_ids[0]
        pos = int(len(fids) * depth)
        seq = torch.cat([fids[:pos], nids, fids[pos:]])
        prompt_ids = torch.cat([seq, tok(" The passcode is", return_tensors="pt").input_ids[0]])
        inp = prompt_ids.unsqueeze(0).to(dev)
        try:
            out = model.generate(inp, max_new_tokens=48, do_sample=False, pad_token_id=tok.eos_token_id)
        except Exception as e:
            print(f"[needle] depth={depth} ERROR {e}", flush=True)
            append_jsonl(OUT, {"metric":"needle","depth":depth,"error":str(e)})
            continue
        gen = tok.decode(out[0][inp.shape[1]:], skip_special_tokens=True)
        hit = needle in gen
        print(f"[needle] depth={depth} hit={hit} gen={gen[:80]!r}", flush=True)
        append_jsonl(OUT, {"metric":"needle","depth":depth,"hit":hit,"needle":needle,"gen_head":gen[:120]})

    print("[done]", flush=True)

if __name__ == "__main__":
    main()
