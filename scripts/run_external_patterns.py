#!/usr/bin/env python3
"""Patrones publicados externos evaluados bajo los mismos controles.

Se senalo que el protocolo solo se ha aplicado a la familia de mascaras propia y pide
"Run at least one independent method under the same controls and show whether the controls change the
conclusions". MInference no es viable aqui (sin wheel para Python 3.14 ni torch 2.13, y su lista de
modelos soportados no incluye ninguno del paper: Qwen2/2.5, LLaMA-3, GLM-4, Yi, Phi-3). Se evaluan en
su lugar los dos patrones publicados clasicos que SI son comparables, porque corren sobre los mismos
modelos, en los mismos contextos y bajo los mismos controles:

  longformer  Beltagy et al. 2020 (ACL). Ventana deslizante DILATADA + globales.
  bigbird     Zaheer et al. 2020 (NeurIPS). Ventana + globales + BLOQUES ALEATORIOS de claves.

El diseno aisla una diferencia por comparacion, que es lo que hace que el experimento informe:

  streamingllm_w92   ventana 92 + 4 globales                 estructura sola, sin contenido
  longformer_d1      ventana 92 + 4 globales, dilatacion 1    == el anterior (control de degeneracion)
  longformer_d2      ventana 92 + 4 globales, dilatacion 2    MISMO presupuesto de claves, el doble de
                                                              campo receptivo. Aislado: solo cambia la
                                                              dilatacion.
  bigbird_r1         ventana 64 + 4 globales + 1 bloque       mismo presupuesto que la receta, pero el
                     aleatorio de 32 claves                   componente libre es aleatorio por BLOQUES
  lg_randtopk        ventana 64 + 4 globales + 32 claves      igual que el anterior pero aleatorio por
                     aleatorias                               CLAVE. Aislado: solo cambia la granularity
                                                              de la aleatoriedad.
  lg_topk            ventana 64 + 4 globales + top-32 por     la receta
                     contenido

BigBird frente a lg_randtopk es la comparacion informativa: ambos gastan el mismo presupuesto en un
componente libre no basado en contenido, y difieren solo en si ese componente se elige por bloques o
por claves. Si el protocolo es correcto, ambos deberian fallar en retrieval de aguja por igual.

Los pair ratios se MIDEN con el contador del interceptor, nunca se asumen: es la disciplina del propio
protocolo y lo unico que hace valida la afirmacion de presupuesto igualado.

Nota de consistencia: run_leaderboard.py usa streamingllm_bm con window=85 y run_ruler_battery.py con
window=topk_budget-4=92. Aqui se usa 92 y se etiqueta con el window explicito en el nombre del metodo,
para que no haya ambiguedad sobre que control es cada fila.

Uso:
  python scripts/run_external_patterns.py --model qwen3.5-0.8b --contexts 2048 4096 \
      --needle-seeds 15 --fidelity-chunks 5 --out results/REV2_external_patterns.jsonl
"""
import argparse
import json
import os
import platform
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, HERE)

from attention_interceptor import PAIR_COUNTER, reset_pair_counter  # noqa: E402
from attention_methods import AttentionMethod  # noqa: E402
from model_loader import MODELS, inspect_model, load_model  # noqa: E402
from run_leaderboard import run_fidelity, run_needle  # noqa: E402
from utils import append_jsonl  # noqa: E402


def build_methods(seed=42):
    """Receta, controles propios y los dos patrones publicados, todos al mismo presupuesto nominal."""
    return [
        ("dense", None),
        # la receta
        ("lg_topk_w64_g4_k32", AttentionMethod(name="local_global_topk", window=64, n_global=4,
                                               topk=32)),
        # control estructuralmente igualado a la receta
        ("lg_randtopk_w64_g4_k32", AttentionMethod(name="local_global_randtopk", window=64,
                                                   n_global=4, topk=32, seed=seed)),
        # BigBird: igual que el anterior pero el componente libre es aleatorio POR BLOQUES
        ("bigbird_w64_g4_r1_b32", AttentionMethod(name="bigbird", window=64, n_global=4,
                                                  n_rand_blocks=1, rand_block_size=32, seed=seed)),
        # estructura sola, ventana ancha (mismo presupuesto que la receta)
        ("streamingllm_w92_g4", AttentionMethod(name="local_global", window=92, n_global=4)),
        # Longformer con dilatacion 1: debe degenerar al anterior salvo por el convenio de ventana
        ("longformer_w92_g4_d1", AttentionMethod(name="longformer", window=92, n_global=4,
                                                 dilation=1)),
        # Longformer con dilatacion 2: mismas claves, doble campo receptivo
        ("longformer_w92_g4_d2", AttentionMethod(name="longformer", window=92, n_global=4,
                                                 dilation=2)),
        # control no estructurado igualado en presupuesto
        ("uniform_random_k96", AttentionMethod(name="uniform_random", topk=96, random_k=0,
                                               seed=seed)),
    ]


def _fmt(v, nd=4):
    """Formatea un numero o devuelve '-' si no existe, para que el log no mienta con None."""
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def platform_info():
    p = torch.cuda.get_device_properties(0)
    return {"gpu": p.name, "capability": f"{p.major}.{p.minor}",
            "vram_gb": round(p.total_memory / 1e9, 2), "torch": torch.__version__,
            "cuda": torch.version.cuda, "python": platform.python_version()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3.5-0.8b", choices=list(MODELS.keys()))
    ap.add_argument("--contexts", type=int, nargs="+", default=[2048, 4096])
    ap.add_argument("--fidelity-chunks", type=int, default=5)
    ap.add_argument("--needle-seeds", type=int, default=15)
    ap.add_argument("--needle-depths", type=float, nargs="+", default=[0.1, 0.5, 0.9])
    ap.add_argument("--gen-tokens", type=int, default=24)
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--no-quantize", action="store_true")
    ap.add_argument("--device-map", default="cuda")
    ap.add_argument("--methods", nargs="+", default=None,
                    help="restringir a un subconjunto de nombres")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None)
    ap.add_argument("--skip-fidelity", action="store_true",
                    help="no medir fidelidad; util cuando solo interesa el brazo de aguja")
    ap.add_argument("--per-probe", action="store_true",
                    help="guardar el resultado de cada sonda (depth, seed, hit) para tests apareados")
    args = ap.parse_args()

    out_path = args.out or os.path.join(HERE, "..", "results", "external_patterns.jsonl")
    out_path = os.path.abspath(out_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    info = platform_info()
    print("plataforma:", json.dumps(info, ensure_ascii=False), flush=True)
    model_id = MODELS[args.model]
    minfo = inspect_model(model_id)
    model, tok = load_model(model_id, dtype=args.dtype,
                            load_in_4bit=not args.no_quantize, device_map=args.device_map)
    model.eval()
    full_layers = minfo.full_attention_layers
    print(f"[load] {model_id} full_attn={len(full_layers)}/{minfo.num_hidden_layers} "
          f"heads={minfo.num_attention_heads}/{minfo.num_key_value_heads} "
          f"head_dim={minfo.head_dim}", flush=True)

    seeds = list(range(1000, 1000 + args.needle_seeds))
    methods = build_methods(args.seed)
    if args.methods:
        methods = [m for m in methods if m[0] in args.methods]
    print(f"metodos: {[n for n, _ in methods]}", flush=True)

    for ctx in args.contexts:
        for name, method in methods:
            t0 = time.time()
            # OJO: run_fidelity y run_needle aplican y restauran el metodo INTERNAMENTE (llaman a
            # apply_method/restore ellos mismos). Aplicarlo aqui ademas es un error silencioso grave:
            # run_fidelity calcula ref_logits ANTES de su propio apply_method, asi que si el modelo ya
            # viene parcheado desde fuera, la referencia "densa" es en realidad sparse y el coseno sale
            # 1.0000 para todos los metodos. Se detecta porque cos=1.0 convive con pair_ratio bajo.
            rec = dict(info)
            rec.update({"experiment": "external_patterns", "model": model_id, "context": ctx,
                        "method": name, "needle_seeds": args.needle_seeds,
                        "needle_depths": args.needle_depths,
                        "fidelity_chunks": args.fidelity_chunks})
            try:
                if not args.skip_fidelity:
                    fid = run_fidelity(model, tok, method, full_layers, ctx,
                                       n_chunks=args.fidelity_chunks)
                    rec.update(fid)

                # run_needle no reinicia el contador, asi que se reinicia aqui para aislar la fase
                reset_pair_counter()
                per_probe = [] if args.per_probe else None
                recall, hits, total = run_needle(model, tok, method, full_layers, ctx,
                                                 args.needle_depths, seeds,
                                                 gen_tokens=args.gen_tokens, records=per_probe)
                rec["needle_recall"] = round(recall, 4)
                rec["needle_hits"] = hits
                rec["needle_total"] = total
                if per_probe is not None:
                    rec["needle_per_probe"] = per_probe
                computed, full = PAIR_COUNTER["computed"], PAIR_COUNTER["full"]
                rec["pair_ratio_needle"] = round(computed / full, 5) if full else None
            except Exception as e:
                rec["error"] = f"{type(e).__name__}: {str(e)[:200]}"
            rec["wall_s"] = round(time.time() - t0, 1)
            append_jsonl(out_path, rec)
            print(f"  ctx={ctx:5d} {name:26s} cos={_fmt(rec.get('cos_mean'))} "
                  f"needle={rec.get('needle_recall','-')} "
                  f"({rec.get('needle_hits','-')}/{rec.get('needle_total','-')}) "
                  f"p_ratio={_fmt(rec.get('pr_mean'))} "
                  f"[{rec['wall_s']}s] {rec.get('error','')}", flush=True)
            torch.cuda.empty_cache()

    print(f"\nDONE -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
