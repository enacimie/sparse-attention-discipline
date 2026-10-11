#!/usr/bin/env python3
"""REV2 — actualiza la celda de SmolLM2-1.7B con la re-medicion apareada a n=66.

Origen de los numeros: datos_rev2/REV2_smollm_ci_n20.jsonl, generado por
scripts/run_ci_repeat.py --model smollm2-1.7b --contexts 2048 --seeds 42..61 --n-samples 5
--methods lg_topk lg_randtopk uniform_random --out results/REV2_smollm_ci_n20.jsonl
(20 seeds, 66 chunks por metodo, cos_list por muestra archivada).

Cambios:
  1. Parrafo "Cross-family dense replication" (4.3): cifras nuevas con el estadistico apareado.
  2. CORRECCION de una comparacion a contextos distintos. El texto decia que la fidelidad absoluta
     de SmolLM2 era menor que la de Qwen3-1.7B (0.958 vs 0.968). El 0.968 es Qwen3-1.7B a
     ctx 1024 (ci_repeat: 0.9673/0.9677/0.9688; robustness: 0.9694). A ctx 2048, que es donde se
     midio SmolLM2, Qwen3-1.7B da 0.9539 (ci_repeat, n=15) o 0.9495 (m1_control, n=5), es decir
     POR DEBAJO de los 0.9586 de SmolLM2. La comparacion se invierte a contexto igualado, asi que
     la conclusion sobre sensibilidad al recipe de entrenamiento no se sostiene y se retira.
  3. 4.1: SmolLM2 deja de entrar como "point estimate without a paired interval"; ya tiene
     estadistico apareado. Se retira el TODO(REV2) correspondiente.
  4. tab:smatch: fila de SmolLM2 y media global (+0.0280 -> +0.0274). El rango (+0.0159 a +0.0565)
     y la mediana (+0.0244) no cambian.
  5. Nombre del artefacto citado.

Cada sustitucion es literal y debe aparecer exactamente una vez; si alguna no encaja, aborta sin
escribir nada.
"""
import sys

PATH = "manuscript_blinded.tex"

CAMBIOS = [
    # --- 1 y 2. parrafo Cross-family dense replication -------------------------
    (
        r"""\paragraph{Cross-family dense replication.} On \texttt{HuggingFaceTB/SmolLM2-1.7B} (dense, Llama
architecture, FineWeb-Edu training, 24 layers, 32/32 MHA, held-out wiki-text, ctx 2048, $n=5$
seeds $\times$ 5 chunks, 4-bit), lg\_topk reaches a cosine of $0.9581\pm0.0062$, the
structurally-matched lg\_randtopk $0.9202\pm0.0568$ (content advantage $+0.042$ [0.013--0.071],
paired $t=3.98$, $p<0.02$), and the budget-matched uniform\_random $0.4163\pm0.0451$ (gap $+0.54$
vs lg\_topk). The pattern thus replicates outside the Qwen family: the content top-$k$ beats the
structurally-matched random control, and unstructured random collapses under the same budget. On
SmolLM2, uniform\_random does not collapse as far as on Qwen3-1.7B ($0.416$ vs $0.190$ at ctx
2048), suggesting that the FineWeb-Edu training recipe produces more sparsity-robust
representations. lg\_randtopk is also noisier ($\pm0.057$ vs $\pm0.006$ for lg\_topk), with
individual chunks dropping to cosine 0.77, which lg\_topk filters out (consistent with
Section~\ref{sec:retrieval}). Absolute fidelity is lower than on Qwen3-1.7B ($0.958$ vs $0.968$),
indicating sensitivity to the training recipe. Data: \texttt{results/smollm\_fidelity.jsonl}.""",
        r"""\paragraph{Cross-family dense replication.} On \texttt{HuggingFaceTB/SmolLM2-1.7B} (dense, Llama
architecture, FineWeb-Edu training, 24 layers, 32/32 MHA, held-out wiki-text, ctx 2048, 4-bit),
lg\_topk reaches a cosine of $0.9586\pm0.0067$ and the structurally-matched lg\_randtopk
$0.9308\pm0.0432$, measured over 20 seeds and 66 held-out chunks per method with the per-sample
cosines archived for all three arms. The content advantage is $+0.0277$, 95\% CI $[+0.018,+0.038]$,
paired $t=5.43$ on 65 degrees of freedom, $p<10^{-6}$, Cohen's $d=0.67$. The budget-matched
uniform\_random is $0.4143\pm0.0508$, a gap of $+0.54$ against lg\_topk. The pattern thus replicates
outside the Qwen family: the content top-$k$ beats the structurally-matched random control, and
unstructured random collapses under the same budget. On SmolLM2, uniform\_random does not collapse as
far as on Qwen3-1.7B ($0.414$ vs $0.190$ at ctx 2048), suggesting that the FineWeb-Edu training recipe
produces more sparsity-robust representations. lg\_randtopk is also far noisier than lg\_topk
($\pm0.043$ vs $\pm0.007$), with individual chunks dropping to cosine 0.78 and 6 of 66 below 0.85,
which lg\_topk filters out (consistent with Section~\ref{sec:retrieval}).

One comparison in this paragraph requires care, and we state it against ourselves. Absolute fidelity
on SmolLM2 is $0.9586$ at ctx 2048; on Qwen3-1.7B at the \textit{same} context it is $0.9539$
(\texttt{ci\_repeat.jsonl}, $n=15$) or $0.9495$ (\texttt{m1\_control.jsonl}, $n=5$), that is, slightly
\textit{lower} than SmolLM2 rather than higher. The two cells also come from different campaigns with
different held-out chunk draws, so they are not strictly matched beyond context. We therefore make no
claim about the training recipe from absolute fidelity levels. Data:
\texttt{results/REV2\_smollm\_ci\_n20.jsonl}.""",
    ),

    # --- 3. 4.1: SmolLM2 ya tiene estadistico apareado; fuera el TODO ----------
    (
        r"""scale. Second, SmolLM2-1.7B enters as a point estimate, $+0.038$ ($0.9581\pm0.0062$ vs
$0.9202\pm0.0568$, $n=25$), without a paired interval.
% TODO(REV2): re-correr SmolLM2-1.7B con scripts/run_ci_repeat.py (guarda cos_list) y restaurar
% aqui el t apareado y el IC recalculados. NO citar t=3.98 / [0.013-0.071] / +0.042: no salen de
% smollm_fidelity.jsonl (diferencia de medias = +0.037945). Ver NOTAS_REVISION.md apartado 3.2.""",
        r"""scale. Second, SmolLM2-1.7B is measured over 66 chunks per method with per-sample cosines archived,
giving $+0.0277$ with a paired 95\% CI of $[+0.018,+0.038]$ ($t=5.43$, $p<10^{-6}$,
Section~\ref{sec:dense}).""",
    ),

    # --- 4. tab:smatch: fila de SmolLM2 ----------------------------------------
    (
        r"SmolLM2-1.7B        & dense  & 2048 & 0.9581 & 0.9202 & $+0.0379$ \\",
        r"SmolLM2-1.7B        & dense  & 2048 & 0.9586 & 0.9308 & $+0.0277$ \\",
    ),
    # --- 4b. media global del caption y del pie --------------------------------
    (
        r"cells: $+0.0159$ to $+0.0565$, median $+0.0244$, mean $+0.0280$. Generated by",
        r"cells: $+0.0159$ to $+0.0565$, median $+0.0244$, mean $+0.0274$. Generated by",
    ),
    (
        r"\multicolumn{5}{l}{median / mean} & $+0.0244$ / $+0.0280$ \\",
        r"\multicolumn{5}{l}{median / mean} & $+0.0244$ / $+0.0274$ \\",
    ),
]


def main():
    src = open(PATH).read()
    for viejo, nuevo in CAMBIOS:
        n = src.count(viejo)
        if n != 1:
            print(f"ABORTA: patron aparece {n} veces (esperado 1).")
            print(f"  {viejo[:110]!r}")
            print("No se escribe nada.")
            sys.exit(1)
        src = src.replace(viejo, nuevo, 1)
        print(f"  ok  {viejo.splitlines()[0][:66]!r}")
    open(PATH, "w").write(src)
    print(f"\n{len(CAMBIOS)} cambios aplicados.")

    # verificacion: no deben quedar las cifras viejas
    restos = []
    for pat in ["0.9581", "0.9202", "+0.0379", "$+0.042$", "t=3.98", "0.013--0.071",
                "smollm\\_fidelity", "+0.0280", "0.958$ vs $0.968", "TODO(REV2): re-correr SmolLM2"]:
        if pat in src:
            restos.append(pat)
    print("cifras viejas que aun aparecen:", restos if restos else "ninguna")


if __name__ == "__main__":
    main()
