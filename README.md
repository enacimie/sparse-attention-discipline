# The Discipline of Sparse Attention — validation artifacts and code

Machine-readable measurements, experiment drivers and analysis code for

> **The Discipline of Sparse Attention: A Budget-Matched Validation from 350M to 27B**
> Eduardo Nacimiento-García and Carina S. González-González, University of La Laguna
> *Neural Computing and Applications*

DOI (always resolves to the current version): [10.5281/zenodo.22229672](https://doi.org/10.5281/zenodo.22229672)

**Start with [`MANIFEST.md`](MANIFEST.md)**, which maps every numbered table and figure of the paper
to the data file and the script that produced it, and states how each attribution was established.

## What the study measures

Hybrid Transformers interleave a few dense full-attention layers with sub-quadratic linear or
convolutional layers. Only the dense layers carry the quadratic cost, so only they need
approximation. This archive contains the measurements behind a controlled protocol for validating
any training-free sparse mask, and the three boundaries it found.

1. **Fidelity.** Under a fixed pair budget the local-window structure dominates. The content top-*k*
   adds a small but consistent gain over a *structurally-matched* random control (+0.016 to +0.057,
   median +0.024, over 15 cells spanning 0.6B to 27B). The much larger gap over an *unstructured*
   random mask is mostly that structure, because unstructured random collapses.
2. **Retrieval.** Fidelity does not predict retrieval. Content-aware key selection preserves exact
   needle retrieval near the dense level at ~9% of the causal pairs, while random and structure-only
   controls hold moderate-to-high logit fidelity yet recall almost nothing. In the 13-task
   RULER-style battery at 520 probes per method per model, six published or control patterns that do
   not select keys by content score 0/270 at each context, against 44/45 and 45/45 for the recipe at
   matched measured budget.
3. **Efficiency.** The pair reduction is a logical budget, not a wall-clock saving. Selection
   requires near-full scoring and current backends run masked attention unfused, so measured speedup
   is negative; retrieval-preserving selection needs per-key granularity that block-sparse kernels
   with blocks of 128 or more cannot provide.

The recipe under test is deliberately simple: a causal local window of 64 tokens, the first 4
positions as sinks, and a content top-*k* of 32 free keys, applied only to the full-attention layers.

## Layout

```
src/                     the attention interceptor, method registry, model loader and harness
scripts/                 one driver per experiment, plus the figure script and the analysis utilities
analysis/                the scripts that assemble the published aggregate tables from the raw rows
results/                 the measurements: one JSONL file per experiment, one row per measurement
results/old_bug/         measurements invalidated by a correction to decode-time key selection (see below)
results/provenance/      the execution log of the notebook-tier campaign, which is the auditable
                         evidence for that platform's driver, library versions and the out-of-memory
                         ceiling reported in the paper
results/figs/            the four figures, regenerable with scripts/plot_paper_figs.py
MANIFEST.md              table and figure -> data file -> script, with the evidence level of each row
environment_manifest.json  the four platforms, their exact software stacks, and what each produced
requirements.txt         the reference-host stack; per-platform versions are in the manifest
LICENSE                  GNU GPL version 3, covering src/, scripts/ and analysis/
LICENSE-data             CC-BY-SA-4.0, covering results/
CITATION.cff
```

Each row of each JSONL file carries at least the model, the method, the task, the context, the seed
and the outcome, so any cell of any table can be traced back to the rows behind it. Files from the
2026 campaigns additionally record the GPU, compute capability, driver, PyTorch, CUDA and Python
versions, peak VRAM and wall-clock **per row**.

## Reproducing a number

The drivers are independent and each writes one file. For example, to regenerate the block-granularity
benchmark behind Table 12:

```bash
python scripts/run_flex_granularity.py --out results/REV2_flex_granularity.jsonl
```

To regenerate the four figures:

```bash
python scripts/plot_paper_figs.py
```

To regenerate the RULER-style battery at n=40 per task (this is a long run; see the header of the
script for the per-model cost):

```bash
bash scripts/rev2_battery.sh
```

The aggregate tables are then assembled from those rows by the scripts in `analysis/`, which print
LaTeX-ready output. For instance `analysis/analisis_bateria_rev2.py` produces Tables 15, 16 and 17,
including the Wilson intervals, the paired effect sizes and the exact McNemar tests.

## Caveats a reuser should know

- **Method labels are not uniform.** Each experiment composed its label from the method name and its
  own configuration, so the recipe appears as `lg_topk_k32`, `local_global_topk`, `lg_topk` and
  `lg_topk_perkey` in different files. Table 2 of the paper is the authoritative mapping.
- **Timings are not comparable across platforms.** Four platforms produced these numbers (an 8 GB
  reference host, a rented 24 GB card, a rented 80 GB card and a two-GPU notebook tier whose cards
  report 14.56 GiB usable). Wall-clock and peak-VRAM figures are comparable only within a platform.
- **`results/old_bug/` is superseded data.** A correction to decode-time key selection invalidated the
  earlier single-forward retrieval numbers. Those rows are archived so the correction can be audited,
  and no number in the paper is drawn from them.
- **Some cells are single-run.** The leaderboard arms at 4B and above and the two Falcon-H1 scales
  carry fidelity over 5 chunks and needle recall over 9 probes. The paper marks these as exploratory;
  the claims that depend on sample size rest on the n=40 batteries and the 45-probe needle arms.
- **Hybrid-model kernels were unavailable on the notebook tier.** `causal_conv1d`,
  `flash-linear-attention` and `mamba_ssm` could not be installed there, so those models fell back to
  reference PyTorch implementations. Fidelity results are unaffected; timings on that platform are not
  comparable to the others.

## Licensing

The software is licensed under the GNU General Public License version 3 (GPL-3.0-only). The
measurements under `results/` are licensed under the Creative Commons Attribution-ShareAlike 4.0
International licence (CC-BY-SA-4.0). Full texts in `LICENSE` and `LICENSE-data`. The two licences
cover disjoint parts of the archive: no file is under both.

## Use of artificial intelligence

Artificial intelligence tools were used for review and translation and for assistance with the scripts
and the analyses, specifically the OpenCode command-line interface together with models of the Qwen3.8
family. The authors reviewed and edited the content and take full responsibility for it.
