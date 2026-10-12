# Manifest: from every table and figure to the data and code behind it

This file is the provenance manifest referred to in the manuscript's Data availability statement.
It maps each numbered table and figure of *The Discipline of Sparse Attention: A Budget-Matched
Validation from 350M to 27B* to the data file and the script that produced it.

Paths are relative to the root of this archive. `results/` holds the measurements, `scripts/` the
experiment drivers, `analysis/` the scripts that assemble the published aggregates from those
measurements, and `src/` the attention interceptor and harness.

## Evidence level

Each row states how the attribution was established.

| code | meaning |
|---|---|
| **V** | verified: the published numbers were recomputed from the named file, or the file's categorical vocabulary (methods, conditions, models, contexts) matches the table exactly |
| **A** | attributed: the file was named by the manuscript itself or by the recorded procedure of the campaign |
| **D** | definitional or derived: the table states conventions, or synthesises results reported elsewhere, so no single file is its source |

## Tables

| # | label | data file(s) | script(s) | ev. |
|---|---|---|---|---|
| 1 | `tab:notation` | — | — | D |
| 2 | `tab:names` | — (naming reference; the identifiers it lists occur in the `method` field of every file in `results/`) | — | D |
| 3 | `tab:protocol` | — | — | D |
| 4 | `tab:9b` | `results/robustness.jsonl` (ctx 2048 and 4096, n=5); `results/scale.jsonl` (the ctx 1024 row, single-run) | `scripts/run_robustness.py`, `scripts/run_scale.py` | A |
| 5 | `tab:4b` | `results/ci_repeat.jsonl` (Qwen3.5-4B-Base, seeds 42-45, per-chunk cosines in `cos_list`) | `scripts/run_ci_repeat.py` | V |
| 6 | `tab:decomp4` | `results/m1_control.jsonl` | `scripts/run_m1_control.py` | V |
| 7 | `tab:smatch` | `results/leaderboard.jsonl`, `results/m1_control.jsonl`, `results/robustness.jsonl`, `results/smollm_fidelity.jsonl`, `results/REV2_smollm_ci_n20.jsonl` | `analysis/analisis_ganancia_contenido.py` | V |
| 8 | `tab:dense` | `results/REV2_smollm_ci_n20.jsonl` (SmolLM2-1.7B, 20 seeds, 66 paired chunk observations per method) | `scripts/run_ci_repeat.py`, `analysis/actualiza_smollm2.py` | V |
| 9 | `tab:flops` | — (analytical, from the model configurations) | `scripts/flops_roofline.py` | V |
| 10 | `tab:consolidated` | `results/flex_model.jsonl` (the 4B, ctx 2048, block-sparse cell) | `scripts/run_flex_model.py` | A |
| 11 | `tab:cost` | `results/flex_model.jsonl`, `results/flex_bench.jsonl` (plus the analytical FLOPs rows) | `scripts/run_flex_model.py`, `scripts/run_flex_bench.py`, `scripts/flops_roofline.py` | A |
| 12 | `tab:granularity` | `results/REV2_flex_isolated.jsonl`, `results/REV2_flex_granularity.jsonl` | `scripts/verify_flex_isolated.py`, `scripts/run_flex_granularity.py` | A |
| 13 | `tab:budget` | `results/m5_budget_curve.jsonl`, `results/m5_budget_curve_4b.jsonl` | `scripts/run_m5_budget_curve.py` | V |
| 14 | `tab:needle` | `results/needle_diag.jsonl` (108 probes) | `scripts/run_needle_diag.py`, `scripts/utility_metrics.py` | V |
| 15 | `tab:ruler` | `results/REV2_battery_qwen3.5-0.8b.jsonl`, `results/REV2_battery_qwen3.5-4b-base.jsonl` (2600 rows each: 13 tasks x 40 seeds x 5 methods) | `scripts/rev2_battery.sh`, `analysis/analisis_bateria_rev2.py` | V |
| 16 | `tab:ruler_effect` | as row 15 | `analysis/analisis_bateria_rev2.py` | V |
| 17 | `tab:ruler_tasks` | as row 15 | `analysis/analisis_bateria_rev2.py` | V |
| 18 | `tab:leaderboard_08b` | `results/leaderboard.jsonl` | `scripts/run_leaderboard.py` | V |
| 19 | `tab:leaderboard_2b` | `results/leaderboard.jsonl` | `scripts/run_leaderboard.py` | V |
| 20 | `tab:leaderboard_4b` | `results/leaderboard.jsonl` (the rows of the notebook-tier campaign) | `scripts/run_leaderboard.py` | A |
| 21 | `tab:leaderboard_9b` | `results/leaderboard.jsonl`; ctx-4096 fidelity from `results/robustness.jsonl` | `scripts/run_leaderboard.py`, `scripts/run_robustness.py` | V |
| 22 | `tab:leaderboard_27b` | `results/leaderboard.jsonl` | `scripts/run_leaderboard.py` | V |
| 23 | `tab:external` | `results/REV2_external_patterns.jsonl` (16 cells, platform recorded per row) | `scripts/run_external_patterns.py`, `analysis/analisis_patrones_externos.py` | V |
| 24 | `tab:kv` | `results/h2o_snapkv.jsonl`, `results/quest.jsonl`, `results/kv_eviction.jsonl` | `scripts/run_h2o_snapkv.py`, `scripts/run_quest.py`, `scripts/run_kv_eviction.py` | V |
| 25 | `tab:knockout` | `results/layer_ablation.jsonl` (conditions `dense`, `ablate_all`, `loi_*`, `ctrl_short`; models 0.8B, 2B-Base, 4B-Base; contexts 256 and 4096) | `scripts/run_layer_ablation.py`, `scripts/run_layer_ablation_probe.py` | V |
| 26 | `tab:decode` | `results/REV2_decode_throughput.jsonl` | `scripts/run_decode_throughput.py` | A |
| 27 | `tab:guide` | — (synthesis of the results above) | — | D |
| 28 | `tab:platforms` | `environment_manifest.json` | — | D |

## Figures

All four figures are drawn by one script, `scripts/plot_paper_figs.py`, which reads only the files
listed here and writes the PDFs.

| # | label | output | data file(s) | ev. |
|---|---|---|---|---|
| 1 | `fig:fidctx` | `fig_fidelity_ctx.pdf` | `results/ci_repeat.jsonl`, `results/robustness.jsonl` | V |
| 2 | `fig:decomp` | `fig_decomposition.pdf` | `results/m1_control.jsonl` | V |
| 3 | `fig:eff` | `fig_efficiency.pdf` | `results/flex_bench.jsonl`, `results/flex_model.jsonl` | V |
| 4 | `fig:budget` | `fig_budget_curve.pdf` | `results/m5_budget_curve.jsonl`, `results/m5_budget_curve_4b.jsonl` | V |

## Results reported in the prose only

| result | data file | script | ev. |
|---|---|---|---|
| Paired needle test, dense vs the recipe at ctx 2048 (b=0, c=6, exact McNemar p=0.0312) | `results/REV2_needle_paired.jsonl` (per-probe outcomes in `needle_per_probe`) | `scripts/run_external_patterns.py` (with `--per-probe`), `analysis/analisis_needle_apareado.py` | V |
| Dense-posterior oracle top-k upper bound (Appendix J) | `results/oracle_topk.jsonl` | `scripts/run_oracle_topk.py` | A |
| QLoRA adaptation (Appendix A) | `results/lora_finetune.jsonl` | `scripts/run_lora_finetune.py` | A |
| Multi-fact retrieval with lexical distractors | `results/multifact_recall.jsonl` | `scripts/run_multifact_recall.py` | A |
| LongBench subset | `results/longbench_subset.jsonl` | `scripts/run_longbench_subset.py` | A |
| Per-layer sensitivity and ablations (Appendix D) | `results/layer_sensitivity.jsonl`, `results/layer_ablation.jsonl` | `scripts/run_layer_sensitivity.py`, `scripts/run_layer_ablation.py` | A |
| Layer-adaptive budget, proof of concept | `results/layer_adaptive.jsonl` | `scripts/run_layer_adaptive.py` | A |
| Generation probes and the same-model noise floor (Appendix F) | `results/gen_baselines.jsonl`, `results/generation.jsonl`, `results/variance.jsonl` | `scripts/run_gen_baselines.py`, `scripts/run_generation.py`, `scripts/run_variance.py` | A |
| Nystrom / landmarks, LSH, bandit selection: measured failures (Appendix C) | `results/lowrank.jsonl`, `results/approx.jsonl`, `results/optimize.jsonl`, `results/selector.jsonl` | `scripts/run_lowrank.py`, `scripts/run_approx.py`, `scripts/run_optimize.py`, `scripts/run_selector.py` | A |
| Domain detail: code and GSM8K (Appendix E) | `results/domains.jsonl`, `results/fidelity_metrics.jsonl` | `scripts/run_domains.py`, `scripts/run_fidelity_metrics.py` | A |
| Needle statistical power and cross-family retrieval: n=108 (6 depths x 18 seeds) at 0.8B and n=54 (6 depths x 9 seeds) at 4B; ctx 4096 at 0.8B; ctx 8192 at 0.8B and 2B-Base; Falcon-H1-0.5B-Base and SmolLM3-3B-Base | `results/needle_power.jsonl` (aggregate rows, one per model x method x context, with `n`, `hits`, `wilson_lo`, `wilson_hi` and the per-probe outcomes in `cases`); `results/needle_power_4096.jsonl` (the only source of the ctx-4096 cells at 0.8B); `results/needle_power_8192.jsonl` (raw ctx-8192 run, consolidated into `needle_power.jsonl`) | `scripts/run_needle_power.py` | V |

## Superseded data

`results/old_bug/` holds the measurements that a correction to decode-time key selection
invalidated. They are archived so that the correction can be audited, and **no number in the
manuscript is drawn from them**. `results/ruler_battery.jsonl` and `results/ruler_mcnemar.jsonl`
are *not* superseded: they are the original three-seed battery, left untouched by the balanced
n=40 re-run that produced `REV2_battery_qwen3.5-*.jsonl`, and they are kept as the record of what
the re-run replaced. `scripts/rev2_battery.sh` deliberately writes to new files rather than to
these.

## Known artefacts in the deposited files

`results/needle_diag.jsonl` contains 108 rows, but only 54 of them carry a value: the file records
two passes of `scripts/run_needle_diag.py` appended one after the other, and the first pass left
`recall: null`. The 54 complete rows are 18 per method (3 depths x 6 seeds), which is the `n=18`
that the caption of Table 14 declares, and they are the only rows used for any published number;
`results/utility_metrics.jsonl` accordingly reports `n_probes: 18`. **Filter on `recall is not null`
before aggregating.** The file is reproduced exactly as deposited in version 1 of this record and is
not corrected here, so that its checksum remains comparable across versions.

`results/ruler_battery.jsonl` contains 490 rows, but the 0.8B arm has 59 rows per method where only
53 are distinct on `(method, task, seed)`: seeds 42, 43 and 44 appear twice for the `fwe` and `cwe`
tasks, because a ten-seed run of those two tasks was appended to an earlier three-seed run without
removing it. The 4B arm (39 rows per method) carries no duplicates. **Deduplicate on
`(method, task, seed)` before aggregating.** Every aggregate published from this file was computed on
the deduplicated set and reproduces exactly: dense 41/53, `lg_topk_k32` 36/53, `lg_randtopk_k32`
12/53, `streamingllm_bm` 13/53 and `uniform_random_m` 2/53 at 0.8B, and dense 33/39, `lg_topk_k32`
32/39, `lg_randtopk_k32` 4/39, `streamingllm_bm` 3/39 and `uniform_random_m` 1/39 at 4B. Aggregating
the 0.8B arm without deduplicating gives 47/59, 39/59, 15/59, 17/59 and 2/59 in the same order, so
the duplication is a trap for a reader rather than an error in any published number. No figure in the
revised manuscript is drawn from this file: Tables 15 to 17 come from the balanced n=40 re-run in
`results/REV2_battery_qwen3.5-*.jsonl`, which uses seeds 1000-1039, gives 65 cells per model at
exactly n=40 and has zero duplicate keys. This file is reproduced exactly as deposited in version 1
of this record and is not corrected here, so that its checksum remains comparable across versions.

`scripts/utility_metrics.py` also contains a retrieval-per-pair branch that reads a `pair_ratio`
field from `results/needle_sparse.jsonl`. That field is not present in the file, so the branch
prints `RPP=inf`. It writes nothing: the RPP figures in the manuscript are computed directly as
recall / p_ratio from the reported working point.

## How file names map between the repository and this record

In the source repository the files keep their directory structure (`results/old_bug/...`). This
record stores files under a single flat level, so a file in a subdirectory is deposited with its
path separator replaced by a double underscore: `results/old_bug/ruler_battery.jsonl` is deposited
as `old_bug__ruler_battery.jsonl`. Files at the top level of `results/` keep their name unchanged.
Every other path in this manifest is the repository path.


## Naming

Method labels in the `method` field do not always equal the canonical name used in the paper,
because each experiment composed its label from the method name and its own configuration. Table 2
of the manuscript (`tab:names`) is the authoritative mapping; for example the recipe appears as
`lg_topk_k32`, `local_global_topk`, `lg_topk` and `lg_topk_perkey` in different files.

## Platforms

`environment_manifest.json` records the four platforms that produced the reported numbers, the exact
software stack of each, what each produced, and the evidence for each attribution. Wall-clock and
peak-VRAM figures are comparable only within a platform, never across platforms.
