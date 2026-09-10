# `experiments/` — how the systems are tested

The research narrative lives in [`../THESIS.md`](../THESIS.md). This page only
says what is in this directory and what may import what.

```
scripts/        every runnable entry point — this is what you type
analysis/       libraries the scripts call: ablations, statistics, reporting
recency_bias/   the Filter Recency-Bias Probe (C1, the primary contribution)
configs/        experiment configuration
results/        GENERATED EVIDENCE — never hand-edited
```

**Source code and generated evidence are kept apart.** Everything under
`results/` is produced by a script and can be rebuilt from the frozen inputs;
nothing under it should ever be edited by hand.

## The one-directional rule

A layer may call the layer below it and may **not** modify it:

```
experiments/  ──calls──▶  architecture/scaf/  ──imports──▶  architecture/rag2/
```

`rag2` importing `scaf` is forbidden, and so is the baseline learning about
publication dates: the thesis measures what the *original* filter does with
evidence of different ages, so if baseline code could see a date, the thing being
measured would cease to exist.
`architecture/rag2/tests/test_metadata_isolation.py` fails the build if that
happens, and a companion test proves the scanner actually fires.

If an experiment appears to need a baseline change, that is a finding to write
down, not a patch to apply.

## Entry points

| Script | What it does | Runs where |
| --- | --- | --- |
| `scripts/freeze_candidates.py` | serialise the reranked candidate set once, for byte-identical replay | GPU machine |
| `scripts/run_comparison.py` | the RAG²-vs-SCAF admission comparison | GPU machine |
| `scripts/run_matched_k.py` | matched-k = 5 selection and generation | GPU machine (`--dry-run` anywhere) |
| `scripts/run_frb_probe.py` | the recency-bias probe, H1 + H4 | GPU machine (`--dry-run` anywhere) |
| `scripts/annotate.py` | the evidence-quality annotation interface | anywhere |
| `scripts/evidence_quality.py` | sample, audit and analyse the annotations | anywhere |
| `scripts/analyse_results.py` | the thesis-readable results report | anywhere |
| `scripts/answer_quality_readiness.py` | whether an answer-quality experiment can run yet | anywhere |
| `scripts/thesis_final_analysis.py` | **rebuild the whole final results package** | anywhere |

Scripts that reach a model or an index refuse rather than downgrade: they check
the frozen candidate digest, the generation endpoint and the annotation pass
before doing anything, and say exactly what is wrong when they stop.

## Results layout

```
results/rag2_vs_scaf_alzheimer/
  comparison_scientific/     THE scientific run. This name is recorded in
                             provenance — do not rename it.
  evidence_quality/          the independent annotation pass, its audit, and
    pilot_anchored/          the invalid anchored pilot, quarantined as history
  analysis/                  all derived analysis, including the final package
  comparison_development/    a superseded development run, kept as audit history
  training_dataset/          filter training data, manifest and provenance
```

Which of these may be reported, and why the rest may not, is decided by the
gates in `analysis/thesis_report.py` and printed in
[`../THESIS.md` §10](../THESIS.md#10-results-actually-obtained).
