# Experiments

Three layers, deliberately kept apart. The rule is one-directional: **a layer may
call the layer below it, and may not modify it.**

```
  3. experiments/scaf/            SCAF EXTENSION             (development runs)
        |  calls, never edits
  2. experiments/recency_bias/    THESIS RECENCY-BIAS STUDY  (not started)
        |  calls, never edits
  1. experiments/baseline/        ORIGINAL RAG2 BASELINE     (runnable now)
        |  calls, never edits
     architecture/scaf/           the SCAF admission policy
     architecture/rag2/           the reproduced RAG2 system
     preprocessing/ + data/       corpus, chunks, MedCPT index
```

Both arms are driven by one runner, so they provably share one corpus, one
candidate set, one generator and one evaluation protocol — only admission
differs:

```bash
python experiments/scripts/freeze_candidates.py --source medcpt --cache <cache>
python experiments/scripts/run_comparison.py --scientific \
    --rag2-filter rag2_perplexity --rag2-checkpoint <dir> --generator huggingface
```

See [`docs/architecture/architecture_map.md`](../docs/architecture/architecture_map.md)
for the full picture, and
[`docs/runbooks/windows_experiment_runbook.md`](../docs/runbooks/windows_experiment_runbook.md)
for the production procedure.

Why the separation is enforced rather than merely intended: the thesis measures
what the *original* filter does with evidence of different ages. If baseline code
ever learned about publication dates, the thing being measured would no longer
exist. So the baseline carries provenance and never reads it, and
`architecture/rag2/tests/test_metadata_isolation.py` fails the build if that changes — it
scans every module under `architecture/rag2/rag2/` for executable references to date, recency
or currency fields, and a companion test proves the scanner actually fires.

---

## 1. `baseline/` — original RAG² baseline

Runs the reproduced system exactly as the paper specifies, over this
repository's corpus. Nothing here is a thesis contribution; it is the comparison
point everything else is measured against.

Entry points live in `architecture/rag2/scripts/` (one CLI per stage). Configuration:
`architecture/rag2/configs/thesis_corpus.yaml`. Start from `architecture/rag2/README.md`, and read
`docs/reproduction/rag2_reproduction_audit.md` first for what is and is not verified.

Every run writes a manifest (resolved config, git commit, model revisions,
seeds, prompt hashes, package versions) next to its outputs. Record the manifest
fingerprint next to any number that reaches the thesis.

**Status: runnable, not yet run.** The MedCPT index is built separately; results
go in `docs/reproduction/reproduction_results.md`, blank until measured.

## 2. `recency_bias/` — thesis recency-bias experiments

The Filter Recency-Bias Probe: does the original RAG² filter admit evidence at
different rates depending on publication date? This layer *observes* the
baseline. It reads the intermediate signals the baseline already emits —
per-candidate ΔPPL, `P([HELPFUL])`, keep/drop decisions, and the
`canonical_date` / `date_precision` / `split_june_2024` provenance carried
through retrieval — and correlates them with date. It does not change how any of
them are produced.

The mechanism that makes this sound is candidate replay: `architecture/rag2/rag2/cache.py`
persists the retrieved-and-reranked candidate set with a retrieval fingerprint,
so every arm provably scores the same evidence population and only the thing
under study differs. That is validity control V3.

**Status: not started.** No code here yet, by instruction — the baseline must be
established first.

## 3. `scaf/` — future SCAF extension

Support–Currency–Authority–Filter: the proposed replacement scoring function,
`A(s) = w1·σ + w2·γ + w3·ρ + w4·τ`, with a three-state currency term and an
abstention/contested gate.

It will arrive as a new implementation of the existing
`rag2.filtering.base.EvidenceFilter` interface — the same seam
`PassthroughFilter` already uses for the paper's "RAG² w/o filter" ablation —
registered under its own key and selected by config. No baseline file changes.

**Status: not started, and deliberately so.** Nothing in this repository
implements recency weighting, authority weighting, currency scoring, supersession
or abstention. Do not add them until the baseline is measured: a correction has
nothing to correct until the bias it targets has been shown to exist.
