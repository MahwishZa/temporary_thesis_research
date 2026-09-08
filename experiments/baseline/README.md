# Baseline experiments — original RAG²

Runs of the reproduced RAG² system, unmodified. This directory holds run
outputs, configs specific to a run, and notes; the implementation lives in
`rag2/` and is not edited from here.

## Running a stage

From the repository root, with the corpus and index already built
(`preprocessing/pmc/build_chunks.py`, then `preprocessing/pmc/embed_chunks.py`):

```bash
# 1-2  rationale -> balanced retrieval -> rerank, cached once
python architecture/rag2/scripts/02_retrieve.py -c architecture/rag2/configs/thesis_corpus.yaml

# 3    perplexity labels (paper Figure 2) on the training split
python architecture/rag2/scripts/03_build_filter_labels.py -c architecture/rag2/configs/thesis_corpus.yaml \
    --candidates cache/candidates/<train cache>.jsonl

# 4    train the Flan-T5 filter (the paper's checkpoint is not distributed)
python architecture/rag2/scripts/04_train_filter.py -c architecture/rag2/configs/thesis_corpus.yaml --init-tokens
python architecture/rag2/scripts/04_train_filter.py -c architecture/rag2/configs/thesis_corpus.yaml \
    --train-file runs/<...>/filter_train.json --select

# 5-6  filter + generate, then score
python architecture/rag2/scripts/05_run_pipeline.py -c architecture/rag2/configs/thesis_corpus.yaml \
    --candidates cache/candidates/<test cache>.jsonl
python architecture/rag2/scripts/06_evaluate.py --predictions runs/<...>/predictions.jsonl
```

`runs/` and `cache/` are gitignored: they are large and machine-specific.

## What to record

For every number that reaches the thesis, record the manifest fingerprint that
produced it. Results and their gaps against the paper belong in
`docs/reproduction/reproduction_results.md` — as measured, never adjusted to close a gap.

## Before trusting a run

- `docs/reproduction/reproduction_results.md` is still blank. A blank row is the honest
  state, not an oversight.
- Check `index_meta.json` reports `production: true`. A stub-encoder index is
  refused by the corpus loader, but check anyway.
- Read `docs/reproduction/rag2_reproduction_audit.md` §10 for what remains unverified.
