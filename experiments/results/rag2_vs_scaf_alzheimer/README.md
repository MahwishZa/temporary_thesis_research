# RAG² vs SCAF — Alzheimer domain

Results directory for the comparison described in
`docs/runbooks/windows_experiment_runbook.md` sections 4A, 5A and 6A.

The experiment has three strictly separated components. Keeping them separate is
what makes the result defensible, so the layout mirrors them.

| Component | Status | Where |
| --- | --- | --- |
| **A. Frozen retrieval corpus** | unchanged, read-only | `data/corpora/`, `indexes/production/` — *not* here |
| **B. Filter training dataset** | built, committed | `training_dataset/` |
| **C. Evaluation questions** | unchanged, evaluation-only | `data/datasets/thesis_questions/dev_questions.jsonl` — *not* here |

B and C share no question. The check is in
`training_dataset/manifest.json` under `evaluation_leakage`, and it is a build
failure, not a warning: the builder refuses to write a manifest if any training
question matches an evaluation question.

## Layout

```
training_dataset/   built by architecture/rag2/scripts/03b_build_alzheimer_filter_labels.py
  manifest.json               every count, source, digest and parameter
  TRAINING_DATA_REPORT.md     generated from the manifest -- read section 1 first
  train.json                  release-schema training file (id/answer/dataset_name/question)
  validation.json             same schema, disjoint questions
  *.provenance.jsonl          per-example: chunk id, document, PMID/PMCID, BM25 rank, role
  question_index.jsonl        one row per question: text, split, source document
  human_validation_subset.jsonl   stratified sample, human_label deliberately blank

filter/             written by architecture/rag2/scripts/verify_filter_checkpoint.py
  checkpoint_verification.json    load test + validation metrics

candidates/         the frozen candidate set's metadata and digest
                    (the .jsonl itself stays in experiments/runs/, which is gitignored)

comparison/         written by experiments/scripts/run_comparison.py --out
  manifest.json               fairness checks, preconditions, both arms' summaries
  per_question.jsonl          full per-question traces for both arms
  paired_comparison.json      the paired view -- what the write-up quotes

report/             the human-readable write-up
```

`rag2/` and `scaf/` do not exist as separate directories: both arms are produced
by one runner in one pass over one frozen candidate set, and their per-question
results live side by side in `comparison/per_question.jsonl` under the `rag2` and
`scaf` keys. Splitting them into separate files would invite reading one arm
without the other, which is exactly what a paired comparison must not encourage.

## What may be claimed from this directory

An Alzheimer-specific supervised training dataset was constructed independently
of the 30-question evaluation set and used to adapt the RAG² evidence-selection
filter. RAG² and SCAF were then evaluated on the same frozen Alzheimer-domain
retrieval corpus and identical frozen candidate evidence.

## What may not

- **Not** an exact reproduction of RAG². The filter is flan-t5-small trained for
  8 epochs on corpus-derived labels; the paper uses flan-t5-large for 40 epochs
  on ΔPPL labels. Both departures are recorded in
  `architecture/rag2/configs/thesis_alzheimer_filter.yaml` and in the run manifest.
- **Not** an accuracy claim. The 30 evaluation questions have no gold answers, so
  no answer-quality metric is computed and none may be reported. The claim is
  about **evidence selection and admission behaviour**.
- **Not** gold labels. The training labels are weakly supervised. See section 1
  of `training_dataset/TRAINING_DATA_REPORT.md`, which states the failure modes.

## Reproducing

`training_dataset/` is committed and deterministic; rebuild it and the `sha256`
values in `manifest.json` should match. Everything else needs the GPU machine
holding `indexes/production/` — follow the runbook.
