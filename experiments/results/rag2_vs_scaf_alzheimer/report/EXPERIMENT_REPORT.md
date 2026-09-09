# RAG² vs SCAF on the Alzheimer corpus — experiment report

**Status: prepared and verified end to end; the scientific run has not been
executed.** Sections 1–4 are complete and their numbers are real. Sections 5–7
are the run itself and are marked `NOT YET RUN`; they must be filled from the
files the runbook produces, not from expectation. Nothing in this file may be
quoted as a result until section 5 carries a frozen-set digest.

Branch: `claude/pubmed-acquisition-pipeline-8aikdr`. Run
`git log --oneline -5` for the commit this file is part of — an earlier draft
named a SHA here, which can never be its own commit's and was stale on arrival.

---

## 1. Design

Three components, kept strictly separate.

| | What | State |
| --- | --- | --- |
| **A** Retrieval corpus | PubMed + PMC + currency pack, chunked, MedCPT index (773,183 × 768) | **Frozen.** Not rebuilt, not re-embedded, not modified. |
| **B** Filter training data | 800 Alzheimer questions built from the corpus | **New.** Section 2. |
| **C** Evaluation questions | `data/datasets/thesis_questions/dev_questions.jsonl`, 30 questions | **Unchanged. Never trained on.** |

B and C are disjoint by construction and by check: the builder refuses to write
a manifest if any training question matches an evaluation question. Zero exact
overlap; highest token Jaccard against any evaluation question is 0.4286, which
is a different question, not a near-duplicate.

The corpus was read to build B and was not written to. `manifest.json` records
`corpus_modified: false`, `index_used: false`, and the corpus `sha256`.

---

## 2. Training dataset — `training_dataset/`

Full detail in `training_dataset/TRAINING_DATA_REPORT.md`, generated from
`manifest.json` so the prose and the numbers cannot diverge.

| | |
| --- | --- |
| questions | 800 |
| examples | 2,466 (train 1,971 / validation 495) |
| questions train / validation | 640 / 160, disjoint, seed 42 |
| positives `[HELPFUL]` | 822 |
| negatives `[NOT_HELPFUL]` | 1,644 (822 hard, 822 easy) |
| positive fraction | 0.3333, identical across train and validation |
| source documents | 800, one per question |
| documents contributing evidence | 2,313 |
| distinct PMIDs / PMCIDs | 2,313 / 1,659 |

### Label class: WEAK SUPERVISION

Not gold. Not human-validated. Not model-generated. Derived deterministically
from corpus structure:

- **questions** — the source document's own title (when interrogative, 20 of
  800) or its `OBJECTIVE:`/`AIM:` statement (780 of 800), the authors' wording
  preserved inside one of four content-free interrogative frames;
- **positives** — an abstract window of that same document, *excluding* the
  window the objective was read from, so the evidence answers the question
  rather than restating it;
- **hard negatives** — BM25 ranks 6–30 over the same Alzheimer corpus,
  excluding the source document. Ranks 1–5 are skipped deliberately: a document
  that ranks top for the question may genuinely answer it, and labelling it
  `[NOT_HELPFUL]` would be a false negative;
- **easy negatives** — a seeded random chunk sharing no distinctive term.

**Known failure modes**, stated rather than hidden:

1. a hard negative may genuinely answer the question — reduced by the rank skip,
   not eliminated;
2. a positive is helpful by construction, not by inspection;
3. positives share a document with the question, so their lexical overlap
   exceeds a genuinely retrieved passage's. Measured: mean Jaccard 0.109
   (positive) vs 0.055 (hard negative) vs 0.015 (easy negative). The gap is real
   and modest; if a trained filter scores word overlap rather than helpfulness,
   this is where it would come from.

**BM25 is used only to mine negatives.** It never scores an evaluation question
and never produces a candidate either arm sees. MedCPT remains the sole
retrieval model of the experiment.

### Human validation — outstanding

`training_dataset/human_validation_subset.jsonl` holds a stratified sample with
`human_label` deliberately blank. Completing it and running
`score_human_validation` (runbook 4A(ii)) converts "weak supervision" from an
assertion into a measured agreement rate. **This has not been done.** Until it
is, the dataset's label quality is argued, not measured.

---

## 3. Filter configuration

`architecture/rag2/configs/thesis_alzheimer_filter.yaml`.

| | Paper | Here | Why |
| --- | --- | --- | --- |
| base model | flan-t5-large (770M) | **flan-t5-small (77M)** | AdamW fp32 costs 16 B/parameter: 12.3 GB for large, 4.0 GB for base, 1.2 GB for small. The card has 4 GB. |
| epochs | 40 | **8** | 1,971 examples at effective batch 16 is ~123 steps/epoch; 40 epochs would mostly memorise. Chosen a priori. |
| learning rate | 3e-5 | 3e-5 | unchanged |
| effective batch | 16 | 16 (4 × 4 accumulation) | preserved |
| max_seq_length / doc_stride | 512 / 128 | 512 / 128 | unchanged |
| seed | — | 42 | recorded |

Mixed precision does not rescue flan-t5-base: fp16 through accelerate keeps
fp32 master weights and fp32 optimizer state. Gradient accumulation reduces
activation memory only. LoRA and 8-bit Adam would fit flan-t5-large but require
editing the authors' training script, which this repository does not do.

**This is not an exact reproduction of RAG², and must never be described as
one.** Two documented departures, both in the run manifest.

Per-epoch checkpoint selection is unavailable: `run_classifier.py` writes each
epoch with `accelerator.save_state`, which produces no `config.json` and cannot
be loaded. `--select` correctly reports that it scored one checkpoint. To choose
an epoch count on evidence, train more than once and compare validation
accuracy — that is selection on the training dataset's validation split, never
on the 30 evaluation questions.

---

## 4. Pipeline verification (done, in-container)

`huggingface.co` is unreachable from the development container, so
`google/flan-t5-small` could not be downloaded and the real filter could not be
trained here. The chain was instead verified against the **real** dataset using
a locally constructed T5, which exercises every step the repository owns:

| Step | Result |
| --- | --- |
| dataset build from the frozen corpus | 800 questions, 2,466 examples, 34 s |
| `04_train_filter.py --init-tokens` | `[HELPFUL]`/`[NOT_HELPFUL]` added as distinct ids |
| the authors' unmodified `run_classifier.py` on `train.json` | trained, wrote a loadable checkpoint |
| checkpoint selection (fix `559f049`) | rejected both `epoch_N` state directories, reported one loadable checkpoint |
| `verify_filter_checkpoint.py` | loaded through `RAG2PerplexityFilter`, scored 48 pairs |
| degenerate-filter guard | **fired correctly** — the throwaway model collapsed to always-`[NOT_HELPFUL]`, and the check refused it |
| `paired_comparison` wiring | ran over 30 questions × 20 candidates via the offline development path |

One real defect was found and fixed: `run_classifier.py` opens
`<output_dir>/logs.log` before creating that directory, so training died with
`FileNotFoundError` before it began. `04_train_filter.py` now creates it.

Tests: **741 passed, 1 skipped** (pmc 301, pubmed 51, rag2 236, scaf 153+1).

---

## 5. Frozen candidate set — `NOT YET RUN`

Requires `indexes/production/`, which exists only on the Windows machine.

| | |
| --- | --- |
| retrieval | MedCPT query encoder + cross-encoder, `--source medcpt` |
| questions | 30 |
| candidate depth | 20 |
| frozen-set digest | *fill from `frozen_candidates.jsonl.meta.json`* |
| `retrieval_is_medcpt` | *must be `true`* |

**One frozen set feeds both arms.** Nothing re-retrieves after this point; that
is what makes the comparison paired. Record the digest and cite it with every
number in section 7.

Before freezing, confirm `rationales.json` is non-empty for all 30 questions:
`retrieval_query()` falls back to the raw question when the rationale is empty,
which is the paper's MedCPT baseline row, not RAG².

---

## 6. Filter checkpoint — `NOT YET RUN`

Fill from `filter/checkpoint_verification.json`.

| | |
| --- | --- |
| checkpoint path | |
| load test | |
| label token ids | |
| validation accuracy | |
| majority-class baseline | 66.67% (the validation split is 33.3% positive) |
| kept fraction | |
| per-class accuracy | |
| training duration | |

**Do not proceed to section 7 if the filter is degenerate or fails to beat
66.67%.** A filter that keeps everything is passthrough wearing a checkpoint;
one that keeps nothing is `no_evidence`. Either makes the RAG² arm
uninformative. Raise the epoch count or rebuild the dataset larger
(`--target-questions 1500`) instead.

---

## 7. Comparison — `NOT YET RUN`

Fill from `comparison/paired_comparison.json` and `comparison/manifest.json`.

| Metric | RAG² | SCAF |
| --- | --- | --- |
| questions | 30 | 30 |
| total candidate chunks | | |
| admitted chunks | | |
| admission rate | | |
| mean admitted / question | | |
| median admitted / question | | |
| min / max admitted | | |

| Paired | |
| --- | --- |
| questions RAG² admits more | |
| questions SCAF admits more | |
| questions tied | |
| mean paired difference (signed) | |
| mean absolute paired difference | |
| mean Jaccard of admitted chunk ids | |
| identical selections | |
| disjoint selections | |

Per-question rows are in `comparison/per_question.jsonl`.

**No significance test is reported.** n=30 without gold answers does not support
an inferential claim.

---

## 8. What may be claimed

> An Alzheimer-specific supervised training dataset was constructed
> independently of the 30-question evaluation set and used to adapt the RAG²
> evidence-selection filter. RAG² and SCAF were then evaluated on the same
> frozen Alzheimer-domain retrieval corpus and identical frozen candidate
> evidence.

The defensible claim is about **evidence selection and admission behaviour**.

## 9. What may not

- **Not** "RAG² is more accurate than SCAF", or the reverse. The 30 questions
  have no gold answers, so no answer-quality metric exists. If answers are
  generated they must be saved as generated outputs and labelled as such.
- **Not** "an exact reproduction of RAG²". flan-t5-small for 8 epochs on
  corpus-derived weak labels is not flan-t5-large for 40 epochs on ΔPPL labels.
- **Not** "gold relevance labels". They are weakly supervised, and until the
  human-validation subset is completed their quality is unmeasured.
- **Not** a full-corpus claim beyond what the index holds. The index is
  773,183 vectors over 42,964 documents; the training dataset was drawn from
  the abstract layer.

## 10. Standing limitation, unchanged

The corpus is overwhelmingly recent: 43,409 policy records with 7 documents
older than 2020 (0.016%). SCAF's currency term γ therefore has very little
dynamic range on this corpus — measured median 0.933 — so this comparison tests
SCAF's admission behaviour, not its recency discrimination. That needs a corpus
with real temporal spread, and no result here should be read as evidence about
recency handling either way.
