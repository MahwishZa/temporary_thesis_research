# Can we evaluate ANSWER quality yet?

> **Generated inspection report.** A *derived* artifact, rebuilt by `experiments/scripts/answer_quality_readiness.py`. It opens files and reports; it runs no retrieval, no generation, no annotation and no experiment, and it creates no reference answer.

Inspected 2026-09-10T18:33:40Z · version `answer-quality-readiness-v1`

## Decision: **PARTIALLY READY**

The machinery exists but the target does not. Answer-evaluation code is present and tested, the generator is pinned, the 30 questions and the frozen candidate set are fixed, and each arm's top-k can be selected from the saved scores without rerunning retrieval. The single missing component is something to grade an answer against: there is no gold answer, no reference answer and no expert answer label anywhere in the repository.

| | |
| --- | --- |
| Answer-quality target exists | **False** |
| Answer-evaluation code exists | **True** |
| Matched-k selection executable now | **True** |

In plain terms: **you can build the experiment, but you cannot yet mark it.** Everything needed to give both systems exactly five passages and produce two answers per question is already in the repository. What is missing is any independent statement of what a good answer to these questions would look like.

## 1. The 30 evaluation questions

`/home/user/thesis_research/data/datasets/thesis_questions/dev_questions.jsonl` — 30 questions, set `alzheimer-dev-v1`.

Fields present on each question record:

| Field | Present on |
| --- | ---: |
| `note` | 30/30 |
| `qid` | 30/30 |
| `question` | 30/30 |
| `set` | 30/30 |
| `time_sensitive` | 30/30 |

**Answer/gold/reference fields: NONE.**

The question set says so itself. Every one of the 30 records carries:

> DEVELOPMENT ONLY. No gold answer: this set compares admission behaviour, not accuracy.

So for each of the 30 questions the repository holds **the question text and nothing else**: no gold answer, no reference answer, no expert label, no answer-quality label.

## 2. Answers that already exist

The completed comparison **did** save its generated answers, which is worth knowing — they are real text, not placeholders:

| | RAG² | SCAF |
| --- | ---: | ---: |
| Answers present | 30 | 30 |
| Abstentions | 0 | 0 |
| Generation errors | 0 | 0 |
| Mean context (chars) | 39.1 | 26904.6 |
| Mean admitted passages | 0.133 | 19.167 |
| Answer length (median chars) | 1329 | 1387 |

**This is the confound.** SCAF's answers were written from roughly 688.1× more context than RAG²'s (26904.6 characters against 39.1). Comparing these 60 answers as they stand would measure context size, not admission policy. That is precisely why a matched-k design is the right next experiment — and why the earlier preliminary answer figures must not be reported.

## 3. Evaluation code that exists

`architecture/rag2/rag2/evaluation.py` is real, tested evaluation machinery — but every one of its metrics needs a target this repository does not have.

| Capability | Functions | Needs | Usable here? |
| --- | --- | --- | --- |
| multiple choice accuracy | `extract_choice, is_correct, accuracy, accuracy_by` | a gold option letter and an options mapping per question | **False** |
| open ended similarity | `rouge_l, bertscore, open_ended_metrics` | a written reference answer per question | **False** |
| filter metrics | `filter_metrics` | gold HELPFUL/NOT_HELPFUL labels | **False** |

- **multiple choice accuracy** — the Alzheimer questions are open-ended and carry neither options nor a gold letter.
- **open ended similarity** — no reference answer exists for any of the 30 questions.
- **filter metrics** — measures the filter, not the answer; and the repository's HELPFUL labels are rule-derived training data.

- **Citation correctness**: False. None exists.
- **Factuality / claim-level evaluation**: False. None exists.
- **Wired into the SCAF comparison**: False. The comparison records answer *counts* only.

The run manifest states the same thing in its own words:

> no gold answers in this question set; only counts are reported. Metrics are absent, not zero.

`with_reference` is **0** for RAG² and **0** for SCAF.

## 4. Things that look like a gold standard but are not

This section exists because these three are easy to reach for, and each would quietly invalidate the experiment.

### `evidence_quality_annotations`

- Path: `experiments/results/rag2_vs_scaf_alzheimer/evidence_quality/annotation_sheet_v2.jsonl`
- Rows: 120
- What it is: human 0/1/2 judgements of whether a PASSAGE is useful for a question
- **Why it is not an answer-quality target:** it grades retrieved evidence, not generated answers. An answer can be wrong from useful passages and right from thin ones.
- Suitable as answer gold: **False**

### `filter_training_weak_labels`

- Path: `experiments/results/rag2_vs_scaf_alzheimer/training_dataset/human_validation_subset.jsonl`
- Rows: 120
- Labels: `{'[NOT_HELPFUL]': 80, '[HELPFUL]': 40}`
- What it is: rule-derived [HELPFUL]/[NOT_HELPFUL] training data for the RAG2 filter
- **Why it is not an answer-quality target:** training data, not evaluation data, and about passages rather than answers. Using it to evaluate would be testing on train.
- Suitable as answer gold: **False**

### `machine_scores`

- Path: `experiments/results/rag2_vs_scaf_alzheimer/comparison_scientific/per_question.jsonl`
- What it is: each system's own SCAF/RAG2 score for each passage
- **Why it is not an answer-quality target:** circular: grading a system's output with that system's own score measures self-consistency, not quality.
- Suitable as answer gold: **False**

## 5. Matched-k feasibility

The proposed design: same 30 questions, same frozen candidate set, same generator, exactly **k = 5** admitted passages per arm, with only the admission policy differing.

| Component | Available now |
| --- | --- |
| same 30 questions | **True** |
| same frozen candidate set | **True** |
| top k selection from frozen scores | **True** |
| same generator configuration recorded | **True** |
| generation must be rerun | **True** |
| answer quality target | **False** |

Every arm-question pair has enough candidates to take a top-5: **60 of 60**.

> choosing each arm's top-k needs no retrieval, no reranking and no index -- the frozen decisions already carry every score. Generation would have to be rerun because the k=5 contexts differ from the ones the completed run used. That is cheap. What is missing is not compute: it is something to grade the answers against.

**Blocked on: answer_quality_target.**

## 6. Limitations of this inspection

- This is a search for artifacts, not a judgement of their scientific quality. A gold answer found would still have needed validating.
- It reports on this repository only. A reference standard may exist outside it — in a supervisor's notes, or a clinical guideline — and would not be visible here.
- It does not verify that the generator is currently reachable, only that its configuration is recorded.
- Absence of a field name is strong but not absolute evidence: a reference stored under an unexpected name would not be matched.

## 7. Recommended next step

**Decide and record the answer-quality standard before generating anything.** Two defensible routes exist, and the cheaper one does not require gold answers at all.

### Route A — blind pairwise preference (recommended)

Regenerate both arms at k = 5 from the frozen candidates, then show a human the question and the two answers **anonymised and in random order**, and ask which is better supported. No gold answer is needed, because the comparison is relative.

Why this is the smallest defensible step:

- it removes the context confound, which is the one thing that makes the current answer data uninterpretable;
- it needs no clinical expertise to *author* a reference, only to *compare* two answers;
- 30 comparisons is roughly an hour of work, against 30 written reference answers;
- the repository already has the machinery: a blind annotation interface, an audit that checks blinding and refuses anchored passes, and a stratified sampling design. The lesson already paid for — **never show the annotator a machine opinion** — applies directly.

Its limitation, which must be stated in the thesis: preference is not correctness. It can show one arm's answers are *preferred*, not that they are *right*.

### Route B — written reference answers

Author a reference answer for each of the 30 questions, grounded in the corpus, and validate it with someone qualified. This unlocks the existing `rouge_l` / `bertscore` / `open_ended_metrics` code directly.

It is more expensive and carries a real risk: a reference written by someone without clinical training, or written after seeing the systems' answers, would be worse than no reference at all. **If Route B is chosen, the references must be written before any matched-k answer is generated, and signed off by a supervisor.**

### What not to do

- Do not reuse the 120 evidence-quality labels as answer labels.
- Do not reuse the filter training weak labels as evaluation gold.
- Do not grade answers with SCAF or RAG² scores.
- Do not generate reference answers with the same model being evaluated.
- Do not report the earlier preliminary answer figures; they came from unmatched contexts.

## 8. Artifacts inspected

```
data/datasets/thesis_questions/dev_questions.jsonl
experiments/results/rag2_vs_scaf_alzheimer/comparison_scientific/per_question.jsonl
experiments/results/rag2_vs_scaf_alzheimer/comparison_scientific/manifest.json
architecture/rag2/rag2/evaluation.py
experiments/results/rag2_vs_scaf_alzheimer/evidence_quality/annotation_sheet_v2.jsonl
experiments/results/rag2_vs_scaf_alzheimer/training_dataset/human_validation_subset.jsonl
```

Regenerate this report with:

```
python experiments/scripts/answer_quality_readiness.py
```

Structured form: `answer_quality_readiness.json` in this directory.
