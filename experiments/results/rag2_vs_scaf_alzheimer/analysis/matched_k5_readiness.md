# Matched-k = 5 answer generation

> **Generated status report.** A *derived* artifact, rebuilt by `python experiments/scripts/run_matched_k.py --document`. It records whether this machine can run the matched-k experiment, and the parts of the design that can be verified without a generator.

Documented 2026-09-10T18:50:18Z

## Status: **BLOCKED**

The experiment **has not been run**. These preconditions failed:

- **frozen candidate file on disk IS the scientific one** — local file records 151b7d5432b124bafb66e057e8b99c1d06c84954d5650951da3fc8c8aa8ccf3d (source: lexical-dev (IDF term overlap))
- **frozen candidate retrieval is production MedCPT** — retrieval_is_medcpt=False

No answers were generated, and no result directory was created. Substituting a different candidate set would produce 60 plausible answers and a clean-looking manifest for **a different experiment**, so the run refuses instead.

## 1. Objective, and why matched-k is necessary

The completed admission comparison measured what each policy naturally admits: SCAF passed about 19 passages per question, RAG² passed 0.13. SCAF's answers were therefore written from roughly **688× more context**. Any answer difference between those two runs is confounded by context size and cannot be attributed to the admission policy.

This experiment hands **both arms exactly 5 passages** from the same frozen candidate set, so the only thing that varies is *which* 5 passages each policy's own score ranked highest.

**It does not replace the admission experiment.** That run answers *how much* evidence each policy admits and remains the answer to that question. This run fixes the amount and asks only about the choice. Neither supersedes the other, and the two must not be conflated in the thesis.

## 2. Selection procedure

Each arm ranks the **same 20 frozen candidates** by **its own score** and takes the top 5. Ties break on `chunk_id`, so the choice is deterministic. This is the procedure already used by `matched_budget` in `evidence_quality.py` and already reported in the results analysis — no new scoring rule, no changed SCAF weight, no tuned threshold.

**Selection deliberately ignores each arm's admission threshold.** RAG² admits only 4 passages across all 600 decisions; a threshold-respecting rule could not reach k=5 for 26 of the 30 questions, and the experiment would collapse back into the confound it exists to remove. The threshold governs *how much* a policy admits, which the original experiment measured. Here the amount is fixed and only the ranking matters.

## 3. Pre-generation checks

| Check | Result | Detail |
| --- | --- | --- |
| scientific decisions file exists | PASS | /home/user/thesis_research/experiments/results/rag2_vs_scaf_alzheimer/comparison_scientific/per_question.jsonl |
| exactly 30 questions | PASS | 30 questions |
| exactly 20 candidates per question per arm | PASS | candidate counts {20: 60} |
| both arms hold identical candidate ids in identical order | PASS |  |
| every arm-question has at least k=5 candidates | PASS |  |
| scientific run digest is the expected one | PASS | run manifest records 316260f04c1720fbc20c1b584ea9f0a093dedbae9fd46476d8a7f45380e86aad |
| frozen candidate file on disk IS the scientific one | **FAIL** | local file records 151b7d5432b124bafb66e057e8b99c1d06c84954d5650951da3fc8c8aa8ccf3d (source: lexical-dev (IDF term overlap)) |
| frozen candidate retrieval is production MedCPT | **FAIL** | retrieval_is_medcpt=False |
| question ids and text match the committed question set | PASS | 30/30 match; question set holds 30 |
| generation configuration identical across arms in the source run | PASS |  |

**All passed: False**

Required frozen digest: `316260f04c1720fbc20c1b584ea9f0a093dedbae9fd46476d8a7f45380e86aad`
Run manifest records: `316260f04c1720fbc20c1b584ea9f0a093dedbae9fd46476d8a7f45380e86aad`
Frozen file on this disk: `151b7d5432b124bafb66e057e8b99c1d06c84954d5650951da3fc8c8aa8ccf3d` (source: lexical-dev (IDF term overlap))

## 4. Selection invariants

These hold regardless of whether generation can run — the selection needs only the saved scores.

| Invariant | Result |
| --- | --- |
| 30 questions selected | PASS |
| rag2: exactly 5 passages for every question | PASS |
| rag2: no duplicate passage within a question | PASS |
| rag2: every selected passage is in that question's frozen candidates | PASS |
| scaf: exactly 5 passages for every question | PASS |
| scaf: no duplicate passage within a question | PASS |
| scaf: every selected passage is in that question's frozen candidates | PASS |
| no cross-question candidate leakage | PASS |

**All passed: True**

## 5. How different is the selected evidence?

Descriptive statistics about what each policy would hand the generator.

| | |
| --- | ---: |
| Questions | 30 |
| k | 5 |
| Questions with **identical** 5-passage sets | 0 |
| Questions where the evidence **differs** | 30 |
| Questions with **no overlap at all** | 9 |
| Questions with some overlap | 21 |
| Mean overlap (of 5) | 0.9 |
| Mean Jaccard | 0.105556 |

Overlap distribution: `{0: 9, 1: 15, 2: 6}`

> descriptive statistics about which passages each policy would hand the generator. They are not answer-quality measurements and must not be reported as one.

In plain terms: at an equal budget of 5 passages the two policies choose **substantially different evidence** — no question gets an identical set, 9 share nothing at all, and on average only 0.9 of 5 passages coincide. That is what makes the comparison worth generating: if the sets were identical the answers would be too, and there would be nothing to measure.

Context-size statistics (characters, tokens) are **not** in this report. They require the candidate text, which lives with the frozen candidate set; they will be recorded in `matched_k5/manifest.json` when the run executes.

## 6. Generator

The run reuses the configuration recorded by the scientific comparison rather than reconstructing it:

| Setting | Value |
| --- | --- |
| `backend` | openai |
| `model` | thesis-llama3-8b-q4 |
| `llm_class` | OpenAIBackend |
| `greedy` | True |
| `max_new_tokens` | 512 |
| `chat_template` | True |
| `dtype` | bfloat16 |
| `prompt_version` | rag2-original-v1 |
| `prompt_fingerprint` | fc6db2781ddc |

Requested for this run: `thesis-llama3-8b-q4` via `openai`.

## 7. What was NOT done

- No answer-quality judgement of any kind.
- No gold answers or reference answers were created.
- No correctness labels, no ROUGE, no BERTScore.
- No human evaluation, and no annotator saw anything.
- No retrieval, reranking or index access.

The blind pairwise evaluation is a **separate later stage**. When this run executes, its output labels each arm by name, so those files must be anonymised before any human sees them.

## 8. Limitations

- Matched-k measures the *ranking*, not the policy as deployed. Neither arm would naturally admit exactly 5 passages.
- 30 questions is small, and they are development questions with no gold answer.
- Even once generated, the answers cannot be scored until an answer-quality target exists — see `answer_quality_readiness.md`.
- A single generator, single decoding pass. No variance estimate.

## 9. Inputs

```
4908e339739dcfbe2e741f843235ea8b13847ef0cf48eeb7933809e0a834f74d  experiments/results/rag2_vs_scaf_alzheimer/comparison_scientific/per_question.jsonl
c5ff80fa6d8287ce4061a3ca394a97164b38774bb5d168f93e92158cb421736f  experiments/results/rag2_vs_scaf_alzheimer/comparison_scientific/manifest.json
6e0a3198293e3331f7f2298541f898bd1514c190527a18d6f78f499d93381439  data/datasets/thesis_questions/dev_questions.jsonl
```

## 10. How to run it

On the machine holding the scientific frozen candidate set and the generator:

```
python experiments/scripts/run_matched_k.py --dry-run
python experiments/scripts/run_matched_k.py \
    --generator openai --generator-model thesis-llama3-8b-q4
```

`--dry-run` runs every check and the selection, generates nothing and writes nothing. The full command writes `experiments/results/rag2_vs_scaf_alzheimer/matched_k5/`. Refresh this report with `--document`.
