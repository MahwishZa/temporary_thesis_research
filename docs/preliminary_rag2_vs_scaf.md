# PRELIMINARY / DEVELOPMENT RESULTS — RAG² vs SCAF

**Not thesis results.** Nothing here validates the thesis or shows SCAF to be
better than RAG². This is a development milestone: it establishes that both
admission policies run end to end on the thesis corpus over an identical frozen
candidate set, and that their decisions are inspectable.

Run date 2026-09-04 · commit recorded in `scaf/runs/manifest.json` · 30 questions ·
600 candidates · frozen-set digest `151b7d54…aa8ccf3d`

---

## 0. Read this first: a corpus-design finding that blocks the research question

The thesis asks whether confidence-based admission **prefers the past**. Answering
that needs evidence from the past. The corpus does not contain any.

| Source | Documents | Pre-2020 | 2020+ |
| --- | --- | --- | --- |
| `pmc/metadata/canonical_dates.csv` (**production**, 43,409 docs) | 43,409 | **7 (0.02 %)** | 43,402 (99.98 %) |
| `pmc/chunks/chunks.jsonl` (container build, 60,874 chunks) | 60,874 | 38 (0.06 %) | 60,836 (99.9 %) |

This is not an acquisition defect. `pubmed/search_queries.txt` applies, by design,

```
("2021/08/30"[Date - Publication] : "2026/08/30"[Date - Publication])
```

to every query — a deliberate five-year window, and a sensible one for a corpus
of *current* Alzheimer evidence.

**The consequence is structural, and it is not something SCAF or any code change
can fix:**

1. **FRB-PAIRS cannot be constructed.** Matched older/newer evidence pairs need
   older evidence. There are 7 pre-2020 documents in the production corpus.
2. **The currency term has almost no dynamic range.** γ = 2^(−age/H) over a corpus
   spanning 2021–2026 with H = 5 years varies only between about 1.0 and 0.5, and
   most of that range is unpopulated. In this run γ's median was 0.933.
3. **A measured "recency bias" would be measuring nothing.** Both arms admit
   almost exclusively post-2020 evidence because that is all there is.

This needs a supervisor decision before the main experiments, not after.
Options, in rough order of cost: widen the acquisition window for a *comparison
stratum* only (leaving the existing corpus frozen); reframe the research question
around supersession and retraction within the current window rather than age;
or construct FRB-PAIRS from an external older-evidence source. **§7 records this
as the primary open question.** Everything else below works.

---

## 1. RAG² reproduction audit — status

Full component-by-component audit: [`rag2_reproduction_audit.md`](rag2_reproduction_audit.md).
Re-verified for this milestone, with emphasis on the filter, since that is the
component SCAF replaces.

| Component | Status | Note |
| --- | --- | --- |
| Overall architecture (4 stages, single pass) | **FAITHFUL** | No iteration, matching §3 |
| Rationale-based query formulation | **FAITHFUL** | Rationale *replaces* the query; verified it is what reaches the encoder |
| Retrieval (MedCPT, 768-d, exact inner product) | **FAITHFUL** | `IndexFlatIP`-equivalent, exact not ANN |
| Balanced retrieval | **FAITHFUL** | Equal quota per corpus; 3 corpora here vs the paper's 4 |
| Reranking (MedCPT cross-encoder) | **FAITHFUL** | Paper/code disagree on *which query* is cross-encoded; both exposed as `retrieval.rerank_query`, default follows the paper |
| Filter model (Flan-T5-large + `[HELPFUL]`/`[NOT_HELPFUL]`) | **FAITHFUL** | Correct class (`AutoModelForSeq2SeqLM`), tokens added and resized |
| **Filter retraining** | **FAITHFUL** | Drives the authors' own `classifier/run_classifier.py`; hyperparameters from Appendix A.3 come from config, not a shell script |
| **Checkpoint loading at inference** | **FAITHFUL** | `RAG2PerplexityFilter` *refuses to run* without a checkpoint — it cannot silently skip filtering |
| Perplexity / ΔPPL, Figure 2 labeling | **MOSTLY FAITHFUL** | Tree matches on all six paths. Eq. 4 literally sums over the *query*; prose and Figure 2 say *rationale*; implementation uses rationale, `filter_training.ppl_target` switches. **Flagged for supervisor.** |
| Threshold τ (top 25 %) | **FAITHFUL** | Value is the paper's; the *population* (global vs per-question) is unstated — `tau_scope`, defaults global |
| Context construction, generator, prompting | **FAITHFUL / APPROXIMATION** | The answer-generation prompt was never published; reconstructed and marked as the largest prompt-level assumption |
| Decoding (greedy, T=0) | **FAITHFUL** | Appendix A.3 |
| Seeds / configuration | **BETTER THAN ORIGINAL** | The authors left `--seed` unset; every run here is seeded and manifested |

**Corrections made this session:** none were needed in the RAG² filter path — the
audit confirmed it. Earlier sessions corrected a false test claim and added the
corpus adapter (audit §6).

**Is it sufficient for this comparison?** For the *architecture*, yes. For a
filtered RAG² arm, **not yet** — the filter checkpoint does not exist. See §2.

---

## 2. What actually ran, and what did not

| Stage | Ran? | Detail |
| --- | --- | --- |
| Frozen candidate retrieval | ✅ real corpus | 30 questions × 20 candidates from `pmc/chunks/chunks.jsonl` |
| Retrieval **model** | ❌ **not MedCPT** | No GPU, no `pmc/index/` and no model weights in this environment. A documented lexical (IDF-overlap) stand-in was used and is stamped `retrieval_is_medcpt: false` in the sidecar. |
| SCAF admission | ✅ **fully real** | Deterministic over real corpus metadata and text |
| RAG² admission (filtered) | ❌ **not run** | Needs the trained Flan-T5 checkpoint, which the RAG² authors do not distribute and which has not been trained yet |
| RAG² admission (**w/o filter**) | ✅ real | The paper's own Table 4 ablation — a defined RAG² configuration, run as Arm A |
| Answer generation | ❌ not run | Needs an 8B backbone. Both arms produce contexts; neither produces answers. |

**Arm A in this run is "RAG² w/o filter", not filtered RAG².** That is stated in
the manifest (`arm_a_filter: passthrough`) and must be stated in any slide made
from these numbers. The admission comparison below is therefore
*SCAF vs no-admission*, which bounds SCAF's selectivity but says nothing yet
about SCAF vs the perplexity filter.

---

## 3. Execution statistics

| | Arm A — RAG² w/o filter | Arm B — SCAF |
| --- | --- | --- |
| Questions attempted | 30 | 30 |
| Succeeded / failed | 30 / 0 | 30 / 0 |
| Mean candidates | 20.0 | 20.0 |
| Mean admitted | 20.0 | 17.87 |
| **Admission rate** | **1.000** | **0.893** |
| Questions with no evidence | 0 | 0 |
| Abstentions | 0 | 0 |
| Answers generated | 0 (no generator) | 0 (no generator) |

SCAF sub-scores across all 600 decisions:

| Signal | min | median | max |
| --- | --- | --- | --- |
| σ support (lexical) | 0.033 | 0.521 | 1.000 |
| γ currency | 0.000 | 0.933 | 1.000 |
| τ authority | 0.400 | 0.450 | 1.000 |
| SCAF score | 0.320 | 0.600 | 0.890 |

Currency states: 479 `current`, 119 `not_time_sensitive`, **2 `retracted`**.

**SCAF's threshold is uncalibrated.** 0.45 was chosen a priori and admits 89 %.
It was deliberately **not** tuned on this set — tuning on the evaluation set is
one of the fairness violations the manifest checks for. Calibration is a
supervisor decision (§7).

---

## 4. Fairness controls — 9/9 passed

Machine-readable in `scaf/runs/manifest.json` under `fairness`.

```
[PASS] same questions in both arms
[PASS] same candidate set and ordering in both arms
[PASS] neither arm retrieved its own candidates
[PASS] same number of candidates scored per question
[PASS] same generator in both arms
[PASS] same decoding settings in both arms
[PASS] same corpus and index in both arms
[PASS] no gold answer reachable from admission records
[PASS] failures reported rather than dropped
```

The third is the one the whole design exists for: each arm records a digest over
candidate identity **and order**, and both are compared against the frozen file.
If an arm ever retrieved for itself, that check fails and the run is refused.

---

## 5. Qualitative examples

**D. SCAF correctly rejects; RAG² w/o filter admits.** Two retracted papers were
retrieved and hard-gated by SCAF:

| Question | Chunk | Date | SCAF | Arm A |
| --- | --- | --- | --- | --- |
| alz-017 (agitation management) | `PMC12808712#abs.w1` | 2025-12 | **REJECT** — "hard gate: evidence is retracted or withdrawn" | ADMIT |
| alz-020 (MMSE tracking) | `PMC9663818#abs.w1` | 2022 | **REJECT** — same | ADMIT |

This is the clearest demonstrable difference in the run, and it is the behaviour
the thesis argues a confidence-only filter cannot produce: retraction is a
property of the *record*, not of how confidently the passage reads.

**C. Both admit.** High-support, current, guideline-tier evidence:
`PMC12893748#abs.w1` (ARIA monitoring, clinical-practice-guideline, score 0.731)
and `PMC12287243#abs.w1` (anti-amyloid monitoring, score 0.815).

**E. SCAF makes a questionable decision — reported, not hidden.** On alz-016
("What is the neuropathological hallmark of Alzheimer disease?"), SCAF rejected
chunks that matched *neuropathological*, *hallmark* and *disease* with σ = 0.033.
The cause is a real weakness of the MVP support scorer: IDF is computed **within
the candidate set**, so when every candidate is equally on-topic those terms
carry almost no weight and σ collapses for all of them. σ is therefore a
*within-set discriminator*, not an absolute topicality measure. The entailment
model the thesis specifies would not have this failure mode. **This is the
strongest argument for replacing σ before any real experiment.**

**A / B (SCAF better / RAG² better) cannot be assessed yet** — no answers were
generated, and Arm A had no filter to be better or worse than.

---

## 6. Recency-oriented descriptive findings

**Formal FRB-PAIRS analysis remains pending**, and per §0 it is currently not
constructible.

| | Arm A — RAG² w/o filter | Arm B — SCAF |
| --- | --- | --- |
| Older (< 2020): candidates / admitted | 1 / 1 (100 %) | 1 / 1 (100 %) |
| Newer (≥ 2020): candidates / admitted | 599 / 599 (100 %) | 599 / 535 (89.3 %) |
| Undated | 0 | 0 |

**These numbers carry no information about recency bias.** With n = 1 in the
older stratum, no comparison is possible. The retrieved date spread was
2012 ×1, 2020 ×1, 2021 ×15, 2022 ×70, 2023 ×88, 2024 ×112, 2025 ×176, 2026 ×137 —
which mirrors the corpus, not any property of either admission policy.

---

## 7. What needs supervisor decision

1. **The corpus date window (§0).** The blocking item. The research question as
   written cannot be answered on a 2021–2026 corpus. Widen a comparison stratum,
   reframe toward supersession/retraction, or source older evidence externally.
2. **SCAF's σ must become entailment-based** before any claim (§5, example E).
3. **SCAF threshold and weight calibration**, on a development split that is not
   the evaluation set.
4. **ΔPPL ambiguity**: Eq. 4 says query, prose and Figure 2 say rationale. The
   implementation uses rationale and exposes the switch. Confirm the reading.
5. **Authority tier ordering** is currently a documented default, not a finding.
   Ablation A12 should test it.
6. **Supersession is unmeasured.** Nothing in the corpus marks one document as
   superseding another, so SCAF reports `supersession: "unknown"` rather than
   claiming currency. Building that map is a prerequisite for the three-state γ.

---

## 8. Reproducing this from a clean checkout

```bash
git clone https://github.com/MahwishZa/thesis_research && cd thesis_research
python3 -m pytest scaf/tests -q                      # 65 tests, no GPU needed

# Stage 1 — freeze the upstream candidate set (development retrieval)
python3 scaf/scripts/freeze_candidates.py --source lexical-dev --depth 20

# Stage 2 — run both admission policies over it
python3 scaf/scripts/run_comparison.py --rag2-filter passthrough
```

Outputs land in `scaf/runs/` (gitignored): `frozen_candidates.jsonl`,
`frozen_candidates.meta.json`, `per_question.jsonl`, `manifest.json`.

**For the real run**, on the machine with the GPU, index and model weights:

```bash
# real MedCPT retrieval + reranking, then freeze it
python rag2\scripts\02_retrieve.py -c rag2\configs\thesis_corpus.yaml
python scaf\scripts\freeze_candidates.py --source medcpt --cache <cache>.jsonl

# train the RAG2 filter (its checkpoint is not distributed by the authors)
python rag2\scripts\03_build_filter_labels.py -c rag2\configs\thesis_corpus.yaml --candidates <cache>.jsonl
python rag2\scripts\04_train_filter.py -c rag2\configs\thesis_corpus.yaml --init-tokens
python rag2\scripts\04_train_filter.py -c rag2\configs\thesis_corpus.yaml --train-file <labels> --select

# then the real comparison
python scaf\scripts\run_comparison.py --rag2-filter rag2_perplexity --rag2-checkpoint <dir>
```

---

## 9. Tests

| Suite | Result |
| --- | --- |
| `scaf/tests` (new) | **65 passed** |
| `rag2/` | 194 passed, 2 skipped (torch-gated) |
| `pmc/` | 355 passed |
| `pubmed/` | 51 passed |

The SCAF tests cover each scorer in isolation, the hard gates, determinism,
finiteness, inspectability, the frozen-set digest (including tamper detection),
and every fairness check — each proven to *fail* when its condition is violated,
not merely to pass when it holds.
