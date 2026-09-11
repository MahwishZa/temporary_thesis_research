# Experimental Results and Analysis

**Evidence admission in retrieval-augmented medical question answering**
Alzheimer's disease domain · 11 September 2026

This document records the experimental work performed to date and what its results
establish. Every quantity is taken from the generated result files; where a figure
appears in more than one file it has been cross-checked against its source, and
agreement or disagreement is reported.

Each experiment carries an explicit result classification:

| Classification | Meaning |
| --- | --- |
| **VALID RESULT** | Executed under the intended conditions; supports the stated claim |
| **PRELIMINARY RESULT** | Executed, but under development rather than final conditions |
| **DESCRIPTIVE ONLY** | Computationally correct, but does not support a causal or comparative claim |
| **INCONCLUSIVE** | Executed with adequate procedure; the estimate does not distinguish the hypotheses |
| **INVALID FOR COMPARISON** | Executed successfully, but the experimental conditions were not matched |
| **NOT EXECUTED** | Designed and implemented, but never run on data |

---

## 1. Research Evaluation Context

Retrieval-augmented generation places a filter between retrieval and answer generation
that decides which retrieved passages reach the answer model. The RAG² system makes that
decision with a small classifier trained on perplexity-differential labels: a passage is
admitted when conditioning on it reduces the base model's perplexity — a *confidence*
signal rather than a measure of evidential support.

The experiments reported here were designed to investigate the behaviour of that
admission stage, and to compare it against an alternative admission policy (SCAF) that
scores support, currency and source authority explicitly. Alzheimer's disease was chosen
as the evaluation domain because its diagnostic criteria, biomarkers and treatment
recommendations changed substantially during the period covered by the corpus.

The experimental design isolates the admission stage: the two policies operate on an
identical, pre-computed candidate set, so retrieval, reranking, generator and decoding
parameters are held constant and only the admission decision differs.

---

## 2. RAG² Reproduction Results

**Classification: VALID RESULT (structural reproduction) · NOT EXECUTED (accuracy benchmarking)**

The RAG² pipeline was reproduced in full: rationale-based query formulation, balanced
retrieval across source corpora, MedCPT dense retrieval and cross-encoder reranking, and
the rationale-guided Flan-T5-large filter.

### Filter configuration as executed

| Setting | Value |
| --- | --- |
| Filter model | Flan-T5-large, retrained (the original checkpoint was not distributed) |
| Checkpoint used | `filter-alz`, file-listing digest `60352885…`, weights 307,760,552 bytes |
| Decision rule | Two-way softmax over the `[HELPFUL]` / `[NOT_HELPFUL]` label logits, admit at ≥ 0.5 |
| Retrieval / reranking | `ncbi/MedCPT-Article-Encoder` with MedCPT cross-encoder |
| Candidate depth | 20 per question |

### Filter training data

| Quantity | Value |
| --- | --- |
| Training examples (total) | 2,466 |
| Train / validation split | 1,971 / 495 |
| Label balance (total) | 822 `[HELPFUL]` · 1,644 `[NOT_HELPFUL]` |
| Negative composition | 822 hard negatives · 822 easy negatives |
| Evaluation leakage check | Performed; build fails rather than warns if a training question matches an evaluation question |

### What this establishes, and what it does not

The reproduction is **structural**: the pipeline stages, the filter architecture and the
decision rule follow the published specification, and the τ = top-25% parameter is
correctly applied to training-label generation rather than to inference.

No accuracy benchmarking was performed. The system was not evaluated on MedQA, MedMCQA
or MMLU-Med, so **no claim is made that this reproduction matches the published accuracy
figures**. The reproduction supports experiments about admission behaviour; it does not
supply a numerical replication of the original paper's headline results.

---

## 3. Production Corpus and Retrieval Results

**Classification: VALID RESULT**

The corpus is domain-scoped to Alzheimer's disease and clinical reasoning, and the same
scoped corpus is used by every experimental arm, making scoping a controlled constant.

| Quantity | Verified value |
| --- | --- |
| PubMed records acquired | 43,409 |
| Documents in the chunk layer | 42,964 |
| Chunks produced | 781,563 |
| Unique chunk texts | 773,183 |
| Exact duplicate chunks | 8,380 (flagged, not deleted) |
| Chunking | 256-word sliding window, 32-word overlap |
| Source composition | 758,867 PMC full text · 22,583 PubMed abstracts · 113 currency pack |
| Publication window | 2021-08-30 to 2026-08-30 (approved search strategy) |
| Queries executed | 42 (recorded in the PubMed search log, one row per query) |
| Retrieval encoder | `ncbi/MedCPT-Article-Encoder` |
| Production index | 773,183 vectors × 768 dimensions, digest `2ab50a1212681abd28f4f250d49f076c36f57d1b3a77ce5aaac093adde770937` |

The approved search strategy document defines 60 numbered query identifiers across 14
thematic groups; 42 of those queries were executed and logged, each with its result
count, retrieved PMID count and new-unique-PMID contribution.

### Temporal composition of the corpus

Publication dates were resolved for all 43,409 documents and recorded with their
precision and source.

| Year | Documents | Share |
| ---: | ---: | ---: |
| ≤ 2020 | 9 | 0.02% |
| 2021 | 2,198 | 5.06% |
| 2022 | 7,003 | 16.13% |
| 2023 | 7,030 | 16.19% |
| 2024 | 8,499 | 19.58% |
| 2025 | 10,671 | 24.58% |
| 2026 | 7,999 | 18.43% |

Only 7 documents predate 2020 (0.016%). A split marker relative to the June 2024 revised
criteria is recorded for every document: **18,707 pre · 22,789 post · 1,913 unknown**.

**Why this materially affects interpretation.** The corpus spans approximately five
years. Any admission component that decays with document age has correspondingly little
dynamic range to exercise within it — a point quantified in §6.

---

## 4. Frozen Candidate-Set Results

**Classification: VALID RESULT (admission behaviour)**

### Candidate population and freezing procedure

The reranked candidate set was computed **once** and serialised, then replayed
byte-identically to both arms. Each arm is a distinct function over one cached candidate
list; neither arm performed its own retrieval.

| Property | Value |
| --- | --- |
| Questions | 30 (fixed Alzheimer's development set) |
| Candidates per question | 20 |
| Total admission decisions | 600 per arm |
| Frozen set digest | `316260f04c1720fbc20c1b584ea9f0a093dedbae9fd46476d8a7f45380e86aad` |
| Retrieval provenance | MedCPT, `retrieval_is_medcpt: true` |
| Run executed | 2026-09-10, Windows 11, Python 3.12.6, torch 2.4.1+cu121 |

### Temporal composition of the frozen candidates

Relative to the June 2024 revised diagnostic criteria, the 600 candidates divide almost
evenly, and every question draws from both sides:

| | Candidates |
| --- | ---: |
| Published before June 2024 | **298** |
| Published from June 2024 onward | **302** |
| Questions whose 20 candidates span both sides | **30 / 30** |

### Fairness verification

Nine fairness conditions were checked automatically and **all nine passed**: identical
questions, identical candidate set and ordering (verified by per-question digests),
neither arm retrieving its own candidates, identical candidate counts, identical
generator, identical decoding settings, identical corpus and index, no gold answer
reachable from the admission records, and failures reported rather than dropped.

### Admission thresholds as configured

| Arm | Policy | Threshold |
| --- | --- | --- |
| RAG² | Perplexity-derived classifier | P(`[HELPFUL]`) ≥ 0.5 |
| SCAF | A(s) = 0.5·σ + 0.3·γ + 0.2·τ | A(s) ≥ 0.45, half-life 5 years |

SCAF's support term σ was computed as lexical coverage weighted by corpus IDF
(`support_method: lexical-coverage-corpus-idf-v2`, `support_is_entailment: false`). The
corroboration term was not implemented and carried weight 0.

### Measured admission counts

| | RAG² | SCAF |
| --- | ---: | ---: |
| Candidates admitted | **4 / 600** | **575 / 600** |
| Admission rate | 0.0067 | 0.9583 |
| Mean admitted per question | 0.133 | 19.167 |
| Questions with no admitted evidence | 26 / 30 | 0 / 30 |
| Abstentions | 0 | 0 |
| Mean context supplied to the generator | 39.1 characters | 26,904.6 characters |

Admission by source category:

| Source | Candidates | RAG² admitted | SCAF admitted |
| --- | ---: | ---: | ---: |
| PMC full text | 355 | 1 (0.28%) | 342 (96.3%) |
| PubMed abstract | 241 | 2 (0.83%) | 229 (95.0%) |
| Currency pack | 4 | 1 (25.0%) | 4 (100%) |

**Cross-check.** These counts were re-derived independently from the 600 raw decision
records and agree exactly with the run manifest (30 questions, 600 candidates, 4 and 575
admissions). No discrepancy was found.

### Paired structure of the decisions

Because both arms scored the same 600 candidates, the decisions form a matched pair:

| | SCAF admitted | SCAF rejected |
| --- | ---: | ---: |
| **RAG² admitted** | 4 | **0** |
| **RAG² rejected** | 571 | 25 |

**Every passage RAG² admitted was also admitted by SCAF.** There is no candidate that
the baseline retained and the proposed policy discarded. The two policies are therefore
not making different trade-offs on this data; one is substantially more permissive than
the other.

### Interpretation

This experiment measures **how much evidence each policy admits at its own configured
operating point**. It establishes that the operating points differ greatly. It does not
establish that either admission set is of higher quality — that question is addressed
separately in §5, and a higher admission rate is not evidence of a better one. An
admission rate of 95.8% is close to applying no filter at all.

The manifest for this run carries the label `PRELIMINARY / DEVELOPMENT RESULTS`,
reflecting that the 30-question set is a development set.

---

## 5. Evidence-Quality Evaluation

**Classification: VALID RESULT, with stated sampling limitations**

### Design

A human annotator judged the usefulness of individual retrieved passages relative to
their question, without seeing any machine score, ranking, admission decision or
suggested label.

| Property | Value |
| --- | --- |
| Passages annotated | 120, drawn from the 600 |
| Annotators | 1 |
| Label scale | 0 = not relevant · 1 = partially useful · 2 = useful |
| Sampling seed | 42 |
| Sampling rule | All RAG² admissions, then round-robin over 9 SCAF × RAG² score-tertile cells; tertile bins fixed before annotation |
| Machine suggestion generated | No |
| Machine suggestion displayed | No |
| Annotation sheet SHA-256 | `19d427cd6a0909056e3e24a1a2aa9a8976f82426cc669eb815b101f474e14ed1` |
| Label fingerprint | `49e2b927212e4a4c39c5756f8a46f4c56b6b3d45a64586ed4b803b32a9688354` |

### Annotation results

| Label | Count | Share |
| --- | ---: | ---: |
| 0 — not relevant | 14 | 11.7% |
| 1 — partially useful | 46 | 38.3% |
| 2 — useful | 60 | 50.0% |

### Human labels against each arm's admission decision

**Classification: DESCRIPTIVE ONLY**

| Subset | n | Mean usefulness | Labels observed |
| --- | ---: | ---: | --- |
| All annotated passages | 120 | 1.383 | — |
| Passages RAG² admitted | 4 | **1.750** | 1, 2, 2, 2 |
| Passages SCAF rejected | 2 | **1.500** | 1, 2 |
| Passages judged *not relevant* (label 0) | 14 | 0 | all 14 admitted by SCAF; **0 admitted by RAG²** |

All 14 passages the annotator judged not relevant were admitted by SCAF, and none was
admitted by RAG².

**These subgroup means are descriptive only and support no comparison.** The RAG²
subgroup contains four passages and the SCAF-rejected subgroup contains two; both are
complete censuses of very small sets rather than samples, and no interval around a mean
of four observations would distinguish them from the overall mean of 1.383. That RAG²
admitted none of the not-relevant passages is a direct consequence of its admitting only
four passages in total, and is not evidence of more selective judgement.

### Integrity audit

An independent audit of the annotation process returned **31 PASS, 2 WARN, 0 FAIL**,
verdict *PASS WITH LIMITATIONS*. Both warnings concern sampling structure rather than
annotation conduct:

1. **The sample does not span the question set evenly.** 29 of 30 questions are
   represented, with up to 8 passages drawn from a single question. The rows are
   therefore not independent observations.
2. **Forcing all RAG² admissions makes the sample non-self-weighting.** 4 of the 120 rows
   were included by rule rather than by chance. Those four are a complete census of RAG²'s
   admissions, not a sample of them, so the RAG²-admitted/rejected contrast describes
   exactly those four passages, and the sample as a whole is not a simple random sample
   of the 600.

### An earlier annotation pass and its status

**Classification: INVALID FOR COMPARISON**

A preliminary annotation pass displayed a machine-generated suggested label on all 120
rows. The human label matched the displayed suggestion on 120 of 120 rows, and the
resulting correlation with the machine score was approximately ρ = 0.63.

That correlation measures the effect of the interface on the annotator, not the
agreement between an independent judgement and the score. When the same association was
measured on the independent pass described above, it fell to 0.12 with an interval
including zero. The earlier pass has been retained in full as a record of what was done;
it does not constitute human validation of any scoring component.

---

## 6. SCAF Analysis

All analyses in this section are exact recomputations of the admission arithmetic on the
recorded decisions. Before any of them was reported, three identities were verified
against the frozen run:

- A(s) rebuilt from the recorded sub-scores reproduces the recorded score for all 600
  candidates (maximum absolute error 8.0 × 10⁻⁷);
- all 600 recorded admit/reject decisions are reproduced exactly;
- the currency term regenerated from each passage's publication date reproduces the
  recorded value (maximum absolute error 5.0 × 10⁻⁷).

### 6.1 Which component drives the admission score

**Classification: VALID RESULT**

| Component | Weight | Observed range | SD | Share of Var A(s) |
| --- | ---: | --- | ---: | ---: |
| Support (σ) | 0.50 | 0.000 – 1.000 | 0.2066 | **83.5%** |
| Currency (γ) | 0.30 | 0.536 – 1.000 | 0.1445 | 14.7% |
| Authority (τ) | 0.20 | 0.400 – 1.000 | 0.0752 | 1.8% |
| Reranker (ρ) | 0.00 | 0.000 – 1.000 | 0.3035 | 0.0% |

By variance, the admission score is dominated by its lexical support term. The currency
term cannot contribute more than it does within this corpus: over a 2021–2026 span at a
five-year half-life, γ never falls below 0.536.

### 6.2 Component ablations

**Classification: VALID RESULT (admission behaviour only)**

| ID | Configuration | Admitted | Decisions changed |
| --- | --- | ---: | ---: |
| — | As executed | 575 / 600 | — |
| A4 | No filter | 600 / 600 | 25 |
| A5 | Currency disabled, weights renormalised | 423 / 600 | 152 |
| A7 | Support excluded, weights renormalised | 600 / 600 | 25 |
| A12a | Authority removed, weights renormalised | 575 / 600 | 4 |
| A12b | Authority ordering fully inverted | 584 / 600 | 15 |

A5 and A7 are reported with weights renormalised because zeroing a weight shrinks A(s)
and makes a fixed threshold effectively stricter — an artifact that would otherwise be
mistaken for an effect of removing the component.

**Authority is not load-bearing in this experiment.** Inverting the entire source
authority ordering changes 15 of 600 decisions (2.5%); removing the component changes 4.

### 6.3 Threshold and half-life sensitivity

**Classification: VALID RESULT**

Neither the admission threshold nor the currency half-life was fitted on validation data.

| Threshold θ | Admitted | | Half-life | Admitted | Decisions changed |
| ---: | ---: | --- | ---: | ---: | ---: |
| 0.30 | 598 (99.7%) | | 1 y | 443 (73.8%) | 132 |
| 0.40 | 589 (98.2%) | | 2 y | 521 (86.8%) | 54 |
| 0.45 | 575 (95.8%) | | 3 y | 551 (91.8%) | 24 |
| 0.50 | 525 (87.5%) | | 5 y | 575 (95.8%) | 0 |
| 0.60 | 371 (61.8%) | | 8 y | 584 (97.3%) | 9 |
| 0.70 | 169 (28.2%) | | 20 y | 589 (98.2%) | 14 |

Admission ranges from 99.7% to 28.2% across plausible thresholds. **The headline figure
of 575/600 is a property of an unfitted threshold as much as of the policy**, and this
sensitivity should accompany any statement of the admission rate.

### 6.4 Component behaviour against the human labels

**Classification: INCONCLUSIVE for support, currency and the admission score**

See §10 for the full statistical treatment. In summary, neither the SCAF admission score
nor its support term nor its currency term shows an association with independent human
usefulness judgements that is distinguishable from zero.

**No SCAF component is validated by these experiments.** Support was implemented as
lexical IDF-weighted overlap rather than as entailment, and its association with human
judgement includes zero. Currency has minimal dynamic range within this corpus. Authority
varies across only a small number of candidates and its ordering is not load-bearing.
Corroboration was not implemented. Supersession information is absent: the field is
recorded as `unknown` for all 480 time-sensitive candidates, so no supersession state was
ever exercised.

---

## 7. Matched-k Results

The matched-k design supplies **both arms with exactly five passages** from the same
frozen candidate set, so that the amount of evidence is held constant and only the choice
of evidence varies.

### Selection rule

Each arm ranks **the same 20 frozen candidates** by **its own score** and takes the top
five, with `chunk_id` as the tie-break so that two runs over the same inputs select
identically. This is the `matched_budget` procedure already used elsewhere in the
analysis; no new scoring rule, weight or threshold was introduced.

Selection deliberately ignores each arm's own admission threshold. Applying RAG²'s
natural threshold would admit 4 passages across all 600 decisions and could not reach
k = 5 for 26 of the 30 questions, which would reintroduce the evidence-quantity
imbalance the design exists to remove.

### 7.1 Selection stage — **Classification: VALID RESULT**

| Property | Value |
| --- | --- |
| k | 5 |
| Questions | 30 |
| Selection invariants | All 8 passed |
| Passages per arm per question | exactly 5 |
| Duplicate passages within a question | none |
| Passages outside that question's frozen candidates | none |
| Cross-question candidate leakage | none |
| Questions where the selected evidence differs | 30 / 30 |
| Questions with identical 5-passage sets | 0 / 30 |
| Questions with no overlap at all | 9 / 30 |
| Questions with some overlap | 21 / 30 |
| Mean overlap | 0.9 of 5 |
| Mean Jaccard | 0.106 |
| Overlap distribution | 0 passages: 9 questions · 1: 15 · 2: 6 |

At an equal budget of five passages the two policies select substantially different
evidence: no question receives an identical set, nine share nothing at all, and on
average fewer than one of five passages coincides.

### 7.2 Generation stage — **Classification: NOT EXECUTED**

**Answer generation under matched-k conditions was not performed.** No matched-k answers
exist and no answer-level evaluation of them was carried out. The selection result above
stands on its own and describes the evidence that would be supplied; it does not describe
any generated answer.

---

## 8. Recency / Filter Recency-Bias Probe Results

**Classification: NOT EXECUTED**

The probe was designed to measure admission asymmetry with respect to evidence age over
matched temporal-counterfactual passage pairs, together with a permutation control in
which publication dates are randomly reassigned. The implementation — pair construction,
the permutation control, the paired estimator and the validity gate — exists and passes
its unit tests, **but it has never been run on data.** The temporal pair set it requires
was not available in this environment.

**No recency-bias result exists.** No statement about whether the admission stage prefers
older or newer evidence is supported by any executed experiment.

### Related offline analysis, reported separately

**Classification: DESCRIPTIVE ONLY**

A descriptive table of admission rate by publication year was computed from the 600
recorded decisions. It is reported here separately from executed experimental results
because it is not the matched-pair measurement described above.

| Year | Candidates | RAG² admitted | SCAF admitted |
| ---: | ---: | ---: | ---: |
| 2021 | 22 | 0 (0.00%) | 19 (86.4%) |
| 2022 | 84 | 0 (0.00%) | 78 (92.9%) |
| 2023 | 116 | 0 (0.00%) | 105 (90.5%) |
| 2024 | 117 | 0 (0.00%) | 115 (98.3%) |
| 2025 | 141 | 2 (1.42%) | 138 (97.9%) |
| 2026 | 120 | 2 (1.67%) | 120 (100%) |

**This table cannot support a recency-bias claim, in either direction.** The passages
compared are not matched on claim, source tier or length, so any difference between years
is confounded with topic, wording and document type. The corpus also spans only about
five years, leaving very little age contrast to detect. RAG²'s four admissions fall in
2025–2026, but four admissions are not evidence of a preference.

---

## 9. Answer-Quality Results

### 9.1 Answers generated under natural admission

**Classification: INVALID FOR COMPARISON**

Sixty answers were generated — 30 per arm — using an identical frozen generator
(`thesis-llama3-8b-q4`), identical greedy decoding, identical maximum output length and
an identical prompt (fingerprint `fc6db2781ddc`). Generation succeeded for all 30
questions in both arms with zero generation failures.

A preliminary summary figure was produced from this run — approximately **1.000 for RAG²
against 0.893 for SCAF** — and is recorded in the analysis document as a preliminary
figure that must not be reported as a headline result. **It is not evidence that RAG²
outperformed SCAF**, for the reasons below. No primary result file records these values as
a validated metric; they survive only as a noted preliminary figure.

**These answers cannot support a comparison of answer quality between the two policies,
for three independent reasons.**

1. **The evidence budgets were not matched.** SCAF's arm received a mean of **19.167
   admitted passages per question**; RAG²'s received **0.133**. In characters of context
   supplied to the generator this is **26,904.6 against 39.1 — a ratio of approximately
   688:1**. The arms therefore differ in the *quantity* of evidence as well as in the
   admission policy, so any difference between the answers is confounded with context
   size.
2. **The design's own context-budget constraint was violated.** The intended maximum is
   five passages per arm. SCAF's arm averaged 19.167, roughly four times that ceiling,
   while RAG² supplied no evidence at all for 26 of 30 questions — meaning those answers
   were produced closed-book rather than from retrieved evidence.
3. **No answer-quality target exists.** The 30-question evaluation set contains no gold
   answer, no reference answer and no expert answer label; the manifest records
   `with_reference: 0` for both arms. No metric of answer correctness can be computed
   against it, so the preliminary figure above is not anchored to any correctness
   standard.

The generated answers are retained as a record of what the pipeline produced. They
establish that the end-to-end pipeline executes and that both arms produce output; they
do not establish anything about the relative quality of that output.

### 9.2 Answer-level correctness evaluation

**Classification: NOT EXECUTED**

No answer-level correctness evaluation, human preference comparison or expert rating was
performed. No claim about answer accuracy, factual correctness, citation correctness or
clinical appropriateness is supported by any executed experiment.

---

## 10. Statistical Analysis

All interval estimates resample **whole questions**, not individual passages. Twenty
candidates share each question, one retrieval and one topic, so treating the 600 rows as
independent observations would understate uncertainty. All analyses use 10,000 resamples
with a fixed seed (20260910), so the intervals reproduce exactly.

### 10.1 Difference in admission rate

**Classification: VALID RESULT**

| Estimate | Value | 95% CI (question-clustered) |
| --- | ---: | --- |
| RAG² admission rate | 0.0067 | [0.0017, 0.0133] |
| SCAF admission rate | 0.9583 | [0.9350, 0.9783] |
| Difference (SCAF − RAG²) | **+0.9517** | **[+0.9283, +0.9733]** |
| Cohen's *h* | 2.567 | large |

An exact McNemar test on the 571 discordant pairs returns p ≈ 2.6 × 10⁻¹⁷². **This
p-value is reported only alongside the clustered interval and is anti-conservative**: it
treats the 600 candidates as independent, which the clustered design shows they are not.

*In plain terms:* the two policies admit very different amounts of evidence, and the size
of that difference is estimated precisely. This is a difference in operating point, not a
measure of quality.

### 10.2 Association between machine scores and human usefulness labels

**Classification: INCONCLUSIVE for four of five signals**

Spearman rank correlation between each machine signal and the human 0/1/2 label, over 120
annotated passages clustered within 29 questions, with Holm–Bonferroni correction applied
across the five comparisons.

| Signal | ρ | 95% CI | p | Holm threshold | Survives correction |
| --- | ---: | --- | ---: | ---: | :---: |
| Reranker rank (ρ term) | **+0.2175** | [+0.0850, +0.3382] | 0.00160 | 0.01000 | **Yes** |
| SCAF support term (σ) | +0.1640 | [−0.0239, +0.3238] | 0.08739 | 0.01250 | No |
| SCAF admission score | +0.1237 | [−0.0418, +0.2759] | 0.14658 | 0.01667 | No |
| SCAF currency term (γ) | −0.0519 | [−0.2690, +0.1721] | 0.65693 | 0.02500 | No |
| RAG² filter score | +0.0170 | [−0.1388, +0.1661] | 0.82932 | 0.05000 | No |

*In plain terms:* the confidence interval for the SCAF admission score includes zero, as
does the interval for its support term, its currency term, and the RAG² filter score. For
those four signals the analysis does not distinguish "orders passages as the annotator
did" from "no association at all". The only signal whose association survives correction
for multiple comparisons is the rank assigned by the frozen MedCPT reranker — a component
of the shared retrieval stage that neither admission policy consults when making its
decision.

### 10.3 Analyses not performed

No test of admission asymmetry with respect to age, no permutation control, no test of
unsupported-claim rate, no mediation analysis, and no non-inferiority test on
time-invariant question answering were carried out. Each requires an instrument or a
reference label that the executed experiments did not produce.

---

## 11. Overall Experimental Findings

### Supported by the data

- The RAG² pipeline was reproduced structurally, with a retrained filter, over an
  Alzheimer's-scoped corpus of 43,409 documents and 773,183 unique chunks.
- The two admission policies operate at very different points: RAG² admitted 4 of 600
  candidates, SCAF 575 of 600, a difference of +0.95 [+0.93, +0.97].
- The RAG² filter admitted almost nothing on this corpus, leaving 26 of 30 questions with
  no retrieved evidence — a measurable over-rejection behaviour at its configured
  threshold.
- RAG²'s admitted set is a strict subset of SCAF's: no passage was kept by the baseline
  and discarded by the proposed policy.
- The SCAF admission score is dominated by its lexical support term, which accounts for
  83.5% of the score's variance; authority accounts for 1.8% and is not load-bearing.
- The admission rate is strongly dependent on an unfitted threshold, ranging from 99.7%
  to 28.2% across plausible values.
- At an equal five-passage budget the two policies select substantially different
  evidence: no identical sets, nine questions with no overlap, mean overlap 0.9 of 5.
- The frozen MedCPT reranker's rank is associated with human-judged passage usefulness
  (ρ = +0.218, [+0.085, +0.338]), and this survives correction for multiple comparisons.

### Not supported by the data

- That SCAF admits evidence of higher quality than RAG². Its association with independent
  human usefulness labels has a confidence interval that includes zero, and it admitted
  all 14 passages the annotator judged not relevant.
- That the RAG² filter score tracks human-judged passage usefulness. Its interval also
  includes zero.
- That either policy produces better answers. No answer-quality target exists for the
  evaluation questions.

### Inconclusive

- Whether the SCAF support term, currency term or overall admission score carries any
  signal about evidence usefulness. The estimates are positive for two of the three but
  not distinguishable from zero at this sample size and clustering.

### Not scientifically interpretable

- The answer comparison from the natural-admission run. The arms received context
  differing by a factor of approximately 688, so the conditions were not matched.
- The preliminary annotation pass in which a suggested label was displayed. Its
  correlation reflects the annotation interface.
- The descriptive admission-by-year table. The passages compared are unmatched and the
  corpus offers minimal age contrast.

### Not yet experimentally tested

- Admission asymmetry with respect to evidence age, and its permutation control. The
  measurement was implemented but not executed.
- Matched-k answer generation and any evaluation of the resulting answers.
- Comparison between a perplexity-derived and a support-derived filter label, which would
  require a second filter trained on the alternative label.
- Any replication on a second model backbone.

---

## Limitations affecting interpretation

1. **Support is lexical, not entailment.** The support term was computed as question-term
   coverage weighted by corpus IDF. It is not a semantic entailment measure, and no claim
   about evidential support rests on it.
2. **Admission counts are operating-point dependent.** Both the 4/600 and the 575/600
   figures are properties of thresholds that were not fitted on validation data.
3. **The two policies have different natural admission rates**, so the natural-admission
   comparison cannot separate policy effects from evidence-quantity effects.
4. **The corpus spans approximately five years**, leaving the currency term a range of
   only [0.536, 1.000] and providing minimal age contrast.
5. **Supersession information is absent.** The field is `unknown` for all 480
   time-sensitive candidates; no supersession or contested state was ever exercised.
6. **Authority varies little.** Most candidates carry no authority tier, and inverting the
   full ordering changes 2.5% of decisions.
7. **The annotation sample is not a simple random sample.** All four RAG² admissions were
   forced into it, 29 of 30 questions are represented, and up to 8 passages come from a
   single question.
8. **One annotator.** No inter-rater agreement could be computed.
9. **No gold answers exist** for the 30-question evaluation set, and no answer-level
   correctness evaluation was completed.
10. **Thirty development questions, one backbone, one decoding pass.** No variance
    estimate across runs is available.
11. **Absolute performance is not comparable to published baselines**, because the corpus
    is domain-scoped. Only within-study contrasts between arms are interpretable.

---

## Provenance of the reported figures

| Quantity | Source file |
| --- | --- |
| Admission counts, fairness checks, configuration, context sizes | `comparison_scientific/manifest.json` |
| Per-candidate decisions and sub-scores (used for cross-checking) | `comparison_scientific/per_question.jsonl` |
| Annotation labels and fingerprint | `evidence_quality/annotation_sheet_v2.jsonl` |
| Audit verdict and warnings | `evidence_quality/audit_report.json` |
| Sampling rule, seed, tertile edges | `evidence_quality/sample_manifest.json` |
| Variance decomposition, ablations, sensitivity sweeps | `analysis/thesis_ablations.json` |
| Clustered intervals, effect sizes, corrected p-values | `analysis/thesis_statistics.json` |
| Matched-k selection statistics and invariants | `analysis/matched_k5_readiness.json` |
| Answer-target availability | `analysis/answer_quality_readiness.json` |
| Corpus counts and composition | `data/corpora/pmc/chunks/chunk_stats.json` |
| Per-document dates and temporal split | `data/corpora/pmc/metadata/canonical_dates.csv` |
| Filter training composition | `training_dataset/manifest.json` |

Admission counts were independently re-derived from the raw decision records and agree
with the run manifest.

### Verification notes on individual figures

Every figure in this document was read from the file named above and, where it appears in
more than one place, checked for agreement. Three points are recorded for completeness:

- **Annotation label fingerprint.** The audit record and the annotation sheet agree with
  each other: sheet SHA-256 `19d427cd6a0909056e3e24a1a2aa9a8976f82426cc669eb815b101f474e14ed1`,
  label fingerprint `49e2b927212e4a4c39c5756f8a46f4c56b6b3d45a64586ed4b803b32a9688354`.
  A value ending `…688351` has circulated in working notes; the repository value ends
  `…688354`, and the repository value is the one used here.
- **Evidence quantity versus context size.** The figures 19.167 and 0.133 are the mean
  number of *admitted passages per question* for SCAF and RAG² respectively. The mean
  *context sizes* supplied to the generator are 26,904.6 and 39.1 characters. The
  approximately 688:1 ratio derives from the character counts; the passage counts give a
  ratio of about 144:1. The two quantities are distinct and are reported separately above.
- **Preliminary answer figures.** The values 1.000 (RAG²) and 0.893 (SCAF) appear only in
  the narrative analysis document, which already records them as preliminary and
  not reportable. No primary result file stores them as a computed metric, and they are
  presented here only as the non-reportable figures they are.
