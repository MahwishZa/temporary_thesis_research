# RAG² vs SCAF — what the current experiment actually shows

> **This file is generated.** It is a *derived* artifact, rebuilt from the frozen scientific artifacts by `experiments/scripts/analyse_results.py`. It is not itself evidence, and writing something here does not make it true. Every number below is computed from the files listed in §3.

Generated 2026-09-10T17:47:42Z · analysis version `scientific-results-v1`

## 1. Executive summary

In one sentence: **the experiment shows that SCAF and RAG² admit very different amounts of evidence, and that neither one's score orders evidence the way a human does.**

- SCAF admits **575 of 600** candidate passages (95.83%). RAG² admits **4** (0.67%).
- **26 of 30 questions receive no evidence at all from RAG².**
- Across the 120 independently annotated passages, the SCAF score's rank association with human-judged usefulness is **0.1237** — weak. RAG²'s is **0.017** — essentially none.
- SCAF-admitted passages average **1.3814** on the 0–2 usefulness scale; the 2 SCAF-rejected passages average **1.5**. The threshold does not separate useful evidence from unuseful evidence here.
- All **14** passages a human judged *not relevant* were **admitted by SCAF**; 0 were admitted by RAG².

**The most important thing to understand about these numbers:** an earlier annotation pass reported a much stronger SCAF association (≈0.63). That pass showed the annotator a word-matching suggestion on every passage, and all 120 of its labels matched that suggestion exactly. Because SCAF's support term is *also* word-matching, that correlation largely measured the annotation interface rather than SCAF. The independent pass analysed here removed the suggestion, and the association fell to 0.1237. **The weak number is the real one.**

## 2. Experimental setup

- **30 Alzheimer questions**, **[20] candidates each**, **600 candidate decisions** in total.
- Both arms scored the **identical frozen candidate set in identical order**. Neither arm did its own retrieval. Same generator, same decoding.
- Fairness checks recorded in the run manifest: `all_passed = True`.
- Scientific preconditions: `all_passed = True`, `reportable = True`.
- The only difference between the arms is the **admission decision**: RAG²'s trained perplexity filter versus SCAF's weighted score against a threshold.

## 3. Data and provenance

| What | Value |
| --- | --- |
| Frozen candidate-set digest (run) | `316260f04c1720fbc20c1b584ea9f0a093dedbae9fd46476d8a7f45380e86aad` |
| Frozen digest recorded for the sample | `316260f04c1720fbc20c1b584ea9f0a093dedbae9fd46476d8a7f45380e86aad` |
| Digests agree | **True** |
| Retrieval is production MedCPT | **True** |
| Annotation sheet SHA-256 | `19d427cd6a0909056e3e24a1a2aa9a8976f82426cc669eb815b101f474e14ed1` |
| Label fingerprint | `49e2b927212e4a4c39c5756f8a46f4c56b6b3d45a64586ed4b803b32a9688354` |
| Annotation pass | `corrected-human-only-v1` |
| Human annotations | 120 (joined to decisions: 120) |
| Sample seed / size / population | 42 / 120 / 600 |
| RAG² admissions forced into sample | 4 |

Source files:

- `decisions` → `/home/user/thesis_research/experiments/results/rag2_vs_scaf_alzheimer/comparison_scientific/per_question.jsonl`
- `run_manifest` → `/home/user/thesis_research/experiments/results/rag2_vs_scaf_alzheimer/comparison_scientific/manifest.json`
- `annotation_sheet` → `/home/user/thesis_research/experiments/results/rag2_vs_scaf_alzheimer/evidence_quality/annotation_sheet_v2.jsonl`
- `sample_manifest` → `/home/user/thesis_research/experiments/results/rag2_vs_scaf_alzheimer/evidence_quality/sample_manifest.json`

The retired anchored pilot in `evidence_quality/pilot_anchored/` was **not** used. It is kept as the historical record of a methodological correction and must never be pooled with these labels.

## 4. Admission behaviour

**Candidate level** (all 600 decisions):

| | Count | Rate |
| --- | ---: | ---: |
| RAG² admitted | 4 | 0.67% |
| SCAF admitted | 575 | 95.83% |
| Both admit | 4 | |
| RAG² only | 0 | |
| SCAF only | 571 | |
| Neither | 25 | |
| Overlap (Jaccard) | 0.006957 | |

**Question level** (30 questions): RAG² supplies evidence for **4** of them and none for **26**. SCAF supplies evidence for **30**, averaging 19.1667 passages per question (range 16–20).

**RAG²-only admissions: 0.** Every passage RAG² admits is also admitted by SCAF. The two policies are not making opposite calls — RAG²'s admissions are a subset of SCAF's.

> admission counts measure how much evidence each policy lets through. They say nothing about whether that evidence is better; that is what the human labels are for.

## 5. Human evidence-quality results

120 passages were read by a person who saw only the question and the passage, and rated each **0** (not relevant), **1** (partially relevant) or **2** (clearly relevant).

Overall: n=120, mean 1.3833, distribution {0: 14, 1: 46, 2: 60}.

| Group | n | Mean usefulness |
| --- | ---: | ---: |
| All annotated | 120 | 1.3833 |
| SCAF admitted | 118 | 1.3814 |
| SCAF rejected | 2 | 1.5 |
| RAG² admitted | 4 | 1.75 |
| RAG² rejected | 116 | 1.3707 |
| SCAF only (SCAF admits, RAG² rejects) | 114 | 1.3684 |
| RAG² only (RAG² admits, SCAF rejects) | 0 | — |
| Both admit | 4 | 1.75 |
| Neither admits | 2 | 1.5 |

**Rank association with the human label** (Spearman, descriptive only):

| Score | ρ | n |
| --- | ---: | ---: |
| `scaf_score` | 0.1237 | 120 |
| `scaf_sigma_support` | 0.164 | 120 |
| `scaf_gamma_currency` | -0.0519 | 120 |
| `scaf_tau_authority` | -0.0164 | 120 |
| `rag2_score` | 0.017 | 120 |

**Mean usefulness by SCAF score band** — if the score ordered evidence quality, this column would rise:

| SCAF score | n | Mean usefulness |
| --- | ---: | ---: |
| below 0.50 | 12 | 1.0833 |
| 0.50 to 0.80 | 102 | 1.4412 |
| 0.80 and above | 6 | 1 |

> These are descriptive associations on a stratified, non-random sample of 120 of 600 decisions, clustered within questions, with all four RAG2 admissions forced in. They are not estimates of a population parameter and no significance test is appropriate. The sigma association in particular is between one lexical measure and a human judgement, not an independent validation of lexical support.

## 6. SCAF component behaviour

Weights in this run: `{'authority': 0.2, 'corroboration': 0.0, 'currency': 0.3, 'support': 0.5}`, threshold `0.45`, support method `['lexical-coverage-corpus-idf-v2']`.

| Term | Distinct values | Range | Admitted − rejected | ρ with human label |
| --- | ---: | --- | ---: | ---: |
| support_sigma | 307 | 0.0–1.0 | 0.3829 | 0.164 |
| currency_gamma | 150 | 0.5359–1.0 | 0.1667 | -0.0519 |
| authority_tau | 5 | 0.4–1.0 | 0.0146 | -0.0164 |
| corroboration_rho | 1 | 0.0–0.0 | 0.0 | n/a (constant) |

Reading this table:

- **Support (σ)** separates admitted from rejected by a wide margin — but it carries half the score, so it does that *by construction*. Its association with the human label is 0.164. It is computed as **lexical/content-word coverage weighted by corpus IDF**, not semantic entailment and not expert relevance.
- **Currency (γ)** varies over 0.5359–1.0 and its association with the human label is -0.0519.
- **Authority (τ)** takes only **5 distinct values** across all 600 decisions and separates admitted from rejected by 0.0146. It is effectively non-discriminative in this corpus.
- **Corroboration (ρ)** is weighted **0.0** and reports `{'not_implemented': 600}`. It contributes nothing and is **not experimentally validated** — it is not implemented.
- The admission gate never fired: `{False: 600}`. Admission is purely score ≥ threshold.

Reduced-vector variants (**not** validated ablations):

| Variant | Admitted | Rate | Max reachable score |
| --- | ---: | ---: | ---: |
| support_only | 426/600 | 0.71 | 1.0 |
| support_currency | 478/600 | 0.796667 | 0.8 |
| support_authority | 147/600 | 0.245 | 0.7 |
| full_scaf | 575/600 | 0.958333 | 1.0 |

> A term that separates admitted from rejected is not thereby validated: support carries half the score, so it separates them by construction. The ablation variants cannot reach 1.0 when a term is removed, so a fixed 0.45 threshold is stricter for them and the counts confound contribution with reachability. Corroboration is weighted 0.0 and reports not_implemented, so it is inactive and untested.

## 7. RAG²-vs-SCAF disagreement

The disagreement is almost entirely one-sided: **571** passages are admitted by SCAF and rejected by RAG², while **0** go the other way. Among annotated rows, SCAF-only passages average 1.3684 usefulness — close to the overall mean of 1.3833, not obviously better or worse.

The 4 passages both arms admit average 1.75, and the 2 neither admits average 1.5. Both cells are far too small to support any inference.

## 8. Matched-k / matched-context analysis

The raw admission counts are not comparable: SCAF passes ~19 passages per question and RAG² passes 0.13. Any downstream difference would be confounded by context size. The honest comparison re-ranks the **same frozen candidates** by each arm's own score and takes the top *k*. This requires no new experiment.

**How much the two arms overlap at matched k:**

| k | Mean Jaccard | Shared passages |
| ---: | ---: | --- |
| 1 | 0.0 | 0/30 |
| 3 | 0.0533 | 8/90 |
| 5 | 0.1056 | 27/150 |
| 8 | 0.2156 | 82/240 |

**At k = 1 the two arms share no passage at all.** They are selecting essentially disjoint evidence.

**Human usefulness of each arm's own top-k** (annotated rows only):

| k | SCAF | RAG² |
| ---: | --- | --- |
| 1 | n=8, mean 1.625 | n=8, mean 1.625 |
| 3 | n=16, mean 1.4375 | n=26, mean 1.5 |
| 5 | n=25, mean 1.4 | n=36, mean 1.4444 |
| 8 | n=41, mean 1.5122 | n=55, mean 1.4364 |

These are close at every k, in both directions. **At matched context budget, neither arm's ranking retrieves better evidence than the other's** in this sample.

> this re-ranks the SAME frozen candidates by each arm's own score; it runs no new retrieval and no new generation. Only annotated rows carry a human label, so the per-k samples are small, unequal between arms, and not a controlled experiment.

The earlier preliminary figures (RAG² ≈ 1.000 vs SCAF ≈ 0.893) came from a run with wildly unmatched context sizes and **must not** be reported as a headline result.

## 9. Retracted and superseded evidence

| Field | Values across all 600 decisions |
| --- | --- |
| `retracted` | {'no': 600} |
| `supersession` | {'unknown': 480, 'n/a': 120} |
| `currency_state` | {'current': 480, 'not_time_sensitive': 120} |

**There are zero retracted candidates in this evaluation set, and supersession is never determined** — it is recorded as `unknown` or `n/a` for every candidate. SCAF's retraction and supersession handling is therefore **completely untested by this experiment**. Nothing can be claimed about it in either direction.

## 10. Temporal / FRB-PAIRS feasibility

Candidate publication years: `{'2021': 22, '2022': 84, '2023': 116, '2024': 117, '2025': 141, '2026': 120}`

- Earliest year present: **2021**
- Candidates before 2020: **0**
- Candidates before 2022: **22** of 600
- γ (currency) range: 0.5359–1.0

**The current corpus cannot support an old-versus-new temporal counterfactual.** A FRB-PAIRS-style design needs pairs where an older passage and a newer passage answer the same question differently. With no candidate older than 2021 and only 22 before 2022, there is almost no old evidence to pair against. This is a **methodological limitation of the corpus**, not a fixable analysis choice, and it is why γ has so little to discriminate on.

## 11. Representative successes

The four passages RAG² admitted — all also admitted by SCAF:

- **eq-0067** (alz-027) — human label **2**
  - SCAF 0.5745 (admitted: True), RAG2 0.5481 (admitted: True)
  - σ 0.439 · γ 0.9167 · τ 0.4 · 2025-05 · pubmed-abstract
  - Q: How is caregiver burden assessed in Alzheimer disease?
  - P: with the evolution of caregiver burden. CONCLUSIONS: Caregivers of patients who received a diagnosis of MCI due to AD report substantial burden, that increased with time. Future studies should investigate caregiver characteristics that may …

- **eq-0044** (alz-012) — human label **2**
  - SCAF 0.5555 (admitted: True), RAG2 0.5457 (admitted: True)
  - σ 0.3509 · γ 1.0 · τ 0.4 · 2026-05-05 · pubmed-abstract
  - Q: What is the evidence for memantine in moderate to severe Alzheimer disease?
  - P: superior memantine effects in moderate-to-severe disease. CONCLUSION: Non-pharmacological interventions demonstrated short-term cognitive benefits primarily in mild Alzheimer's disease populations and should be interpreted as adjunctive sym…

- **eq-0008** (alz-028) — human label **2**
  - SCAF 0.5853 (admitted: True), RAG2 0.5243 (admitted: True)
  - σ 0.3925 · γ 0.9969 · τ 0.45 · 2025-12-23 · pmc-fulltext
  - Q: What is the differential diagnosis for rapidly progressive dementia?
  - P: Imaging biomarkers provide structural and molecular and metabolic insights into DLB, crucial for its differential diagnosis, especially at early stages.…

- **eq-0064** (alz-010) — human label **1**
  - SCAF 0.4951 (admitted: True), RAG2 0.5037 (admitted: True)
  - σ 0.1502 · γ 1.0 · τ 0.6 · 2026-04-16 · currency-pack
  - Q: What is the recommended monitoring schedule for patients receiving anti-amyloid antibodies?
  - P: to increase with higher antibody doses and more rapid titration schedules. These side effects can limit dosage and treatment duration [[REF]]. Furthermore, as mentioned, the drugs require repeated intravenous administration and close monito…

## 12. Representative failures

**SCAF admitted these, and a human judged them not relevant.** These are the highest-scoring such cases, so they are SCAF's worst mistakes, not its marginal ones:

- **eq-0072** (alz-015) — human label **0**
  - SCAF 0.8132 (admitted: True), RAG2 0.452 (admitted: False)
  - σ 0.857 · γ 0.9824 · τ 0.45 · 2025-11 · pmc-fulltext
  - Q: Which modifiable risk factors are associated with dementia prevention?
  - P: Dementia is a major cause of disability and dependence in older adults, highlighting the urgent need to identify those at high risk early. Numerous dementia risk scores have been developed as statistical models, with a few being adapted int…

- **eq-0118** (alz-005) — human label **0**
  - SCAF 0.6761 (admitted: True), RAG2 0.2746 (admitted: False)
  - σ 0.7112 · γ 0.8017 · τ 0.4 · 2024-05-27 · pubmed-abstract
  - Q: How is cerebrospinal fluid p-tau181 used in Alzheimer diagnosis?
  - P: Alzheimer's disease (AD), a primary cause of dementia globally, is traditionally diagnosed via cerebrospinal fluid (CSF) measures and positron emission tomography (PET). The invasiveness, cost, and limited accessibility of these methods hav…

- **eq-0085** (alz-028) — human label **0**
  - SCAF 0.6459 (admitted: True), RAG2 0.3576 (admitted: False)
  - σ 0.7604 · γ 0.6189 · τ 0.4 · 2022-07 · pubmed-abstract
  - Q: What is the differential diagnosis for rapidly progressive dementia?
  - P: Alzheimer's disease (AD) is a progressive neurodegenerative disease and the single commonest cause of dementia. Many other diseases can, however, cause dementia, and differential diagnosis can be challenging, especially in early disease sta…

**SCAF rejected these** (the only rejections in the annotated sample):

- **eq-0069** (alz-022) — human label **2**
  - SCAF 0.399 (admitted: False), RAG2 0.2763 (admitted: False)
  - σ 0.2728 · γ 0.6085 · τ 0.4 · 2022-06-01 · pubmed-abstract
  - Q: What blood tests should be ordered in the initial dementia workup?
  - P: PURPOSE OF REVIEW: This article discusses how fluid biomarkers can augment the routine dementia evaluation and improve diagnostic accuracy. The tests that are currently available and the indications for their use are described. Further, tes…

- **eq-0003** (alz-022) — human label **1**
  - SCAF 0.4398 (admitted: False), RAG2 0.2985 (admitted: False)
  - σ 0.2728 · γ 0.7446 · τ 0.4 · 2023-11 · pubmed-abstract
  - Q: What blood tests should be ordered in the initial dementia workup?
  - P: Early and accurate diagnosis are crucial to appropriate care, ensuring timely intervention, and planning for future needs of patients with dementia. Dementia is a clinical diagnosis and should include comprehensive evaluation of patient cog…

**Useful evidence SCAF scored lowest** — passages a human called clearly relevant that SCAF ranked at the bottom:

- **eq-0069** (alz-022) — human label **2**
  - SCAF 0.399 (admitted: False), RAG2 0.2763 (admitted: False)
  - σ 0.2728 · γ 0.6085 · τ 0.4 · 2022-06-01 · pubmed-abstract
  - Q: What blood tests should be ordered in the initial dementia workup?
  - P: PURPOSE OF REVIEW: This article discusses how fluid biomarkers can augment the routine dementia evaluation and improve diagnostic accuracy. The tests that are currently available and the indications for their use are described. Further, tes…

- **eq-0007** (alz-022) — human label **2**
  - SCAF 0.4848 (admitted: True), RAG2 0.44 (admitted: False)
  - σ 0.4203 · γ 0.6156 · τ 0.45 · 2022 · pmc-fulltext
  - Q: What blood tests should be ordered in the initial dementia workup?
  - P: Testing an individual's blood and other fluids and checking levels of different chemicals, hormones, and vitamins can help diagnose or rule out potential causes of symptoms. Mini strokes, cancers, and other conditions that may partly underp…

- **eq-0078** (alz-017) — human label **2**
  - SCAF 0.4897 (admitted: True), RAG2 0.2866 (admitted: False)
  - σ 0.4767 · γ 0.5713 · τ 0.4 · 2021-12-17 · pubmed-abstract
  - Q: How should agitation in Alzheimer dementia be managed?
  - P: BACKGROUND: Typical and atypical antipsychotics are widely used to treat agitation and psychosis in dementia. However, whether or not they are beneficial is uncertain. Some trials have yielded negative results and effectiveness may be outwe…

> deterministic: highest SCAF score first for the admitted-but-irrelevant case, highest human label for SCAF rejections, highest RAG2 score for RAG2 admissions, lowest SCAF score for underrated evidence. No example was chosen by hand.

## 13. Claims supported by the current evidence

These are safe to write in the thesis, with the stated scope:

1. **The two policies admit vastly different amounts of evidence.** SCAF 95.83% versus RAG² 0.67% of 600 candidate decisions.
2. **RAG²'s filter leaves most questions with no evidence at all** — 26 of 30 questions receive zero admitted passages.
3. **The two policies select near-disjoint evidence.** At matched k = 1 they share no passage; Jaccard rises only to 0.2156 at k = 8.
4. **Neither score orders evidence the way a human does, in this sample.** SCAF ρ = 0.1237, RAG² ρ = 0.017.
5. **SCAF's corroboration term is inactive** (weight 0.0, `{'not_implemented': 600}`) and its authority term is near-constant (5 distinct values across 600 decisions).
6. **Showing an annotator a lexical suggestion inflated the measured SCAF–human association roughly fivefold** (≈0.63 anchored versus 0.1237 independent). This is a genuine methodological finding and is worth reporting as one.

## 14. Descriptive-only findings

- The 4 RAG²-admitted passages average 1.75 usefulness, above the overall mean of 1.3833. **n = 4.** This is a description of four passages, not evidence that RAG² admits better evidence. All four were forced into the sample by the sampling rule.
- The 2 SCAF-rejected passages average 1.5. **n = 2.** Nothing follows from two passages.
- All 14 passages judged not relevant were SCAF-admitted. Suggestive of a permissive threshold, but the annotated rows are a stratified sample, not a census.
- The score-band table in §5 does not rise monotonically. Suggestive, and consistent with the weak ρ, but the bands are small.

## 15. Claims that are NOT established

Do **not** write any of these:

| Claim | Status |
| --- | --- |
| SCAF improves evidence quality | **Not established.** The association with human judgement is weak and the threshold does not separate quality. |
| SCAF is superior to RAG² | **Not established.** At matched k neither ranks better; admitting more is not admitting better. |
| SCAF improves answer accuracy | **Requires additional experiment.** No answer quality was measured; there are no gold answers. |
| SCAF reduces outdated evidence | **Not established.** No retracted or superseded candidate exists in the set to reduce. |
| RAG² has recency bias | **Not established.** Nothing here measures temporal behaviour, and the corpus has almost no old evidence. |
| SCAF improves citation quality | **Requires additional experiment.** Not measured. |
| SCAF reduces unsupported claims | **Requires additional experiment.** Not measured. |
| SCAF's currency/authority/corroboration components are validated | **Not established.** γ weakly discriminative, τ near-constant, ρ inactive. |

## 16. Limitations

1. **Only 120 of 600 decisions were annotated**, by a stratified rule, not at random.
2. **All 4 RAG² admissions were forced into the sample.** They are a census of RAG²'s admissions, not a sample of them, and the sample is not self-weighting.
3. **The rows are clustered**: 120 passages from 29 questions, up to 8 from one. The effective sample size is nearer the number of questions than the number of rows. No clustering correction is applied and no p-values are reported.
4. **One annotator, no second rater.** There is no inter-annotator agreement figure, so label reliability is unknown.
5. **σ and the human label are not fully independent constructs.** σ is lexical overlap; a human judging relevance also responds to shared terminology. Their association is partly mechanical.
6. **No answer-level outcome was measured** — no accuracy, no citation quality, no hallucination rate.
7. **The corpus is temporally narrow** (§10), which limits both γ and any temporal experiment.
8. **Retraction and supersession are untested** (§9).

## 17. Is another experiment necessary?

**Yes — one is, if the thesis wants to claim SCAF helps.** The current evidence supports claims about *admission behaviour* and a negative result about *score-versus-human ordering*. It cannot support any claim about answer quality, because answer quality was never measured.

| Experiment | Priority | Question it answers |
| --- | --- | --- |
| **Matched-k answer quality**: fix both arms at the same k (e.g. 5 passages), generate answers, score them against a reference | **Essential** | Does SCAF's evidence selection produce better answers when context size is held constant? This is the claim the thesis actually wants. |
| **Second annotator on the same 120 rows** | **Essential** | How reliable are these labels? Without it, every number in §5 rests on one person. |
| **Threshold sweep against human labels** | Useful but optional | Is there *any* SCAF threshold that separates useful from unuseful evidence, or is the score simply not informative? |
| **Seed a retraction/supersession probe set** | Useful but optional | Does SCAF's currency machinery do anything when there is something to catch? |
| **Extend the corpus backwards in time** | Future work | Makes γ and any FRB-PAIRS temporal design possible at all. |
| **Implement and weight corroboration (ρ)** | Future work | ρ is currently dead weight in the formula. |

## 18. Recommended next step

**Run the matched-k answer-quality experiment.** Everything else is secondary. The thesis currently has a well-documented negative result and a well-documented behavioural difference; what it lacks is any measurement of whether the behavioural difference *matters* for the answers a reader would receive. Matched-k removes the context-size confound that made the earlier preliminary comparison uninterpretable.

If time does not allow it, the honest thesis is the one written from §13 and §15: a careful measurement instrument, a clear negative finding, and an explicit account of what remains unproven. That is a legitimate MS thesis.

## 19. Exact source files and identifiers

```
decisions            /home/user/thesis_research/experiments/results/rag2_vs_scaf_alzheimer/comparison_scientific/per_question.jsonl
run_manifest         /home/user/thesis_research/experiments/results/rag2_vs_scaf_alzheimer/comparison_scientific/manifest.json
annotation_sheet     /home/user/thesis_research/experiments/results/rag2_vs_scaf_alzheimer/evidence_quality/annotation_sheet_v2.jsonl
sample_manifest      /home/user/thesis_research/experiments/results/rag2_vs_scaf_alzheimer/evidence_quality/sample_manifest.json
frozen digest        316260f04c1720fbc20c1b584ea9f0a093dedbae9fd46476d8a7f45380e86aad
sheet sha256         19d427cd6a0909056e3e24a1a2aa9a8976f82426cc669eb815b101f474e14ed1
label fingerprint    49e2b927212e4a4c39c5756f8a46f4c56b6b3d45a64586ed4b803b32a9688354
annotation pass      corrected-human-only-v1
analysis version     scientific-results-v1
generated            2026-09-10T17:47:42Z
```

Regenerate with:

```
python experiments/scripts/analyse_results.py
```

The machine-readable form of every number above is `scientific_results_summary.json` in this directory.
