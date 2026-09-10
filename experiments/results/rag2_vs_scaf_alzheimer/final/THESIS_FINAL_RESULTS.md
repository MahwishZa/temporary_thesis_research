# Thesis final results — RAG² vs SCAF admission policy

> **Generated artifact.** Rebuilt by `python experiments/scripts/thesis_final_analysis.py`. Every number is derived from the frozen scientific run and the independent annotation pass; nothing here retrieves, generates, samples or annotates.

Built 2026-09-10T22:18:43Z  ·  source run `experiments/results/rag2_vs_scaf_alzheimer/comparison_scientific/per_question.jsonl`

## The one-paragraph version

The admission-policy comparison harness is sound and its upstream fairness is proved rather than asserted. What it measured is that the two policies sit at wildly different operating points — RAG² admits 4 of 600 candidates, the implemented SCAF admits 575 — and that **neither policy's score is distinguishable from noise as a predictor of human-judged evidence usefulness.** The only signal that survives multiplicity correction is the frozen MedCPT reranker's own rank, which is the term the SCAF implementation had accidentally dropped. The thesis's *primary* contribution — the Filter Recency-Bias Probe — was never built, and the reason it stalled is a correctable misreading of the proposal, not a dead end.

## 1. Reportability gates

A result may be reported only if every condition holds. Non-reportable results are **kept as audit history** and must never appear as findings.

| Result | Reportable | Blocking condition |
| --- | :---: | --- |
| Natural-admission comparison (RAG2 vs SCAF) | **YES** | — |
| Answer-quality comparison from the natural-admission run | no | fairness constraints passed; the evaluation labels support the claim being made; no known invalidating issue remains |
| Evidence-quality association with human labels | **YES** | — |
| Anchored evidence-quality pilot | no | fairness constraints passed; the evaluation labels support the claim being made; no known invalidating issue remains |
| Offline ablations (A4, A5, A7, A11, A12) | **YES** | — |
| Filter Recency-Bias Probe (H1-H4), the proposal's primary claim | no | the experiment was actually executed, not inferred; the production corpus and index were used; the scientific frozen candidate set was used; fairness constraints passed; the intended generator was used; provenance is complete; the evaluation labels support the claim being made; no known invalidating issue remains |
| Matched-k = 5 answer comparison | no | the experiment was actually executed, not inferred; the intended generator was used; provenance is complete; the evaluation labels support the claim being made; no known invalidating issue remains |

**Answer-quality comparison from the natural-admission run** — Two independent blockers. (1) Context confound: SCAF's answers were written from ~688x more context (26,905 vs 39 chars mean), so the arms differ in evidence QUANTITY as well as policy, and the proposal's fairness guarantee 2 caps every arm at five passages. (2) No answer-quality label of any kind exists: the 30 questions carry no gold answer and no human judged any answer.

**Anchored evidence-quality pilot** — A lexical suggestion was shown on 120/120 rows and the label matched it on 120/120. The correlation measured the interface. The independent pass put the same association at 0.12 with an interval that includes zero. Retained as audit history only.

**Filter Recency-Bias Probe (H1-H4), the proposal's primary claim** — Not executed. The probe pipeline now exists -- pair construction, the permutation control, the paired bootstrap and the blocking gate, with 36 offline tests -- but it has never been run on data. H1 and H4 (the proposal's Minimum Viable Implementation, 5.4) need only MedChangeQA plus the already-trained filter. H2 additionally needs a second filter trained on the entailment label, which does not exist.

**Matched-k = 5 answer comparison** — The selection half is executed and passes every invariant. Generation has not run. Even once it does, no answer-quality label exists, so the result can support 'the evidence differs' and 'the answers differ', never 'the answers are better'.

## 2. What each policy admitted

| | SCAF admitted | SCAF rejected |
| --- | ---: | ---: |
| **RAG² admitted** | 4 | 0 |
| **RAG² rejected** | 571 | 25 |

- RAG² admission rate **0.0067**, question-clustered 95% CI [+0.0017, +0.0133]
- SCAF admission rate **0.9583**, question-clustered 95% CI [+0.9350, +0.9783]
- Difference **+0.9517 [+0.9283, +0.9733]**
- Cohen's *h* = 2.567 (large)
- Exact McNemar on 571 discordant pairs: p = 2.59e-172 — **anti-conservative**, reported only beside the clustered interval

> This measures how much evidence each policy admits at its own configured operating point. It does not measure whether the admitted evidence is better, and a larger admission rate is not a better one -- 96% admission is close to no filtering at all.

`rag2_only = 0`: RAG²'s admitted set is a strict subset of SCAF's. There is no passage the baseline kept and the proposed policy discarded, so the two policies are not making different trade-offs here — one is simply far more permissive.

## 3. Does any score track human-judged evidence quality?

120 annotated passages in 29 question clusters, labels {'0': 14, '1': 46, '2': 60}, 10,000 question-clustered bootstrap resamples, Holm–Bonferroni across the five comparisons.

| Signal | Spearman ρ | 95% CI (clustered) | p | Holm threshold | Survives |
| --- | ---: | --- | ---: | ---: | :---: |
| `rho_rerank` | +0.2175 | [+0.0850, +0.3382] | 0.00160 | 0.01000 | **YES** |
| `sigma_support` | +0.1640 | [-0.0239, +0.3238] | 0.08739 | 0.01250 | no |
| `scaf_score` | +0.1237 | [-0.0418, +0.2759] | 0.14658 | 0.01667 | no |
| `gamma_currency` | -0.0519 | [-0.2690, +0.1721] | 0.65693 | 0.02500 | no |
| `rag2_score` | +0.0170 | [-0.1388, +0.1661] | 0.82932 | 0.05000 | no |

**This is the study's central result.** The two admission scores the thesis compares — SCAF's `scaf_score` and RAG²'s filter probability — both have intervals that include zero. So does SCAF's lexical support term and its currency term. The only signal that measurably orders passages the way the annotator did is `rho_rerank`, the rank-normalised MedCPT reranker score: **the term the SCAF implementation had dropped, and the one neither arm consults at admission.**

> Descriptive association only. These labels were produced independently of every machine score (annotation pass 'corrected-human-only-v1', no suggestion generated or shown), which is what makes them usable at all -- but a correlation between a lexical support score and a human topicality judgement is not evidence that the support score measures evidential support, and the sample is 120 non-randomly-selected passages clustered in 29 questions.

## 4. What actually moves the SCAF admission score?

| Term | Weight | Raw range | Raw SD | Weighted SD | Share of Var(A) |
| --- | ---: | --- | ---: | ---: | ---: |
| support | 0.50 | [0.000, 1.000] | 0.2066 | 0.1033 | 83.5% |
| currency | 0.30 | [0.536, 1.000] | 0.1445 | 0.0433 | 14.7% |
| rerank | 0.00 | [0.000, 1.000] | 0.3035 | 0.0000 | 0.0% |
| authority | 0.20 | [0.400, 1.000] | 0.0752 | 0.0150 | 1.8% |

The implemented SCAF is **84% a lexical-overlap filter by variance.** Currency contributes 14.7% and cannot contribute more: over a 2021–2026 corpus with a five-year half-life, γ never falls below 0.536. Authority contributes 1.8%. ρ contributes nothing because its weight is zero.

## 5. Ablations

Counterfactual re-scoring of the frozen run: exact, offline, no model and no generation. Admission only.

| ID | Configuration | Admitted | Questions with no evidence | Decisions changed |
| --- | --- | ---: | ---: | ---: |
| RUN | As run (SCAF, lexical support) | 575/600 (0.958) | 0 | 0 |
| A4 | No filter | 600/600 (1.000) | 0 | 25 |
| A5 | Support only, currency disabled | 147/600 (0.245) | 6 | 428 |
| A7 | Currency only, support excluded | 5/600 (0.008) | 27 | 570 |
| A12a | Authority removed | 478/600 (0.797) | 0 | 97 |
| A5 | Support only, currency disabled (renormalised) | 423/600 (0.705) | 0 | 152 |
| A7 | Currency only, support excluded (renormalised) | 600/600 (1.000) | 0 | 25 |
| A12a | Authority removed (renormalised) | 575/600 (0.958) | 0 | 4 |
| A12b | Authority ordering inverted | 584/600 (0.973) | 0 | 15 |
| RHO | Corrected rho (reranker) at w=0.1 | 570/600 (0.950) | 0 | 15 |
| RHO | Corrected rho (reranker) at w=0.2 | 547/600 (0.912) | 0 | 40 |
| RHO | Corrected rho (reranker) at w=0.3 | 530/600 (0.883) | 0 | 59 |

Reading these: **A12b inverts the entire source-authority ordering and changes 15 of 600 decisions.** The proposal treats that ordering as a tested variable (A12); in this run it is not load-bearing. **A5 and A7 are reported twice** — once with the term's weight zeroed and once renormalised — because zeroing a weight shrinks A(s) and silently makes a fixed threshold stricter, which would masquerade as an effect of removing the term.

Ablations that cannot be run from these artifacts, and why:

- **A1** — perplexity against entailment label -- needs both filter backbones; the trained checkpoint is not in this repository
- **A2** — permutation control -- needs FRB-PAIRS
- **A3** — second backbone replication -- needs a second trained filter
- **A6** — prompt with and without date annotation -- needs generation
- **A8** — contested state removed -- the contested state is not implemented
- **A9** — abstention gate removed -- abstention never fired (0 questions)
- **A10** — teacher against distilled student -- no entailment teacher exists
- **A13** — hard supersession gate restored -- no passage is marked superseded in this corpus (supersession is 'unknown' for all 480 time-sensitive candidates), so the gate has nothing to act on
- **A14** — generator transfer -- needs generation

## 6. Sensitivity (A11)

Neither the threshold nor the half-life was fitted on validation data, so the honest statement is how much the headline depends on them.

| θ | Admitted | | Half-life | Admitted | Decisions changed |
| ---: | ---: | --- | ---: | ---: | ---: |
| 0.30 | 598/600 (0.997) | | 1 y | 443/600 (0.738) | 132 |
| 0.35 | 596/600 (0.993) | | 2 y | 521/600 (0.868) | 54 |
| 0.40 | 589/600 (0.982) | | 3 y | 551/600 (0.918) | 24 |
| 0.45 | 575/600 (0.958) | | 5 y | 575/600 (0.958) | 0 |
| 0.50 | 525/600 (0.875) | | 8 y | 584/600 (0.973) | 9 |
| 0.55 | 466/600 (0.777) | | 10 y | 586/600 (0.977) | 11 |
| 0.60 | 371/600 (0.618) | | 20 y | 589/600 (0.982) | 14 |
| 0.65 | 259/600 (0.432) | | | | |
| 0.70 | 169/600 (0.282) | | | | |

Admission runs from 99.7% to 28.2% across plausible thresholds. **The headline number 575/600 is a property of an unfitted threshold as much as of the policy.**

## 7. The temporal question

Corpus span: `2021` … `2026-08-25` (5.15 years).

| Year | Candidates | RAG² admitted | SCAF admitted |
| --- | ---: | ---: | ---: |
| 2021 | 22 | 0 (0.0000) | 19 (0.8636) |
| 2022 | 84 | 0 (0.0000) | 78 (0.9286) |
| 2023 | 116 | 0 (0.0000) | 105 (0.9052) |
| 2024 | 117 | 0 (0.0000) | 115 (0.9829) |
| 2025 | 141 | 2 (0.0142) | 138 (0.9787) |
| 2026 | 120 | 2 (0.0167) | 120 (1.0000) |

> DESCRIPTIVE ONLY, AND CONFOUNDED. These passages are not matched on claim, tier or length, so this is not the admission asymmetry the thesis defines and must not be reported as a recency-bias result. The corpus also spans only ~5 years with no pre-2021 stratum, so there is almost no age contrast to detect even descriptively.

**The blocker is not the one previously recorded.** The repository concluded that FRB-PAIRS cannot be built because the corpus has no pre-2021 stratum. That is true of the corpus and irrelevant to the design: proposal §5.2 specifies FRB-PAIRS as derived from **MedChangeQA**, an externally authored peer-reviewed dataset, behind an explicit *provenance firewall* — the primary claim is required to rest on external material precisely so it cannot be circular. The corpus was never meant to supply the pairs. This is the single highest-value correction available to the project.

## 8. The thesis's own questions

**1. What was reproduced from RAG2?**

The full pipeline: rationale-based query formulation, balanced retrieval over four corpora, MedCPT dense retrieval and cross-encoder reranking, and the rationale-guided Flan-T5-large filter with the released two-way softmax over [HELPFUL] / [NOT_HELPFUL] at a 0.5 decision boundary. The filter checkpoint was retrained, as the paper's was never distributed. What was NOT reproduced is the paper's headline: no accuracy was measured on MedQA, MedMCQA or MMLU-Med, so the reproduction is structural, not numerical.

*Status: done, structurally verified, numerically unverified*

**2. What exactly was changed by SCAF?**

The admission stage only, at the EvidenceFilter seam. Everything upstream is byte-identical by replay. But SCAF as implemented is not SCAF as proposed: sigma is lexical overlap rather than entailment, the contested state and the Grounded/Flagged/Abstain output policy are absent, and rho -- defined in the proposal as the rank-normalised reranker score -- had been misread as 'corroboration' and hard-coded to zero, removing the term entirely. That misreading is corrected in this pass.

*Status: partially implemented; the divergences are now itemised*

**3. Was the RAG2/SCAF comparison fair?**

Upstream, yes and provably: same 30 questions, same 600 candidates in the same order with identical per-question digests, same generator, same prompt fingerprint, same decoding, nine of nine fairness checks passed. Downstream, no: the proposal's fairness guarantee 2 requires every arm to receive at most five passages, and SCAF's arm received a mean of 19.2 against RAG2's 0.13. The admission comparison is therefore fair as a measurement of admission and confounded as a comparison of answers.

*Status: fair for admission, confounded for answers*

**4. What evidence did each policy admit?**

RAG2 admitted 4 of 600 candidates (0.0067); SCAF admitted 575 (0.9583). The difference is +0.9517 with a question-clustered 95% CI of [+0.9283, +0.9733]. RAG2's admitted set is a strict subset of SCAF's -- there is no passage RAG2 admitted that SCAF rejected. RAG2 left 26 of 30 questions with no evidence at all, which is closed-book generation rather than abstention.

*Status: measured and reportable*

**5. Did SCAF demonstrably improve evidence quality?**

No. Against independent human usefulness labels the SCAF admission score gives rho = +0.1237, 95% CI [-0.0418, +0.2759], Holm does not survive, and its support term gives rho = +0.1640, 95% CI [-0.0239, +0.3238], Holm does not survive -- both intervals include zero. RAG2's filter score gives rho = +0.0170, 95% CI [-0.1388, +0.1661], Holm does not survive, also indistinguishable from none. The one signal whose association survives Holm-Bonferroni is the frozen reranker rank: rho = +0.2175, 95% CI [+0.0850, +0.3382], Holm survives. That is the term SCAF's implementation had dropped, and neither arm uses it at admission. Note also that all 14 passages the annotator judged NOT relevant were admitted by SCAF.

*Status: tested; hypothesis NOT supported*

**6. Did SCAF demonstrably improve answer quality?**

Unanswerable with the current artifacts, and not because the experiment failed. No gold answers, no reference answers, no human answer judgements and no claim-level labels exist for these 30 questions. 60 answers were generated in the natural-admission run, but under a 688x context imbalance. The matched-k = 5 design removes that confound and is fully prepared, but even it can only establish that the answers DIFFER, never that they are better, until an answer-level evaluation exists.

*Status: not measured; no instrument exists*

**7. Was recency bias demonstrated, refuted, or left unresolved?**

Left unresolved, and it is the proposal's primary claim. The Filter Recency-Bias Probe was never built. The corpus cannot support it -- 2021-2026 only, so gamma never falls below 0.536 and there is no older stratum -- but the proposal never intended it to: FRB-PAIRS is specified as derived from MedChangeQA, an external peer-reviewed dataset, behind an explicit provenance firewall. Descriptively, RAG2's four admissions all fall in 2025-2026, which points AWAY from the hypothesised direction, but four admissions is not evidence of anything. The probe is now implemented and tested offline; H1 and H4 need MedChangeQA and the filter that is already trained, and nothing else.

*Status: not executed; pipeline implemented, one dataset away*

**8. What did the human evidence evaluation show?**

120 passages, one annotator, labels 0/1/2 distributed 14/46/60, produced with no machine suggestion generated or displayed. It showed that the machine scores order passages roughly the way the reranker does and not the way either admission policy does. It also showed that admission volume and evidence quality are different things: SCAF admitted every passage the annotator rejected. Two design limits carried from the audit: 29 of 30 questions are represented with up to 8 passages from one question, and all four RAG2 admissions were forced into the sample, so it is not a simple random sample of the 600.

*Status: done and reportable, with stated limitations*

**9. What did the answer evaluation show?**

Nothing: it has not been run. See question 6.

*Status: not executed*

**10. Which hypotheses were supported?**

None of the proposal's numbered hypotheses (H1-H7) was tested, because each needs an instrument that does not exist here: FRB-PAIRS for H1-H4, claim-level answer labels for H5-H6, gold answers for H7. What IS established, and was not a hypothesis: the two policies admit at radically different rates; SCAF's score is dominated by its lexical support term (83.5% of the score's variance); and the reranker rank is the only signal that measurably tracks human usefulness judgements.

*Status: no hypothesis supported; three findings established*

**11. Which hypotheses were not supported?**

The implicit hypothesis behind SCAF -- that an entailment-and-currency admission score selects better evidence than a confidence-derived one -- is NOT supported by this data. Its association with human labels does not clear zero. This is a genuine negative result about the implemented policy. It is not a test of the proposed policy, because the entailment support term and the contested state were never built.

*Status: one negative result, correctly scoped*

**12. What remains a limitation?**

(a) sigma is lexical, not entailment -- the single largest gap between proposal and implementation; (b) the corpus spans five years, so the currency term has almost no dynamic range and the temporal question cannot be asked of it; (c) no answer-quality instrument of any kind; (d) one annotator, 120 non-randomly selected passages clustered in 29 questions; (e) the admission threshold 0.45 and half-life 5 y were never fitted on validation data, and the sweeps show admission moving from 99.7% to 28.2% across plausible thresholds; (f) the contested state, the supersession table and the abstention gate are unimplemented or never fired; (g) 30 development questions, one backbone, one decoding pass, no variance estimate.

*Status: itemised*

**13. What can the student safely claim in the thesis?**

Claimable, with the artifacts to back each one: (1) a faithful structural reproduction of the RAG2 pipeline over an Alzheimer-scoped corpus with a retrained filter; (2) that filter admits almost nothing on this corpus -- 4 of 600 -- which is a measured over-rejection finding in the proposal's own error taxonomy; (3) a working, fully provenanced admission-policy comparison harness with byte-identical upstream replay; (4) the implemented SCAF admits 575 of 600, and its score is ~84% lexical support by variance; (5) neither admission score's association with independent human usefulness labels is distinguishable from zero, while the frozen reranker's rank is; (6) the corpus as approved cannot support the temporal question, quantified exactly. NOT claimable: that SCAF is better than RAG2, that SCAF improves evidence or answer quality, that recency bias exists or does not, or any statement about answer correctness.

*Status: six defensible claims, four explicitly forbidden*

## 9. Provenance

```
c5ff80fa6d8287ce4061a3ca394a97164b38774bb5d168f93e92158cb421736f  experiments/results/rag2_vs_scaf_alzheimer/comparison_scientific/manifest.json
4908e339739dcfbe2e741f843235ea8b13847ef0cf48eeb7933809e0a834f74d  experiments/results/rag2_vs_scaf_alzheimer/comparison_scientific/per_question.jsonl
19d427cd6a0909056e3e24a1a2aa9a8976f82426cc669eb815b101f474e14ed1  experiments/results/rag2_vs_scaf_alzheimer/evidence_quality/annotation_sheet_v2.jsonl
```

| | |
| --- | --- |
| annotation pass | `corrected-human-only-v1` |
| annotation sheet sha256 | `19d427cd6a0909056e3e24a1a2aa9a8976f82426cc669eb815b101f474e14ed1` |
| generator | `thesis-llama3-8b-q4` |
| production MedCPT index digest | `2ab50a1212681abd28f4f250d49f076c36f57d1b3a77ce5aaac093adde770937` |
| prompt fingerprint | `fc6db2781ddc` |
| run manifest records | `316260f04c1720fbc20c1b584ea9f0a093dedbae9fd46476d8a7f45380e86aad` |
| scientific frozen candidate digest | `316260f04c1720fbc20c1b584ea9f0a093dedbae9fd46476d8a7f45380e86aad` |
| source run git commit | `dc52745d639ba45d5d7727416e604a96cbc9f2a9` |

Reproduction check on the frozen run (the precondition for every re-scoring above):

- PASS — A(s) rebuilt from the recorded sub-scores for all 600 (max abs error 8.00e-07)
- PASS — every recorded admit/reject is reproduced (0 mismatches)
- PASS — gamma regenerated from the publication date matches the run (max abs error 4.99e-07 at as_of=2026.0, H=5.0)
- PASS — every candidate carries a reranker rank (20 distinct ranks)

## 10. What remains, and where it must run

| Work | Blocker | Where it runs |
| --- | --- | --- |
| Run the Filter Recency-Bias Probe for H1 + H4 — the proposal's primary contribution C1 and its Minimum Viable Implementation (§5.4). Pipeline implemented and tested; needs data. | MedChangeQA is not in the repository and this container cannot reach huggingface.co (the proxy returns 403). Everything else it needs — pair construction, the permutation control, the estimator, and the already-trained filter — is in place. | Student's Windows machine: download MedChangeQA, then `python experiments/scripts/run_frb_probe.py --dataset <path> --checkpoint architecture\rag2\runs\filter-alz --equivalence-band 0.05`. Validate pair construction anywhere first with --dry-run. |
| Train the second filter on the entailment label (ablation A1 and H2, the primary causal claim) | needs a GPU and the entailment teacher; only the perplexity-labelled filter exists. H1 and H4 do not depend on this. | Student's Windows machine (RTX 2050) |
| Matched-k = 5 answer generation (60 answers) | needs the scientific frozen candidate TEXT and the local Ollama generator; this container has neither | Student's Windows machine |
| Blind pairwise answer evaluation | requires the matched-k answers to exist first | Anywhere, once the answers exist |
| Replace σ with an entailment model (the largest proposal-to-implementation gap) | needs a trained NLI model; scoped as future work by the implementation's own docstring | Student's Windows machine |

### Not tested, and why

- **H1/H2/H3 (admission asymmetry)** — needs FRB-PAIRS; not constructed
- **H4 (permutation control)** — needs FRB-PAIRS; not constructed
- **H5 (unsupported-claim rate)** — needs claim-level answer labels; none exist
- **H6 (mediation by retrieval recall)** — needs relevance labels per arm
- **H7 (non-inferiority on time-invariant QA)** — needs gold answers; none exist

## 11. Claim discipline

Statements this package does **not** support, listed so they cannot be read into it:

- Do NOT claim that SCAF is superior to RAG² because it admits more passages — admission volume is not evidence quality, and 96% admission is close to no filtering at all.
- Do NOT claim that SCAF improves evidence quality — its association with independent human labels has a 95% interval that includes zero.
- Do NOT claim that SCAF improves answer accuracy — no answer-level correctness label exists for any of the 30 questions.
- Do NOT claim that RAG² exhibits recency bias — the probe that would test this was never built, and the four admissions it made are all recent, if anything pointing the other way.
- Do NOT claim that the human labels validate the SCAF scoring formula — they were collected independently of it, which is what makes them usable, and they do not corroborate it.
- Do NOT claim that a passage is factually correct because the annotator marked it relevant — the labels are passage-level usefulness judgements, not correctness judgements.
- Do NOT claim that the matched-k results prove superiority — different evidence producing different answers is not better answers.
- Do NOT claim that this reproduction matches the RAG² paper's reported accuracy — no accuracy was measured, and the reproduction is structural.
