# The thesis, end to end

*Does the Filter Prefer the Past? Measuring and Correcting Recency Bias in
Confidence-Derived Evidence Utility Signals for Retrieval-Augmented Alzheimer's
Clinical Reasoning*

**This is the one document that explains the whole research.** Everything else in
the repository is either code, generated evidence, a formal external source, or
an operational runbook. If you read one file, read this one.

Controlling sources, in this order of authority:
[`docs/MS_Thesis_Proposal.pdf`](docs/MS_Thesis_Proposal.pdf) and Sohn et al.,
*Rationale-Guided Retrieval Augmented Generation for Medical Question Answering*,
NAACL 2025, pp. 12739–12753. Where this document and either source disagree, the
source wins and the disagreement is a bug in this document.

**Status line, stated once and honestly:** the research infrastructure is built
and tested; the two experiments that carry the thesis's primary claims have
**not been executed**. Nothing below describes code existence as evidence. See
§11 for the exact distinction and §12 for what remains.

---

## Contents

1. [Motivation and the research problem](#1-motivation-and-the-research-problem)
2. [Research questions and hypotheses](#2-research-questions-and-hypotheses)
3. [RAG² — the baseline, and why it is the baseline](#3-rag--the-baseline-and-why-it-is-the-baseline)
4. [The suspected defect: admission, not retrieval](#4-the-suspected-defect-admission-not-retrieval)
5. [C1 — the Filter Recency-Bias Probe (the primary contribution)](#5-c1--the-filter-recency-bias-probe-the-primary-contribution)
6. [C2 — SCAF (the corrective policy)](#6-c2--scaf-the-corrective-policy)
7. [Corpus, datasets, retrieval and reranking](#7-corpus-datasets-retrieval-and-reranking)
8. [Experimental design: arms, fairness, controls, ablations](#8-experimental-design-arms-fairness-controls-ablations)
9. [Evaluation: evidence quality, answer quality, statistics](#9-evaluation-evidence-quality-answer-quality-statistics)
10. [Results actually obtained](#10-results-actually-obtained)
11. [Status ledger: implemented / tested / executed / reportable](#11-status-ledger)
12. [What remains, and exactly where it runs](#12-what-remains-and-exactly-where-it-runs)
13. [Limitations](#13-limitations)
14. [Reproducibility](#14-reproducibility)
15. [Repository map](#15-repository-map)

---

## 1. Motivation and the research problem

Large language models fail in clinical settings in two persistent ways:
they hallucinate, and they rely on knowledge frozen at pre-training time.
Retrieval-augmented generation addresses both by conditioning generation on
retrieved documents. But retrieval introduces failure modes of its own, and the
largest expert evaluation of medical RAG to date reports that standard retrieval
augmentation *degrades* medical performance under several conditions: only 22% of
top-16 retrieved passages were judged relevant, and 31% of queries returned no
relevant passage at all.

Between retrieval and generation sits a stage that is rarely examined: **evidence
admission** — the decision about which retrieved passages actually reach the
answer model. RAG² makes that decision with a small classifier trained on labels
derived from the perplexity differential ΔPPL = PPL(x) − PPL(x, d), admitting a
passage when conditioning on it *reduces the model's perplexity*.

**The problem this thesis addresses.** That criterion measures **belief shift**,
not **evidential support**, and belief shift is not neutral with respect to
passage content. Two independently established findings bear on this: model
adoption of retrieved evidence is confidence-mediated, and models exhibit
confirmation bias toward evidence partially consistent with parametric memory. If
a filter's training signal rewards confidence gain, and confidence gain is
elevated by passages consonant with the model's pre-training-era beliefs, then
**the filter may be directionally biased toward older evidence.** This has never
been measured.

**Why Alzheimer's disease.** Its evidence base moved substantially between 2023
and 2025 — revised diagnostic criteria, plasma biomarkers elevated to diagnostic
sufficiency, appropriate-use recommendations for lecanemab and donanemab with
mandatory APOE genotyping and ARIA monitoring. Staleness here has direct clinical
consequence. And the knowledge does not move monotonically: an April 2026
Cochrane review of 17 trials concluded amyloid-beta-targeting antibodies probably
produce little to no clinically meaningful cognitive difference, and was
contested within days — furnishing a live test case for a *contested* evidence
state.

## 2. Research questions and hypotheses

| ID | Research question | Hypothesis |
| --- | --- | --- |
| RQ1 | Do confidence-derived utility signals admit passages asymmetrically with respect to age, holding content, tier and length constant? | **H1:** Δ(perplexity) > 0 · **H2:** Δ(perplexity) > Δ(support) |
| RQ2 | Does any asymmetry replicate across filter backbones? | **H3:** asymmetry replicates with the same sign |
| RQ3 | Does an entailment-derived label reduce unsupported and superseded claims? | **H5:** unsupported-claim rate falls |
| RQ4 | Is any reduction attributable to admission policy rather than retrieval quality? | **H6:** a direct arm effect persists after conditioning on recall |
| RQ5 | Does an explicit contested state improve clinical appropriateness on unresolved questions? | blinded expert rubric |
| RQ6 | Is accuracy preserved on time-invariant medical QA? | **H7:** non-inferiority within a pre-registered margin δ |

**H4 is a validity gate, not a finding:** on the permutation control, Δ = 0.
Should the control fail, H1–H3 are uninterpretable and must not be reported.

**Contributions.** **C1 (primary)** is a measurement instrument, the Filter
Recency-Bias Probe. **C2 (secondary)** is a corrective admission policy, SCAF.
The proposal sequences the probe *before* SCAF deliberately: "If the primary
claim fails, that must be discovered in month five, when a pivot is inexpensive,
rather than in month nine."

> **The project built C2 first and has not yet run C1.** That is the single most
> important fact about the current state of this research. §12 is the correction.

## 3. RAG² — the baseline, and why it is the baseline

RAG² is the *foundational system*: the thesis measures a property of its
admission stage and proposes a replacement for it. It is reproduced faithfully
and is never "improved".

Three components, all reproduced in [`architecture/rag2/`](architecture/rag2/):

1. **Rationale-based query formulation.** The model-generated chain-of-thought
   rationale *replaces* the question as the retrieval query; the original query is
   deliberately excluded, because concatenating both exceeds the retriever's
   maximum length.
2. **Balanced retrieval.** Equal passage quotas from four corpora (PubMed, PMC
   full text, clinical practice guidelines, textbooks), then MedCPT cross-encoder
   reranking of the initial query against each snippet.
3. **Rationale-guided filtering.** A Flan-T5-large classifier (770M) trained on
   perplexity-differential labels, with τ fixed at the top 25% of differentials.
   At inference it emits `[HELPFUL]` / `[NOT_HELPFUL]`, decided by a two-way
   softmax over just those two label logits — reproduced exactly as released.

**Two properties of the base paper are load-bearing for this thesis.** It
motivates itself with hallucination and outdated knowledge but **measures
neither**; and it contains **no temporal representation of any kind** — no
publication date, guideline version or supersession relation anywhere in its
corpus, index, retriever, reranker or filter. That absence is the opening.

**The filter checkpoint was never distributed**, so it was retrained here. Per
the proposal, both filters are trained on general medical QA and applied
zero-shot to Alzheimer's items, mirroring the base paper's own cross-dataset
transfer claim.

Component-by-component verification:
[`docs/rag2_reproduction_audit.md`](docs/rag2_reproduction_audit.md). What the
paper fixes, leaves ambiguous, or omits:
[`docs/rag2_reproduction.md`](docs/rag2_reproduction.md).

**No accuracy was measured.** The reproduction is structural, not numerical. No
MedQA / MedMCQA / MMLU-Med run has been executed, so this repository cannot claim
to match the paper's reported numbers, and does not.

## 4. The suspected defect: admission, not retrieval

The literature establishes that confidence-based utility signals exist, are
widely used, and that generators exhibit confirmation bias. It does **not**
establish whether the *selection stage itself* is temporally asymmetric, nor does
it provide any mechanism for representing evidential disagreement. Those two gaps
define this thesis.

The design deliberately **freezes retrieval** to buy causal attribution, at the
cost of a lower achievable ceiling. The proposal states this tension explicitly
rather than leaving it for an examiner to raise.

## 5. C1 — the Filter Recency-Bias Probe (the primary contribution)

**What it measures.** For a filter *f* over matched temporal-counterfactual
passage pairs:

```
Δ = E_pairs[ P(admit | older) − P(admit | newer) ]
```

Δ > 0 means the filter preferentially admits **older** evidence (H1). A pair is
two passages addressing the same claim on opposite sides of a known change point,
matched on source tier and on token length within a tolerance band.

**The two filters must be identical except for the label function.** Same
backbone, same parameter count, same training data, same hyperparameters — that
identity is what licenses attributing the measured asymmetry to the *signal*
rather than to capacity, architecture or retrieval. The baseline label reproduces
the base paper; the proposed label is entailment against the verbalised gold
answer at training time.

### MedChangeQA and the provenance firewall

FRB-PAIRS is derived from **MedChangeQA** (Vladika et al., *Facts Fade Fast*,
Findings of EMNLP 2025 — 512 changed-verdict pairs). This is not a convenience.
Proposal §5.2 makes it mandatory:

> "[DES] Provenance separation is mandatory and is the single most important
> correction in this revision. An earlier formulation authored evaluation items,
> the supersession table and the currency corpus from the same twenty-odd
> documents, rendering a positive result unfalsifiable by construction."

There is a **provenance firewall**: the primary claim rests on the externally
authored lane; thesis-curated Alzheimer material supports replication and case
study only. **The lanes must not cross.**
[`experiments/recency_bias/frb_pairs.py`](experiments/recency_bias/frb_pairs.py)
enforces this in code — `load_dataset` refuses a thesis-curated `--provenance`
rather than trusting the operator to remember.

> **A correction to an earlier conclusion in this repository.** The README, the
> preliminary write-up and the Windows runbook all recorded that FRB-PAIRS
> *cannot be constructed* because the production corpus holds only 7 pre-2020
> documents, and listed "source it externally" as one option among three awaiting
> a supervisor decision. That is true of the corpus and beside the point: the
> corpus was never the intended source, and the proposal already decides. Those
> documents have been corrected.

**Validity control V1 (H4).** The pair set is duplicated with the older/newer
roles randomly reassigned; on the permuted set Δ must collapse into a
**pre-specified equivalence band**. This is an equivalence test, not a failure to
reject — a wide interval that merely contains zero does not pass. V1 is
**blocking**: if it fails, the probe is responding to prose-era features rather
than to recency, and the primary result is withheld rather than explained.

**Status: implemented and unit-tested, never executed on data.** See §12.1.

## 6. C2 — SCAF (the corrective policy)

SCAF replaces confidence-derived utility with an explicit, inspectable admission
score at the one seam the baseline already exposes
(`rag2.filtering.base.EvidenceFilter`), so both arms run on an identical
candidate set with nothing else changed.

```
A(s) = w₁·σ(s) + w₂·γ(s,q) + w₃·ρ(s) + w₄·τ(s)
```

### σ — Support

Entailment-derived support, *as specified*. σ(s) is the maximum entailment
probability over the hypothesis set: verbalised answer options for
multiple-choice items, or up to eight atomic claims decomposed from the
rationale.

**As implemented, σ is lexical, not entailment** — question-term coverage
combined with corpus-IDF-weighted overlap. This is the single largest gap between
proposal and implementation. Every decision record carries `support_method` so no
reader can mistake one for the other, and no thesis claim about support may rest
on it.

### γ — Currency (three states)

```
γ(s, q, t_q) =
    0                                if retracted(s) or withdrawn(s)
    1                                if ψ(q) = 0            (claim does not age)
    δ · 2^(−(t_q − date(s))/H)       if superseded(s, t_q)
    2^(−(t_q − date(s))/H)           otherwise
```

Two design decisions carry justification. **Hard rejection is confined to
retraction and formal withdrawal**, which are objective and externally
verifiable; supersession only *down-weights*, because claim-class tags come from
an automatic matcher and a hard gate would irreversibly delete correctly relevant
evidence in proportion to the tagging error. **The decay is conditioned on ψ(q)**
— an unconditional recency prior would penalise correct older sources on
time-invariant questions (APOE biology, neuropathological staging), trading a
currency gain for an accuracy loss.

### ρ — Rank-normalised reranker score

The proposal's notation table defines ρ(s) as the **rank-normalised reranker
score**, with §4.5 giving the reason for rank rather than min-max normalisation:
so that A(s) is comparable across queries and one global admission threshold is
well defined.

> **A corrected implementation error.** An earlier version of
> `architecture/scaf/policy.py` read ρ as *corroboration* and hard-coded it to
> `0.0`, silently removing an entire term from the admission score. ρ is now
> computed from the frozen reranker's rank and recorded on every decision. Its
> weight still defaults to `0.0`, so **the correction changed no admission
> decision**: the proposal marks weight selection `[OPEN]`, and choosing a value
> here would both fabricate a parameter and violate fairness guarantee 5 (no
> test-set tuning). `w_ρ` must be fitted on validation data and frozen before any
> test run. Corroboration remains a separate, still-unimplemented concept with
> its own weight key.

### τ — Authority

Source-tier weight, treated as a **tested variable, not an assumed ordering**
(ablation A12). Excluding it would let a recent preprint outrank a current
guideline; fixing it would encode a position the April 2026 Cochrane episode
shows to be contestable. The tier→weight map therefore lives in configuration and
appears in every run manifest; changing it is a config edit, never a code edit.

### Admission, gates and output policy

Algorithm SCAF-ADMIT: build hypotheses → score σ and γ → drop only
retracted/withdrawn → score A(s) → threshold at θ_admit → **test contested before
supersession** (reversing the order would classify a live controversy as
resolved) → take top-k → emit one of four states: **Grounded**, **Flagged**,
**Contested**, **Abstain**.

**Implemented vs proposed**, stated plainly:

| Algorithm step | Implemented? |
| --- | --- |
| σ from entailment | **No** — lexical stand-in |
| γ three-state, retraction hard-gate | Yes |
| A(s) with all four terms | **Partial** — ρ now computed, weight 0 by design |
| θ_admit threshold | Yes |
| Contested state, one representative per position | **No** |
| top-k final cap | **Partial** — `max_admit` defaults to 0 (no cap) |
| Abstain on empty or on max σ < θ_support | **Partial** — abstains only when empty |
| Flagged on max γ < θ_stale | **No** |

## 7. Corpus, datasets, retrieval and reranking

The base paper's 564.2 GB corpus is not tractable on the assumed hardware, so the
corpus is **domain-scoped** — and the identical scoped corpus is used by every
arm, making scoping a controlled constant rather than a confound.

| | |
| --- | --- |
| PubMed records acquired | 43,409 |
| Documents in the chunk layer | 42,964 |
| Chunks | 781,563 (773,183 unique) |
| Source categories | `pubmed-abstract` 22,583 · `pmc-fulltext` 758,867 · `currency-pack` 113 |
| Publication window | 2021-08-30 → 2026-08-30 (approved strategy) |
| Retriever / reranker | `ncbi/MedCPT-Article-Encoder`, MedCPT cross-encoder |
| Production index | 773,183 × 768, digest `2ab50a12…` |
| Evaluation questions | 30 fixed Alzheimer development questions, **no gold answers** |

Chunking is a 256-token sliding window with 32-token overlap, sized against the
article encoder's 512-token limit with headroom for a prepended title and section
header. Every passage carries chunk and document identifiers, source tier,
publication date, retraction and withdrawal flags, claim classes and character
span. Document identity is PMID-primary; DOI is a consistency check, never an
identity key.

**Immutable by instruction:** `data/corpora/pmc/pmc_oa_inventory.csv` (the
acquisition record), `data/corpora/pubmed/search_queries.txt` (the approved
search strategy), and the currency-pack XML snapshot pinned by MD5.

**The baseline never reads the temporal metadata.** It is carried through
retrieval and ignored, because the thesis measures what the *original* filter
does with evidence of different ages — if baseline code learned about dates, the
thing being measured would cease to exist.
`architecture/rag2/tests/test_metadata_isolation.py` fails the build if any
baseline module gains an executable reference to a date, recency or currency
field, and a companion test proves the scanner actually fires.

## 8. Experimental design: arms, fairness, controls, ablations

**Arms.** B0 closed-book · B1 retrieval and rerank, no filter · **B2 the
foundational RAG² baseline** · B3 current state of the art (support-supervised
filtering with query reformulation) · B4 agentic reference point, for context
only · **P the proposed SCAF policy**.

**Fairness guarantees** (proposal §6.3):

1. **Upstream identity.** The reranked candidate set is computed once, serialised
   and replayed byte-identically. Arms are distinct functions over one cached
   candidate list. This is validity control **V3**.
2. **Matched context budget. All arms receive at most five passages.**
3. **Matched filter capacity.** Baseline and proposed filters share backbone,
   size, training data volume and hyperparameters; only the label function
   differs.
4. **Prompt parity.** One template without date annotations for the main
   comparison, so gains are attributable to which passages were admitted rather
   than to date cues.
5. **No test-set tuning.** Thresholds and weights are selected on validation data
   and frozen before any test run.
6. **Contamination control.** Filters are trained only on general medical
   training splits; evaluation sets are finalised after training data is frozen.

The frozen candidate set for the scientific run is digest
`316260f0…`; a stale lexical-development set (`151b7d54…`,
`retrieval_is_medcpt: False`) exists and every runner refuses it explicitly.

**Validity controls.** V1 permutation control (blocking) · V2 backbone
replication · V3 cached candidate replay · V4 decoupled verifier from a different
model family, so a claim the filter mis-scores is not mis-verified identically ·
V5 negative control set of claims stable across the window.

**Ablations.** A1–A3 are load-bearing and scheduled first: A1 perplexity against
entailment label (the primary causal claim), A2 permutation control, A3 second
backbone. A4 no filter · A5 currency disabled · A6 date-annotated prompting · A7
support excluded · A8 contested state removed · A9 abstention removed · A10
teacher vs distilled student · A11 sensitivity sweep · A12 authority ordering
varied and removed · A13 hard supersession gate restored · A14 generator transfer.

## 9. Evaluation: evidence quality, answer quality, statistics

### Evidence-quality annotation

120 passages drawn from the scientific run, labelled 0/1/2 for usefulness by one
annotator, distributed 14/46/60. Annotation pass `corrected-human-only-v1`, sheet
SHA-256 `19d427cd…`. **No machine suggestion was generated or displayed.**

That last point is not a detail. An earlier **anchored pilot** displayed a
lexical suggestion on 120/120 rows and the label matched it on 120/120; the
correlation it produced (ρ ≈ 0.63) measured the interface, not the scorer. The
independent pass put the same association at 0.12 with an interval including
zero. The pilot is retained under
`experiments/results/rag2_vs_scaf_alzheimer/evidence_quality/pilot_anchored/` as
audit history and is **not** scientific human validation.

Two limitations carried from the audit (31 PASS / 2 WARN / 0 FAIL): 29 of 30
questions are represented, with up to 8 passages from a single question; and all
four RAG² admissions were *forced* into the sample, so it is a census of those
four rather than a sample of them, and the whole is not a simple random sample of
the 600.

### Answer-level evaluation

The 30 questions carry **no gold answer, no reference answer and no expert
label**. Conventional accuracy is therefore unavailable and is not simulated. The
defensible instrument is **blind pairwise preference**, clearly labelled as
preference and not correctness, with arm identity, admission policy, model
identity, scores, thresholds and any machine suggestion hidden from the
evaluator, A/B order randomised, and an explicit tie option.

**No such harness exists yet**, and none has been run. See §12.

### Statistical methodology

Claims are clustered within questions, so a naive per-claim test is
anti-conservative. Every interval in this project therefore **resamples whole
questions**, never passages: 20 candidates share each question, so the effective
sample size is far closer to 30 than to 600.

- H1/H2/H3: paired bootstrap, 10,000 resamples, unit = matched pair.
- H4: equivalence test against a pre-specified band.
- Currency metrics: McNemar's exact test on paired outcomes; Cohen's *h*.
- Multiplicity: Holm–Bonferroni across the primary family at family-wise 0.05.
- **Mandatory dual reporting:** every rate twice — conditional on answering, and
  with abstentions counted as failures.

Seeds are fixed (`20260910`), so intervals reproduce exactly.

### Reportability rules

A result may be reported only if **all** hold: it was actually executed rather
than inferred; the production corpus and index were used; the scientific frozen
candidate set was used; fairness constraints passed; the intended generator was
used; provenance is complete; the evaluation labels support the specific claim
being made; and no known invalidating issue remains.

A non-reportable result is **kept, not deleted** — it is audit history — and must
never appear in the thesis as a finding. The gates are machine-evaluated in
`experiments/analysis/thesis_report.py`.

## 10. Results actually obtained

Everything in this section is derived from the completed admission comparison and
the independent annotation pass. Machine-readable form:
[`experiments/results/rag2_vs_scaf_alzheimer/analysis/thesis_final_results.json`](experiments/results/rag2_vs_scaf_alzheimer/analysis/thesis_final_results.json).
Regenerate with `python experiments/scripts/thesis_final_analysis.py`.

**Precondition, verified before any of it.** The frozen run reproduces its own
arithmetic: A(s) rebuilds from the recorded sub-scores to 8.0 × 10⁻⁷ across all
600 candidates, all 600 admit/reject decisions reproduce, and γ regenerates from
the publication date to 5.0 × 10⁻⁷. Without this, no re-scoring below would mean
anything; the runner refuses if it fails.

### 10.1 Admission behaviour — reportable

| | SCAF admitted | SCAF rejected |
| --- | ---: | ---: |
| **RAG² admitted** | 4 | **0** |
| **RAG² rejected** | 571 | 25 |

RAG² 4/600 (0.0067) · SCAF 575/600 (0.9583) · difference **+0.9517**,
question-clustered 95% CI [+0.9283, +0.9733] · Cohen's *h* = 2.567 (large).
Exact McNemar p ≈ 2.6 × 10⁻¹⁷² is reported only beside the clustered interval and
labelled anti-conservative.

**RAG²'s admitted set is a strict subset of SCAF's.** The two policies are not
trading off differently; one is simply far more permissive. RAG² answered 26 of
30 questions with zero admitted passages — closed-book generation, not
abstention, and a *filter over-rejection* finding in the proposal's own error
taxonomy.

### 10.2 Evidence quality — reportable, and the central result

Holm–Bonferroni across five comparisons, question-clustered bootstrap:

| Signal | ρ | 95% CI | Survives Holm |
| --- | ---: | --- | :---: |
| `rho_rerank` | **+0.2175** | [+0.0850, +0.3382] | **YES** |
| `sigma_support` | +0.1640 | [−0.0239, +0.3238] | no |
| `scaf_score` | +0.1237 | [−0.0418, +0.2759] | no |
| `gamma_currency` | −0.0519 | [−0.2690, +0.1721] | no |
| `rag2_score` | +0.0170 | [−0.1388, +0.1661] | no |

**Neither admission policy's score is distinguishable from noise as a predictor
of human-judged evidence usefulness.** The only signal surviving correction is
the frozen MedCPT reranker's rank — the term SCAF's implementation had dropped,
and the one neither arm consults at admission. All 14 passages the annotator
judged *not relevant* were admitted by SCAF.

### 10.3 What moves the SCAF score — reportable

| Term | Weight | Raw range | Share of Var(A) |
| --- | ---: | --- | ---: |
| support | 0.50 | [0.000, 1.000] | **83.5%** |
| currency | 0.30 | [0.536, 1.000] | 14.7% |
| authority | 0.20 | [0.400, 1.000] | 1.8% |
| rerank | 0.00 | [0.000, 1.000] | 0.0% |

The implemented SCAF is, by variance, **84% a lexical-overlap filter**. Currency
cannot contribute more than it does: over a 2021–2026 corpus at a five-year
half-life, γ never falls below 0.536.

### 10.4 Ablations — reportable (admission only)

Exact counterfactual re-scoring of the frozen run; no model, no generation.

| ID | Configuration | Admitted | Decisions changed |
| --- | --- | ---: | ---: |
| — | as run | 575/600 | — |
| A4 | no filter | 600/600 | 25 |
| A5 | currency disabled (renormalised) | 423/600 | 152 |
| A7 | support excluded (renormalised) | 600/600 | 25 |
| A12a | authority removed (renormalised) | 575/600 | 4 |
| A12b | **authority ordering fully inverted** | 584/600 | **15** |

A5 and A7 are reported both zeroed and renormalised, because zeroing a weight
shrinks A(s) and silently makes a fixed threshold stricter — which would
masquerade as the effect of removing the term. **A12 is not load-bearing in this
run:** inverting the entire source-authority ordering moves 15 of 600 decisions.

**A11 sensitivity.** Admission runs 99.7% (θ = 0.30) → 28.2% (θ = 0.70). Neither
θ = 0.45 nor H = 5 y was fitted on validation data, so 575/600 is a property of
an unfitted threshold as much as of the policy.

### 10.5 Non-reportable, retained as audit history

| Result | Why it cannot be reported |
| --- | --- |
| Answer comparison from the natural-admission run | SCAF's answers used ~688× more context (26,905 vs 39 mean chars), violating fairness guarantee 2; **and** no answer-quality label exists |
| Anchored evidence-quality pilot | suggestion shown on 120/120 rows, label matched on 120/120 — it measured the interface |
| Matched-k = 5 | selection executed and passes every invariant; **generation has not run** |
| Filter Recency-Bias Probe | **not executed** |

## 11. Status ledger

The distinction this table draws is the point of it. *Implemented* means code
exists. *Tested* means unit tests pass. *Executed* means it ran on real data and
produced output. **Only executed work can be evidence.**

| Component | Implemented | Tested | Executed | Reportable |
| --- | :---: | :---: | :---: | :---: |
| RAG² pipeline reproduction | yes | yes | yes (structural) | structural only |
| RAG² accuracy vs the paper | yes | yes | **no** | — |
| Frozen candidate replay (V3) | yes | yes | yes | yes |
| SCAF admission policy | partial (§6) | yes | yes | yes, for admission |
| Admission comparison | yes | yes | yes | **yes** |
| Evidence-quality annotation | yes | yes | yes | **yes** |
| Offline ablations A4/A5/A7/A11/A12 | yes | yes | yes | **yes** |
| Clustered statistics | yes | yes | yes | **yes** |
| Matched-k = 5 selection | yes | yes | yes | yes |
| Matched-k = 5 generation | yes | yes | **no** | no |
| FRB probe (C1) | yes | yes | **no** | no |
| Entailment filter (H2 / A1) | **no** | — | no | no |
| Answer-level evaluation | **no** | — | no | no |
| Meerkat-7B replication (H3 / A3) | **no** | — | no | no |
| Contested state, output policy | **no** | — | no | no |

**Not feasible in the current environment** (no network to HuggingFace, no GPU,
no local generator, wrong frozen candidate set on disk): everything in the
"Executed: no" rows. They are blocked on resources, not on design.

## 12. What remains, and exactly where it runs

### 12.1 FRB probe, H1 + H4 — **mandatory**

This is proposal §5.4's *Minimum Viable Implementation*: "a complete and
defensible thesis should all later phases fail". It needs MedChangeQA and the
filter that is **already trained** — nothing else.

Download MedChangeQA (arXiv:2509.04304) to `data/datasets/medchangeqa/`, then
validate pair construction with no model loaded:

```powershell
python experiments\scripts\run_frb_probe.py `
    --dataset data\datasets\medchangeqa\medchangeqa.jsonl --dry-run
```

If the column names differ, map only those that differ — a wrong map fails loudly
rather than yielding zero pairs:

```powershell
python experiments\scripts\run_frb_probe.py `
    --dataset data\datasets\medchangeqa\medchangeqa.jsonl --dry-run `
    --field-map '{\"older_text\": \"abstract_old\", \"newer_date\": \"year_new\"}'
```

Then run it. **Fix `--equivalence-band` before looking at any output** — §7.3
requires the band pre-registered; choosing it afterwards turns an equivalence
test into a formality:

```powershell
python experiments\scripts\run_frb_probe.py `
    --dataset data\datasets\medchangeqa\medchangeqa.jsonl `
    --checkpoint architecture\rag2\runs\filter-alz `
    --equivalence-band 0.05
```

Writes `experiments/results/recency_bias/frb_probe.json`. Exit code 0 **only** if
the permutation control passes; H1 is withheld otherwise.

### 12.2 Matched-k = 5 generation — mandatory for any answer-level claim

The only proposal-compliant configuration: fairness guarantee 2 caps every arm at
five passages, which the natural-admission run violated at 19.2. Needs the real
frozen candidate **text** (`316260f0…`) and the local generator.

```powershell
$env:OPENAI_BASE_URL = "http://localhost:11434/v1"
$env:OPENAI_API_KEY  = "ollama"
ollama list                      # confirm thesis-llama3-8b-q4 is served
python experiments\scripts\run_matched_k.py --dry-run
python experiments\scripts\run_matched_k.py `
    --generator openai --generator-model thesis-llama3-8b-q4
```

`backend: openai` here means a **local** OpenAI-compatible server. The endpoint
guard refuses if `OPENAI_BASE_URL` is unset or points at a commercial endpoint —
never bypass it; an unset variable would silently send thesis questions to
`api.openai.com`.

### 12.3 Remaining, in priority order

| Work | Blocker | Mandatory? |
| --- | --- | --- |
| Entailment-labelled second filter (H2 / A1) | GPU + entailment teacher | mandatory for C1's full claim; H1/H4 do not depend on it |
| Blind pairwise answer evaluation | needs §12.2's answers **and** a harness that does not exist | secondary — §5.4 excludes expert evaluation from the MVP |
| σ replaced by a real entailment model | trained NLI model | mandatory before any thesis claim about support |
| Meerkat-7B replication (H3 / A3) | second trained filter | secondary robustness; must not delay C1 |

## 13. Limitations

1. **σ is lexical, not entailment** — the single largest proposal-to-
   implementation gap. No claim about support may rest on the current scorer.
2. **The corpus has almost no age contrast.** 7 of 43,409 documents predate 2020,
   so γ ∈ [0.536, 1.000]. An age comparison *within this corpus* would measure
   almost nothing. This does **not** block FRB-PAIRS (§5).
3. **No answer-quality instrument of any kind** exists for the 30 questions.
4. **One annotator, 120 non-randomly-selected passages** clustered in 29
   questions, with all four RAG² admissions forced in.
5. **θ = 0.45 and H = 5 y were never fitted on validation data.**
6. **Contested state, supersession table and abstention gate** are unimplemented
   or never fired; supersession is `unknown` for all 480 time-sensitive
   candidates, and abstention fired on 0 of 30 questions.
7. **30 development questions, one backbone, one decoding pass**, no variance
   estimate.
8. **Absolute performance is not comparable to published baselines** because the
   corpus is domain-scoped; only within-study arm contrasts are interpretable.
9. **The contested detector cannot reliably distinguish genuine contradiction
   from scope difference** ("effective in early-stage" vs "not effective in
   moderate" is a qualifier, not a conflict). Acknowledged, not solved.

## 14. Reproducibility

```bash
python experiments/scripts/thesis_final_analysis.py   # rebuild the results package
python run_tests.py                                   # every offline suite
```

Identity of the inputs, recorded in the package and checked by tests:

```
frozen candidate set   316260f04c1720fbc20c1b584ea9f0a093dedbae9fd46476d8a7f45380e86aad
production index       2ab50a1212681abd28f4f250d49f076c36f57d1b3a77ce5aaac093adde770937
annotation sheet       19d427cd6a0909056e3e24a1a2aa9a8976f82426cc669eb815b101f474e14ed1
annotation pass        corrected-human-only-v1
generator              thesis-llama3-8b-q4, prompt fingerprint fc6db2781ddc
source run commit      dc52745d639ba45d5d7727416e604a96cbc9f2a9
```

The analysis **refuses** to run against a non-scientific candidate set, or
against an annotation sheet from the anchored pilot. Bootstraps are seeded, so
intervals reproduce exactly.

**Passing tests are not scientific validation.** They establish that the code
does what it says, not that the science is right.

Building the corpus and baseline from scratch (GPU machine only) is documented
step by step in [`docs/windows_runbook.md`](docs/windows_runbook.md). Running the
annotation interface: [`docs/annotation_runbook.md`](docs/annotation_runbook.md).

## 15. Repository map

```
THESIS.md              ← you are here; the whole research story
README.md              one-page entry point

data/                  corpora and question sets (large assets gitignored)
preprocessing/         acquisition → parsing → QC → chunking → indexing
architecture/          THE SYSTEMS UNDER STUDY
  rag2/                  reproduced RAG² baseline (vendored + adapted)
  scaf/                  the SCAF admission policy
experiments/           HOW THEY ARE TESTED
  scripts/               every runnable entry point — this is what you type
  analysis/              libraries the scripts call: ablations, statistics, reporting
  recency_bias/          the FRB probe (C1)
  configs/               experiment configuration
  results/               GENERATED EVIDENCE — never hand-edited
indexes/                built MedCPT indexes (gitignored)
docs/                   external source, specialist references, runbooks, QC history
run_tests.py            all offline suites, each in its own process
```

**The dependency rule that makes the comparison mean anything:**

```
  scaf  ──imports──▶  rag2      ALLOWED
  rag2  ──imports──▶  scaf      FORBIDDEN
```

Machine-checked, not merely intended.

`experiments/results/rag2_vs_scaf_alzheimer/` holds:
`comparison_scientific/` (the scientific run — this name is recorded in
provenance and must not be renamed) · `evidence_quality/` (annotation + audit,
with the invalid pilot quarantined beneath it) · `analysis/` (all derived
analysis and the final results package) · `comparison_development/` (a superseded
development run and a stale checkpoint-verification artifact, kept as history) ·
`training_dataset/` (the filter training data, with its manifest and provenance).
