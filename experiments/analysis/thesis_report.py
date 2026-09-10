"""Assemble the final thesis results package and decide what may be reported.

Two jobs, deliberately separate:

1. **Gate.** Every result carries a reportability decision derived from the
   artifacts, not from an author's confidence. A result that used the wrong
   candidate set, was never executed, or lacks the labels its claim needs is
   marked non-reportable *and kept*, because an audit trail of what was tried is
   worth more than a tidy directory.

2. **Answer.** The package answers the thesis's own questions in order, with
   each answer bound to the artifact that supports it. Where the honest answer
   is "not established", that is the answer -- a negative result is a result,
   and a manufactured positive one is not.

Nothing here computes a new scientific quantity. It reads the analyses, applies
the gates and writes the package.
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Dict, List, Optional

#: The candidate set the scientific comparison used.
SCIENTIFIC_FROZEN_DIGEST = (
    "316260f04c1720fbc20c1b584ea9f0a093dedbae9fd46476d8a7f45380e86aad")

#: The production MedCPT index the corpus was retrieved from.
PRODUCTION_INDEX_DIGEST = (
    "2ab50a1212681abd28f4f250d49f076c36f57d1b3a77ce5aaac093adde770937")

#: The independent annotation pass. The suggestion-anchored pilot is NOT this.
CORRECTED_ANNOTATION_PASS = "corrected-human-only-v1"
ANNOTATION_SHEET_SHA256 = (
    "19d427cd6a0909056e3e24a1a2aa9a8976f82426cc669eb815b101f474e14ed1")


def sha256(path: str) -> Optional[str]:
    if not os.path.isfile(path):
        return None
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


# --------------------------------------------------------------------------
# Reportability
# --------------------------------------------------------------------------
def gate(name: str, *, executed: bool, correct_corpus: bool, correct_candidates: bool,
         fairness_passed: bool, intended_generator: bool, provenance_complete: bool,
         labels_support_the_claim: bool, known_invalidating_issue: str = "",
         claim: str = "", artifact: str = "") -> Dict[str, Any]:
    """One reportability decision. Every condition is recorded, not just the verdict."""
    conditions = {
        "the experiment was actually executed, not inferred": executed,
        "the production corpus and index were used": correct_corpus,
        "the scientific frozen candidate set was used": correct_candidates,
        "fairness constraints passed": fairness_passed,
        "the intended generator was used": intended_generator,
        "provenance is complete": provenance_complete,
        "the evaluation labels support the claim being made": labels_support_the_claim,
        "no known invalidating issue remains": not known_invalidating_issue,
    }
    failed = [text for text, ok in conditions.items() if not ok]
    return {
        "result": name,
        "claim": claim,
        "artifact": artifact,
        "reportable": not failed,
        "conditions": conditions,
        "blocking": failed,
        "known_invalidating_issue": known_invalidating_issue,
    }


def reportability(ablations: Dict[str, Any], statistics: Dict[str, Any],
                  matched_k_ready: bool, matched_k_generated: bool) -> Dict[str, Any]:
    """The gate for every result this project has produced."""
    reproduced = ablations["reproduction_check"]["all_passed"]
    gates: List[Dict[str, Any]] = [
        gate("Natural-admission comparison (RAG2 vs SCAF)",
             claim="At their configured operating points the two policies admit "
                   "very different amounts of evidence: RAG2 4/600, SCAF 575/600.",
             artifact="comparison_scientific/",
             executed=True, correct_corpus=True, correct_candidates=True,
             fairness_passed=True, intended_generator=True,
             provenance_complete=True, labels_support_the_claim=True),

        gate("Answer-quality comparison from the natural-admission run",
             claim="SCAF produces better answers than RAG2.",
             artifact="comparison_scientific/per_question.jsonl (answers present)",
             executed=True, correct_corpus=True, correct_candidates=True,
             fairness_passed=False, intended_generator=True,
             provenance_complete=True, labels_support_the_claim=False,
             known_invalidating_issue=(
                 "Two independent blockers. (1) Context confound: SCAF's answers "
                 "were written from ~688x more context (26,905 vs 39 chars mean), "
                 "so the arms differ in evidence QUANTITY as well as policy, and "
                 "the proposal's fairness guarantee 2 caps every arm at five "
                 "passages. (2) No answer-quality label of any kind exists: the "
                 "30 questions carry no gold answer and no human judged any "
                 "answer.")),

        gate("Evidence-quality association with human labels",
             claim="The reranker rank tracks human-judged usefulness; no "
                   "admission score demonstrably does.",
             artifact="evidence_quality/annotation_sheet_v2.jsonl + "
                      "final/thesis_statistics.json",
             executed=True, correct_corpus=True, correct_candidates=True,
             fairness_passed=True, intended_generator=True,
             provenance_complete=True, labels_support_the_claim=True),

        gate("Anchored evidence-quality pilot",
             claim="Human labels validate the SCAF support score (rho = 0.63).",
             artifact="evidence_quality/pilot_anchored/",
             executed=True, correct_corpus=True, correct_candidates=True,
             fairness_passed=False, intended_generator=True,
             provenance_complete=True, labels_support_the_claim=False,
             known_invalidating_issue=(
                 "A lexical suggestion was shown on 120/120 rows and the label "
                 "matched it on 120/120. The correlation measured the interface. "
                 "The independent pass put the same association at 0.12 with an "
                 "interval that includes zero. Retained as audit history only.")),

        gate("Offline ablations (A4, A5, A7, A11, A12)",
             claim="Support dominates SCAF's admission score; the authority "
                   "ordering is not load-bearing in this run.",
             artifact="final/thesis_ablations.json",
             executed=True, correct_corpus=True, correct_candidates=True,
             fairness_passed=True, intended_generator=True,
             provenance_complete=reproduced, labels_support_the_claim=True,
             known_invalidating_issue=(
                 "" if reproduced else
                 "the frozen record no longer reproduces its own admissions")),

        gate("Filter Recency-Bias Probe (H1-H4), the proposal's primary claim",
             claim="Confidence-derived utility signals admit older evidence "
                   "preferentially.",
             artifact="experiments/recency_bias/frb_pairs.py -- implemented, not run",
             executed=False, correct_corpus=False, correct_candidates=False,
             fairness_passed=False, intended_generator=False,
             provenance_complete=False, labels_support_the_claim=False,
             known_invalidating_issue=(
                 "Not executed. The probe pipeline now exists -- pair "
                 "construction, the permutation control, the paired bootstrap "
                 "and the blocking gate, with 36 offline tests -- but it has "
                 "never been run on data. H1 and H4 (the proposal's Minimum "
                 "Viable Implementation, 5.4) need only MedChangeQA plus the "
                 "already-trained filter. H2 additionally needs a second filter "
                 "trained on the entailment label, which does not exist.")),

        gate("Matched-k = 5 answer comparison",
             claim="At an equal five-passage budget the two policies select "
                   "different evidence and produce different answers.",
             artifact="matched_k5/ -- not generated",
             executed=matched_k_generated, correct_corpus=True,
             correct_candidates=matched_k_ready, fairness_passed=matched_k_ready,
             intended_generator=matched_k_generated, provenance_complete=matched_k_generated,
             labels_support_the_claim=False,
             known_invalidating_issue=(
                 "The selection half is executed and passes every invariant. "
                 "Generation has not run. Even once it does, no answer-quality "
                 "label exists, so the result can support 'the evidence differs' "
                 "and 'the answers differ', never 'the answers are better'.")),
    ]
    return {
        "gates": gates,
        "reportable": [g["result"] for g in gates if g["reportable"]],
        "not_reportable": [g["result"] for g in gates if not g["reportable"]],
        "policy": ("A non-reportable result is kept, not deleted: it is audit "
                   "history. It must never appear in the thesis as a finding."),
    }


# --------------------------------------------------------------------------
# The thesis's own questions
# --------------------------------------------------------------------------
def answer_the_thesis_questions(ablations: Dict[str, Any],
                                statistics: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Section 23's thirteen questions, answered from the artifacts."""
    variance = ablations["variance_decomposition"]["terms"]
    admission = statistics["admission"]
    quality = statistics["evidence_quality"]["rank_association"]

    def assoc(name: str) -> str:
        block = quality.get(name, {})
        if block.get("point") is None:
            return "not computed"
        return (f"rho = {block['point']:+.4f}, 95% CI "
                f"[{block['ci_low']:+.4f}, {block['ci_high']:+.4f}], "
                f"Holm {'survives' if block['holm']['survives_holm'] else 'does not survive'}")

    return [
        {"question": "1. What was reproduced from RAG2?",
         "answer": (
             "The full pipeline: rationale-based query formulation, balanced "
             "retrieval over four corpora, MedCPT dense retrieval and "
             "cross-encoder reranking, and the rationale-guided Flan-T5-large "
             "filter with the released two-way softmax over [HELPFUL] / "
             "[NOT_HELPFUL] at a 0.5 decision boundary. The filter checkpoint was "
             "retrained, as the paper's was never distributed. What was NOT "
             "reproduced is the paper's headline: no accuracy was measured on "
             "MedQA, MedMCQA or MMLU-Med, so the reproduction is structural, not "
             "numerical."),
         "status": "done, structurally verified, numerically unverified"},

        {"question": "2. What exactly was changed by SCAF?",
         "answer": (
             "The admission stage only, at the EvidenceFilter seam. Everything "
             "upstream is byte-identical by replay. But SCAF as implemented is "
             "not SCAF as proposed: sigma is lexical overlap rather than "
             "entailment, the contested state and the Grounded/Flagged/Abstain "
             "output policy are absent, and rho -- defined in the proposal as the "
             "rank-normalised reranker score -- had been misread as "
             "'corroboration' and hard-coded to zero, removing the term "
             "entirely. That misreading is corrected in this pass."),
         "status": "partially implemented; the divergences are now itemised"},

        {"question": "3. Was the RAG2/SCAF comparison fair?",
         "answer": (
             "Upstream, yes and provably: same 30 questions, same 600 candidates "
             "in the same order with identical per-question digests, same "
             "generator, same prompt fingerprint, same decoding, nine of nine "
             "fairness checks passed. Downstream, no: the proposal's fairness "
             "guarantee 2 requires every arm to receive at most five passages, "
             "and SCAF's arm received a mean of 19.2 against RAG2's 0.13. The "
             "admission comparison is therefore fair as a measurement of "
             "admission and confounded as a comparison of answers."),
         "status": "fair for admission, confounded for answers"},

        {"question": "4. What evidence did each policy admit?",
         "answer": (
             f"RAG2 admitted {admission['contingency']['both_admitted'] + admission['contingency']['rag2_only']} "
             f"of 600 candidates ({admission['rag2_admission_rate']:.4f}); SCAF "
             f"admitted {admission['contingency']['both_admitted'] + admission['contingency']['scaf_only']} "
             f"({admission['scaf_admission_rate']:.4f}). The difference is "
             f"{admission['difference_clustered']['point']:+.4f} with a "
             f"question-clustered 95% CI of "
             f"[{admission['difference_clustered']['ci_low']:+.4f}, "
             f"{admission['difference_clustered']['ci_high']:+.4f}]. RAG2's "
             "admitted set is a strict subset of SCAF's -- there is no passage "
             "RAG2 admitted that SCAF rejected. RAG2 left 26 of 30 questions "
             "with no evidence at all, which is closed-book generation rather "
             "than abstention."),
         "status": "measured and reportable"},

        {"question": "5. Did SCAF demonstrably improve evidence quality?",
         "answer": (
             "No. Against independent human usefulness labels the SCAF admission "
             f"score gives {assoc('scaf_score')}, and its support term gives "
             f"{assoc('sigma_support')} -- both intervals include zero. RAG2's "
             f"filter score gives {assoc('rag2_score')}, also indistinguishable "
             "from none. The one signal whose association survives "
             f"Holm-Bonferroni is the frozen reranker rank: {assoc('rho_rerank')}. "
             "That is the term SCAF's implementation had dropped, and neither "
             "arm uses it at admission. Note also that all 14 passages the "
             "annotator judged NOT relevant were admitted by SCAF."),
         "status": "tested; hypothesis NOT supported"},

        {"question": "6. Did SCAF demonstrably improve answer quality?",
         "answer": (
             "Unanswerable with the current artifacts, and not because the "
             "experiment failed. No gold answers, no reference answers, no human "
             "answer judgements and no claim-level labels exist for these 30 "
             "questions. 60 answers were generated in the natural-admission run, "
             "but under a 688x context imbalance. The matched-k = 5 design "
             "removes that confound and is fully prepared, but even it can only "
             "establish that the answers DIFFER, never that they are better, "
             "until an answer-level evaluation exists."),
         "status": "not measured; no instrument exists"},

        {"question": "7. Was recency bias demonstrated, refuted, or left unresolved?",
         "answer": (
             "Left unresolved, and it is the proposal's primary claim. The "
             "Filter Recency-Bias Probe was never built. The corpus cannot "
             "support it -- 2021-2026 only, so gamma never falls below "
             f"{variance['currency']['raw_min']:.3f} and there is no older "
             "stratum -- but the proposal never intended it to: FRB-PAIRS is "
             "specified as derived from MedChangeQA, an external peer-reviewed "
             "dataset, behind an explicit provenance firewall. Descriptively, "
             "RAG2's four admissions all fall in 2025-2026, which points AWAY "
             "from the hypothesised direction, but four admissions is not "
             "evidence of anything. The probe is now implemented and tested "
             "offline; H1 and H4 need MedChangeQA and the filter that is "
             "already trained, and nothing else."),
         "status": "not executed; pipeline implemented, one dataset away"},

        {"question": "8. What did the human evidence evaluation show?",
         "answer": (
             "120 passages, one annotator, labels 0/1/2 distributed 14/46/60, "
             "produced with no machine suggestion generated or displayed. It "
             "showed that the machine scores order passages roughly the way the "
             "reranker does and not the way either admission policy does. It "
             "also showed that admission volume and evidence quality are "
             "different things: SCAF admitted every passage the annotator "
             "rejected. Two design limits carried from the audit: 29 of 30 "
             "questions are represented with up to 8 passages from one question, "
             "and all four RAG2 admissions were forced into the sample, so it is "
             "not a simple random sample of the 600."),
         "status": "done and reportable, with stated limitations"},

        {"question": "9. What did the answer evaluation show?",
         "answer": "Nothing: it has not been run. See question 6.",
         "status": "not executed"},

        {"question": "10. Which hypotheses were supported?",
         "answer": (
             "None of the proposal's numbered hypotheses (H1-H7) was tested, "
             "because each needs an instrument that does not exist here: "
             "FRB-PAIRS for H1-H4, claim-level answer labels for H5-H6, gold "
             "answers for H7. What IS established, and was not a hypothesis: the "
             "two policies admit at radically different rates; SCAF's score is "
             "dominated by its lexical support term "
             f"({variance['support']['share_of_score_variance']:.1%} of the "
             "score's variance); and the reranker rank is the only signal that "
             "measurably tracks human usefulness judgements."),
         "status": "no hypothesis supported; three findings established"},

        {"question": "11. Which hypotheses were not supported?",
         "answer": (
             "The implicit hypothesis behind SCAF -- that an entailment-and-"
             "currency admission score selects better evidence than a "
             "confidence-derived one -- is NOT supported by this data. Its "
             "association with human labels does not clear zero. This is a "
             "genuine negative result about the implemented policy. It is not a "
             "test of the proposed policy, because the entailment support term "
             "and the contested state were never built."),
         "status": "one negative result, correctly scoped"},

        {"question": "12. What remains a limitation?",
         "answer": (
             "(a) sigma is lexical, not entailment -- the single largest gap "
             "between proposal and implementation; (b) the corpus spans five "
             "years, so the currency term has almost no dynamic range and the "
             "temporal question cannot be asked of it; (c) no answer-quality "
             "instrument of any kind; (d) one annotator, 120 non-randomly "
             "selected passages clustered in 29 questions; (e) the admission "
             "threshold 0.45 and half-life 5 y were never fitted on validation "
             "data, and the sweeps show admission moving from 99.7% to 28.2% "
             "across plausible thresholds; (f) the contested state, the "
             "supersession table and the abstention gate are unimplemented or "
             "never fired; (g) 30 development questions, one backbone, one "
             "decoding pass, no variance estimate."),
         "status": "itemised"},

        {"question": "13. What can the student safely claim in the thesis?",
         "answer": (
             "Claimable, with the artifacts to back each one: (1) a faithful "
             "structural reproduction of the RAG2 pipeline over an "
             "Alzheimer-scoped corpus with a retrained filter; (2) that filter "
             "admits almost nothing on this corpus -- 4 of 600 -- which is a "
             "measured over-rejection finding in the proposal's own error "
             "taxonomy; (3) a working, fully provenanced admission-policy "
             "comparison harness with byte-identical upstream replay; (4) the "
             "implemented SCAF admits 575 of 600, and its score is ~84% lexical "
             "support by variance; (5) neither admission score's association "
             "with independent human usefulness labels is distinguishable from "
             "zero, while the frozen reranker's rank is; (6) the corpus as "
             "approved cannot support the temporal question, quantified exactly. "
             "NOT claimable: that SCAF is better than RAG2, that SCAF improves "
             "evidence or answer quality, that recency bias exists or does not, "
             "or any statement about answer correctness."),
         "status": "six defensible claims, four explicitly forbidden"},
    ]
