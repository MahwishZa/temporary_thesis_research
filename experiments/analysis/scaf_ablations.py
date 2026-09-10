"""Counterfactual re-scoring of the completed comparison: the proposal's ablations.

What this module is
-------------------
The scientific run recorded, for every one of the 600 candidates, each SCAF
sub-score (sigma, gamma, tau), the weights, the threshold, the reranker rank and
the publication date. Admission is a deterministic function of exactly those
numbers. So a question of the form

    "what would this policy have admitted with the currency term switched off?"

can be answered *exactly*, offline, with no model, no index, no generator and no
GPU -- by recomputing the admission arithmetic on the frozen record.

That is what this module does. It is re-scoring, never re-running: no retrieval,
no reranking, no generation, and no candidate is added, removed or re-ordered.

Verified before use
-------------------
Two identities are checked against the frozen run before any ablation is
reported, and a failure refuses rather than degrades:

* ``A(s) = w_sigma*sigma + w_gamma*gamma + w_rho*rho + w_tau*tau`` reproduces the
  recorded ``scaf_score`` for all 600 candidates, and the recorded admit/reject
  for all 600;
* ``gamma = 2^(-age/H)`` with ``age = max(0, as_of - date)`` and
  ``as_of = 2026-01-01`` reproduces every recorded gamma. This is what licenses
  sweeping the half-life: gamma is regenerated from the date, not rescaled.

What it cannot answer
---------------------
Admission only. Every number here is "which passages would have been admitted",
never "would the answer have been better" -- that needs generation, which needs
the frozen candidate text and the local generator. Ablations A1 (perplexity
against entailment label) and A3 (second backbone) need a filter that is not in
this repository and are out of reach here by construction, not by omission.
"""

from __future__ import annotations

import json
import math
import os
import re
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

#: The as-of date the scientific run scored against, recovered from the frozen
#: record and verified to reproduce every gamma to < 1e-6. Stored as a year
#: fraction because that is the unit the currency curve uses.
AS_OF_YEAR = 2026.0

#: Half-life and discount the scientific run used.
RUN_HALF_LIFE_YEARS = 5.0
RUN_THRESHOLD = 0.45
RUN_WEIGHTS: Dict[str, float] = {"support": 0.5, "currency": 0.3,
                                 "rerank": 0.0, "authority": 0.2}

#: Candidate depth per question in the scientific run.
RUN_DEPTH = 20

TERMS = ("support", "currency", "rerank", "authority")


class NotTheScientificRun(Exception):
    """The frozen record does not reproduce its own admissions."""


def _year_fraction(value: str) -> Optional[float]:
    """A YYYY / YYYY-MM / YYYY-MM-DD date as a float year.

    **Must stay identical to ``scaf.policy._year_fraction``.** It is duplicated
    rather than imported so this analysis layer keeps no dependency on the
    policy's import chain, and ``test_scaf_ablations.py`` asserts the two agree
    across a date grid -- a silent divergence here would re-derive gamma on a
    different calendar from the one that produced the run, which is precisely
    the class of bug this module exists to correct.
    """
    text = (value or "").strip()
    match = re.match(r"^(\d{4})(?:-(\d{2}))?(?:-(\d{2}))?", text)
    if not match:
        return None
    year = int(match.group(1))
    if not 1500 <= year <= 2200:
        return None
    if match.group(2):
        month = min(12, max(1, int(match.group(2))))
        day = int(match.group(3)) if match.group(3) else 15
        return year + ((month - 1) + (min(28, max(1, day)) - 1) / 30.0) / 12.0
    return year + 0.5


def rank_normalised(rank: Optional[int], depth: int = RUN_DEPTH) -> float:
    """rho: rank 1 -> 1.0, rank ``depth`` -> 0.0. Missing rank -> neutral 0.5."""
    if rank is None or depth <= 1:
        return 0.5
    clamped = min(max(int(rank), 1), depth)
    return (depth - clamped) / (depth - 1)


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
def load_records(per_question_path: str) -> List[Dict[str, Any]]:
    """One flat record per candidate, carrying everything an ablation needs."""
    out: List[Dict[str, Any]] = []
    with open(per_question_path, "r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            rag2_by_chunk = {d["chunk_id"]: d for d in row["rag2"]["decisions"]}
            for decision in row["scaf"]["decisions"]:
                detail = decision["detail"]
                currency = detail["currency_detail"]
                rag2 = rag2_by_chunk.get(decision["chunk_id"], {})
                out.append({
                    "qid": row["qid"],
                    "question": row["question"],
                    "chunk_id": decision["chunk_id"],
                    "time_sensitive": bool(currency["time_sensitive"]),
                    "canonical_date": currency.get("canonical_date") or "",
                    "date_precision": currency.get("date_precision") or "",
                    "state": currency.get("state") or "",
                    "retracted": currency.get("retracted"),
                    "supersession": currency.get("supersession"),
                    "rerank_rank": decision.get("rerank_rank"),
                    "source_category": decision.get("source_category") or "",
                    "authority_tier_label": decision.get("authority_tier_label") or "",
                    "sigma": float(detail["sigma_support"]),
                    "gamma": float(detail["gamma_currency"]),
                    "tau": float(detail["tau_authority"]),
                    "rho": rank_normalised(decision.get("rerank_rank")),
                    "recorded_score": float(detail["scaf_score"]),
                    "recorded_threshold": float(detail["threshold"]),
                    "recorded_weights": dict(detail["weights"]),
                    "recorded_gate": detail.get("gate") or "",
                    "scaf_admit": bool(decision["keep"]),
                    "rag2_admit": bool(rag2.get("keep", False)),
                    "rag2_score": float(rag2.get("score", 0.0)),
                })
    return out


def verify_reproduction(records: Sequence[Dict[str, Any]],
                        tolerance: float = 1e-5) -> Dict[str, Any]:
    """Prove the frozen record reproduces its own scores and admissions.

    Every ablation below is only as trustworthy as this check. It is separate
    from the ablations so it can be reported as a precondition rather than
    assumed inside one.
    """
    checks: List[Dict[str, Any]] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        checks.append({"check": name, "pass": bool(ok), "detail": detail})

    worst_score = 0.0
    admission_mismatch = 0
    for record in records:
        weights = record["recorded_weights"]
        rebuilt = (weights.get("support", 0.0) * record["sigma"]
                   + weights.get("currency", 0.0) * record["gamma"]
                   + weights.get("authority", 0.0) * record["tau"])
        worst_score = max(worst_score, abs(rebuilt - record["recorded_score"]))
        expected_admit = rebuilt >= record["recorded_threshold"]
        if record["recorded_gate"] == "retracted":
            expected_admit = False
        if expected_admit != record["scaf_admit"]:
            admission_mismatch += 1
    check(f"A(s) rebuilt from the recorded sub-scores for all {len(records)}",
          worst_score <= tolerance, f"max abs error {worst_score:.2e}")
    check("every recorded admit/reject is reproduced",
          admission_mismatch == 0, f"{admission_mismatch} mismatches")

    worst_gamma = 0.0
    for record in records:
        if not record["time_sensitive"]:
            continue
        rebuilt = currency_at(record, RUN_HALF_LIFE_YEARS)
        worst_gamma = max(worst_gamma, abs(rebuilt - record["gamma"]))
    check("gamma regenerated from the publication date matches the run",
          worst_gamma <= 1e-4,
          f"max abs error {worst_gamma:.2e} at as_of={AS_OF_YEAR}, "
          f"H={RUN_HALF_LIFE_YEARS}")

    ranks = {r["rerank_rank"] for r in records}
    check("every candidate carries a reranker rank",
          None not in ranks and ranks <= set(range(1, RUN_DEPTH + 1)),
          f"{len(ranks)} distinct ranks")

    return {"all_passed": all(c["pass"] for c in checks), "checks": checks,
            "candidates": len(records)}


# --------------------------------------------------------------------------
# Re-scoring
# --------------------------------------------------------------------------
def currency_at(record: Dict[str, Any], half_life_years: float,
                superseded_discount: float = 0.5) -> float:
    """gamma regenerated from the publication date at an arbitrary half-life.

    Follows the proposal's three-state definition as the run implemented it:
    retracted -> 0; a claim that does not age (psi(q) = 0) -> 1; otherwise the
    decay curve, discounted if the corpus marks the passage superseded.
    """
    if str(record.get("retracted", "")).lower() in ("yes", "true", "1"):
        return 0.0
    if not record["time_sensitive"]:
        return 1.0
    year = _year_fraction(record["canonical_date"])
    if year is None:
        return 0.5                      # undated: flagged, never guessed
    age = max(0.0, AS_OF_YEAR - year)
    gamma = 2.0 ** (-age / float(half_life_years))
    supersession = str(record.get("supersession") or "")
    if supersession not in ("", "unknown", "n/a", "none"):
        gamma *= float(superseded_discount)
    return gamma


def score_records(records: Sequence[Dict[str, Any]],
                  weights: Dict[str, float],
                  threshold: float,
                  half_life_years: Optional[float] = None,
                  superseded_discount: float = 0.5,
                  authority_map: Optional[Dict[str, float]] = None,
                  renormalise: bool = False) -> List[Dict[str, Any]]:
    """Admission under a counterfactual configuration.

    ``renormalise`` rescales the supplied weights to sum to 1. It matters:
    zeroing a term shrinks A(s) toward zero, so a fixed threshold silently
    becomes *stricter* -- an apparent "effect of removing the term" that is
    really an effect of moving the scale. Both readings are reported by the
    ablation suite rather than one being chosen.
    """
    active = {term: float(weights.get(term, 0.0)) for term in TERMS}
    if renormalise:
        total = sum(active.values())
        if total > 0:
            active = {k: v / total for k, v in active.items()}
    out: List[Dict[str, Any]] = []
    for record in records:
        gamma = (record["gamma"] if half_life_years is None
                 else currency_at(record, half_life_years, superseded_discount))
        tau = (record["tau"] if authority_map is None
               else _authority_at(record, authority_map))
        score = (active["support"] * record["sigma"]
                 + active["currency"] * gamma
                 + active["rerank"] * record["rho"]
                 + active["authority"] * tau)
        admit = score >= threshold
        if str(record.get("retracted", "")).lower() in ("yes", "true", "1"):
            admit = False               # the one hard gate, preserved
        out.append({**record, "score": score, "gamma_used": gamma,
                    "tau_used": tau, "admit": admit})
    return out


def _authority_at(record: Dict[str, Any], mapping: Dict[str, float]) -> float:
    tier = record["authority_tier_label"]
    if tier and tier in mapping:
        return float(mapping[tier])
    category = record["source_category"]
    if category in mapping:
        return float(mapping[category])
    return float(mapping.get("", 0.4))


# --------------------------------------------------------------------------
# Summaries
# --------------------------------------------------------------------------
def summarise(scored: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Admission counts, plus the per-question vector the statistics need."""
    per_question: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for record in scored:
        per_question[record["qid"]].append(record)
    rates = {qid: sum(1 for r in rows if r["admit"]) / len(rows)
             for qid, rows in per_question.items()}
    admitted = sum(1 for r in scored if r["admit"])
    return {
        "candidates": len(scored),
        "admitted": admitted,
        "admission_rate": round(admitted / len(scored), 6) if scored else 0.0,
        "questions": len(per_question),
        "questions_with_no_evidence": sum(1 for v in rates.values() if v == 0.0),
        "mean_admitted_per_question": round(
            admitted / len(per_question), 3) if per_question else 0.0,
        "per_question_admission_rate": {q: round(v, 6) for q, v in sorted(rates.items())},
    }


def agreement_with_run(scored: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """How many admission decisions this configuration changes, and which way."""
    flipped_in = sum(1 for r in scored if r["admit"] and not r["scaf_admit"])
    flipped_out = sum(1 for r in scored if not r["admit"] and r["scaf_admit"])
    return {"decisions_changed": flipped_in + flipped_out,
            "newly_admitted": flipped_in, "newly_rejected": flipped_out,
            "unchanged": len(scored) - flipped_in - flipped_out}


def variance_decomposition(records: Sequence[Dict[str, Any]],
                           weights: Dict[str, float] = None) -> Dict[str, Any]:
    """Which term actually moves A(s)?

    A term with a large weight but no spread across candidates cannot change any
    admission decision, however important it looks in the formula. This reports
    the spread of each *weighted* term, which is what admission responds to.
    """
    weights = dict(weights or RUN_WEIGHTS)
    columns = {"support": [r["sigma"] for r in records],
               "currency": [r["gamma"] for r in records],
               "rerank": [r["rho"] for r in records],
               "authority": [r["tau"] for r in records]}
    out: Dict[str, Any] = {"weights": weights, "terms": {}}
    weighted_variance_total = 0.0
    for term, values in columns.items():
        weight = float(weights.get(term, 0.0))
        mean = sum(values) / len(values)
        variance = sum((v - mean) ** 2 for v in values) / len(values)
        weighted = (weight ** 2) * variance
        weighted_variance_total += weighted
        out["terms"][term] = {
            "weight": weight,
            "raw_mean": round(mean, 6),
            "raw_sd": round(math.sqrt(variance), 6),
            "raw_min": round(min(values), 6),
            "raw_max": round(max(values), 6),
            "weighted_sd": round(weight * math.sqrt(variance), 6),
            "weighted_variance": weighted,
        }
    for term, block in out["terms"].items():
        block["share_of_score_variance"] = round(
            block.pop("weighted_variance") / weighted_variance_total, 6
        ) if weighted_variance_total > 0 else 0.0
    out["note"] = (
        "Shares are of the variance of A(s) under the run's weights, treating "
        "the terms as independent; they answer 'what moves admission', not "
        "'what matters clinically'. A term with share 0 cannot have changed a "
        "single admission decision in this run.")
    return out


# --------------------------------------------------------------------------
# The ablation suite
# --------------------------------------------------------------------------
def _ablation(name: str, ident: str, question: str, records, weights, threshold,
              **kwargs) -> Dict[str, Any]:
    scored = score_records(records, weights, threshold, **kwargs)
    return {"id": ident, "name": name, "question_answered": question,
            "weights": {t: float(weights.get(t, 0.0)) for t in TERMS},
            "threshold": threshold,
            "renormalised": bool(kwargs.get("renormalise")),
            "half_life_years": kwargs.get("half_life_years", RUN_HALF_LIFE_YEARS),
            "summary": summarise(scored),
            "vs_run": agreement_with_run(scored)}


def ablation_suite(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """The proposal's ablations that admission arithmetic alone can answer."""
    w = dict(RUN_WEIGHTS)
    t = RUN_THRESHOLD
    results: List[Dict[str, Any]] = []

    results.append(_ablation(
        "As run (SCAF, lexical support)", "RUN",
        "what the scientific run did", records, w, t))

    results.append(_ablation(
        "No filter", "A4", "does any filter help?",
        records, w, 0.0))

    for renorm in (False, True):
        results.append(_ablation(
            "Support only, currency disabled" + (" (renormalised)" if renorm else ""),
            "A5", "contribution of currency",
            records, {**w, "currency": 0.0}, t, renormalise=renorm))
        results.append(_ablation(
            "Currency only, support excluded" + (" (renormalised)" if renorm else ""),
            "A7", "contribution of support",
            records, {**w, "support": 0.0}, t, renormalise=renorm))
        results.append(_ablation(
            "Authority removed" + (" (renormalised)" if renorm else ""),
            "A12a", "contribution of source authority",
            records, {**w, "authority": 0.0}, t, renormalise=renorm))

    # A12: the ordering itself is the tested variable, not just its presence.
    inverted = {"clinical-practice-guideline": 0.40, "appropriate-use-criteria": 0.45,
                "diagnostic-criteria": 0.45, "consensus-recommendation": 0.55,
                "": 1.00, "currency-pack": 0.40, "pmc-fulltext": 0.55,
                "pubmed-abstract": 0.60}
    results.append(_ablation(
        "Authority ordering inverted", "A12b",
        "recency against authority trade-off: is the ordering load-bearing?",
        records, w, t, authority_map=inverted))

    # The rho correction: what the corrected term would do if it were weighted.
    for weight in (0.1, 0.2, 0.3):
        results.append(_ablation(
            f"Corrected rho (reranker) at w={weight}", "RHO",
            "effect of the term the earlier implementation dropped",
            records, {**w, "rerank": weight}, t, renormalise=True))

    return {"ablations": results,
            "guard": ("Admission only. No answer was generated or evaluated in "
                      "any of these configurations, so none of them supports a "
                      "claim about answer quality."),
            "not_executable_here": {
                "A1": "perplexity against entailment label -- needs both filter "
                      "backbones; the trained checkpoint is not in this repository",
                "A2": "permutation control -- needs FRB-PAIRS",
                "A3": "second backbone replication -- needs a second trained filter",
                "A6": "prompt with and without date annotation -- needs generation",
                "A8": "contested state removed -- the contested state is not implemented",
                "A9": "abstention gate removed -- abstention never fired (0 questions)",
                "A10": "teacher against distilled student -- no entailment teacher exists",
                "A13": "hard supersession gate restored -- no passage is marked "
                       "superseded in this corpus (supersession is 'unknown' for all 480 "
                       "time-sensitive candidates), so the gate has nothing to act on",
                "A14": "generator transfer -- needs generation"}}


def sensitivity(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Ablation A11: does the result survive the parameters nobody has fitted?"""
    w, t = dict(RUN_WEIGHTS), RUN_THRESHOLD
    thresholds = []
    for value in (0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70):
        summary = summarise(score_records(records, w, value))
        thresholds.append({"threshold": value,
                           "admitted": summary["admitted"],
                           "admission_rate": summary["admission_rate"],
                           "questions_with_no_evidence":
                               summary["questions_with_no_evidence"]})
    half_lives = []
    for value in (1.0, 2.0, 3.0, 5.0, 8.0, 10.0, 20.0):
        scored = score_records(records, w, t, half_life_years=value)
        summary = summarise(scored)
        half_lives.append({"half_life_years": value,
                           "admitted": summary["admitted"],
                           "admission_rate": summary["admission_rate"],
                           "questions_with_no_evidence":
                               summary["questions_with_no_evidence"],
                           "vs_run": agreement_with_run(scored)["decisions_changed"]})
    return {"threshold_sweep": thresholds, "half_life_sweep": half_lives,
            "note": ("The run's threshold (0.45) and half-life (5 y) were not "
                     "fitted on validation data. Fairness guarantee 5 requires "
                     "they be selected and frozen before a test run; until then "
                     "these sweeps are the honest statement of how much the "
                     "headline admission numbers depend on them.")}


def admission_by_age(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Descriptive admission by publication year. **Not** the FRB-PAIRS probe.

    FRB-PAIRS compares *matched* older/newer passages making the same claim, so
    age is the only difference. This compares unmatched passages of different
    ages that also differ in topic, wording, journal and length, so any
    difference it shows is confounded by all of those. It is reported because a
    reader will ask, and refusing to show it invites a worse guess.
    """
    by_year_rag2: Dict[str, List[bool]] = defaultdict(list)
    by_year_scaf: Dict[str, List[bool]] = defaultdict(list)
    for record in records:
        year = (record["canonical_date"] or "")[:4] or "unknown"
        by_year_rag2[year].append(record["rag2_admit"])
        by_year_scaf[year].append(record["scaf_admit"])
    rows = []
    for year in sorted(set(by_year_rag2) | set(by_year_scaf)):
        rag2, scaf = by_year_rag2[year], by_year_scaf[year]
        rows.append({
            "year": year, "candidates": len(rag2),
            "rag2_admitted": sum(rag2),
            "rag2_rate": round(sum(rag2) / len(rag2), 6) if rag2 else None,
            "scaf_admitted": sum(scaf),
            "scaf_rate": round(sum(scaf) / len(scaf), 6) if scaf else None})
    years = [_year_fraction(r["canonical_date"]) for r in records]
    return {
        "by_year": rows,
        "corpus_span": {"earliest": min(r["canonical_date"] for r in records),
                        "latest": max(r["canonical_date"] for r in records),
                        "distinct_years": len(rows),
                        "span_years": round(max(y for y in years if y)
                                            - min(y for y in years if y), 2)},
        "is_frb_pairs": False,
        "guard": ("DESCRIPTIVE ONLY, AND CONFOUNDED. These passages are not "
                  "matched on claim, tier or length, so this is not the "
                  "admission asymmetry the thesis defines and must not be "
                  "reported as a recency-bias result. The corpus also spans "
                  "only ~5 years with no pre-2021 stratum, so there is almost "
                  "no age contrast to detect even descriptively."),
    }


def build_ablations(per_question_path: str) -> Dict[str, Any]:
    """Everything, with the reproduction check as a precondition."""
    records = load_records(per_question_path)
    reproduction = verify_reproduction(records)
    if not reproduction["all_passed"]:
        raise NotTheScientificRun(
            "the frozen record does not reproduce its own admissions: "
            + "; ".join(c["check"] for c in reproduction["checks"] if not c["pass"]))
    return {
        "source": os.path.basename(per_question_path),
        "reproduction_check": reproduction,
        "as_of_year": AS_OF_YEAR,
        "run_configuration": {"weights": RUN_WEIGHTS, "threshold": RUN_THRESHOLD,
                              "half_life_years": RUN_HALF_LIFE_YEARS,
                              "candidate_depth": RUN_DEPTH},
        "variance_decomposition": variance_decomposition(records),
        "ablations": ablation_suite(records),
        "sensitivity": sensitivity(records),
        "admission_by_age": admission_by_age(records),
    }
