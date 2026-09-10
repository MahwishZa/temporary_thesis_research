"""Read the frozen RAG2-vs-SCAF artifacts and say what they actually show.

Every number here is computed from files that already exist. Nothing is
retrieved, scored, generated, sampled or annotated: this module opens the
completed comparison and the completed independent annotation, joins them, and
counts. It writes nothing itself -- ``experiments/scripts/analyse_results.py``
decides where the output goes.

The join is on ``(qid, chunk_id)``, which both the decision file and the blind
annotation sheet carry. That deliberately avoids needing ``annotation_key.jsonl``
(gitignored, machine-local), so the analysis reproduces anywhere the repository
is checked out.

One warning belongs at the top rather than buried in a limitations section.
SCAF's support term is lexical overlap between question and passage; the first
annotation pass was collected through an interface that displayed a lexical
suggestion, and every one of its 120 labels matched that suggestion. Its
correlations are therefore not measurements of SCAF. **Only the corrected
human-only pass may be used**, and this module refuses any sheet not marked as
that pass.
"""

from __future__ import annotations

import hashlib
import json
import statistics
import time
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Sequence

from experiments.analysis.evidence_quality import (
    VALID_LABELS,
    component_diagnostics,
    load_decisions,
    matched_budget,
    question_clustering,
    spearman,
)

ANALYSIS_VERSION = "scientific-results-v1"
CORRECTED_PASS_LABEL = "corrected-human-only-v1"

#: The score fields whose rank association with the human label is reported.
SCORE_FIELDS = (
    ("scaf_score", "scaf_score"),
    ("scaf_sigma_support", "scaf_sigma_support"),
    ("scaf_gamma_currency", "scaf_gamma_currency"),
    ("scaf_tau_authority", "scaf_tau_authority"),
    ("rag2_score", "rag2_score"),
)


class NotTheCorrectedPass(Exception):
    """Raised when asked to analyse anything but the independent pass."""


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(65536), b""):
            digest.update(block)
    return digest.hexdigest()


def _label_fingerprint(rows: Sequence[Dict[str, Any]]) -> str:
    payload = "\n".join(f"{r.get('annotation_id')}={r.get('human_label')}"
                        for r in sorted(rows, key=lambda r: str(r.get("annotation_id"))))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _summary(values: Sequence[float]) -> Dict[str, Any]:
    if not values:
        return {"n": 0}
    return {
        "n": len(values),
        "mean": round(statistics.mean(values), 4),
        "median": statistics.median(values),
        "distribution": dict(sorted(Counter(values).items())),
    }


# --------------------------------------------------------------------------
def provenance(per_question_path: str, sheet_path: str, manifest_path: str,
               comparison_manifest_path: str,
               decisions: Sequence[Dict[str, Any]],
               sheet_rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Exactly which artifacts this analysis read, and how to recognise them."""
    with open(manifest_path, "r", encoding="utf-8") as handle:
        sample_manifest = json.load(handle)
    with open(comparison_manifest_path, "r", encoding="utf-8") as handle:
        run_manifest = json.load(handle)

    raw = _read_jsonl(per_question_path)
    return {
        "analysis_version": ANALYSIS_VERSION,
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "sources": {
            "decisions": per_question_path,
            "run_manifest": comparison_manifest_path,
            "annotation_sheet": sheet_path,
            "sample_manifest": manifest_path,
        },
        "scientific_run": {
            "frozen_set_digest": run_manifest.get("frozen_set_digest"),
            "created_at": run_manifest.get("created_at"),
            "reportable": run_manifest.get("reportable"),
            "fairness_all_passed": (run_manifest.get("fairness") or {}).get("all_passed"),
            "scientific_all_passed": (run_manifest.get("scientific") or {}).get("all_passed"),
        },
        "sample": {
            "frozen_set_digest": sample_manifest.get("frozen_set_digest"),
            "retrieval_is_medcpt": sample_manifest.get("retrieval_is_medcpt"),
            "seed": sample_manifest.get("seed"),
            "population": sample_manifest.get("population"),
            "sampled": sample_manifest.get("sampled"),
            "forced_rag2_admitted": sample_manifest.get("forced_rag2_admitted"),
            "scaf_tertile_edges": sample_manifest.get("scaf_tertile_edges"),
            "rag2_tertile_edges": sample_manifest.get("rag2_tertile_edges"),
        },
        "counts": {
            "questions": len(raw),
            "candidates_per_question": sorted({r.get("num_candidates") for r in raw}),
            "decisions": len(decisions),
            "human_annotations": len(sheet_rows),
        },
        "annotation_sheet_sha256": _sha256(sheet_path),
        "label_fingerprint": _label_fingerprint(sheet_rows),
        "annotation_pass": sorted({str(r.get("annotation_pass")) for r in sheet_rows}),
        "digests_agree": (sample_manifest.get("frozen_set_digest")
                          == run_manifest.get("frozen_set_digest")),
    }


def admission_behaviour(decisions: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """How much each policy admits -- counted, never interpreted as quality."""
    total = len(decisions)
    rag2 = sum(1 for d in decisions if d["rag2_admitted"])
    scaf = sum(1 for d in decisions if d["scaf_admitted"])
    cells = Counter((d["rag2_admitted"], d["scaf_admitted"]) for d in decisions)
    both, rag2_only = cells[(True, True)], cells[(True, False)]
    scaf_only, neither = cells[(False, True)], cells[(False, False)]

    per_question: Dict[str, List[int]] = defaultdict(lambda: [0, 0])
    for d in decisions:
        per_question[d["qid"]][0] += bool(d["rag2_admitted"])
        per_question[d["qid"]][1] += bool(d["scaf_admitted"])
    rag2_counts = sorted(v[0] for v in per_question.values())
    scaf_counts = sorted(v[1] for v in per_question.values())

    return {
        "candidate_level": {
            "decisions": total,
            "rag2_admitted": rag2,
            "scaf_admitted": scaf,
            "rag2_admission_rate": round(rag2 / total, 6) if total else None,
            "scaf_admission_rate": round(scaf / total, 6) if total else None,
            "both_admit": both,
            "rag2_only": rag2_only,
            "scaf_only": scaf_only,
            "neither_admits": neither,
            "jaccard": round(both / (total - neither), 6) if total - neither else None,
        },
        "question_level": {
            "questions": len(per_question),
            "questions_with_any_rag2_evidence": sum(1 for c in rag2_counts if c),
            "questions_with_any_scaf_evidence": sum(1 for c in scaf_counts if c),
            "questions_with_zero_rag2_evidence": sum(1 for c in rag2_counts if not c),
            "rag2_admissions_per_question": {
                "min": rag2_counts[0], "max": rag2_counts[-1],
                "mean": round(statistics.mean(rag2_counts), 4)},
            "scaf_admissions_per_question": {
                "min": scaf_counts[0], "max": scaf_counts[-1],
                "mean": round(statistics.mean(scaf_counts), 4)},
        },
        "guard": ("admission counts measure how much evidence each policy lets "
                  "through. They say nothing about whether that evidence is "
                  "better; that is what the human labels are for."),
    }


def human_evidence_quality(joined: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Human-judged quality, sliced by what each policy decided."""
    labels = [j["human_label"] for j in joined]
    groups = {
        "all_annotated": lambda d: True,
        "scaf_admitted": lambda d: d["scaf_admitted"],
        "scaf_rejected": lambda d: not d["scaf_admitted"],
        "rag2_admitted": lambda d: d["rag2_admitted"],
        "rag2_rejected": lambda d: not d["rag2_admitted"],
        "scaf_only": lambda d: d["scaf_admitted"] and not d["rag2_admitted"],
        "rag2_only": lambda d: d["rag2_admitted"] and not d["scaf_admitted"],
        "both_admit": lambda d: d["scaf_admitted"] and d["rag2_admitted"],
        "neither_admits": lambda d: not d["scaf_admitted"] and not d["rag2_admitted"],
    }
    out: Dict[str, Any] = {
        "overall": _summary(labels),
        "by_group": {name: _summary([j["human_label"] for j in joined if pred(j)])
                     for name, pred in groups.items()},
        "rank_association": {},
    }
    for name, field in SCORE_FIELDS:
        out["rank_association"][name] = {
            "spearman": spearman([j[field] for j in joined], labels),
            "n": len(joined),
        }
    zeros = [j for j in joined if j["human_label"] == 0]
    out["irrelevant_passages"] = {
        "n": len(zeros),
        "scaf_admitted": sum(1 for j in zeros if j["scaf_admitted"]),
        "rag2_admitted": sum(1 for j in zeros if j["rag2_admitted"]),
    }
    out["by_scaf_score_band"] = {}
    for name, lo, hi in (("below_0.50", 0.0, 0.50), ("0.50_to_0.80", 0.50, 0.80),
                         ("0.80_and_above", 0.80, 1.01)):
        band = [j["human_label"] for j in joined if lo <= j["scaf_score"] < hi]
        out["by_scaf_score_band"][name] = _summary(band)
    out["guard"] = (
        "These are descriptive associations on a stratified, non-random sample "
        "of 120 of 600 decisions, clustered within questions, with all four RAG2 "
        "admissions forced in. They are not estimates of a population parameter "
        "and no significance test is appropriate. The sigma association in "
        "particular is between one lexical measure and a human judgement, not an "
        "independent validation of lexical support.")
    return out


def scaf_components(decisions: Sequence[Dict[str, Any]],
                    joined: Sequence[Dict[str, Any]],
                    raw_details: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Whether each SCAF term actually varies and actually discriminates."""
    out: Dict[str, Any] = {"terms": {}}
    for field, name in (("scaf_sigma_support", "support_sigma"),
                        ("scaf_gamma_currency", "currency_gamma"),
                        ("scaf_tau_authority", "authority_tau"),
                        ("scaf_rho_corroboration", "corroboration_rho")):
        values = [d[field] for d in decisions]
        admitted = [d[field] for d in decisions if d["scaf_admitted"]]
        rejected = [d[field] for d in decisions if not d["scaf_admitted"]]
        out["terms"][name] = {
            "distinct_values": len(set(values)),
            "min": round(min(values), 4), "max": round(max(values), 4),
            "median": round(statistics.median(values), 4),
            "mean_when_admitted": round(statistics.mean(admitted), 4) if admitted else None,
            "mean_when_rejected": round(statistics.mean(rejected), 4) if rejected else None,
            "admitted_minus_rejected": (
                round(statistics.mean(admitted) - statistics.mean(rejected), 4)
                if admitted and rejected else None),
            "spearman_with_human_label": spearman(
                [j[field] for j in joined], [j["human_label"] for j in joined]),
        }
    if raw_details:
        out["weights"] = raw_details[0].get("weights")
        out["threshold"] = raw_details[0].get("threshold")
        out["support_method"] = sorted({d.get("support_method") for d in raw_details})
        out["corroboration_status"] = dict(Counter(
            str(d.get("corroboration_status")) for d in raw_details))
        out["gate_fired"] = dict(Counter(bool(d.get("gate")) for d in raw_details))
    out["ablation_variants"] = component_diagnostics(decisions)["variants"]
    out["guard"] = (
        "A term that separates admitted from rejected is not thereby validated: "
        "support carries half the score, so it separates them by construction. "
        "The ablation variants cannot reach 1.0 when a term is removed, so a fixed "
        "0.45 threshold is stricter for them and the counts confound contribution "
        "with reachability. Corroboration is weighted 0.0 and reports "
        "not_implemented, so it is inactive and untested.")
    return out


def matched_context(decisions: Sequence[Dict[str, Any]],
                    annotated: Dict[Any, int]) -> Dict[str, Any]:
    """Each arm's own top-k by its own score -- the only matched comparison the
    frozen decisions support without running a new experiment."""
    budget = matched_budget(decisions)
    by_question: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for d in decisions:
        by_question[d["qid"]].append(d)

    quality: Dict[str, Any] = {}
    for k in (1, 3, 5, 8):
        entry: Dict[str, Any] = {}
        for arm in ("scaf", "rag2"):
            labels = []
            for candidates in by_question.values():
                top = sorted(candidates, key=lambda x: -x[f"{arm}_score"])[:k]
                labels += [annotated[(t["qid"], t["chunk_id"])] for t in top
                           if (t["qid"], t["chunk_id"]) in annotated]
            entry[arm] = _summary(labels)
        quality[f"k={k}"] = entry
    return {
        "overlap": budget["k"],
        "human_quality_of_each_arms_top_k": quality,
        "guard": ("this re-ranks the SAME frozen candidates by each arm's own "
                  "score; it runs no new retrieval and no new generation. Only "
                  "annotated rows carry a human label, so the per-k samples are "
                  "small, unequal between arms, and not a controlled experiment."),
    }


def temporal_feasibility(decisions: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Can this corpus support an old-vs-new temporal counterfactual at all?"""
    years = Counter(str(d["canonical_date"])[:4] for d in decisions if d["canonical_date"])
    gamma = [d["scaf_gamma_currency"] for d in decisions]
    ordered = sorted(years)
    return {
        "candidate_year_distribution": dict(sorted(years.items())),
        "earliest_year": ordered[0] if ordered else None,
        "latest_year": ordered[-1] if ordered else None,
        "candidates_before_2020": sum(v for y, v in years.items() if y < "2020"),
        "candidates_before_2022": sum(v for y, v in years.items() if y < "2022"),
        "gamma_range": [round(min(gamma), 4), round(max(gamma), 4)],
        "gamma_distinct_values": len(set(gamma)),
        "time_sensitive_decisions": dict(Counter(bool(d["time_sensitive"])
                                                 for d in decisions)),
    }


def retraction_census(raw_details: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """What the run actually recorded about retracted or superseded evidence."""
    currency = [d.get("currency_detail") or {} for d in raw_details]
    return {
        "retracted": dict(Counter(str(c.get("retracted")) for c in currency)),
        "supersession": dict(Counter(str(c.get("supersession")) for c in currency)),
        "currency_state": dict(Counter(str(c.get("state")) for c in currency)),
        "date_precision": dict(Counter(str(c.get("date_precision")) for c in currency)),
    }


def examples(joined: Sequence[Dict[str, Any]], per_case: int = 3) -> Dict[str, Any]:
    """Representative rows, chosen by a stated rule rather than by hand."""
    def pick(rows, key, limit=per_case):
        return [{
            "annotation_id": j["annotation_id"], "qid": j["qid"],
            "human_label": j["human_label"],
            "scaf_score": round(j["scaf_score"], 4), "scaf_admitted": j["scaf_admitted"],
            "rag2_score": round(j["rag2_score"], 4), "rag2_admitted": j["rag2_admitted"],
            "sigma": round(j["scaf_sigma_support"], 4),
            "gamma": round(j["scaf_gamma_currency"], 4),
            "tau": round(j["scaf_tau_authority"], 4),
            "canonical_date": j["canonical_date"], "source_category": j["source_category"],
            "question": j["question"],
            "passage_excerpt": " ".join(str(j["candidate_text"]).split())[:240],
        } for j in sorted(rows, key=key)[:limit]]

    return {
        "scaf_admitted_but_human_says_irrelevant": pick(
            [j for j in joined if j["scaf_admitted"] and j["human_label"] == 0],
            lambda j: -j["scaf_score"]),
        "scaf_rejected": pick(
            [j for j in joined if not j["scaf_admitted"]], lambda j: -j["human_label"]),
        "rag2_admitted": pick(
            [j for j in joined if j["rag2_admitted"]], lambda j: -j["rag2_score"], 4),
        "lowest_scaf_score_that_humans_called_clearly_relevant": pick(
            [j for j in joined if j["human_label"] == 2], lambda j: j["scaf_score"]),
        "selection_rule": ("deterministic: highest SCAF score first for the "
                           "admitted-but-irrelevant case, highest human label for "
                           "SCAF rejections, highest RAG2 score for RAG2 "
                           "admissions, lowest SCAF score for underrated evidence. "
                           "No example was chosen by hand."),
    }


# --------------------------------------------------------------------------
def build_results(per_question_path: str, sheet_path: str, manifest_path: str,
                  comparison_manifest_path: str,
                  require_corrected_pass: bool = True) -> Dict[str, Any]:
    """Every section, computed from the frozen artifacts. Reads only."""
    decisions = load_decisions(per_question_path)
    sheet_rows = _read_jsonl(sheet_path)

    passes = {str(r.get("annotation_pass")) for r in sheet_rows}
    if require_corrected_pass and passes != {CORRECTED_PASS_LABEL}:
        raise NotTheCorrectedPass(
            f"annotation_pass is {sorted(passes)}, expected {[CORRECTED_PASS_LABEL]}. "
            "The retired anchored pilot reproduced a displayed lexical suggestion on "
            "every row and cannot be used for scientific interpretation.")

    by_pair = {(d["qid"], d["chunk_id"]): d for d in decisions}
    joined, unmatched = [], []
    for row in sheet_rows:
        key = (row.get("qid"), row.get("chunk_id"))
        decision = by_pair.get(key)
        raw = str(row.get("human_label", "")).strip()
        if decision is None or not raw.isdigit() or int(raw) not in VALID_LABELS:
            unmatched.append(row.get("annotation_id"))
            continue
        joined.append({**decision, **{
            "annotation_id": row.get("annotation_id"),
            "candidate_text": row.get("candidate_text", ""),
            "human_label": int(raw)}})

    raw_details = [d["detail"] for r in _read_jsonl(per_question_path)
                   for d in r["scaf"]["decisions"] if d.get("detail")]
    annotated = {(j["qid"], j["chunk_id"]): j["human_label"] for j in joined}

    return {
        "provenance": provenance(per_question_path, sheet_path, manifest_path,
                                 comparison_manifest_path, decisions, sheet_rows),
        "join": {"annotations": len(sheet_rows), "joined": len(joined),
                 "unmatched": unmatched},
        "admission_behaviour": admission_behaviour(decisions),
        "human_evidence_quality": human_evidence_quality(joined),
        "scaf_components": scaf_components(decisions, joined, raw_details),
        "matched_context": matched_context(decisions, annotated),
        "temporal_feasibility": temporal_feasibility(decisions),
        "retraction_census": retraction_census(raw_details),
        "clustering": question_clustering(joined),
        "examples": examples(joined),
    }
