"""Evidence-quality validation: do higher machine scores mean better evidence?

The completed RAG2-vs-SCAF comparison measured *how much* evidence each policy
admits. The threshold sweep then showed that quantity is governed by where each
decision boundary happens to fall in its own score distribution, and is
reversible by moving that boundary. Neither result says anything about whether
either policy ranks *better* evidence higher, because the repository contains no
evidence-quality signal at all: no gold answers, no relevance labels, no expert
annotations.

This module builds that missing signal from one human pass over a stratified
sample of the 600 saved decisions, and then tests the only question that decides
whether SCAF is a viable extension rather than a differently-calibrated one:

    does a higher SCAF score correspond to better human-judged evidence?

Two design decisions worth stating plainly
------------------------------------------
**The annotation file is blind.** Machine scores are split into a separate key
file and joined back by ``annotation_id`` afterwards. Showing an annotator the
SCAF score beside the passage would contaminate the very signal being collected
-- the labels would partly measure the annotator's deference to the score. The
machine values are preserved in full; they are simply not visible while judging.

**Score is analysed separately from admission.** A threshold changes the
admission rate without changing the underlying ranking, so the primary analysis
is the rank association between the continuous score and the human label.
Admission is reported as a secondary, threshold-dependent view.

Nothing here labels anything automatically. No function in this module writes a
value into ``human_label``.
"""

from __future__ import annotations

import json
import math
import os
import random
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

#: Bump when the label definitions change; recorded in every artifact.
SCHEMA_VERSION = "evidence-quality-v1"

#: The ordinal scale the annotator fills in. Deliberately three points: a finer
#: scale costs agreement without adding resolution at n=120.
LABELS: Dict[int, str] = {
    0: "Not relevant -- does not help answer the question",
    1: "Partially relevant -- weak, indirect or background support",
    2: "Clearly relevant -- useful supporting evidence",
}

VALID_LABELS = frozenset(LABELS)


# --------------------------------------------------------------------------
# Reading the completed comparison
# --------------------------------------------------------------------------
def load_decisions(per_question_path: str) -> List[Dict[str, Any]]:
    """Flatten the saved comparison into one record per (question, candidate).

    Both arms scored the identical frozen candidates in identical order, so the
    two decision lists are joined on ``chunk_id``. A mismatch means the file is
    not the paired artifact it claims to be, and is raised rather than patched.
    """
    records: List[Dict[str, Any]] = []
    with open(per_question_path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            scaf_by_id = {d["chunk_id"]: d for d in row["scaf"]["decisions"]}
            if sorted(scaf_by_id) != sorted(d["chunk_id"] for d in row["rag2"]["decisions"]):
                raise ValueError(
                    f"{row['qid']}: the two arms did not score the same chunk ids; "
                    "this file is not a paired comparison")
            for rag2 in row["rag2"]["decisions"]:
                scaf = scaf_by_id[rag2["chunk_id"]]
                detail = scaf.get("detail") or {}
                records.append({
                    "qid": row["qid"],
                    "question": row["question"],
                    "time_sensitive": row.get("time_sensitive"),
                    "chunk_id": rag2["chunk_id"],
                    "rerank_rank": rag2.get("rerank_rank"),
                    "source_category": rag2.get("source_category", ""),
                    "canonical_date": rag2.get("canonical_date", ""),
                    "authority_tier_label": rag2.get("authority_tier_label", ""),
                    "rag2_score": float(rag2["score"]),
                    "rag2_admitted": bool(rag2["keep"]),
                    "scaf_score": float(scaf["score"]),
                    "scaf_admitted": bool(scaf["keep"]),
                    "scaf_sigma_support": _num(detail.get("sigma_support")),
                    "scaf_gamma_currency": _num(detail.get("gamma_currency")),
                    "scaf_tau_authority": _num(detail.get("tau_authority")),
                    "scaf_rho_corroboration": _num(detail.get("rho_corroboration")),
                })
    return records


def _num(value: Any) -> Optional[float]:
    return None if value is None else float(value)


def attach_candidate_text(records: Sequence[Dict[str, Any]],
                          frozen_path: str) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Join candidate text in from the frozen candidate set.

    ``per_question.jsonl`` records chunk ids, not passages, so the frozen file is
    the only place the text lives. It is read, never written. Returns the
    enriched records and the ids that could not be resolved -- an unresolved id
    means the frozen file is not the one the comparison ran against, which the
    caller must treat as a blocker rather than annotate around.
    """
    text_by_id: Dict[str, Dict[str, Any]] = {}
    with open(frozen_path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            frozen = json.loads(line)
            for candidate in frozen.get("candidates", []):
                text_by_id[str(candidate.get("chunk_id", ""))] = candidate

    enriched: List[Dict[str, Any]] = []
    missing: List[str] = []
    for record in records:
        candidate = text_by_id.get(record["chunk_id"])
        if candidate is None:
            missing.append(record["chunk_id"])
            continue
        enriched.append({**record,
                         "candidate_text": str(candidate.get("text", "")),
                         "title": str(candidate.get("title", "")),
                         "pmid": str(candidate.get("pmid", "")),
                         "pmcid": str(candidate.get("pmcid", ""))})
    return enriched, missing


# --------------------------------------------------------------------------
# Sampling -- the rule is fixed here, before any label exists
# --------------------------------------------------------------------------
def tertile_edges(values: Sequence[float]) -> Tuple[float, float]:
    """The two cut points splitting ``values`` into thirds.

    Computed over **all 600 decisions**, not over the sample, and fixed before
    annotation begins. Choosing bins after seeing labels would let the bin
    boundaries manufacture a trend.
    """
    ordered = sorted(values)
    if not ordered:
        return (0.0, 0.0)
    return (ordered[len(ordered) // 3], ordered[2 * len(ordered) // 3])


def bin_of(value: float, edges: Tuple[float, float]) -> str:
    low, high = edges
    if value < low:
        return "low"
    return "medium" if value < high else "high"


def stratified_sample(records: Sequence[Dict[str, Any]], size: int = 120,
                      seed: int = 42) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Sample across the score range of both arms, deterministically.

    The stratification rule, declared in advance:

    1. every decision RAG2 admitted is included -- there are only a handful, and
       the RAG2-admits/SCAF-rejects cell is otherwise unobservable;
    2. the remainder is drawn evenly across the nine (SCAF tertile x RAG2
       tertile) cells, so the sample spans both score ranges rather than
       clustering where the policies happen to agree;
    3. cells are filled round-robin, so an under-populated cell cannot starve
       the others;
    4. ordering is by ``annotation_id``, which is assigned after shuffling with
       ``seed``, so the annotator meets the rows in an order uncorrelated with
       either score.
    """
    rng = random.Random(seed)
    scaf_edges = tertile_edges([r["scaf_score"] for r in records])
    rag2_edges = tertile_edges([r["rag2_score"] for r in records])

    forced = [r for r in records if r["rag2_admitted"]]
    forced_ids = {(r["qid"], r["chunk_id"]) for r in forced}

    cells: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for record in records:
        if (record["qid"], record["chunk_id"]) in forced_ids:
            continue
        cells[(bin_of(record["scaf_score"], scaf_edges),
               bin_of(record["rag2_score"], rag2_edges))].append(record)
    for pool in cells.values():
        pool.sort(key=lambda r: (r["qid"], r["chunk_id"]))
        rng.shuffle(pool)

    chosen: List[Dict[str, Any]] = list(forced)
    order = sorted(cells)
    while len(chosen) < size and any(cells[key] for key in order):
        for key in order:
            if len(chosen) >= size:
                break
            if cells[key]:
                chosen.append(cells[key].pop())

    rng.shuffle(chosen)
    for index, record in enumerate(chosen, start=1):
        record["annotation_id"] = f"eq-{index:04d}"
        record["scaf_bin"] = bin_of(record["scaf_score"], scaf_edges)
        record["rag2_bin"] = bin_of(record["rag2_score"], rag2_edges)

    provenance = {
        "schema_version": SCHEMA_VERSION,
        "requested_size": size,
        "sampled": len(chosen),
        "seed": seed,
        "population": len(records),
        "scaf_tertile_edges": [round(e, 6) for e in scaf_edges],
        "rag2_tertile_edges": [round(e, 6) for e in rag2_edges],
        "forced_rag2_admitted": len(forced),
        "cell_counts": {f"scaf_{a}|rag2_{b}": sum(
            1 for r in chosen if r["scaf_bin"] == a and r["rag2_bin"] == b)
            for a in ("low", "medium", "high") for b in ("low", "medium", "high")},
        "rule": "all RAG2 admissions, then round-robin over 9 SCAF x RAG2 tertile "
                "cells; bins from tertiles of all 600 decisions, fixed before "
                "annotation",
    }
    return chosen, provenance


# --------------------------------------------------------------------------
# The annotation file (blind) and its key
# --------------------------------------------------------------------------
BLIND_FIELDS = ("annotation_id", "qid", "question", "chunk_id", "title",
                "candidate_text", "canonical_date", "source_category",
                "human_label", "human_notes")


def split_blind_and_key(records: Sequence[Dict[str, Any]]
                        ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Separate what the annotator sees from what the machine decided.

    The blind rows carry the passage, its date and source -- everything needed to
    judge relevance -- and an empty ``human_label``. Every machine value is kept,
    in the key, joined by ``annotation_id``.
    """
    blind, key = [], []
    for record in records:
        blind.append({field: ("" if field in ("human_label", "human_notes")
                              else record.get(field, ""))
                      for field in BLIND_FIELDS})
        key.append({k: v for k, v in record.items()
                    if k not in ("question", "candidate_text", "title",
                                 "human_label", "human_notes")})
    return blind, key


# --------------------------------------------------------------------------
# Completeness and validity -- missing is never silently a zero
# --------------------------------------------------------------------------
def check_annotations(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Report completeness. Missing labels are counted, never imputed."""
    total = len(rows)
    seen: Counter = Counter(str(r.get("annotation_id", "")) for r in rows)
    duplicates = sorted(a for a, n in seen.items() if n > 1)

    completed, missing, invalid = [], [], []
    for row in rows:
        raw = str(row.get("human_label", "")).strip()
        if not raw:
            missing.append(row.get("annotation_id"))
            continue
        try:
            value = int(raw)
        except ValueError:
            invalid.append((row.get("annotation_id"), raw))
            continue
        (completed if value in VALID_LABELS else invalid).append(
            row.get("annotation_id") if value in VALID_LABELS else (row.get("annotation_id"), raw))

    return {
        "schema_version": SCHEMA_VERSION,
        "total_rows": total,
        "completed": len(completed),
        "missing": len(missing),
        "invalid": len(invalid),
        "duplicate_annotation_ids": duplicates,
        "missing_ids": [m for m in missing][:20],
        "invalid_entries": invalid[:20],
        "ready_for_analysis": total > 0 and not missing and not invalid and not duplicates,
        "label_distribution": dict(Counter(
            int(str(r.get("human_label", "")).strip())
            for r in rows if str(r.get("human_label", "")).strip().isdigit()
            and int(str(r.get("human_label")).strip()) in VALID_LABELS)),
    }


# --------------------------------------------------------------------------
# Statistics -- small, interpretable, dependency-free
# --------------------------------------------------------------------------
def _ranks(values: Sequence[float]) -> List[float]:
    """Average ranks, so ties do not distort the correlation."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    position = 0
    while position < len(order):
        end = position
        while end + 1 < len(order) and values[order[end + 1]] == values[order[position]]:
            end += 1
        shared = (position + end) / 2 + 1
        for index in range(position, end + 1):
            ranks[order[index]] = shared
        position = end + 1
    return ranks


def spearman(x: Sequence[float], y: Sequence[float]) -> Optional[float]:
    """Spearman rank correlation. ``None`` when it is undefined.

    Rank-based because the human label is ordinal and the machine scores are not
    linearly comparable to it -- only their ordering is.
    """
    if len(x) != len(y) or len(x) < 3:
        return None
    rx, ry = _ranks(x), _ranks(y)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return None if den == 0 else round(num / den, 4)


def _summary(values: Sequence[float]) -> Dict[str, Any]:
    if not values:
        return {"n": 0}
    ordered = sorted(values)
    middle = len(ordered) // 2
    return {
        "n": len(ordered),
        "mean": round(sum(ordered) / len(ordered), 4),
        "median": float(ordered[middle] if len(ordered) % 2
                        else (ordered[middle - 1] + ordered[middle]) / 2),
        "min": ordered[0], "max": ordered[-1],
    }


def suggestion_anchoring(rows: Sequence[Dict[str, Any]],
                         human: Sequence[int]) -> Dict[str, Any]:
    """How far the human labels track the suggestion the annotator was shown.

    This is not a curiosity, it is the main threat to the headline result. The
    interface's suggestion is lexical overlap between question and passage;
    SCAF's support term sigma is *also* lexical overlap. If the annotator largely
    pressed whatever the suggestion said, then a sigma-versus-human correlation
    is partly an artefact of the interface rather than a measurement of SCAF, and
    since sigma is the term carrying SCAF's overall score, so is that.

    Nothing here can *remove* the anchoring; it can only make its size visible.
    Read a high ``agreement_rate`` as a warning about the labels, never as
    confirmation that the suggestion was right.

    ``shown`` and ``hidden`` are split on ``ai_suggestion_shown``, which the
    interface writes on every row it saves. Rows annotated before that field
    existed, or by hand, report under ``unknown``.
    """
    paired = [(r, h) for r, h in zip(rows, human)
              if str(r.get("ai_suggested_label", "")).strip().isdigit()]
    out: Dict[str, Any] = {
        "rows_with_a_recorded_suggestion": len(paired),
        "rows_without_a_recorded_suggestion": len(rows) - len(paired),
    }
    if not paired:
        out["note"] = ("no ai_suggested_label on any row: either the sheet was "
                       "filled in by hand, or it predates the annotation "
                       "interface. Anchoring cannot be assessed from this file.")
        return out

    def _shown(record: Dict[str, Any]) -> str:
        value = record.get("ai_suggestion_shown")
        return {True: "shown", False: "hidden"}.get(value, "unknown")

    groups: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
    for record, label in paired:
        groups[_shown(record)].append(
            (int(str(record["ai_suggested_label"]).strip()), label))
    groups["all"] = [pair for key in ("shown", "hidden", "unknown")
                     for pair in groups.get(key, [])]

    for key in ("all", "shown", "hidden", "unknown"):
        pairs = groups.get(key) or []
        if not pairs:
            continue
        agree = sum(1 for a, h in pairs if a == h)
        out[key] = {
            "n": len(pairs),
            "agreement_rate": round(agree / len(pairs), 4),
            "mean_signed_difference": round(
                sum(h - a for a, h in pairs) / len(pairs), 4),
            "human_above_suggestion": sum(1 for a, h in pairs if h > a),
            "human_below_suggestion": sum(1 for a, h in pairs if h < a),
            "spearman_suggestion_vs_human": spearman([a for a, _ in pairs],
                                                     [h for _, h in pairs]),
            "suggestion_distribution": dict(sorted(Counter(
                a for a, _ in pairs).items())),
        }

    out["guard"] = (
        "A high agreement rate does NOT validate the suggestion rule. It means "
        "the labels are not independent of it, which inflates any association "
        "between the human label and SCAF's lexical support term. Rows annotated "
        "with the suggestion hidden are the only ones free of this effect; if "
        "there are none, the whole sample carries it.")
    return out


def question_clustering(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """How the sampled passages group by question.

    Passages drawn from the same question are not independent observations: they
    share a query, a retrieval, and whatever makes that question easy or hard.
    The correlations reported here treat all rows as exchangeable, so the
    effective sample size is nearer the number of questions than the number of
    rows. This block records the sizes so a reader can see the gap rather than
    having to assume it away.
    """
    per_question = Counter(str(r.get("qid", "")) for r in rows)
    sizes = sorted(per_question.values())
    return {
        "rows": len(rows),
        "questions_represented": len(per_question),
        "rows_per_question": {"min": sizes[0] if sizes else 0,
                              "max": sizes[-1] if sizes else 0,
                              "mean": round(sum(sizes) / len(sizes), 4) if sizes else 0},
        "guard": ("rows within a question are not independent; the reported n is "
                  "a count of passages, not of independent observations, and no "
                  "clustering correction is applied"),
    }


def analyse(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """The evidence-quality analysis. Requires completed labels.

    Reports rank association first (does the *score* rank better evidence
    higher?) and admission second (does the *threshold* separate better
    evidence?), because those are different claims and only the first is a
    property of the policy rather than of its operating point.
    """
    labelled = [r for r in rows if str(r.get("human_label", "")).strip().isdigit()
                and int(str(r["human_label"]).strip()) in VALID_LABELS]
    if len(labelled) < 3:
        return {"error": "fewer than 3 valid labels; nothing can be estimated",
                "labelled": len(labelled)}

    human = [int(str(r["human_label"]).strip()) for r in labelled]
    out: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "labelled_rows": len(labelled),
        "human_label_distribution": dict(sorted(Counter(human).items())),
        "rank_association_with_human_label": {},
        "by_score_bin": {},
        "by_admission": {},
        "disagreement_cases": {},
    }

    # -- A/B/C/D: does a higher score mean a better passage? ----------------
    for name, field in (("scaf_score", "scaf_score"),
                        ("scaf_sigma_support", "scaf_sigma_support"),
                        ("scaf_gamma_currency", "scaf_gamma_currency"),
                        ("scaf_tau_authority", "scaf_tau_authority"),
                        ("rag2_score", "rag2_score")):
        pairs = [(float(r[field]), h) for r, h in zip(labelled, human)
                 if r.get(field) is not None]
        out["rank_association_with_human_label"][name] = {
            "spearman": spearman([p[0] for p in pairs], [p[1] for p in pairs]),
            "n": len(pairs),
        }

    # -- score bins: predeclared tertiles, assigned at export time ----------
    for arm in ("scaf", "rag2"):
        buckets: Dict[str, List[int]] = defaultdict(list)
        for record, label in zip(labelled, human):
            buckets[str(record.get(f"{arm}_bin", "unknown"))].append(label)
        out["by_score_bin"][arm] = {
            level: _summary(buckets[level]) for level in ("low", "medium", "high")
            if buckets.get(level)}

    # -- admission: the threshold-dependent view ---------------------------
    for arm in ("scaf", "rag2"):
        admitted = [h for r, h in zip(labelled, human) if r.get(f"{arm}_admitted")]
        rejected = [h for r, h in zip(labelled, human) if not r.get(f"{arm}_admitted")]
        out["by_admission"][arm] = {"admitted": _summary(admitted),
                                    "rejected": _summary(rejected)}

    # -- G: where the two policies disagree --------------------------------
    for name, predicate in (
        ("scaf_admits_rag2_rejects", lambda r: r.get("scaf_admitted") and not r.get("rag2_admitted")),
        ("rag2_admits_scaf_rejects", lambda r: r.get("rag2_admitted") and not r.get("scaf_admitted")),
        ("both_admit", lambda r: r.get("scaf_admitted") and r.get("rag2_admitted")),
        ("neither_admits", lambda r: not r.get("scaf_admitted") and not r.get("rag2_admitted")),
    ):
        out["disagreement_cases"][name] = _summary(
            [h for r, h in zip(labelled, human) if predicate(r)])

    out["suggestion_anchoring"] = suggestion_anchoring(labelled, human)
    out["question_clustering"] = question_clustering(labelled)

    out["interpretation_guard"] = (
        "Rank association is the primary result: it asks whether the continuous "
        "score orders evidence as a human does, independently of any threshold. "
        "The admission rows are threshold-dependent and change if the threshold "
        "moves. Neither establishes answer quality, temporal bias or clinical "
        "appropriateness, none of which is measured here.")
    return out


# --------------------------------------------------------------------------
# Offline diagnostics -- no rerun, no annotation needed
# --------------------------------------------------------------------------
def matched_budget(records: Sequence[Dict[str, Any]], ks: Sequence[int] = (1, 3, 5, 8)
                   ) -> Dict[str, Any]:
    """Admission and overlap when both arms are capped at the same k.

    The proposal's fairness guarantee 2 caps every arm at five passages. The
    completed run did not apply that cap, so this recomputes it from the saved
    rankings: within each question, each arm's own top-k by its own score.
    Diagnostic only -- it says nothing about evidence quality.
    """
    by_qid: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_qid[record["qid"]].append(record)

    out: Dict[str, Any] = {"note": "top-k by each arm's own score, same frozen "
                                   "candidates and ordering; no rerun", "k": {}}
    for k in ks:
        overlaps, both = [], 0
        for candidates in by_qid.values():
            scaf = {r["chunk_id"] for r in sorted(
                candidates, key=lambda r: -r["scaf_score"])[:k]}
            rag2 = {r["chunk_id"] for r in sorted(
                candidates, key=lambda r: -r["rag2_score"])[:k]}
            union = scaf | rag2
            if union:
                overlaps.append(len(scaf & rag2) / len(union))
            both += len(scaf & rag2)
        out["k"][str(k)] = {
            "questions": len(by_qid),
            "passages_per_arm_per_question": k,
            "mean_jaccard": round(sum(overlaps) / len(overlaps), 4) if overlaps else None,
            "shared_passages_total": both,
            "possible_total": k * len(by_qid),
        }
    return out


def component_diagnostics(records: Sequence[Dict[str, Any]], threshold: float = 0.45,
                          weights: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
    """Admission under reduced SCAF weight vectors, recomputed from saved sigma/gamma/tau.

    **Diagnostics, not validated ablations**, for three reasons:

    1. Two of the three terms are weak on this corpus. gamma has almost no
       dynamic range (very few pre-2020 documents) and tau is largely a
       source-category constant. Removing a term with no range measures nothing
       about the model it implements.
    2. rho is not implemented (w=0) and is absent from every variant, so no
       variant is the policy the proposal specifies.
    3. **The variants do not share a score range**, so one fixed threshold does
       not mean the same thing across them: dropping currency caps the maximum
       achievable score at 0.8 and dropping authority caps it at 0.7, which makes
       the same 0.45 progressively stricter. Admission differences between
       variants therefore confound "this term contributed" with "this variant
       could not reach the threshold". Compare rankings, not counts.
    """
    weights = weights or {"support": 0.5, "currency": 0.3, "authority": 0.2}
    variants = {
        "support_only": {"support": 1.0, "currency": 0.0, "authority": 0.0},
        "support_currency": {"support": weights["support"], "currency": weights["currency"],
                             "authority": 0.0},
        "support_authority": {"support": weights["support"], "currency": 0.0,
                              "authority": weights["authority"]},
        "full_scaf": dict(weights),
    }
    out: Dict[str, Any] = {"threshold": threshold, "caveat":
                           "offline diagnostic; rho is not implemented (w=0) and is "
                           "absent from every variant; variants have different "
                           "maximum achievable scores, so a fixed threshold is "
                           "stricter for the reduced ones -- see max_possible_score",
                           "variants": {}}
    for name, weight in variants.items():
        admitted = 0
        for record in records:
            score = (weight["support"] * (record.get("scaf_sigma_support") or 0.0)
                     + weight["currency"] * (record.get("scaf_gamma_currency") or 0.0)
                     + weight["authority"] * (record.get("scaf_tau_authority") or 0.0))
            admitted += score >= threshold
        out["variants"][name] = {
            "weights": weight, "admitted": admitted, "of": len(records),
            "max_possible_score": round(sum(weight.values()), 4),
            "rate": round(admitted / len(records), 6) if records else 0.0}
    return out


def time_sensitivity_split(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Admission split by the question set's own ``time_sensitive`` flag.

    The flag is read from the committed question set, never inferred.
    """
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[str(record.get("time_sensitive"))].append(record)
    out: Dict[str, Any] = {"note": "flag taken from the committed question set; "
                                   "no classification invented here", "groups": {}}
    for key, items in sorted(groups.items()):
        out["groups"][key] = {
            "questions": len({r["qid"] for r in items}),
            "decisions": len(items),
            "scaf_admitted": sum(1 for r in items if r["scaf_admitted"]),
            "rag2_admitted": sum(1 for r in items if r["rag2_admitted"]),
            "mean_scaf_score": round(sum(r["scaf_score"] for r in items) / len(items), 4),
            "mean_rag2_score": round(sum(r["rag2_score"] for r in items) / len(items), 4),
            "mean_gamma_currency": round(sum((r.get("scaf_gamma_currency") or 0.0)
                                             for r in items) / len(items), 4),
        }
    return out


def qualitative_cases(records: Sequence[Dict[str, Any]], per_case: int = 2
                      ) -> Dict[str, List[Dict[str, Any]]]:
    """Representative decisions under a transparent, symmetric selection rule.

    Every category takes the extreme cases by score, and the categories are
    deliberately symmetric between the arms so the selection cannot favour
    either. Nothing here is evidence of quality -- these are cases to *read*.
    """
    def top(items: Sequence[Dict[str, Any]], key, reverse=True):
        return [{k: r[k] for k in ("qid", "chunk_id", "scaf_score", "rag2_score",
                                   "scaf_admitted", "rag2_admitted",
                                   "source_category", "canonical_date")}
                for r in sorted(items, key=key, reverse=reverse)[:per_case]]

    scaf_only = [r for r in records if r["scaf_admitted"] and not r["rag2_admitted"]]
    rag2_only = [r for r in records if r["rag2_admitted"] and not r["scaf_admitted"]]
    both = [r for r in records if r["scaf_admitted"] and r["rag2_admitted"]]
    neither = [r for r in records if not r["scaf_admitted"] and not r["rag2_admitted"]]
    return {
        "rule": [f"highest and lowest scoring {per_case} per category; categories "
                 "are symmetric between arms"],
        "scaf_admits_rag2_rejects_highest_scaf": top(scaf_only, lambda r: r["scaf_score"]),
        "scaf_admits_rag2_rejects_lowest_scaf": top(scaf_only, lambda r: r["scaf_score"], False),
        "rag2_admits_scaf_rejects": top(rag2_only, lambda r: r["rag2_score"]),
        "both_admit_highest_scaf": top(both, lambda r: r["scaf_score"]),
        "neither_admits_highest_scaf": top(neither, lambda r: r["scaf_score"]),
        "neither_admits_highest_rag2": top(neither, lambda r: r["rag2_score"]),
    }
