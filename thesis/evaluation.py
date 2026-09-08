#!/usr/bin/env python3
"""Evaluation protocol, shared by every condition.

One protocol across arms is the other half of a controlled comparison: if the
baseline and the recency arm were scored differently, a difference in the numbers
would not be attributable to the arms.

What is measured depends on what the arm produces. A retrieval-only arm has no
answer, so answer metrics are absent rather than zero -- reporting 0.0 accuracy
for a condition that never attempted an answer would be a fabricated number.

Open-ended generation metrics (ROUGE-L, BERTScore) are not reimplemented here;
``rag2.evaluation`` already has them, defined from the base paper's appendix, and
this module calls them.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Mapping, Optional, Sequence

from .conditions.base import ConditionResult
from .queries import QuerySet


def retrieval_report(results: Sequence[ConditionResult]) -> Dict[str, Any]:
    """Admission behaviour: how much evidence survived, and from where.

    This is the metric every arm has, and the one the recency probe reads: an
    admission-rate shift by source or by date is exactly the effect the thesis
    is looking for.
    """
    admitted = sum(len(r.admitted) for r in results)
    rejected = sum(len(r.rejected) for r in results)
    considered = admitted + rejected
    sources: Counter = Counter()
    for result in results:
        for evidence in result.admitted:
            sources[evidence.source_category or "unknown"] += 1
    return {
        "queries": len(results),
        "evidence_considered": considered,
        "evidence_admitted": admitted,
        "evidence_rejected": rejected,
        "admission_rate": (admitted / considered) if considered else None,
        "mean_admitted_per_query": (admitted / len(results)) if results else 0.0,
        "queries_with_no_evidence": sum(1 for r in results if not r.admitted),
        "admitted_by_source": dict(sources.most_common()),
    }


def answer_report(
    results: Sequence[ConditionResult], query_set: QuerySet, metrics: Sequence[str] = ()
) -> Optional[Dict[str, Any]]:
    """Open-ended answer metrics, or ``None`` when the arm produced no answers."""
    answered = [r for r in results if r.answer]
    if not answered:
        return None

    references = {q.query_id: q.answer for q in query_set if q.answer}
    paired = [(r.answer, references[r.query_id]) for r in answered if r.query_id in references]
    report: Dict[str, Any] = {
        "answered": len(answered),
        "with_reference": len(paired),
    }
    if not paired:
        report["note"] = "no reference answers in the query set; only counts are reported"
        return report

    from ._bootstrap import ensure_rag2_importable

    ensure_rag2_importable()
    from rag2.evaluation import open_ended_metrics

    wanted = [m for m in metrics if m in ("rouge_l", "bertscore")] or ["rouge_l"]
    report.update(
        open_ended_metrics([c for c, _ in paired], [r for _, r in paired], metrics=wanted)
    )
    return report


def evaluate(
    results: Sequence[ConditionResult], query_set: QuerySet, metrics: Sequence[str] = ()
) -> Dict[str, Any]:
    """Score one condition's results under the shared protocol."""
    report: Dict[str, Any] = {
        "protocol": "thesis.evaluation.evaluate",
        "metrics_requested": list(metrics),
        "retrieval": retrieval_report(results),
    }
    answers = answer_report(results, query_set, metrics)
    if answers is not None:
        report["answers"] = answers
    else:
        report["answers"] = None
        report["answers_note"] = (
            "this condition produced no answers; answer metrics are absent rather than zero"
        )
    return report


def compare(reports: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    """Put two or more arms' reports side by side.

    Refuses to compare arms that did not see the same evidence population: a
    difference between arms scoring different candidates is not a finding about
    admission policy.
    """
    digests = {
        name: report.get("candidate_digest") for name, report in reports.items()
    }
    distinct = {d for d in digests.values() if d}
    comparable = len(distinct) <= 1
    return {
        "arms": list(reports),
        "candidate_digests": digests,
        "same_evidence_population": comparable,
        "warning": None if comparable else (
            "arms scored different candidate sets, so differences between them are not "
            "attributable to admission policy. Replay one candidate set across arms "
            "(retrieval.replay_from)."
        ),
        "admission_rate": {
            name: (report.get("retrieval") or {}).get("admission_rate")
            for name, report in reports.items()
        },
    }
