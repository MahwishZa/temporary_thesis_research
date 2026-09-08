#!/usr/bin/env python3
"""Baseline condition: retrieval only, no evidence filter.

This is the control arm. It admits the top-k candidates exactly as retrieval
ordered them and produces no answer, so any difference a later arm shows is
attributable to that arm's admission policy rather than to retrieval.

It is deliberately temporally blind. Provenance -- including
``canonical_date`` -- is carried into every admitted record so downstream
analysis can use it, and this module never reads it.
``thesis/tests/test_condition_isolation.py`` enforces that at token level.
"""

from __future__ import annotations

from typing import Any, Dict

from ..queries import ResearchQuery
from ..retrieval import RetrievalResult
from .base import AdmittedEvidence, ConditionResult, ExperimentCondition, register_condition


class RetrievalOnlyCondition(ExperimentCondition):
    """Admit the reranked top-k; do not filter, do not generate."""

    name = "baseline"
    generates = False

    def run(self, query: ResearchQuery, retrieved: RetrievalResult) -> ConditionResult:
        candidates = self.apply_policy(retrieved)
        admitted = [
            AdmittedEvidence.from_candidate(candidate, rank=index + 1)
            for index, candidate in enumerate(candidates)
        ]
        return ConditionResult(
            query_id=query.query_id,
            condition=self.name,
            query=retrieved.query,
            admitted=admitted,
            rejected=[],
            answer="",
            answer_source="none (baseline admits evidence but does not generate)",
            candidate_digest=retrieved.candidate_digest,
            retrieval_fingerprint=retrieved.retrieval_fingerprint,
            temporal_policy=self.policy.name,
            diagnostics={
                "candidates_seen": len(retrieved.candidates),
                "final_top_k": self.config.retrieval.final_top_k,
                "admitted": len(admitted),
                "admitted_by_source": _by_source(admitted),
            },
        )


def _by_source(admitted) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for evidence in admitted:
        counts[evidence.source_category or "unknown"] = (
            counts.get(evidence.source_category or "unknown", 0) + 1
        )
    return dict(sorted(counts.items()))


@register_condition("baseline")
def _build(config, policy=None) -> ExperimentCondition:
    return RetrievalOnlyCondition(config, policy)
