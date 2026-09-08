#!/usr/bin/env python3
"""Experimental conditions: the arms of the controlled comparison.

A condition decides **which evidence is admitted and how an answer is produced**.
It does not decide what evidence exists -- that is the corpus -- nor which
candidates retrieval returns. Holding those fixed while varying only admission is
what makes the comparison controlled, and the architecture enforces it
structurally: every condition receives an already-retrieved
:class:`~thesis.retrieval.RetrievalResult` and cannot retrieve for itself.

    corpus (frozen, digest-checked)
      -> retrieval (exact, balanced, replayable)      <- identical across arms
        -> temporal policy                            <- 'none' for the baseline
          -> CONDITION: admission + generation        <- the only thing that varies
            -> ConditionResult (+ provenance)

Conditions are registered by name and selected by config, so adding SCAF later
means registering one more -- not restructuring anything.
"""

from __future__ import annotations

import abc
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

from ..config import ThesisConfig
from ..queries import ResearchQuery
from ..recency import TemporalPolicy, build_policy
from ..retrieval import RetrievalResult


@dataclass
class AdmittedEvidence:
    """One piece of evidence a condition admitted, with why it was admitted."""

    chunk_id: str
    document_id: str
    source_category: str
    text: str
    rank: int
    retrieval_score: Optional[float] = None
    rerank_score: Optional[float] = None
    admission_score: Optional[float] = None
    admission_label: str = ""
    #: Corpus provenance, carried verbatim. The baseline never reads it.
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_candidate(cls, candidate: Mapping[str, Any], rank: int, **extra: Any) -> "AdmittedEvidence":
        consumed = {
            "chunk_id", "document_id", "source_category", "text",
            "retrieval_score", "rerank_score", "retrieval_rank",
        }
        return cls(
            chunk_id=str(candidate.get("chunk_id", "")),
            document_id=str(candidate.get("document_id", "")),
            source_category=str(candidate.get("source_category", "")),
            text=str(candidate.get("text", "")),
            rank=rank,
            retrieval_score=candidate.get("retrieval_score"),
            rerank_score=candidate.get("rerank_score"),
            metadata={k: v for k, v in candidate.items() if k not in consumed},
            **extra,
        )


@dataclass
class ConditionResult:
    """What one condition produced for one query."""

    query_id: str
    condition: str
    query: str = ""
    admitted: List[AdmittedEvidence] = field(default_factory=list)
    rejected: List[AdmittedEvidence] = field(default_factory=list)
    answer: str = ""
    answer_source: str = ""
    candidate_digest: str = ""
    retrieval_fingerprint: str = ""
    temporal_policy: str = "none"
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query_id": self.query_id,
            "condition": self.condition,
            "query": self.query,
            "admitted": [e.to_dict() for e in self.admitted],
            "rejected": [e.to_dict() for e in self.rejected],
            "answer": self.answer,
            "answer_source": self.answer_source,
            "candidate_digest": self.candidate_digest,
            "retrieval_fingerprint": self.retrieval_fingerprint,
            "temporal_policy": self.temporal_policy,
            "diagnostics": self.diagnostics,
        }


class ExperimentCondition(abc.ABC):
    """One arm of the comparison."""

    name: str = ""
    #: Whether this arm generates an answer, or stops at evidence admission.
    generates: bool = False

    def __init__(self, config: ThesisConfig, policy: Optional[TemporalPolicy] = None) -> None:
        self.config = config
        self.policy = policy if policy is not None else build_policy(
            config.recency.policy, config.recency.options
        )

    @abc.abstractmethod
    def run(self, query: ResearchQuery, retrieved: RetrievalResult) -> ConditionResult:
        """Admit evidence (and optionally answer) for one already-retrieved query."""

    # -- shared helpers ---------------------------------------------------
    def apply_policy(self, retrieved: RetrievalResult) -> List[Dict[str, Any]]:
        """Run the configured temporal policy over the candidate set.

        Every condition goes through this, including the baseline -- whose policy
        is the identity. Routing all arms through one call keeps "the temporal
        policy is the only difference" true by construction.
        """
        return self.policy.apply(
            retrieved.top(self.config.retrieval.final_top_k),
            context={"query_id": retrieved.query_id},
        )

    def describe(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "class": type(self).__name__,
            "generates": self.generates,
            "temporal_policy": self.policy.describe(),
        }


_REGISTRY: Dict[str, Callable[..., ExperimentCondition]] = {}


def register_condition(key: str):
    def decorator(factory: Callable[..., ExperimentCondition]):
        if key in _REGISTRY:
            raise ValueError(f"condition {key!r} already registered")
        _REGISTRY[key] = factory
        return factory

    return decorator


def available_conditions() -> List[str]:
    _load_builtin()
    return sorted(_REGISTRY)


def build_condition(
    config: ThesisConfig, policy: Optional[TemporalPolicy] = None
) -> ExperimentCondition:
    _load_builtin()
    name = config.condition.name
    if name not in _REGISTRY:
        raise KeyError(f"unknown condition {name!r}; registered: {sorted(_REGISTRY)}")
    return _REGISTRY[name](config, policy)


def _load_builtin() -> None:
    from . import rag2_condition, recency_aware, retrieval_only  # noqa: F401  (registration)
