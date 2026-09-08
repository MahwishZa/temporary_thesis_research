#!/usr/bin/env python3
"""The evaluation query set: one population, shared by every condition.

A controlled comparison needs the arms to differ in exactly one thing. Two of the
three inputs are already pinned -- the corpus is frozen and digest-checked, the
retrieval is exact and replayable -- and this module pins the third.

Format is JSONL, one object per line::

    {"query_id": "Q001",
     "query": "Does lecanemab slow cognitive decline in early Alzheimer's disease?",
     "metadata": {"topic": "disease-modifying therapy"}}

``answer`` and ``references`` are optional and used only by evaluation; a query
set with neither is still valid and yields retrieval-level metrics only.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence

from .provenance import stable_hash


@dataclass
class ResearchQuery:
    """One evaluation query. Identity is ``query_id`` and it is never rewritten."""

    query_id: str
    query: str
    answer: Optional[str] = None
    references: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any], index: int = 0) -> "ResearchQuery":
        text = payload.get("query", payload.get("question", payload.get("text", "")))
        if not str(text).strip():
            raise ValueError(f"query record {index} has no query text")
        return cls(
            query_id=str(payload.get("query_id", payload.get("id", f"q{index:04d}"))),
            query=str(text),
            answer=payload.get("answer"),
            references=list(payload.get("references", []) or []),
            metadata=dict(payload.get("metadata", {}) or {}),
        )


@dataclass
class QuerySet:
    """An ordered, identified set of queries.

    ``digest`` covers the ids and texts, so two runs claiming the same query set
    can be shown to have used it. Order is preserved as written: retrieval is
    deterministic, and so is the sequence it is applied in.
    """

    queries: List[ResearchQuery] = field(default_factory=list)
    name: str = ""
    version: str = ""
    path: str = ""

    def __iter__(self) -> Iterator[ResearchQuery]:
        return iter(self.queries)

    def __len__(self) -> int:
        return len(self.queries)

    @property
    def digest(self) -> str:
        return stable_hash([(q.query_id, q.query) for q in self.queries])

    def describe(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "path": self.path,
            "size": len(self.queries),
            "digest": self.digest,
        }


def load_query_set(
    path: str, name: str = "", version: str = "", limit: Optional[int] = None
) -> QuerySet:
    """Read a JSONL (or JSON list) query file."""
    if not path:
        raise ValueError("queries.path is empty: an experimental condition needs a query set")
    if not os.path.exists(path):
        raise FileNotFoundError(f"query set not found: {path}")

    records: List[Mapping[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        if path.endswith(".jsonl") or path.endswith(".ndjson"):
            for line in handle:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        else:
            payload = json.load(handle)
            records = payload["queries"] if isinstance(payload, Mapping) else payload

    queries = [ResearchQuery.from_dict(r, i) for i, r in enumerate(records)]
    seen: Dict[str, int] = {}
    for index, query in enumerate(queries):
        if query.query_id in seen:
            raise ValueError(
                f"duplicate query_id {query.query_id!r} at records {seen[query.query_id]} "
                f"and {index}: ids must be unique or results cannot be joined across arms"
            )
        seen[query.query_id] = index
    if limit:
        queries = queries[:limit]
    return QuerySet(
        queries=queries, name=name or os.path.basename(path), version=version, path=path
    )


def in_memory_query_set(records: Sequence[Mapping[str, Any]], name: str = "inline") -> QuerySet:
    """Build a query set directly. Used by tests and the smoke pipeline."""
    return QuerySet(
        queries=[ResearchQuery.from_dict(r, i) for i, r in enumerate(records)], name=name
    )


def normalise_query(query: ResearchQuery) -> str:
    """The retrieval-facing representation of a query.

    Whitespace normalisation only. The architecture keeps this function as the
    single place query representation is decided, so that a future change (query
    expansion, a rationale-based reformulation as in RAG2 section 3.3) has one
    home and shows up in the run record rather than being spread across callers.
    """
    return " ".join(query.query.split())
