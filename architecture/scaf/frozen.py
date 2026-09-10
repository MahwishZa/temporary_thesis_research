"""Frozen candidate sets -- the control that makes the comparison mean anything.

The thesis compares two *admission policies*. For that comparison to be about
admission, everything upstream of admission has to be held identical: same
question, same query, same corpus, same index, same retrieval depth, same
reranker, same candidate order. If RAG2 and SCAF each retrieved for themselves,
any difference in the answers could be a retrieval difference, and the result
would say nothing.

So the upstream half runs **once**:

    question -> query/rationale -> retrieval -> rerank  ==>  FrozenCandidateSet

and both arms then consume that record. A frozen set carries a digest over the
candidate identities *and their order*; both arms verify it before scoring, so
"the two arms saw the same evidence" is a checked fact rather than an assumption.

This is validity control V3 in the thesis. ``rag2.cache`` already persists
candidate sets for the baseline's own use; this module adds the part the
comparison needs -- a single immutable artifact carrying the SCAF-relevant
provenance (publication date, precision, authority tier, currency-pack
membership, retraction) alongside the retrieval and rerank scores.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence

FROZEN_FORMAT_VERSION = 1

#: Provenance every candidate must carry. Losing any of these makes either the
#: SCAF policy or the recency analysis impossible, so absence is an error at
#: freeze time rather than a silent gap at analysis time.
REQUIRED_CANDIDATE_FIELDS = (
    "chunk_id", "document_id", "source_category", "canonical_date",
    "date_precision", "authority_tier_label", "in_currency_pack", "retracted",
    "retrieval_rank", "text",
)


@dataclass
class FrozenCandidate:
    """One retrieved-and-reranked passage, with everything both arms may read."""

    chunk_id: str
    document_id: str = ""
    pmcid: str = ""
    pmid: str = ""
    text: str = ""
    title: str = ""
    source_category: str = ""
    canonical_date: str = ""
    date_precision: str = ""
    split_june_2024: str = ""
    authority_tier_label: str = ""
    guideline_family: str = ""
    in_currency_pack: str = ""
    retracted: str = ""
    eligibility_status: str = ""
    license_code: str = ""
    section_heading: str = ""
    retrieval_rank: int = 0
    retrieval_score: float = 0.0
    rerank_rank: int = 0
    rerank_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        payload = {k: v for k, v in self.__dict__.items() if k != "metadata"}
        payload["metadata"] = dict(self.metadata)
        return payload

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "FrozenCandidate":
        known = {k: payload.get(k, getattr(cls, k, "")) for k in cls.__annotations__
                 if k != "metadata"}
        known["metadata"] = dict(payload.get("metadata", {}))
        return cls(**known)

    def to_evidence(self):
        """The baseline's Evidence type, with all provenance in metadata.

        Both arms build their inputs through here, so neither can see a field the
        other cannot.
        """
        from rag2.schema import Evidence

        meta = {
            "canonical_date": self.canonical_date,
            "date_precision": self.date_precision,
            "split_june_2024": self.split_june_2024,
            "authority_tier_label": self.authority_tier_label,
            "guideline_family": self.guideline_family,
            "in_currency_pack": self.in_currency_pack,
            "retracted": self.retracted,
            "eligibility_status": self.eligibility_status,
            "license_code": self.license_code,
            "section_heading": self.section_heading,
            "pmcid": self.pmcid,
            "pmid": self.pmid,
            "retrieval_score": self.retrieval_score,
            "rerank_score": self.rerank_score,
            **self.metadata,
        }
        return Evidence(
            text=self.text,
            source=self.source_category,
            doc_id=self.document_id or None,
            passage_id=self.chunk_id,
            rank=self.rerank_rank or self.retrieval_rank,
            metadata=meta,
        )


@dataclass
class FrozenCandidateSet:
    """The immutable upstream result for one question."""

    qid: str
    question: str
    query: str = ""                 # the rationale, when one was generated
    candidates: List[FrozenCandidate] = field(default_factory=list)
    question_metadata: Dict[str, Any] = field(default_factory=dict)

    def digest(self) -> str:
        return candidate_digest(self.candidates)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "qid": self.qid,
            "question": self.question,
            "query": self.query,
            "question_metadata": self.question_metadata,
            "candidate_count": len(self.candidates),
            "candidate_digest": self.digest(),
            "candidates": [c.to_dict() for c in self.candidates],
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "FrozenCandidateSet":
        frozen = cls(
            qid=payload["qid"],
            question=payload.get("question", ""),
            query=payload.get("query", ""),
            candidates=[FrozenCandidate.from_dict(c) for c in payload.get("candidates", [])],
            question_metadata=dict(payload.get("question_metadata", {})),
        )
        recorded = payload.get("candidate_digest")
        if recorded and recorded != frozen.digest():
            raise ValueError(
                f"frozen candidate set {frozen.qid!r} failed its digest check: "
                f"recorded {recorded}, recomputed {frozen.digest()}. The file has been "
                "edited or truncated; both arms would no longer be scoring the same set."
            )
        return frozen

    def to_question(self):
        """The baseline's Question type. ``time_sensitive`` reaches SCAF here."""
        from rag2.schema import Question

        return Question(
            qid=self.qid,
            question=self.question,
            options=dict(self.question_metadata.get("options", {})),
            answer=self.question_metadata.get("answer"),
            metadata=dict(self.question_metadata),
        )


def candidate_digest(candidates: Sequence[FrozenCandidate]) -> str:
    """Digest over candidate identity *and order* -- what V3 must hold fixed."""
    h = hashlib.sha256()
    for c in candidates:
        h.update(str(c.chunk_id).encode("utf-8"))
        h.update(b"\x00")
        h.update(str(c.rerank_rank or c.retrieval_rank).encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


def set_digest(sets: Sequence[FrozenCandidateSet]) -> str:
    """One digest over the whole frozen file. Goes in the experiment manifest."""
    h = hashlib.sha256()
    for s in sorted(sets, key=lambda x: x.qid):
        h.update(s.qid.encode("utf-8"))
        h.update(b"\x00")
        h.update(s.digest().encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


def validate(sets: Sequence[FrozenCandidateSet]) -> List[str]:
    """Problems that would invalidate the comparison. Empty list means usable."""
    problems: List[str] = []
    seen_qids: set = set()
    for s in sets:
        if s.qid in seen_qids:
            problems.append(f"{s.qid}: duplicate question id")
        seen_qids.add(s.qid)
        if not s.candidates:
            problems.append(f"{s.qid}: no candidates -- both arms would see an empty set")
        ids = [c.chunk_id for c in s.candidates]
        if len(set(ids)) != len(ids):
            duplicates = sorted({i for i in ids if ids.count(i) > 1})
            problems.append(f"{s.qid}: duplicate chunk_ids {duplicates[:5]}")
        for c in s.candidates:
            for required in REQUIRED_CANDIDATE_FIELDS:
                value = getattr(c, required, None)
                if required == "text" and not str(value).strip():
                    problems.append(f"{s.qid}/{c.chunk_id}: empty text")
                elif value is None:
                    problems.append(f"{s.qid}/{c.chunk_id}: missing {required}")
            for name, value in (("retrieval_score", c.retrieval_score),
                                ("rerank_score", c.rerank_score)):
                if value != value or value in (float("inf"), float("-inf")):
                    problems.append(f"{s.qid}/{c.chunk_id}: {name} is not finite")
    return problems


# --------------------------------------------------------------------------
# Persistence
# --------------------------------------------------------------------------
def save(path: str, sets: Sequence[FrozenCandidateSet],
         provenance: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Write the frozen sets (JSONL) plus a ``.meta.json`` sidecar."""
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    ordered = sorted(sets, key=lambda s: s.qid)
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        for s in ordered:
            handle.write(json.dumps(s.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")

    meta = {
        "format_version": FROZEN_FORMAT_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "questions": len(ordered),
        "candidates_total": sum(len(s.candidates) for s in ordered),
        "candidates_per_question": sorted({len(s.candidates) for s in ordered}),
        "frozen_set_digest": set_digest(ordered),
        "provenance": dict(provenance or {}),
    }
    with open(f"{os.path.splitext(path)[0]}.meta.json", "w",
              encoding="utf-8", newline="\n") as handle:
        json.dump(meta, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return meta


def load(path: str, expected_digest: Optional[str] = None) -> List[FrozenCandidateSet]:
    """Load frozen sets, verifying each one's digest and optionally the file's.

    ``from_dict`` raises on a per-question digest mismatch; ``expected_digest``
    additionally pins the whole file, which is what an experiment manifest cites.
    """
    sets: List[FrozenCandidateSet] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                sets.append(FrozenCandidateSet.from_dict(json.loads(line)))
    if expected_digest:
        actual = set_digest(sets)
        if actual != expected_digest:
            raise ValueError(
                f"frozen candidate file {path} has digest {actual}, but the manifest "
                f"cites {expected_digest}. The two arms would not be comparable."
            )
    return sets


def read_meta(path: str) -> Optional[Dict[str, Any]]:
    meta_path = f"{os.path.splitext(path)[0]}.meta.json"
    if not os.path.exists(meta_path):
        return None
    with open(meta_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def from_rag2_candidate_sets(candidate_sets: Iterable[Any],
                             questions: Dict[str, Any],
                             depth: int = 20) -> List[FrozenCandidateSet]:
    """Convert ``rag2.schema.CandidateSet``s (the baseline's own upstream output).

    This is the bridge: the real run produces candidates through the baseline's
    MedCPT retrieval and reranking, and freezes them here without re-retrieving.

    ``depth`` keeps the first N candidates per question, in the order the cache
    already holds them. It is a truncation of an existing ranking, never a
    reordering and never a re-ranking: the cache arrives sorted by the baseline's
    MedCPT cross-encoder, so ``[:depth]`` is exactly the top-N of that ranking.
    Previously the whole cache was frozen regardless, so a cache built at a
    different ``retrieval.final_top_k`` silently produced a frozen set of that
    size while the sidecar recorded the requested depth -- the two could
    disagree, and the manifest's ``candidate_depth`` was then wrong.
    """
    out: List[FrozenCandidateSet] = []
    for cs in candidate_sets:
        question = questions.get(cs.qid)
        candidates: List[FrozenCandidate] = []
        for rank, ev in enumerate(cs.candidates[:depth], start=1):
            meta = dict(ev.metadata or {})
            candidates.append(FrozenCandidate(
                chunk_id=str(ev.passage_id or ""),
                document_id=str(ev.doc_id or ""),
                pmcid=str(meta.pop("pmcid", "")),
                pmid=str(meta.pop("pmid", "")),
                text=ev.text,
                title=str(meta.pop("title", "")),
                source_category=str(ev.source or ""),
                canonical_date=str(meta.pop("canonical_date", "")),
                date_precision=str(meta.pop("date_precision", "")),
                split_june_2024=str(meta.pop("split_june_2024", "")),
                authority_tier_label=str(meta.pop("authority_tier_label", "")),
                guideline_family=str(meta.pop("guideline_family", "")),
                in_currency_pack=str(meta.pop("in_currency_pack", "")),
                retracted=str(meta.pop("retracted", "")),
                eligibility_status=str(meta.pop("eligibility_status", "")),
                license_code=str(meta.pop("license_code", "")),
                section_heading=str(meta.pop("section_heading", "")),
                retrieval_rank=rank,
                retrieval_score=float(meta.pop("retrieval_score", 0.0) or 0.0),
                rerank_rank=int(getattr(ev, "rank", rank) or rank),
                rerank_score=float(getattr(ev, "rerank_score", 0.0) or 0.0),
                metadata=meta,
            ))
        out.append(FrozenCandidateSet(
            qid=cs.qid,
            question=getattr(question, "question", "") if question else "",
            query=getattr(cs, "rationale", "") or "",
            candidates=candidates,
            question_metadata=dict(getattr(question, "metadata", {}) or {}),
        ))
    return out
