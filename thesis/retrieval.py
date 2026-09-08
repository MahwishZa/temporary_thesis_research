#!/usr/bin/env python3
"""Retrieval service: a thin, provenance-preserving facade over ``pmc/retrieve.py``.

``pmc/retrieve.py`` is the approved implementation and is not reimplemented here.
It already provides everything the architecture needs, and the properties it
guarantees are the ones the thesis depends on:

* **exact flat search**, never approximate, so a candidate set is reproducible;
* **balanced retrieval** -- an equal quota per source category before merging
  (base paper 3.4), which is RAG2's retriever-bias mitigation;
* **total, stable ordering** -- ties break on ``chunk_id``;
* **candidate persistence and replay** with a digest, so two conditions can be
  *proven* to have scored the same evidence population (validity control V3).

What this module adds is the seam: one call shape for every condition, evidence
records that keep their corpus provenance, and a fingerprint recorded in the run
record. It deliberately does not add caching, re-ranking variants or scoring of
its own -- that would put retrieval behaviour in two places.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence

from ._bootstrap import pmc_retrieve_module
from .config import ThesisConfig
from .provenance import ModelStamp, check_evidence_provenance, stable_hash
from .queries import ResearchQuery, normalise_query


@dataclass
class RetrievalResult:
    """Candidates for one query, with the provenance needed to replay them."""

    query_id: str
    query: str
    candidates: List[Dict[str, Any]] = field(default_factory=list)
    retrieval_fingerprint: str = ""
    candidate_digest: str = ""
    replayed_from: str = ""

    def top(self, k: Optional[int]) -> List[Dict[str, Any]]:
        return list(self.candidates) if k is None else list(self.candidates[:k])

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query_id": self.query_id,
            "query": self.query,
            "candidates": self.candidates,
            "retrieval_fingerprint": self.retrieval_fingerprint,
            "candidate_digest": self.candidate_digest,
            "replayed_from": self.replayed_from,
        }


class RetrievalService:
    """Encodes queries and retrieves candidates over the frozen index."""

    def __init__(
        self,
        config: ThesisConfig,
        encoder: Optional[Any] = None,
        index: Optional[Any] = None,
    ) -> None:
        self.config = config
        self.retrieval = config.retrieval
        self._pmc = pmc_retrieve_module()
        self._encoder = encoder
        self._index = index
        self._query_encoder = None
        rag2_config = config.load_rag2_config()
        self.models = ModelStamp(
            query_encoder=getattr(rag2_config.retrieval, "query_encoder", "") if rag2_config else "",
            article_encoder=getattr(rag2_config.retrieval, "article_encoder", "") if rag2_config else "",
            cross_encoder=getattr(rag2_config.retrieval, "reranker", "") if rag2_config else "",
            generator=getattr(rag2_config.llm, "model", "") if rag2_config else "",
            generator_revision=getattr(rag2_config.llm, "revision", "") if rag2_config else "",
            filter_kind=getattr(rag2_config.filter, "kind", "") if rag2_config else "",
            filter_checkpoint=getattr(rag2_config.filter, "checkpoint", "") if rag2_config else "",
        )

    # -- identity ---------------------------------------------------------
    def fingerprint(self) -> str:
        """Covers everything that determines the candidate set.

        ``final_top_k`` is excluded on purpose: a candidate set retrieved at
        depth 32 legitimately serves a k=8 arm, and forcing a re-retrieval for a
        depth sweep would break the shared-population guarantee it exists to
        provide.
        """
        return stable_hash(
            {
                "per_category": self.retrieval.per_category,
                "rerank": self.retrieval.rerank,
                "query_encoder": self.models.query_encoder,
                "cross_encoder": self.models.cross_encoder if self.retrieval.rerank else "",
                "index_dir": self.config.corpus.index_dir,
                "source_categories": list(self.config.corpus.source_categories),
            }
        )

    # -- index ------------------------------------------------------------
    def index(self):
        if self._index is None:
            self._index = self._pmc.Index(self.config.path(self.config.corpus.index_dir))
        return self._index

    def encode(self, text: str) -> Sequence[float]:
        """Encode a query with the MedCPT **query** encoder.

        MedCPT is asymmetric: ``ncbi/MedCPT-Article-Encoder`` embedded the chunks
        and ``ncbi/MedCPT-Query-Encoder`` must embed the queries searched against
        them. ``pmc.embed_chunks.get_encoder("medcpt", ...)`` returns the *article*
        encoder -- correct for indexing, wrong here -- so this constructs
        ``MedCPTEncoder`` with the query-encoder id from the RAG2 config instead.
        The class is reused, so query vectors still come from the same
        implementation that produced the index.

        An encoder may be injected; tests and the offline smoke path do that to
        stay deterministic without model weights.
        """
        if self._encoder is not None:
            return self._encoder.encode([text])[0]
        if self._query_encoder is None:  # pragma: no cover - needs weights
            from pmc import embed_chunks  # lazy: torch + transformers + weights

            model_id = self.models.query_encoder or embed_chunks.QUERY_ENCODER
            self._query_encoder = embed_chunks.MedCPTEncoder(model_id, None, 1)
        return self._query_encoder.encode([text])[0]

    # -- retrieval --------------------------------------------------------
    def retrieve(self, query: ResearchQuery) -> RetrievalResult:
        """Balanced retrieval (+ optional rerank) for one query."""
        text = normalise_query(query)
        if self.retrieval.replay_from:
            return self.replay(query)

        index = self.index()
        vector = self.encode(text)
        candidates = self._pmc.balanced_retrieve(
            index, vector, self.retrieval.per_category,
            categories=self.config.corpus.source_categories,
        )
        if self.retrieval.rerank:
            candidates = self._pmc.rerank(text, candidates, self.models.cross_encoder)

        problems = check_evidence_provenance(candidates)
        if problems:
            raise RuntimeError(
                "retrieved candidates lost corpus provenance, so results from them could not "
                f"be traced: {problems[:3]}"
            )
        return RetrievalResult(
            query_id=query.query_id,
            query=text,
            candidates=candidates,
            retrieval_fingerprint=self.fingerprint(),
            candidate_digest=self._pmc.candidate_digest(candidates),
        )

    def replay(self, query: ResearchQuery) -> RetrievalResult:
        """Reload a saved candidate set instead of retrieving.

        This is how an arm proves it scored the same evidence population as
        another. The digest is re-derived from the reloaded candidates and
        checked, so a corrupted or edited file fails rather than quietly
        producing a different population.
        """
        path = self._candidate_path(query.query_id, self.config.path(self.retrieval.replay_from))
        payload = self._pmc.replay_candidates(path)
        candidates = payload["candidates"]
        if not self._pmc.verify_replay(path, candidates):
            raise RuntimeError(
                f"candidate replay digest mismatch for {path}: the saved candidate set does not "
                "hash to its recorded digest, so arms replaying it are not comparable"
            )
        return RetrievalResult(
            query_id=query.query_id,
            query=payload.get("query", normalise_query(query)),
            candidates=candidates,
            retrieval_fingerprint=self.fingerprint(),
            candidate_digest=payload.get("digest", self._pmc.candidate_digest(candidates)),
            replayed_from=path,
        )

    def save(self, result: RetrievalResult, directory: str) -> str:
        """Persist a candidate set for later replay by another condition.

        ``pmc.retrieve.save_candidates`` records the index identity and the
        retrieval parameters alongside the candidates, so a replayed arm can be
        shown to have used the same index and depth -- not merely the same
        candidate list.
        """
        from pathlib import Path

        os.makedirs(directory, exist_ok=True)
        path = self._candidate_path(result.query_id, directory)
        self._pmc.save_candidates(
            Path(path), result.query_id, result.query, result.candidates,
            self.index_meta(),
            {
                "per_category": self.retrieval.per_category,
                "final_top_k": self.retrieval.final_top_k,
                "rerank": self.retrieval.rerank,
                "query_encoder": self.models.query_encoder,
                "cross_encoder": self.models.cross_encoder if self.retrieval.rerank else "",
                "source_categories": list(self.config.corpus.source_categories),
                "retrieval_fingerprint": result.retrieval_fingerprint,
            },
        )
        return path

    def index_meta(self) -> Dict[str, Any]:
        """Identity of the index actually searched."""
        try:
            return dict(self.index().meta)
        except Exception:
            return {}

    @staticmethod
    def _candidate_path(query_id: str, directory: str) -> str:
        safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in query_id)
        return os.path.join(directory, f"{safe}.json")
