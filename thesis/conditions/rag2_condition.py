#!/usr/bin/env python3
"""RAG2 condition: the reproduced baseline, called rather than reimplemented.

This module contains **no** RAG2 algorithm. Filtering, prompting, perplexity,
thresholding and generation all live in ``rag2/rag2/`` and are reached through
the interfaces that package already exposes:

    rag2.filtering.base.build_filter   -> the trained Flan-T5 filter, or an ablation
    rag2.filtering.base.EvidenceFilter -> .apply(question, candidates)
    rag2.schema.Evidence / Question    -> its record types
    rag2.generation.generate_answers   -> answer construction
    rag2.prompts.PromptSet             -> the paper's prompts, unmodified

What this file does is translate: thesis candidate dicts in, RAG2 ``Evidence``
objects across the seam, ``ConditionResult`` out. Translation is the only thing
it is allowed to do. In particular it must not:

* re-implement or "improve" the filter's scoring, threshold or prompt;
* re-rank, dedupe or truncate candidates beyond the configured ``final_top_k``;
* pass anything into the filter beyond evidence text -- the filter in the paper
  saw a (question, snippet) pair and nothing else, and feeding it corpus
  metadata would change what is being reproduced.

That last point is why ``_to_rag2_evidence`` copies text into ``Evidence.text``
and provenance into ``Evidence.metadata``: RAG2's prompt builder renders only the
text, and its own metadata-isolation test guarantees that stays true.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence

from .._bootstrap import ensure_rag2_importable
from ..queries import ResearchQuery
from ..retrieval import RetrievalResult
from .base import AdmittedEvidence, ConditionResult, ExperimentCondition, register_condition


class RAG2Condition(ExperimentCondition):
    """Runs the reproduced RAG2 filter (and optionally generation) on candidates."""

    name = "rag2"
    generates = True

    def __init__(self, config, policy=None, evidence_filter=None, llm=None) -> None:
        super().__init__(config, policy)
        ensure_rag2_importable()
        self.rag2_config = config.load_rag2_config()
        if self.rag2_config is None:
            raise ValueError(
                "the rag2 condition needs thesis config 'rag2_config' to point at a RAG2 "
                "configuration file (e.g. rag2/configs/thesis_corpus.yaml)"
            )
        self.prompts = self.rag2_config.prompt_set()
        self._filter = evidence_filter
        self._llm = llm

    # -- lazily built so a wiring test can inject stubs -------------------
    def evidence_filter(self):
        if self._filter is None:
            from rag2.filtering.base import build_filter

            self._filter = build_filter(self.rag2_config.filter, self.prompts)
        return self._filter

    def llm(self):
        if self._llm is None:  # pragma: no cover - needs model weights
            from rag2.llm.base import build_llm

            self._llm = build_llm(self.rag2_config.llm)
        return self._llm

    # -- the seam ---------------------------------------------------------
    def run(self, query: ResearchQuery, retrieved: RetrievalResult) -> ConditionResult:
        candidates = self.apply_policy(retrieved)
        question = _to_rag2_question(query)
        evidence = [_to_rag2_evidence(c, i + 1) for i, c in enumerate(candidates)]

        kept, decisions = self.evidence_filter().apply(question, evidence)
        kept_ids = {id(e) for e in kept}

        admitted: List[AdmittedEvidence] = []
        rejected: List[AdmittedEvidence] = []
        for rank, (candidate, item, decision) in enumerate(
            zip(candidates, evidence, decisions), start=1
        ):
            record = AdmittedEvidence.from_candidate(
                candidate,
                rank=rank,
                admission_score=decision.score,
                admission_label=decision.label,
            )
            (admitted if id(item) in kept_ids else rejected).append(record)

        answer, answer_source = "", "filtering only (generation not requested)"
        if self.config.condition.name == self.name and self._should_generate():
            answer = self._generate(question, kept)
            answer_source = f"rag2.generation via {self.rag2_config.llm.model}"

        return ConditionResult(
            query_id=query.query_id,
            condition=self.name,
            query=retrieved.query,
            admitted=admitted,
            rejected=rejected,
            answer=answer,
            answer_source=answer_source,
            candidate_digest=retrieved.candidate_digest,
            retrieval_fingerprint=retrieved.retrieval_fingerprint,
            temporal_policy=self.policy.name,
            diagnostics={
                "candidates_seen": len(retrieved.candidates),
                "final_top_k": self.config.retrieval.final_top_k,
                "filtered": len(evidence),
                "admitted": len(admitted),
                "rejected": len(rejected),
                "filter": self.evidence_filter().describe(),
                "on_empty": self.rag2_config.filter.on_empty,
            },
        )

    def _should_generate(self) -> bool:
        return bool(self.config.evaluation.metrics) and "answer" in " ".join(
            self.config.evaluation.metrics
        )

    def _generate(self, question, kept) -> str:  # pragma: no cover - needs weights
        from rag2.generation import generate_answers

        return generate_answers(
            self.llm(), [question], [list(kept)],
            config=self.rag2_config.generation, prompts=self.prompts,
        )[0]


# ---------------------------------------------------------------------------
# Translation across the seam
# ---------------------------------------------------------------------------
def _to_rag2_question(query: ResearchQuery):
    """A thesis research query as a RAG2 ``Question``.

    RAG2's benchmarks are multiple choice; the thesis's research queries are
    open-ended. ``options`` is therefore left empty, which RAG2's prompt builder
    renders as a bare question -- the correct behaviour for an open-ended query,
    and one that does not require touching RAG2 to obtain.
    """
    from rag2.schema import Question

    return Question(
        qid=query.query_id,
        question=query.query,
        options={},
        answer=query.answer,
        dataset="thesis",
        split="eval",
        metadata=dict(query.metadata),
    )


def _to_rag2_evidence(candidate: Mapping[str, Any], rank: int):
    """A retrieved chunk as a RAG2 ``Evidence``.

    Text goes in ``text`` -- the only field RAG2 renders into a prompt. Corpus
    provenance, publication date included, goes in ``metadata``, which RAG2
    carries and never reads.
    """
    from rag2.schema import Evidence

    consumed = {"text", "chunk_id", "document_id", "source_category",
                "retrieval_score", "rerank_score", "retrieval_rank"}
    return Evidence(
        text=str(candidate.get("text", "")),
        source=str(candidate.get("source_category", "")),
        doc_id=str(candidate.get("document_id", "")) or None,
        passage_id=str(candidate.get("chunk_id", "")) or None,
        retrieval_score=candidate.get("retrieval_score"),
        rerank_score=candidate.get("rerank_score"),
        rank=rank,
        metadata={k: v for k, v in candidate.items() if k not in consumed},
    )


@register_condition("rag2")
def _build(config, policy=None) -> ExperimentCondition:
    return RAG2Condition(config, policy)
