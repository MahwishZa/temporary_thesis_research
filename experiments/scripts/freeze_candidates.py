#!/usr/bin/env python3
"""Stage 1 of the comparison: run the upstream pipeline once and freeze it.

    question -> query/rationale -> retrieval -> rerank  ==>  frozen candidate set

Two sources, and the distinction matters for what may be reported:

``--source medcpt``  the real path. Reads a candidate cache produced by the
    baseline's own ``rag2/scripts/02_retrieve.py`` (MedCPT query encoding,
    balanced retrieval over pmc/index, MedCPT cross-encoder reranking) and
    freezes it verbatim. This is the only source whose output may back a
    thesis claim.

``--source lexical-dev``  an offline stand-in for development only. Ranks chunks
    by IDF-weighted term overlap instead of MedCPT. It exercises the whole
    downstream comparison without model weights or the index, and it stamps
    ``retrieval_is_medcpt: false`` into the sidecar so its output can never be
    mistaken for the real thing.

Either way the result is immutable: both arms consume it and a digest over
candidate identity *and* order is recorded.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterator, List

_ROOT = Path(__file__).resolve().parent.parent.parent
for _p in (str(_ROOT), str(_ROOT / "rag2")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from scaf.frozen import FrozenCandidate, FrozenCandidateSet, save, validate  # noqa: E402
from scaf.policy import tokenize                                            # noqa: E402

DEFAULT_QUESTIONS = _ROOT / "scaf" / "data" / "dev_questions.jsonl"
DEFAULT_CHUNKS = _ROOT / "pmc" / "chunks" / "chunks.jsonl"
DEFAULT_OUT = _ROOT / "scaf" / "runs" / "frozen_candidates.jsonl"


def read_questions(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def iter_chunks(path: Path) -> Iterator[Dict[str, Any]]:
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


# ---------------------------------------------------------------------------
# Development retriever -- NOT MedCPT
# ---------------------------------------------------------------------------
def lexical_rank(questions: List[Dict[str, Any]], chunks_path: Path, depth: int,
                 max_chunks: int = 0) -> Dict[str, List[Dict[str, Any]]]:
    """IDF-weighted overlap ranking. Deterministic; ties break on chunk_id.

    One pass over the chunk file scoring every question, so a 60k-780k chunk
    file is read once rather than once per question.
    """
    q_tokens = {q["qid"]: set(tokenize(q["question"])) for q in questions}
    document_freq: Counter = Counter()
    kept: List[Dict[str, Any]] = []

    for i, chunk in enumerate(iter_chunks(chunks_path)):
        if max_chunks and i >= max_chunks:
            break
        if chunk.get("duplicate_of"):
            continue
        if str(chunk.get("eligibility_status", "")) == "excluded":
            continue
        tokens = set(tokenize(chunk.get("text", "")))
        # Only keep chunks that touch at least one question, to bound memory.
        if not any(tokens & tq for tq in q_tokens.values()):
            continue
        document_freq.update(tokens)
        kept.append({"chunk": chunk, "tokens": tokens})

    n = len(kept) or 1
    idf = {t: math.log((n + 1) / (c + 0.5)) for t, c in document_freq.items()}

    out: Dict[str, List[Dict[str, Any]]] = {}
    for question in questions:
        qid, terms = question["qid"], q_tokens[question["qid"]]
        total = sum(idf.get(t, 1.0) for t in terms) or 1.0
        scored = []
        for entry in kept:
            hit = sum(idf.get(t, 1.0) for t in (terms & entry["tokens"]))
            if hit > 0:
                scored.append((hit / total, entry["chunk"]))
        scored.sort(key=lambda pair: (-pair[0], pair[1]["chunk_id"]))
        out[qid] = [{"score": s, "chunk": c} for s, c in scored[:depth]]
    return out


def corpus_document_frequency(questions: List[Dict[str, Any]], chunks_path: Path,
                              max_chunks: int = 0) -> Dict[str, Any]:
    """Document frequency of the question terms over the corpus, plus corpus size.

    SCAF's support scorer needs corpus-level IDF (its v1 used within-candidate-set
    IDF and collapsed). Only terms appearing in some question can affect sigma, so
    only those are counted -- the table stays a few hundred entries instead of
    hundreds of thousands.
    """
    terms: set = set()
    for question in questions:
        terms.update(tokenize(question["question"]))
    counts = {t: 0 for t in terms}
    size = 0
    with chunks_path.open(encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if max_chunks and i >= max_chunks:
                break
            line = line.strip()
            if not line:
                continue
            chunk = json.loads(line)
            if chunk.get("duplicate_of"):
                continue
            size += 1
            present = set(tokenize(chunk.get("text", ""))) & terms
            for token in present:
                counts[token] += 1
    return {"document_frequency": counts, "corpus_size": size}


def to_frozen(question: Dict[str, Any], ranked: List[Dict[str, Any]]) -> FrozenCandidateSet:
    candidates = []
    for rank, item in enumerate(ranked, start=1):
        chunk = item["chunk"]
        candidates.append(FrozenCandidate(
            chunk_id=str(chunk.get("chunk_id", "")),
            document_id=str(chunk.get("document_id", "")),
            pmcid=str(chunk.get("pmcid", "")),
            pmid=str(chunk.get("pmid", "")),
            text=str(chunk.get("text", "")),
            title=str(chunk.get("title", "")),
            source_category=str(chunk.get("source_category", "")),
            canonical_date=str(chunk.get("canonical_date", "")),
            date_precision=str(chunk.get("date_precision", "")),
            split_june_2024=str(chunk.get("split_june_2024", "")),
            authority_tier_label=str(chunk.get("authority_tier_label", "")),
            guideline_family=str(chunk.get("guideline_family", "")),
            in_currency_pack=str(chunk.get("in_currency_pack", "")),
            retracted=str(chunk.get("retracted", "")),
            eligibility_status=str(chunk.get("eligibility_status", "")),
            license_code=str(chunk.get("license_code", "")),
            section_heading=str(chunk.get("section_heading", "")),
            retrieval_rank=rank,
            retrieval_score=round(float(item["score"]), 6),
            rerank_rank=rank,
            rerank_score=round(float(item["score"]), 6),
        ))
    meta = {k: v for k, v in question.items() if k not in ("qid", "question")}
    return FrozenCandidateSet(qid=question["qid"], question=question["question"],
                              query=question["question"], candidates=candidates,
                              question_metadata=meta)


def from_rag2_cache(cache_path: Path, questions: List[Dict[str, Any]]) -> List[FrozenCandidateSet]:
    """Freeze the baseline's own MedCPT candidate cache. The real path."""
    from rag2.cache import iter_candidates
    from rag2.schema import Question

    from scaf.frozen import from_rag2_candidate_sets

    lookup = {
        q["qid"]: Question(qid=q["qid"], question=q["question"], options={},
                           metadata={k: v for k, v in q.items()
                                     if k not in ("qid", "question")})
        for q in questions
    }
    return from_rag2_candidate_sets(list(iter_candidates(str(cache_path))), lookup)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Freeze the upstream candidate set.")
    ap.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    ap.add_argument("--chunks", type=Path, default=DEFAULT_CHUNKS)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--source", choices=["medcpt", "lexical-dev"], default="lexical-dev",
                    help="'medcpt' freezes a rag2 candidate cache (the real path); "
                         "'lexical-dev' ranks lexically offline (development only)")
    ap.add_argument("--cache", type=Path, help="rag2 candidate cache, for --source medcpt")
    ap.add_argument("--depth", type=int, default=20, help="candidates kept per question")
    ap.add_argument("--max-chunks", type=int, default=0,
                    help="scan only the first N chunks (development speed-up)")
    args = ap.parse_args(argv)

    questions = read_questions(args.questions)
    print(f"questions: {len(questions)}  from {args.questions}")

    if args.source == "medcpt":
        if not args.cache:
            raise SystemExit("--source medcpt requires --cache <rag2 candidate cache>")
        frozen_sets = from_rag2_cache(args.cache, questions)
        provenance = {
            "retrieval_is_medcpt": True,
            "source": "rag2 candidate cache",
            "cache": str(args.cache),
        }
    else:
        if not args.chunks.exists():
            raise SystemExit(
                f"chunk layer not found: {args.chunks}\n"
                "Build it with pmc/build_chunks.py, or pass --chunks.")
        ranked = lexical_rank(questions, args.chunks, args.depth, args.max_chunks)
        frozen_sets = [to_frozen(q, ranked.get(q["qid"], [])) for q in questions]
        provenance = {
            "retrieval_is_medcpt": False,
            "source": "lexical-dev (IDF term overlap)",
            "warning": "DEVELOPMENT RETRIEVAL. Not MedCPT. Results from this frozen "
                       "set describe wiring and admission behaviour only, and must "
                       "not be reported as thesis retrieval results.",
            "chunks": str(args.chunks),
            "max_chunks_scanned": args.max_chunks or "all",
        }

    empty = [s.qid for s in frozen_sets if not s.candidates]
    if empty:
        print(f"  WARNING: {len(empty)} question(s) retrieved nothing: {empty[:5]}")

    problems = validate(frozen_sets)
    if problems:
        print(f"\nFAILED validation ({len(problems)} problem(s)):")
        for p in problems[:20]:
            print(f"  - {p}")
        return 1

    # Corpus statistics for SCAF's support scorer. Computed from the SAME chunk
    # file the candidates came from, and carried in the frozen sidecar so the
    # comparison cannot silently score with different statistics.
    stats = {"document_frequency": {}, "corpus_size": 0}
    if args.chunks.exists():
        stats = corpus_document_frequency(questions, args.chunks, args.max_chunks)
        print(f"  corpus stats: {len(stats['document_frequency'])} question terms "
              f"over {stats['corpus_size']:,} chunks")
    else:
        print("  WARNING: no chunk file; SCAF support will fall back to coverage only")

    provenance.update({"depth": args.depth, "questions_file": str(args.questions),
                       "corpus_size": stats["corpus_size"],
                       "document_frequency": stats["document_frequency"]})
    meta = save(str(args.out), frozen_sets, provenance=provenance)
    print(f"\nFrozen candidates -> {args.out}")
    print(f"  questions            {meta['questions']}")
    print(f"  candidates total     {meta['candidates_total']}")
    print(f"  per question         {meta['candidates_per_question']}")
    print(f"  frozen_set_digest    {meta['frozen_set_digest']}")
    if not provenance["retrieval_is_medcpt"]:
        print("\n  WARNING: development retrieval (not MedCPT). Wiring only.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
