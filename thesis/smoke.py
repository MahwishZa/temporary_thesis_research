#!/usr/bin/env python3
"""Offline end-to-end wiring check.

Runs the real orchestrator, the real corpus loader, the real retrieval facade,
the real conditions and the real provenance layer, over a tiny synthetic corpus
and a deterministic stand-in encoder. No model weights, no GPU, no network, and
**it never touches the production corpus or index** -- it builds its own fixture
in a temporary directory and deletes it.

What it demonstrates, in the order the task requires:

  1. a research query enters the system;
  2. the query reaches the retrieval layer;
  3. retrieval returns evidence (exact search over the fixture index);
  4. corpus metadata and provenance survive retrieval;
  5. the selected experimental condition is invoked;
  6. the pipeline produces a structured result;
  7. the result records the corpus/config provenance needed to reproduce it.

What it does **not** demonstrate: retrieval quality, filter behaviour on real
evidence, or any research finding. The encoder is a hash, so the ranking is
meaningless by construction. It proves the parts are connected, nothing more.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from typing import Any, Dict, List

from .conditions.base import build_condition
from .config import ThesisConfig, load_config
from .corpus import CorpusHandle
from .pipeline import run_pipeline
from .queries import in_memory_query_set
from .retrieval import RetrievalService

DIM = 32


class HashEncoder:
    """Deterministic stand-in for MedCPT. sha256-seeded, stable across processes."""

    production = False
    name = "smoke-hash-encoder"
    dim = DIM

    def encode(self, texts) -> List[List[float]]:
        vectors = []
        for text in texts:
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            raw = [((digest[i % len(digest)] / 255.0) - 0.5) for i in range(DIM)]
            norm = sum(v * v for v in raw) ** 0.5 or 1.0
            vectors.append([v / norm for v in raw])
        return vectors


def build_fixture(root: str, chunks: int = 12) -> Dict[str, Any]:
    """A miniature corpus + index in the on-disk shape pmc/ produces."""
    chunk_dir = os.path.join(root, "chunks")
    index_dir = os.path.join(root, "index")
    os.makedirs(chunk_dir, exist_ok=True)
    os.makedirs(index_dir, exist_ok=True)

    categories = ["pubmed-abstract", "pmc-fulltext", "currency-pack"]
    encoder = HashEncoder()
    records, manifest, vectors = [], [], []
    for i in range(chunks):
        category = categories[i % len(categories)]
        record = {
            "chunk_id": f"CHUNK{i:04d}",
            "document_id": f"PMC{9000000 + i}",
            "source_category": category,
            "title": f"Synthetic Alzheimer study {i}",
            "text": f"Synthetic evidence passage {i} concerning amyloid and cognition.",
            "canonical_date": f"{2015 + (i % 10)}-03-01",
            "date_precision": "day",
            "split_june_2024": "before" if (2015 + (i % 10)) < 2024 else "after",
            "authority_tier": "primary",
            "row": i,
        }
        records.append(record)
        # The manifest is what retrieval actually returns, so the fixture must
        # carry the same fields pmc/embed_chunks.py writes -- including the
        # temporal ones. A thinner manifest would hide a real capability loss.
        manifest.append({k: v for k, v in record.items() if k != "text"})
        vectors.append(encoder.encode([record["text"]])[0])

    with open(os.path.join(chunk_dir, "chunks.jsonl"), "w", encoding="utf-8", newline="\n") as h:
        for record in records:
            h.write(json.dumps(record, ensure_ascii=False) + "\n")

    digest = hashlib.sha256(
        "".join(r["chunk_id"] + r["text"] for r in records).encode("utf-8")
    ).hexdigest()
    stats = {
        "chunks": len(records), "documents_chunked": len(records),
        "unique_chunk_texts": len(records), "window_words": 256, "overlap_words": 32,
        "chunk_digest": digest,
    }
    with open(os.path.join(chunk_dir, "chunk_stats.json"), "w", encoding="utf-8", newline="\n") as h:
        json.dump(stats, h, indent=2)

    import struct

    with open(os.path.join(index_dir, "embeddings.f32"), "wb") as h:
        for vector in vectors:
            h.write(struct.pack(f"<{DIM}f", *vector))
    with open(os.path.join(index_dir, "index_manifest.jsonl"), "w", encoding="utf-8", newline="\n") as h:
        for row in manifest:
            h.write(json.dumps(row, ensure_ascii=False) + "\n")
    with open(os.path.join(index_dir, "index_meta.json"), "w", encoding="utf-8", newline="\n") as h:
        json.dump({"encoder": encoder.name, "dim": DIM, "vectors": len(vectors),
                   "production": False, "chunk_digest": digest,
                   "content_digest": digest[:32]}, h, indent=2)
    return {"root": root, "digest": digest, "chunks": len(records)}


def smoke_config(root: str, fixture: Dict[str, Any], condition: str = "baseline") -> ThesisConfig:
    return load_config(None, {
        "name": f"smoke-{condition}",
        "rag2_config": "rag2/configs/thesis_corpus.yaml",
        "corpus": {
            "chunks_path": os.path.join(root, "chunks", "chunks.jsonl"),
            "chunk_stats_path": os.path.join(root, "chunks", "chunk_stats.json"),
            "index_dir": os.path.join(root, "index"),
            "expected_chunk_digest": fixture["digest"],
            "expected_chunk_count": fixture["chunks"],
            # The fixture index is a stub by construction; a wiring test may say so.
            "require_production_index": False,
        },
        "retrieval": {"per_category": 2, "final_top_k": 4, "rerank": False},
        "condition": {"name": condition},
        "output": {"dir": os.path.join(root, "runs", "{condition}")},
    })


def run_smoke(verbose: bool = True) -> int:
    root = tempfile.mkdtemp(prefix="thesis-smoke-")
    failures: List[str] = []

    def check(label: str, ok: bool, detail: str = "") -> None:
        print(f"  [{'ok  ' if ok else 'FAIL'}] {label}{(' -- ' + detail) if detail else ''}")
        if not ok:
            failures.append(label)

    try:
        fixture = build_fixture(root)
        config = smoke_config(root, fixture)
        encoder = HashEncoder()

        print("1-2. query enters the system and reaches retrieval")
        queries = in_memory_query_set([
            {"query_id": "SMOKE-1",
             "query": "Does amyloid burden predict cognitive decline in early Alzheimer's disease?"},
            {"query_id": "SMOKE-2", "query": "What is the evidence for anti-amyloid therapy?"},
        ], name="smoke")
        corpus = CorpusHandle.open(config)
        check("corpus opens and its digest is verified", corpus.digest_verified,
              f"digest {fixture['digest'][:12]}...")

        retrieval = RetrievalService(config, encoder=encoder)
        outcome = run_pipeline(config, corpus=corpus, query_set=queries, retrieval=retrieval)

        print("3. MedCPT-shaped retrieval returns evidence")
        admitted = [e for r in outcome.results for e in r.admitted]
        check("evidence retrieved for every query",
              all(r.admitted for r in outcome.results),
              f"{len(admitted)} admitted across {len(outcome.results)} queries")
        check("balanced across source categories",
              len({e.source_category for e in admitted}) > 1,
              str(sorted({e.source_category for e in admitted})))

        print("4. metadata and provenance survive retrieval")
        first = admitted[0]
        check("chunk and document identity preserved",
              bool(first.chunk_id and first.document_id), f"{first.chunk_id} / {first.document_id}")
        check("publication date carried as metadata",
              bool(first.metadata.get("canonical_date")), str(first.metadata.get("canonical_date")))
        carried = outcome.record.retrieval["temporal_fields_carried"]
        check("all temporal fields survive retrieval, so a recency arm is possible",
              carried["complete"], f"missing: {carried['missing'] or 'none'}")

        print("5-6. condition invoked; structured result produced")
        check("condition recorded on every result",
              all(r.condition == "baseline" for r in outcome.results))
        check("results serialise", isinstance(outcome.results[0].to_dict(), dict))
        check("evaluation report produced",
              outcome.report["retrieval"]["evidence_admitted"] == len(admitted))

        print("7. provenance sufficient to reproduce the result")
        record = outcome.record
        check("corpus digest recorded and verified",
              record.corpus.chunk_digest == fixture["digest"] and record.corpus.digest_verified)
        check("model identities recorded",
              record.models.query_encoder == "ncbi/MedCPT-Query-Encoder",
              record.models.query_encoder)
        check("config + retrieval fingerprints recorded",
              bool(record.config_fingerprint and record.retrieval_fingerprint))
        check("git commit recorded", bool(record.git.get("commit")))
        check("run correctly marked NOT reportable (stub index)",
              not record.is_reportable()[0], "; ".join(record.is_reportable()[1])[:70])
        check("outputs written",
              all(os.path.exists(os.path.join(outcome.output_dir, f))
                  for f in ("results.jsonl", "report.json", "run_record.json")))

        print("8. the baseline is temporally blind")
        condition = build_condition(config)
        check("baseline temporal policy is the identity",
              condition.policy.name == "none" and not condition.policy.reads_dates)

        print("9. determinism: same inputs, same candidate population")
        second = run_pipeline(smoke_config(root, fixture), corpus=corpus, query_set=queries,
                              retrieval=RetrievalService(smoke_config(root, fixture), encoder=HashEncoder()),
                              write=False)
        check("candidate digests reproduce",
              [r.candidate_digest for r in outcome.results] ==
              [r.candidate_digest for r in second.results])

        print()
        if failures:
            print(f"SMOKE TEST FAILED: {failures}")
            return 1
        print("SMOKE TEST PASSED -- architecture wired end to end")
        print("(hash encoder over a synthetic fixture: rankings are meaningless by construction)")
        return 0
    finally:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(run_smoke())
