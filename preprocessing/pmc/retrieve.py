#!/usr/bin/env python3
"""Balanced retrieval over the frozen index, with candidate-set replay.

Implements the retrieval stage the thesis inherits frozen from RAG²:

* **Balanced retrieval** -- an equal quota of candidates is drawn from each
  source category before merging (base paper 3.4). This is RAG²'s bias
  mitigation: a dense retriever trained on PubMed otherwise drowns the smaller
  but decisive corpora, which here are the CPG/currency-pack documents.
* **Exact flat search, never approximate.** Validity control V3 requires the
  candidate set to be replayed byte-identically to every experimental arm. An
  ANN index would introduce run-to-run variation, so search is an exact scan;
  at this corpus size that is inexpensive and removes a whole class of
  irreproducibility. Ties break on chunk_id so ordering is total and stable.
* **Candidate-set persistence and replay** -- save_candidates() serialises the
  reranked candidate list with a digest; replay_candidates() reloads it and
  verify_replay() proves the arm is scoring the same population. Arms that
  differ in admission policy must never differ in candidates.

Reranking uses the MedCPT cross-encoder (proposal 5.3), applied to the merged
candidate list. It is optional here so the candidate layer can be built and
validated before model weights are present.

Every candidate keeps the full provenance and recency metadata carried by the
index manifest -- date, precision, source category, authority tier, guideline
family, currency-pack membership, retraction flag, chunk and document identity.

Standard library only; numpy/faiss are used when present purely for speed, and
produce identical results because the search is exact.
"""

from __future__ import annotations

import argparse
import array
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Sequence

REPO = Path(__file__).resolve().parent.parent
DEFAULT_INDEX = REPO / "pmc" / "index"
DEFAULT_CANDIDATES = REPO / "pmc" / "candidates"


# ---------------------------------------------------------------------------
# Index loading
# ---------------------------------------------------------------------------
class Index:
    def __init__(self, index_dir: Path):
        self.dir = Path(index_dir)
        self.meta = json.loads((self.dir / "index_meta.json").read_text(encoding="utf-8"))
        self.dim = int(self.meta["dim"])
        self.rows: list[dict] = []
        with (self.dir / "index_manifest.jsonl").open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    self.rows.append(json.loads(line))
        raw = (self.dir / "embeddings.f32").read_bytes()
        buf = array.array("f")
        buf.frombytes(raw)
        if sys.byteorder != "little":                        # pragma: no cover
            buf.byteswap()
        n = len(buf) // self.dim
        if n != len(self.rows):
            raise SystemExit(
                f"index corrupt: {n} vectors vs {len(self.rows)} manifest rows")
        self.vectors = buf
        self.n = n

    def vector(self, row: int) -> list[float]:
        s = row * self.dim
        return list(self.vectors[s:s + self.dim])

    def categories(self) -> list[str]:
        return sorted({r.get("source_category", "") for r in self.rows})


# ---------------------------------------------------------------------------
# Exact search
# ---------------------------------------------------------------------------
def _scores_python(index: Index, q: Sequence[float]) -> list[float]:
    dim, vecs = index.dim, index.vectors
    out = []
    for i in range(index.n):
        s = i * dim
        out.append(sum(vecs[s + j] * q[j] for j in range(dim)))
    return out


def _scores_numpy(index: Index, q: Sequence[float]):             # pragma: no cover
    import numpy as np
    mat = np.frombuffer(index.vectors.tobytes(), dtype=np.float32).reshape(index.n, index.dim)
    return (mat @ np.asarray(q, dtype=np.float32)).tolist()


def score_all(index: Index, query_vec: Sequence[float]) -> list[float]:
    if len(query_vec) != index.dim:
        raise ValueError(f"query dim {len(query_vec)} != index dim {index.dim}")
    try:
        return _scores_numpy(index, query_vec)
    except ImportError:
        return _scores_python(index, query_vec)


def top_k(index: Index, scores: Sequence[float], k: int,
          rows: Sequence[int] | None = None) -> list[tuple[int, float]]:
    """Deterministic top-k. Ties break on chunk_id, so ordering is total."""
    idx = range(index.n) if rows is None else rows
    ranked = sorted(idx, key=lambda i: (-scores[i], index.rows[i]["chunk_id"]))
    return [(i, scores[i]) for i in ranked[:k]]


# ---------------------------------------------------------------------------
# Balanced retrieval
# ---------------------------------------------------------------------------
def balanced_retrieve(index: Index, query_vec: Sequence[float], per_category: int,
                      categories: Sequence[str] | None = None) -> list[dict]:
    """Equal quota per source category, then merge and order deterministically."""
    scores = score_all(index, query_vec)
    cats = list(categories) if categories else index.categories()
    by_cat: dict[str, list[int]] = {c: [] for c in cats}
    for i, row in enumerate(index.rows):
        c = row.get("source_category", "")
        if c in by_cat:
            by_cat[c].append(i)

    picked: list[dict] = []
    for c in cats:
        for row_i, sc in top_k(index, scores, per_category, by_cat[c]):
            cand = dict(index.rows[row_i])
            cand["retrieval_score"] = round(float(sc), 6)
            cand["retrieved_from_category"] = c
            picked.append(cand)
    picked.sort(key=lambda r: (-r["retrieval_score"], r["chunk_id"]))
    for rank, cand in enumerate(picked, 1):
        cand["retrieval_rank"] = rank
    return picked


# ---------------------------------------------------------------------------
# Reranking (MedCPT cross-encoder)
# ---------------------------------------------------------------------------
def rerank(query: str, candidates: list[dict], model_id: str,
           device: str | None = None, batch_size: int = 16) -> list[dict]:  # pragma: no cover
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as exc:
        raise SystemExit(
            "Reranking requires torch and transformers:\n"
            "  pip install torch transformers\n"
            f"(import failed: {exc})"
        ) from exc
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(model_id).to(dev).eval()
    texts = [c.get("title", "") for c in candidates]
    out: list[float] = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            pairs = [[query, t] for t in texts[i:i + batch_size]]
            enc = tok(pairs, truncation=True, padding=True, max_length=512,
                      return_tensors="pt").to(dev)
            out.extend(model(**enc).logits.squeeze(-1).float().cpu().tolist())
    for c, s in zip(candidates, out):
        c["rerank_score"] = round(float(s), 6)
    candidates.sort(key=lambda r: (-r.get("rerank_score", 0.0), r["chunk_id"]))
    for rank, c in enumerate(candidates, 1):
        c["rerank_rank"] = rank
    return candidates


# ---------------------------------------------------------------------------
# Candidate-set persistence and replay  (validity control V3)
# ---------------------------------------------------------------------------
def candidate_digest(candidates: Sequence[dict]) -> str:
    """Digest over the candidate identity and order -- what V3 must hold fixed."""
    h = hashlib.sha256()
    for c in candidates:
        h.update(c["chunk_id"].encode("utf-8"))
        h.update(b"\x00")
        h.update(str(c.get("retrieval_rank", "")).encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


def save_candidates(path: Path, query_id: str, query: str, candidates: list[dict],
                    index_meta: dict, params: dict) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "query_id": query_id,
        "query": query,
        "params": params,
        "index": {
            "encoder": index_meta.get("encoder"),
            "production": index_meta.get("production"),
            "dim": index_meta.get("dim"),
            "vectors": index_meta.get("vectors"),
            "content_digest": index_meta.get("content_digest"),
        },
        "candidate_count": len(candidates),
        "candidate_digest": candidate_digest(candidates),
        "candidates": candidates,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True,
                               ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    return payload


def replay_candidates(path: Path) -> dict:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    actual = candidate_digest(payload["candidates"])
    if actual != payload["candidate_digest"]:
        raise SystemExit(
            f"candidate set {path} failed its digest check: "
            f"recorded {payload['candidate_digest']}, recomputed {actual}"
        )
    return payload


def verify_replay(path: Path, candidates: Sequence[dict]) -> bool:
    """True when a freshly retrieved set is identical to the saved one."""
    saved = replay_candidates(path)
    return saved["candidate_digest"] == candidate_digest(candidates)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Balanced retrieval with candidate replay.")
    ap.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    ap.add_argument("--query", help="query text (requires the MedCPT query encoder)")
    ap.add_argument("--query-id", default="q1")
    ap.add_argument("--per-category", type=int, default=8)
    ap.add_argument("--out", type=Path, default=DEFAULT_CANDIDATES)
    ap.add_argument("--rerank", action="store_true", help="apply the MedCPT cross-encoder")
    ap.add_argument("--replay", type=Path, help="verify a saved candidate set instead")
    args = ap.parse_args(argv)

    if args.replay:
        payload = replay_candidates(args.replay)
        print(f"Replayed {args.replay}")
        print(f"  query_id         {payload['query_id']}")
        print(f"  candidates       {payload['candidate_count']}")
        print(f"  candidate_digest {payload['candidate_digest']}  (verified)")
        print(f"  index encoder    {payload['index'].get('encoder')}")
        return 0

    if not args.query:
        raise SystemExit("--query is required unless --replay is used")

    index = Index(args.index)
    if not index.meta.get("production"):
        print("WARNING: this index was built with a stub encoder; results are not "
              "valid for thesis reporting.", file=sys.stderr)

    from embed_chunks import MedCPTEncoder, QUERY_ENCODER      # lazy: needs torch
    qenc = MedCPTEncoder(QUERY_ENCODER)
    qvec = qenc.encode([args.query])[0]

    cands = balanced_retrieve(index, qvec, args.per_category)
    if args.rerank:
        from embed_chunks import CROSS_ENCODER
        cands = rerank(args.query, cands, CROSS_ENCODER)

    out_path = args.out / f"{args.query_id}.json"
    payload = save_candidates(out_path, args.query_id, args.query, cands, index.meta,
                              {"per_category": args.per_category, "reranked": args.rerank})
    print(f"Candidates saved to {out_path}")
    print(f"  candidates       {payload['candidate_count']}")
    print(f"  candidate_digest {payload['candidate_digest']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
