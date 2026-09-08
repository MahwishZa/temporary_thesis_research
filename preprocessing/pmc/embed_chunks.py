#!/usr/bin/env python3
"""Deterministic embedding + index construction over the frozen chunk layer.

Reads pmc/chunks/chunks.jsonl (frozen, unmodified) and writes an additive index
layer under pmc/index/:

    embeddings.f32      row-major float32 vectors, one row per chunk
    index_manifest.jsonl  row -> chunk_id + the metadata retrieval needs
    index_meta.json     model id, dim, counts, digests, production flag

Model choice is NOT a convenience decision. The base paper uses MedCPT for dense
retrieval and MedCPT cross-encoding for reranking, and the thesis proposal
(5.3) pins both as "baseline fidelity" because the retriever is deliberately
frozen -- it is the central internal-validity guarantee. So the production
encoder here is MedCPT and nothing else:

    documents : ncbi/MedCPT-Article-Encoder
    queries   : ncbi/MedCPT-Query-Encoder      (used by retrieve.py)
    reranking : ncbi/MedCPT-Cross-Encoder      (used by retrieve.py)

A deterministic stub encoder exists for offline testing only. It refuses to
produce a production index unless --allow-stub is passed, and it stamps
index_meta.json with production=false so a stub index can never be mistaken for
a real one downstream.

Determinism: chunks are processed in sorted chunk_id order, vectors are float32,
and the manifest records a digest over (chunk_id, vector bytes). Re-running on
the same chunks with the same model reproduces the same files.

Running it on a GPU
-------------------
The production corpus is ~780k chunks, which is many hours of work, so the run
is built to survive being interrupted:

* **Resumable by default.** Output is flushed every batch and a restart picks up
  at the last complete row, replaying the digest over what is already there so a
  resumed index carries the same content_digest as an uninterrupted one. Pass
  ``--restart`` to start over.
* **The device is explicit.** ``--device cuda`` is a *requirement* that fails
  with an actionable message; it will not quietly run on the CPU, which is what
  a ``+cpu`` torch build otherwise does.
* **OOM backs off, it does not crash or move to CPU.** On CUDA OOM the batch
  size halves and the run continues.

Standard library only, except the optional production backend (torch +
transformers), which is imported lazily and only when actually used.
"""

from __future__ import annotations

import argparse
import array
import hashlib
import json
import sys
from pathlib import Path
from typing import Iterable, Iterator

REPO = Path(__file__).resolve().parent.parent
DEFAULT_CHUNKS = REPO / "pmc" / "chunks" / "chunks.jsonl"
DEFAULT_OUT = REPO / "pmc" / "index"

ARTICLE_ENCODER = "ncbi/MedCPT-Article-Encoder"
QUERY_ENCODER = "ncbi/MedCPT-Query-Encoder"
CROSS_ENCODER = "ncbi/MedCPT-Cross-Encoder"
MEDCPT_DIM = 768
MEDCPT_MAX_TOKENS = 512

# Conservative by design. MedCPT-Article-Encoder is BERT-base (~110M params,
# ~440 MB in fp32); at 512 tokens the activations, not the weights, dominate.
# On a 4 GiB card the model, the CUDA context and one batch must all fit, so 8
# leaves real headroom for the long-chunk batches that dynamic padding produces.
# Raise it with --batch-size on a larger card; encode() halves it on OOM.
DEFAULT_BATCH_SIZE = 8

# Metadata every retrieved candidate must still carry. Losing any of these would
# make the recency-bias study impossible to run later.
MANIFEST_FIELDS = [
    "chunk_id", "document_id", "pmcid", "pmid", "doi", "title",
    "source_category", "eligibility_status", "canonical_date", "date_precision",
    "date_source", "split_june_2024", "authority_tier_label", "guideline_family",
    "in_currency_pack", "retracted", "license_code", "license_band",
    "location", "section_id", "section_heading", "imrad", "word_count",
    "text_sha256", "duplicate_of",
]


# ---------------------------------------------------------------------------
# Encoders
# ---------------------------------------------------------------------------
class StubEncoder:
    """Deterministic hash-based vectors. Testing only -- never a real index."""

    name = "stub-sha256"
    production = False

    def __init__(self, dim: int = 64):
        self.dim = dim

    def encode(self, texts: list[str]) -> list[list[float]]:
        out = []
        for t in texts:
            vals: list[float] = []
            counter = 0
            while len(vals) < self.dim:
                h = hashlib.sha256(f"{counter}\x00{t}".encode("utf-8")).digest()
                for i in range(0, len(h), 2):
                    if len(vals) >= self.dim:
                        break
                    vals.append((int.from_bytes(h[i:i + 2], "big") / 65535.0) - 0.5)
                counter += 1
            norm = sum(v * v for v in vals) ** 0.5 or 1.0
            out.append([v / norm for v in vals])
        return out


def resolve_device(requested: str | None, torch) -> str:      # pragma: no cover
    """Resolve --device, and never quietly pretend CPU is what was asked for.

    The old behaviour was ``device or ("cuda" if available else "cpu")``. On a
    machine whose torch is a ``+cpu`` build that silently selects CPU, which is
    how a 781k-chunk job ends up running on a CPU by accident and has to be
    killed. So:

        "cuda"  a requirement -- if CUDA is unavailable this raises and says why
        "cpu"   an explicit choice
        "auto"  the old fallback, but it prints which device it picked

    Anything else (``cuda:0``) is passed through to torch untouched.
    """
    choice = (requested or "auto").strip().lower()
    if choice == "auto":
        if torch.cuda.is_available():
            return "cuda"
        print("  device: CUDA not available, falling back to CPU (pass --device cuda "
              "to make this an error instead)", file=sys.stderr)
        return "cpu"
    if choice.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit(
            "--device cuda was requested but torch.cuda.is_available() is False.\n"
            f"  torch {torch.__version__} is installed; a '+cpu' build has no CUDA support.\n"
            "  Install a CUDA build matching your driver, e.g.:\n"
            "    pip uninstall -y torch\n"
            "    pip install torch==2.4.1+cu121 --index-url "
            "https://download.pytorch.org/whl/cu121\n"
            "  Then re-check with: python -c \"import torch; print(torch.cuda.is_available())\""
        )
    return choice


class MedCPTEncoder:
    """The production encoder. Requires torch + transformers; GPU strongly advised."""

    production = True

    def __init__(self, model_id: str = ARTICLE_ENCODER, device: str | None = None,
                 batch_size: int = DEFAULT_BATCH_SIZE, min_batch_size: int = 1):
        try:
            import torch                                    # noqa: F401
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:                           # pragma: no cover
            raise SystemExit(
                "MedCPT requires torch and transformers:\n"
                "  pip install torch transformers\n"
                f"(import failed: {exc})"
            ) from exc
        import torch
        self.torch = torch
        self.name = model_id
        self.dim = MEDCPT_DIM
        self.batch_size = batch_size
        self.min_batch_size = max(1, min_batch_size)
        self.device = resolve_device(device, torch)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id).to(self.device).eval()

    def describe_device(self) -> str:                        # pragma: no cover
        torch = self.torch
        if not str(self.device).startswith("cuda"):
            return f"{self.device} (CPU)"
        index = torch.cuda.current_device()
        name = torch.cuda.get_device_name(index)
        total = torch.cuda.get_device_properties(index).total_memory / (1024 ** 3)
        return f"{self.device} -> {name}, {total:.1f} GiB VRAM"

    def _encode_batch(self, batch: list[str]):               # pragma: no cover
        torch = self.torch
        enc = self.tokenizer(batch, truncation=True, padding=True,
                             max_length=MEDCPT_MAX_TOKENS,
                             return_tensors="pt").to(self.device)
        # MedCPT uses the [CLS] representation.
        vecs = self.model(**enc).last_hidden_state[:, 0, :]
        vecs = torch.nn.functional.normalize(vecs, p=2, dim=1)
        return vecs.float().cpu().tolist()

    def encode(self, texts: list[str]) -> list[list[float]]:  # pragma: no cover
        """Encode, halving the batch on CUDA OOM rather than dying.

        A 4 GiB card running 512-token sequences can meet a batch it cannot fit
        even when the configured size is usually fine, because padding is
        dynamic and a batch of long chunks is far heavier than an average one.
        Backing off keeps the run alive; ``self.batch_size`` stays reduced so
        the rest of the run does not repeat the same failure. It never falls
        back to CPU -- a silent device change mid-run would make the index
        inconsistent and take days.
        """
        torch = self.torch
        out: list[list[float]] = []
        index = 0
        with torch.no_grad():
            while index < len(texts):
                size = self.batch_size
                while True:
                    batch = texts[index:index + size]
                    try:
                        out.extend(self._encode_batch(batch))
                        break
                    except torch.cuda.OutOfMemoryError:
                        if size <= self.min_batch_size:
                            raise SystemExit(
                                f"CUDA out of memory even at batch size {size}. "
                                "Free VRAM (close other GPU programs) or lower "
                                "--batch-size further, then re-run -- the job resumes "
                                "from the last completed row."
                            )
                        size = max(self.min_batch_size, size // 2)
                        self.batch_size = size
                        torch.cuda.empty_cache()
                        print(f"  CUDA OOM: batch size reduced to {size}", file=sys.stderr)
                index += len(batch)
        return out


def get_encoder(kind: str, dim: int, device: str | None, batch_size: int,
                min_batch_size: int = 1):
    if kind == "medcpt":
        return MedCPTEncoder(ARTICLE_ENCODER, device, batch_size, min_batch_size)
    if kind == "stub":
        return StubEncoder(dim)
    raise SystemExit(f"unknown encoder: {kind}")


# ---------------------------------------------------------------------------
# Chunk reading
# ---------------------------------------------------------------------------
def read_chunks(path: Path, skip_duplicates: bool) -> list[dict]:
    """All chunks in deterministic (sorted chunk_id) order.

    Exact-duplicate chunk texts were flagged, not deleted, by build_chunks.
    Embedding them again adds nothing to the index, so they are skipped by
    default -- but the chunk layer itself is untouched and the duplicate is
    still resolvable through duplicate_of.
    """
    rows: list[dict] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            c = json.loads(line)
            if skip_duplicates and c.get("duplicate_of"):
                continue
            rows.append(c)
    rows.sort(key=lambda c: c["chunk_id"])
    return rows


_COMPOSE_EMBED_TEXT = None


def embed_text_of(chunk: dict) -> str:
    """The exact string the encoder sees. One shared rule with the chunk layer.

    The import is resolved once. It used to run per chunk, which pushed a copy
    of this directory onto ``sys.path`` on every call -- 781k entries over a
    production run.
    """
    global _COMPOSE_EMBED_TEXT
    if _COMPOSE_EMBED_TEXT is None:
        here = str(Path(__file__).resolve().parent)
        if here not in sys.path:
            sys.path.insert(0, here)
        from build_chunks import compose_embed_text
        _COMPOSE_EMBED_TEXT = compose_embed_text
    return _COMPOSE_EMBED_TEXT(chunk.get("title", ""), chunk.get("section_heading", ""),
                               chunk.get("text", ""))


def batched(items: list, size: int) -> Iterator[list]:
    for i in range(0, len(items), size):
        yield items[i:i + size]


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------
def resume_state(vec_path: Path, man_path: Path, dim: int):
    """Rows already complete in *both* output files, with the digest replayed.

    Returns ``(rows, digest, by_category, last_chunk_id)``. A run killed
    mid-write can leave a partial vector or a partial manifest line, so a row
    counts only when its manifest entry parses, declares the row number it sits
    at, and has a whole vector behind it. Both files are then truncated to that
    boundary, which is what makes appending safe.

    The digest is replayed over the kept rows rather than recomputed at the end,
    so a resumed index carries the same content_digest as an uninterrupted one.
    """
    empty = (0, hashlib.sha256(), {}, "")
    if not vec_path.exists() or not man_path.exists():
        return empty

    row_bytes = dim * 4
    available_rows = vec_path.stat().st_size // row_bytes
    digest = hashlib.sha256()
    by_category: dict[str, int] = {}
    kept = 0
    last_chunk_id = ""
    manifest_bytes = 0

    with vec_path.open("rb") as vf, man_path.open("rb") as mf:
        while kept < available_rows:
            position = mf.tell()
            raw_line = mf.readline()
            if not raw_line:
                break
            text = raw_line.decode("utf-8", "replace").strip()
            if not text:
                manifest_bytes = mf.tell()
                continue
            try:
                row = json.loads(text)
            except ValueError:                      # partial final line
                mf.seek(position)
                break
            if int(row.get("row", -1)) != kept:     # misaligned: stop at the last good row
                mf.seek(position)
                break
            vector = vf.read(row_bytes)
            if len(vector) != row_bytes:            # partial final vector
                mf.seek(position)
                break
            chunk_id = str(row.get("chunk_id", ""))
            digest.update(chunk_id.encode("utf-8"))
            digest.update(vector)
            category = row.get("source_category", "")
            by_category[category] = by_category.get(category, 0) + 1
            last_chunk_id = chunk_id
            kept += 1
            manifest_bytes = mf.tell()

    if kept == 0:
        return empty
    # Drop anything past the last complete row so the append lands cleanly.
    with man_path.open("r+b") as fh:
        fh.truncate(manifest_bytes)
    with vec_path.open("r+b") as fh:
        fh.truncate(kept * row_bytes)
    return kept, digest, by_category, last_chunk_id


def build(chunks_path: Path, out_dir: Path, encoder, batch_size: int,
          skip_duplicates: bool = True, resume: bool = False,
          progress_every: int = 0, limit: int = 0) -> dict:
    chunks = read_chunks(chunks_path, skip_duplicates)
    if limit:
        chunks = chunks[:limit]
    out_dir.mkdir(parents=True, exist_ok=True)
    vec_path = out_dir / "embeddings.f32"
    man_path = out_dir / "index_manifest.jsonl"

    digest = hashlib.sha256()
    n = 0
    by_category: dict[str, int] = {}

    if resume:
        n, digest, by_category, last_chunk_id = resume_state(vec_path, man_path, encoder.dim)
        if n:
            if n > len(chunks):
                raise SystemExit(
                    f"{vec_path} already holds {n} rows but the chunk layer has only "
                    f"{len(chunks)}. This index was built from different chunks; move it "
                    "aside or pass --restart."
                )
            if chunks[n - 1]["chunk_id"] != last_chunk_id:
                raise SystemExit(
                    f"resume mismatch at row {n - 1}: the index has chunk_id "
                    f"{last_chunk_id!r} but the chunk layer has "
                    f"{chunks[n - 1]['chunk_id']!r}. The chunk file changed since this "
                    "index was started; move the index aside or pass --restart."
                )
            print(f"  resuming at row {n:,} of {len(chunks):,} "
                  f"({100.0 * n / len(chunks):.1f}% already done)")
            chunks = chunks[n:]

    total = n + len(chunks)
    vec_mode, man_mode = ("ab", "a") if n else ("wb", "w")
    next_report = progress_every

    with vec_path.open(vec_mode) as vf, \
            man_path.open(man_mode, encoding="utf-8", newline="\n") as mf:
        for batch in batched(chunks, batch_size):
            vectors = encoder.encode([embed_text_of(c) for c in batch])
            if len(vectors) != len(batch):
                raise SystemExit("encoder returned the wrong number of vectors")
            for chunk, vec in zip(batch, vectors):
                if len(vec) != encoder.dim:
                    raise SystemExit(
                        f"encoder returned dim {len(vec)}, expected {encoder.dim}")
                buf = array.array("f", vec)
                if sys.byteorder != "little":                # pragma: no cover
                    buf.byteswap()
                raw = buf.tobytes()
                vf.write(raw)
                digest.update(chunk["chunk_id"].encode("utf-8"))
                digest.update(raw)
                row = {k: chunk.get(k, "") for k in MANIFEST_FIELDS}
                row["row"] = n
                mf.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                cat = chunk.get("source_category", "")
                by_category[cat] = by_category.get(cat, 0) + 1
                n += 1
            # Flush every batch so an interrupted run loses at most one batch;
            # resume_state() trims any partial tail on the next start.
            vf.flush()
            mf.flush()
            if progress_every and n >= next_report:
                print(f"  {n:,} / {total:,} vectors ({100.0 * n / total:.1f}%)", flush=True)
                next_report = n + progress_every

    meta = {
        "encoder": encoder.name,
        "production": bool(getattr(encoder, "production", False)),
        "dim": encoder.dim,
        "vectors": n,
        "vectors_by_source_category": dict(sorted(by_category.items())),
        "skipped_exact_duplicates": skip_duplicates,
        "dtype": "float32",
        "byte_order": "little",
        "normalized": True,
        "similarity": "inner-product (cosine on normalized vectors)",
        "search": "exact flat scan -- no approximate index, so results are reproducible",
        "content_digest": digest.hexdigest(),
        "chunks_source": str(chunks_path.name),
        "query_encoder": QUERY_ENCODER,
        "cross_encoder": CROSS_ENCODER,
    }
    if limit:
        # A truncated index must never be mistaken for the production one.
        meta["partial_index_limit"] = limit
    device = getattr(encoder, "device", None)
    if device:
        meta["device"] = str(device)
    (out_dir / "index_meta.json").write_text(
        json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8",
        newline="\n")
    return meta


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Embed the frozen chunk layer.")
    ap.add_argument("--chunks", type=Path, default=DEFAULT_CHUNKS)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--encoder", choices=["medcpt", "stub"], default="medcpt")
    ap.add_argument("--dim", type=int, default=64, help="stub encoder dimension only")
    ap.add_argument("--device", default="auto",
                    help="auto (default) | cuda | cpu | cuda:N. 'cuda' is a requirement: "
                         "it fails loudly rather than silently running on the CPU")
    ap.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
                    help=f"sequences per forward pass (default {DEFAULT_BATCH_SIZE}, "
                         "sized for a 4 GiB GPU; halved automatically on CUDA OOM)")
    ap.add_argument("--min-batch-size", type=int, default=1,
                    help="floor for the OOM backoff before giving up")
    ap.add_argument("--restart", action="store_true",
                    help="discard any existing index and embed from row 0 "
                         "(default is to resume where the last run stopped)")
    ap.add_argument("--limit", type=int, default=0,
                    help="embed only the first N chunks -- for a GPU smoke test")
    ap.add_argument("--progress-every", type=int, default=5000,
                    help="print progress every N vectors (0 to silence)")
    ap.add_argument("--include-duplicates", action="store_true",
                    help="also embed chunks flagged as exact duplicates")
    ap.add_argument("--allow-stub", action="store_true",
                    help="permit writing a non-production stub index")
    args = ap.parse_args(argv)

    if args.batch_size < 1:
        raise SystemExit("--batch-size must be at least 1")

    if args.encoder == "stub" and not args.allow_stub:
        raise SystemExit(
            "Refusing to write a stub index without --allow-stub.\n"
            f"The thesis requires {ARTICLE_ENCODER}; use --encoder medcpt."
        )

    encoder = get_encoder(args.encoder, args.dim, args.device, args.batch_size,
                          args.min_batch_size)
    if hasattr(encoder, "describe_device"):                  # pragma: no cover
        print(f"  encoder: {encoder.name}")
        print(f"  device:  {encoder.describe_device()}")
        print(f"  batch:   {encoder.batch_size}")

    # MedCPTEncoder may already have reduced its own batch size; the stub has none.
    effective_batch = getattr(encoder, "batch_size", args.batch_size)
    meta = build(args.chunks, args.out, encoder, effective_batch,
                 skip_duplicates=not args.include_duplicates,
                 resume=not args.restart,
                 progress_every=max(0, args.progress_every),
                 limit=max(0, args.limit))
    print(f"Index written to {args.out}")
    for k, v in meta.items():
        print(f"  {k:30} {v}")
    if not meta["production"]:
        print("\n  WARNING: non-production stub index. Not valid for thesis results.")
    if meta.get("partial_index_limit"):
        print(f"\n  WARNING: partial index -- only the first {meta['partial_index_limit']} "
              "chunks were embedded (--limit). Not the production index.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
