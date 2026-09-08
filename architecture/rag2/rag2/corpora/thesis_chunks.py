"""Corpus backed by the thesis chunk layer built under ``pmc/``.

The baseline's other loader (:mod:`rag2.corpora.json_corpus`) reads the article
/embedding layout of the authors' release. This repository's corpus is built by
``pmc/build_chunks.py`` and ``pmc/embed_chunks.py`` instead, which produce a
different but equivalent pair of artifacts:

    pmc/chunks/chunks.jsonl      one JSON object per chunk, carrying ``text``
    pmc/index/embeddings.f32     row-major float32 MedCPT vectors, 768-dim
    pmc/index/index_manifest.jsonl  one row per vector: chunk_id + provenance
    pmc/index/index_meta.json    encoder id, dim, counts, digests, production flag

Two properties of that layout drive this module:

* **The manifest carries no text.** ``embed_chunks.MANIFEST_FIELDS`` deliberately
  omits it, so the vectors and their provenance can be shipped without the
  corpus body. Text is therefore joined back in from ``chunks.jsonl`` on
  ``chunk_id``.
* **Row order is the alignment contract.** ``index_manifest.jsonl`` line *i*
  describes row *i* of ``embeddings.f32``. This loader preserves that mapping;
  a mismatch between the two files is a hard error, never a silent misalignment.

Balanced retrieval, one corpus per source category
--------------------------------------------------
The paper draws an equal quota from each of four corpora (section 3.4). This
repository's corpus carries an equivalent ``source_category`` field
(``pubmed-abstract``, ``pmc-fulltext``, ``currency-pack``), so one instance of
this loader is configured **per category** and balanced retrieval proceeds
unchanged. ``source_category: ""`` loads every row as a single corpus, which
disables balance and is only useful for diagnostics.

Provenance policy
-----------------
Every manifest field except the ones that become ``Evidence`` identity
(``chunk_id``, ``document_id``, ``title``, ``text``) is preserved verbatim in
``Evidence.metadata`` -- publication dates, authority tier, currency-pack
membership and retraction flags included. **No baseline component reads any of
it**; it is carried so the thesis's later recency-bias experiments have it
without re-running retrieval. ``tests/test_metadata_isolation.py`` enforces that.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Iterator, List, Optional, Tuple

from ..config import CorpusConfig
from ..schema import Evidence
from .base import Corpus, register_corpus

#: Manifest fields that become Evidence identity rather than metadata.
_IDENTITY_FIELDS = ("chunk_id", "document_id", "title", "row", "text")

#: Default vector width of the thesis index (MedCPT article encoder).
DEFAULT_DIM = 768

#: Rows per embedding shard. Bounds peak memory when a category holds hundreds
#: of thousands of vectors; the search result is identical either way because
#: ``retrieval.index`` merges shards by score.
DEFAULT_SHARD_ROWS = 200_000


def _read_jsonl(path: str) -> Iterator[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


class ThesisChunkCorpus(Corpus):
    """One source category of the ``pmc/`` chunk layer, as a retrievable corpus."""

    def __init__(self, config: CorpusConfig) -> None:
        options = dict(config.options or {})
        self.name = config.name
        self.config = config
        self.index_dir = str(options.get("index_dir", ""))
        self.chunks_path = str(options.get("chunks_path", ""))
        self.source_category = str(options.get("source_category", config.name or ""))
        self.dim = int(options.get("dim", 0)) or DEFAULT_DIM
        self.shard_rows = int(options.get("shard_rows", DEFAULT_SHARD_ROWS))
        self.require_production = bool(options.get("require_production_index", True))
        self.drop_duplicates = bool(options.get("drop_duplicate_chunks", True))

        if not self.index_dir or not self.chunks_path:
            raise ValueError(
                f"corpus {config.name!r}: the thesis_chunks loader needs both "
                "options.index_dir (holding embeddings.f32 + index_manifest.jsonl) "
                "and options.chunks_path (pmc/chunks/chunks.jsonl)"
            )

        self.manifest_path = os.path.join(self.index_dir, "index_manifest.jsonl")
        self.vectors_path = os.path.join(self.index_dir, "embeddings.f32")
        self.meta_path = os.path.join(self.index_dir, "index_meta.json")
        missing = [
            p
            for p in (self.manifest_path, self.vectors_path, self.chunks_path)
            if not os.path.exists(p)
        ]
        if missing:
            raise FileNotFoundError(
                f"corpus {config.name!r}: {len(missing)} required file(s) not found, first: "
                f"{missing[0]}. Build the chunk layer with pmc/build_chunks.py and the index "
                "with pmc/embed_chunks.py before running retrieval."
            )

        self.index_meta: Dict[str, Any] = {}
        if os.path.exists(self.meta_path):
            with open(self.meta_path, "r", encoding="utf-8") as handle:
                self.index_meta = json.load(handle)
            meta_dim = int(self.index_meta.get("dim", 0))
            if meta_dim:
                self.dim = meta_dim
            if self.require_production and not self.index_meta.get("production", False):
                raise ValueError(
                    f"corpus {config.name!r}: {self.meta_path} reports production=false, i.e. the "
                    f"index was built with a stub encoder ({self.index_meta.get('encoder')!r}). "
                    "Stub vectors are not valid for reported results. Rebuild with "
                    "pmc/embed_chunks.py --encoder medcpt, or set "
                    "options.require_production_index=false for a wiring test."
                )

        self._rows: List[Dict[str, Any]] = []      # manifest rows for this category
        self._vector_rows: List[int] = []          # their row numbers in embeddings.f32
        self._texts: Dict[str, str] = {}
        self._load_manifest()
        self._load_texts()

    # -- loading -----------------------------------------------------------
    def _load_manifest(self) -> None:
        expected_row = 0
        seen_categories: Dict[str, int] = {}
        for line_no, row in enumerate(_read_jsonl(self.manifest_path)):
            category = str(row.get("source_category", ""))
            seen_categories[category] = seen_categories.get(category, 0) + 1
            # ``row`` is written by embed_chunks and must match the file order:
            # it is the index into embeddings.f32.
            declared = row.get("row", line_no)
            if int(declared) != expected_row:
                raise ValueError(
                    f"corpus {self.name!r}: {self.manifest_path} line {line_no + 1} declares "
                    f"row={declared} but is at position {expected_row}. The manifest must be in "
                    "embeddings.f32 row order; retrieval would decode to the wrong passage."
                )
            expected_row += 1
            if self.source_category and row.get("source_category") != self.source_category:
                continue
            if self.drop_duplicates and row.get("duplicate_of"):
                continue
            self._rows.append(row)
            self._vector_rows.append(int(declared))

        self.total_index_rows = expected_row
        expected_bytes = self.total_index_rows * self.dim * 4
        actual_bytes = os.path.getsize(self.vectors_path)
        if actual_bytes != expected_bytes:
            raise ValueError(
                f"corpus {self.name!r}: {self.vectors_path} is {actual_bytes} bytes but the "
                f"manifest declares {self.total_index_rows} rows of {self.dim} float32 "
                f"({expected_bytes} bytes). The index and its manifest are out of step."
            )
        if not self._rows:
            known = dict(sorted(seen_categories.items()))
            raise ValueError(
                f"corpus {self.name!r}: no manifest rows matched source_category="
                f"{self.source_category!r} (of {self.total_index_rows} rows). Check "
                f"options.source_category against the values in {self.manifest_path} "
                f"(categories present, with row counts: {known})."
            )

    def _load_texts(self) -> None:
        """Join chunk text back in on ``chunk_id``; the manifest carries none."""
        wanted = {row["chunk_id"] for row in self._rows}
        for chunk in _read_jsonl(self.chunks_path):
            chunk_id = chunk.get("chunk_id")
            if chunk_id in wanted:
                self._texts[chunk_id] = chunk.get("text", "")
        missing = [row["chunk_id"] for row in self._rows if row["chunk_id"] not in self._texts]
        if missing:
            raise ValueError(
                f"corpus {self.name!r}: {len(missing)} indexed chunk(s) have no text in "
                f"{self.chunks_path}, first: {missing[0]!r}. The index and the chunk layer "
                "were built from different runs; rebuild one of them."
            )

    # -- Corpus ------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._rows)

    def passage(self, index: int) -> Evidence:
        if index < 0 or index >= len(self._rows):
            raise IndexError(
                f"{self.name}: passage index {index} out of range (0..{len(self._rows) - 1})"
            )
        row = self._rows[index]
        chunk_id = row["chunk_id"]
        metadata = {k: v for k, v in row.items() if k not in _IDENTITY_FIELDS}
        return Evidence(
            text=self._texts.get(chunk_id, ""),
            source=self.name,
            doc_id=str(row.get("document_id", "")) or None,
            passage_id=str(chunk_id),
            corpus_index=index,
            metadata=metadata,
        )

    def embedding_shards(self) -> Iterator[Tuple[int, Any]]:
        """Yield ``(offset, matrix)`` covering this category's vectors in order.

        ``embeddings.f32`` is memory-mapped, so only the selected rows are ever
        materialised even when the file holds the whole corpus.
        """
        import numpy as np

        full = np.memmap(
            self.vectors_path, dtype="float32", mode="r", shape=(self.total_index_rows, self.dim)
        )
        step = max(self.shard_rows, 1)
        for start in range(0, len(self._vector_rows), step):
            rows = self._vector_rows[start : start + step]
            yield start, np.ascontiguousarray(full[rows], dtype=np.float32)

    def describe(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "loader": "thesis_chunks",
            "source_category": self.source_category,
            "passages": len(self._rows),
            "index_rows_total": self.total_index_rows,
            "dim": self.dim,
            "encoder": self.index_meta.get("encoder"),
            "production_index": self.index_meta.get("production"),
            "index_content_digest": self.index_meta.get("content_digest"),
            "drop_duplicate_chunks": self.drop_duplicates,
        }


def index_categories(index_dir: str) -> Dict[str, int]:
    """Row count per ``source_category`` in an index manifest. A setup helper."""
    counts: Dict[str, int] = {}
    for row in _read_jsonl(os.path.join(index_dir, "index_manifest.jsonl")):
        category = str(row.get("source_category", ""))
        counts[category] = counts.get(category, 0) + 1
    return dict(sorted(counts.items()))


@register_corpus("thesis_chunks")
def _build_thesis_chunks(config: CorpusConfig) -> Corpus:
    return ThesisChunkCorpus(config)


def build_thesis_corpora(
    index_dir: str,
    chunks_path: str,
    categories: Optional[List[str]] = None,
    **options: Any,
) -> List[ThesisChunkCorpus]:
    """One corpus per source category, ready for balanced retrieval."""
    names = categories if categories is not None else list(index_categories(index_dir))
    return [
        ThesisChunkCorpus(
            CorpusConfig(
                name=name,
                loader="thesis_chunks",
                options={
                    "index_dir": index_dir,
                    "chunks_path": chunks_path,
                    "source_category": name,
                    **options,
                },
            )
        )
        for name in names
    ]
