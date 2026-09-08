#!/usr/bin/env python3
"""Stable corpus interface over the frozen PMC chunk layer.

The corpus is an established asset: built by ``pmc/build_chunks.py``, embedded by
``pmc/embed_chunks.py``, verified by ``pmc/verify_index.py``. This module does
not rebuild, re-chunk or re-embed any of it. It opens what is on disk, checks it
is the corpus the configuration claims, and exposes each chunk through one
record shape that the rest of the architecture can depend on:

    chunk_id, document_id, source_category, text,
    canonical_date, date_precision, publication metadata,
    plus every other field the corpus pipeline already established.

**Digest verification is the point.** ``CorpusHandle.open`` compares the live
corpus against ``corpus.expected_chunk_digest`` / ``expected_chunk_count`` and
raises on a mismatch. A result whose corpus cannot be identified is not a
result, and silently recording whatever digest happened to be present would make
the mismatch invisible -- exactly the failure mode provenance exists to prevent.

Note on the current repository state: the committed
``pmc/chunks/chunk_stats.json`` records a *partial* container run (60,874 chunks
over 76 parsed records), not the production build. That is documented in
``docs/rag2_reproduction_audit.md`` section 10. Configure
``corpus.expected_chunk_digest`` with the production digest and this loader will
tell you, loudly, whether the corpus on the machine is that one.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence

from .config import CorpusSection, ThesisConfig
from .provenance import CorpusStamp

#: Fields the corpus pipeline establishes that must survive into evidence.
#: Temporal fields are carried here and read only by a temporal policy.
PASSTHROUGH_FIELDS = (
    # Identity
    "chunk_id", "document_id", "pmcid", "pmid", "doi", "title",
    # Category and eligibility
    "source_category", "eligibility_status",
    # Temporal -- carried by every stage, read only by a temporal policy
    "canonical_date", "date_precision", "date_source", "split_june_2024",
    # Authority / provenance overlays established by the M1-M4 metadata pass
    "authority_tier_label", "guideline_family", "in_currency_pack", "retracted",
    "license_code", "license_band",
    # Structure
    "location", "section_id", "section_heading", "imrad", "word_count",
    "text_sha256", "duplicate_of", "row",
)


class CorpusError(RuntimeError):
    """Raised when the corpus on disk is not the corpus the config claims."""


@dataclass
class CorpusHandle:
    """An opened, identity-checked view of the frozen chunk corpus."""

    chunks_path: str
    index_dir: str
    stats: Dict[str, Any] = field(default_factory=dict)
    index_meta: Dict[str, Any] = field(default_factory=dict)
    digest_verified: bool = False
    source_categories: List[str] = field(default_factory=list)

    # -- construction -----------------------------------------------------
    @classmethod
    def open(cls, config: ThesisConfig, require_index: Optional[bool] = None) -> "CorpusHandle":
        section: CorpusSection = config.corpus
        chunks_path = config.path(section.chunks_path)
        stats_path = config.path(section.chunk_stats_path)
        index_dir = config.path(section.index_dir)

        stats = _read_json(stats_path)
        index_meta = _read_json(os.path.join(index_dir, "index_meta.json"))

        handle = cls(
            chunks_path=chunks_path,
            index_dir=index_dir,
            stats=stats,
            index_meta=index_meta,
            source_categories=list(section.source_categories),
        )
        handle._verify(section, require_index)
        return handle

    def _verify(self, section: CorpusSection, require_index: Optional[bool]) -> None:
        """Refuse a corpus that is not the one the configuration names."""
        expected_digest = (section.expected_chunk_digest or "").strip()
        actual_digest = str(
            self.stats.get("chunk_digest")
            or self.stats.get("digest")
            or self.index_meta.get("chunk_digest")
            or ""
        ).strip()

        if expected_digest:
            if not actual_digest:
                raise CorpusError(
                    f"corpus.expected_chunk_digest is set to {expected_digest[:12]}... but the "
                    f"corpus on disk reports no digest. {self.chunks_path} and "
                    f"{os.path.join(self.index_dir, 'index_meta.json')} must come from a build "
                    "that recorded one; rebuild with pmc/build_chunks.py."
                )
            if actual_digest != expected_digest:
                raise CorpusError(
                    f"corpus digest mismatch: configuration expects {expected_digest}, the "
                    f"corpus on disk is {actual_digest}. These are different corpora; a result "
                    "produced from the second cannot be reported against the first."
                )
            self.digest_verified = True

        for label, key, expected in (
            ("chunk", "chunks", section.expected_chunk_count),
            ("document", "documents_chunked", section.expected_document_count),
        ):
            if expected is None:
                continue
            actual = self.stats.get(key)
            if actual is not None and int(actual) != int(expected):
                raise CorpusError(
                    f"corpus {label} count mismatch: configuration expects {expected}, "
                    f"{os.path.basename(self.chunks_path)}'s stats report {actual}."
                )

        wants_index = section.require_production_index if require_index is None else require_index
        if wants_index:
            if not self.index_meta:
                raise CorpusError(
                    f"no index_meta.json under {self.index_dir}. Build the MedCPT index with "
                    "pmc/embed_chunks.py, or set corpus.require_production_index=false for a "
                    "wiring test (never for a reported result)."
                )
            if not self.index_meta.get("production", False):
                raise CorpusError(
                    f"the index under {self.index_dir} was built with a stub encoder "
                    f"({self.index_meta.get('encoder')!r}), not "
                    "ncbi/MedCPT-Article-Encoder. It is valid for wiring tests only."
                )

    # -- record access ----------------------------------------------------
    def iter_chunks(self, limit: Optional[int] = None) -> Iterator[Dict[str, Any]]:
        """Stream chunk records. The corpus is not copied into memory."""
        if not os.path.exists(self.chunks_path):
            raise CorpusError(
                f"chunk file not found: {self.chunks_path}. It is gitignored and rebuilt "
                "deterministically with pmc/build_chunks.py."
            )
        count = 0
        with open(self.chunks_path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                yield normalise_chunk(json.loads(line))
                count += 1
                if limit and count >= limit:
                    return

    def describe(self) -> Dict[str, Any]:
        return {
            "chunks_path": self.chunks_path,
            "index_dir": self.index_dir,
            "digest_verified": self.digest_verified,
            "stats": self.stats,
            "index_meta": {
                k: self.index_meta.get(k) for k in ("encoder", "dim", "vectors", "production")
            },
        }

    def stamp(self) -> CorpusStamp:
        """The provenance stamp recorded in every run record."""
        return CorpusStamp(
            chunk_digest=str(
                self.stats.get("chunk_digest") or self.index_meta.get("chunk_digest") or ""
            ),
            chunk_count=self.stats.get("chunks"),
            document_count=self.stats.get("documents_chunked"),
            unique_chunk_texts=self.stats.get("unique_chunk_texts"),
            window_words=self.stats.get("window_words"),
            overlap_words=self.stats.get("overlap_words"),
            source_categories=list(self.source_categories),
            index_encoder=str(self.index_meta.get("encoder", "")),
            index_content_digest=str(self.index_meta.get("content_digest", "")),
            production_index=self.index_meta.get("production"),
            digest_verified=self.digest_verified,
        )


def normalise_chunk(raw: Mapping[str, Any]) -> Dict[str, Any]:
    """One record shape for the rest of the architecture.

    Known fields are surfaced at the top level; anything else the corpus pipeline
    established is preserved under ``extra`` rather than dropped, so a later
    stage can use metadata this module has never heard of.
    """
    record: Dict[str, Any] = {"text": str(raw.get("text", ""))}
    for key in PASSTHROUGH_FIELDS:
        if key in raw:
            record[key] = raw[key]
    record.setdefault("chunk_id", "")
    record.setdefault("document_id", "")
    record.setdefault("source_category", "")
    consumed = set(PASSTHROUGH_FIELDS) | {"text"}
    extra = {k: v for k, v in raw.items() if k not in consumed}
    if extra:
        record["extra"] = extra
    return record


def _read_json(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload if isinstance(payload, dict) else {}
