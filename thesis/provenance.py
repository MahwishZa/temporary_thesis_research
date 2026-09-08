#!/usr/bin/env python3
"""Provenance: what a research result must carry to be reproducible.

The requirement this implements is that any number reaching the thesis can be
traced back to exactly the corpus and configuration that produced it. Concretely,
every run record answers:

    which corpus?        chunk digest, chunk/document counts, index digest
    which evidence?      chunk_id, document_id, source_category, publication date
    which retrieval?     encoder ids, per-category quota, top-k, rerank on/off
    which condition?     the experimental arm, and its temporal policy
    which query?         query id, and the query set's name/version
    which software?      git commit, package versions, config fingerprint, seed

Two design choices worth stating.

**The corpus digest is verified, not recorded.** Writing down whatever digest the
corpus happens to have would make a mismatch invisible. ``thesis.corpus``
compares the live corpus against ``corpus.expected_chunk_digest`` and refuses to
proceed on a mismatch, so the digest in a record is a checked claim.

**Provenance is carried, not merged into the payload.** Evidence keeps its own
metadata dict end to end; this module snapshots identity, never rewrites it. The
baseline must be able to carry a publication date it never reads.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping, Optional, Sequence

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#: Packages whose versions change results and are therefore recorded.
TRACKED_PACKAGES = ("torch", "transformers", "numpy", "faiss", "accelerate", "sentencepiece")


def stable_hash(payload: Any, length: int = 16) -> str:
    """Deterministic short digest of a JSON-serialisable payload."""
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:length]


def git_state(repo_dir: str = REPO_ROOT) -> Dict[str, Any]:
    def _run(*args: str) -> Optional[str]:
        try:
            return subprocess.check_output(
                ["git", *args], cwd=repo_dir, stderr=subprocess.DEVNULL, text=True
            ).strip()
        except Exception:
            return None

    status = _run("status", "--porcelain")
    return {
        "commit": _run("rev-parse", "HEAD"),
        "branch": _run("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(status) if status is not None else None,
    }


def package_versions() -> Dict[str, Optional[str]]:
    versions: Dict[str, Optional[str]] = {}
    for name in TRACKED_PACKAGES:
        try:
            versions[name] = str(getattr(__import__(name), "__version__", "unknown"))
        except Exception:
            versions[name] = None
    return versions


def environment() -> Dict[str, Any]:
    return {
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "packages": package_versions(),
    }


@dataclass
class CorpusStamp:
    """Identity of the evidence corpus a result came from."""

    chunk_digest: str = ""
    chunk_count: Optional[int] = None
    document_count: Optional[int] = None
    unique_chunk_texts: Optional[int] = None
    window_words: Optional[int] = None
    overlap_words: Optional[int] = None
    source_categories: List[str] = field(default_factory=list)
    index_encoder: str = ""
    index_content_digest: str = ""
    production_index: Optional[bool] = None
    #: False when expected_chunk_digest was empty, i.e. nothing was checked.
    digest_verified: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ModelStamp:
    """Encoder and generator identities, as configured."""

    query_encoder: str = ""
    article_encoder: str = ""
    cross_encoder: str = ""
    generator: str = ""
    generator_revision: str = ""
    filter_kind: str = ""
    filter_checkpoint: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RunRecord:
    """The provenance envelope written beside every run's outputs."""

    run_name: str
    condition: str
    temporal_policy: str = "none"
    created_at: str = ""
    seed: Optional[int] = None
    config_fingerprint: str = ""
    retrieval_fingerprint: str = ""
    query_set: Dict[str, Any] = field(default_factory=dict)
    corpus: Optional[CorpusStamp] = None
    models: Optional[ModelStamp] = None
    retrieval: Dict[str, Any] = field(default_factory=dict)
    git: Dict[str, Any] = field(default_factory=dict)
    environment: Dict[str, Any] = field(default_factory=dict)
    config: Dict[str, Any] = field(default_factory=dict)
    notes: str = ""

    def __post_init__(self) -> None:
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        if not self.git:
            self.git = git_state()
        if not self.environment:
            self.environment = environment()

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["corpus"] = self.corpus.to_dict() if self.corpus else None
        payload["models"] = self.models.to_dict() if self.models else None
        return payload

    def write(self, directory: str, filename: str = "run_record.json") -> str:
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, filename)
        with open(path, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(self.to_dict(), handle, indent=2, ensure_ascii=False, default=str)
            handle.write("\n")
        return path

    def is_reportable(self) -> tuple:
        """Whether this run may back a number in the thesis, and why not.

        A run is reportable only when the corpus digest was actually verified and
        the index is a production (non-stub) build. Anything else is a wiring
        test -- useful, but not evidence.
        """
        reasons: List[str] = []
        if self.corpus is None:
            reasons.append("no corpus stamp recorded")
        else:
            if not self.corpus.digest_verified:
                reasons.append(
                    "corpus digest was not verified (corpus.expected_chunk_digest is empty)"
                )
            if self.corpus.production_index is False:
                reasons.append("index was built with a stub encoder, not MedCPT")
        if self.git.get("dirty"):
            reasons.append("working tree was dirty at run time")
        return (not reasons, reasons)


#: Provenance keys an Evidence record must retain end to end. The retrieval and
#: condition layers are tested against this list, so dropping one fails a test
#: rather than silently producing untraceable evidence.
REQUIRED_EVIDENCE_FIELDS = (
    "chunk_id",
    "document_id",
    "source_category",
)

#: Fields carried for the thesis's later temporal work. The baseline must carry
#: them and must not read them.
CARRIED_TEMPORAL_FIELDS = (
    "canonical_date",
    "date_precision",
    "split_june_2024",
)


def temporal_fields_carried(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Which temporal fields actually survive into retrieved evidence.

    The thesis's recency work reads these off candidates. Whether they arrive
    depends on the index manifest built by ``pmc/embed_chunks.py``, which is a
    separate artefact from the chunk file -- so "the corpus has dates" does not
    by itself mean "retrieval returns them". Recording the answer in every run
    record turns a silent capability loss into a visible fact: a candidate set
    without dates cannot support a recency arm, and the run record says so
    rather than the experiment failing later for unclear reasons.
    """
    present = {field: 0 for field in CARRIED_TEMPORAL_FIELDS}
    for record in records:
        for field_name in CARRIED_TEMPORAL_FIELDS:
            if record.get(field_name) not in (None, ""):
                present[field_name] += 1
    total = len(records)
    return {
        "records": total,
        "present": present,
        "complete": bool(total) and all(count == total for count in present.values()),
        "missing": [f for f, count in present.items() if count < total],
    }


def check_evidence_provenance(records: Sequence[Mapping[str, Any]]) -> List[str]:
    """Return a list of provenance problems; empty means every record is traceable."""
    problems: List[str] = []
    for index, record in enumerate(records):
        for key in REQUIRED_EVIDENCE_FIELDS:
            if not record.get(key):
                problems.append(f"record {index}: missing {key}")
    return problems
