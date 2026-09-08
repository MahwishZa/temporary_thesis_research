#!/usr/bin/env python3
"""Thesis-level configuration.

Deliberately *not* a second configuration system. RAG2 already has one
(``rag2.config``: dataclasses, YAML, a ``base:`` inheritance chain and
``section.key=value`` overrides), and it is the mechanism the baseline's
reproducibility manifests depend on. This module reuses that machinery and adds
only the keys the architecture needs above RAG2's level -- which corpus, which
query set, which experimental condition, which temporal policy, where results go.

The RAG2 configuration is *referenced*, never copied: ``rag2_config`` points at
a file under ``rag2/configs/``, and RAG2's own loader parses it. So the settings
that govern reproduction-critical behaviour keep exactly one definition, in the
tree that owns them.

    thesis config  (this file)          configs/thesis/architecture.yaml
      corpus:      digest, paths            |
      queries:     query set                | rag2_config: -> rag2/configs/thesis_corpus.yaml
      condition:   which arm                |      retrieval:  MedCPT ids, top-k, balance
      recency:     temporal policy          |      filter:     checkpoint, on_empty
      evaluation:  metrics                  |      llm:        backend, model, temperature
      output:      run directory            |      cache:      candidate replay
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from typing import Any, Dict, List, Mapping, Optional, Type, TypeVar, get_type_hints

from ._bootstrap import rag2_config_module

T = TypeVar("T")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------
@dataclass
class CorpusSection:
    """Where the frozen evidence corpus lives, and what it must be.

    ``expected_chunk_digest`` is the provenance contract: the digest of the
    production chunk build a result is allowed to come from. It is *verified*
    at load time, not assumed -- see thesis.corpus. An empty string means
    "unverified", which is honest for a wiring test and unacceptable for a
    reported result.
    """

    chunks_path: str = "pmc/chunks/chunks.jsonl"
    chunk_stats_path: str = "pmc/chunks/chunk_stats.json"
    index_dir: str = "pmc/index"
    expected_chunk_digest: str = ""
    expected_chunk_count: Optional[int] = None
    expected_document_count: Optional[int] = None
    require_production_index: bool = True
    # Source categories the corpus pipeline defines. Balanced retrieval draws an
    # equal quota from each (base paper 3.4).
    source_categories: List[str] = field(
        default_factory=lambda: ["pubmed-abstract", "pmc-fulltext", "currency-pack"]
    )


@dataclass
class QueriesSection:
    """The evaluation query set. Identical across conditions, by construction."""

    path: str = ""
    name: str = ""
    version: str = ""
    limit: Optional[int] = None


@dataclass
class RetrievalSection:
    """Retrieval settings owned at the architecture level.

    The encoder identities live in the RAG2 config (one definition); what varies
    per experiment is depth and whether reranking runs.
    """

    per_category: int = 10
    final_top_k: int = 8
    rerank: bool = True
    candidates_dir: str = "pmc/candidates"
    # Replay a saved candidate set instead of retrieving. This is how two
    # conditions are proven to score the same evidence population.
    replay_from: str = ""


@dataclass
class RecencySection:
    """The temporal-policy boundary.

    ``policy: none`` is the baseline: dates are carried and never read. Any other
    value names a policy in thesis.recency's registry. The thesis has not yet
    fixed its recency method, so no policy that scores or reweights by date is
    implemented here -- see thesis/recency.py.
    """

    policy: str = "none"
    options: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConditionSection:
    """Which experimental arm to run."""

    name: str = "baseline"
    notes: str = ""


@dataclass
class EvaluationSection:
    metrics: List[str] = field(default_factory=lambda: ["retrieval_report"])
    references_path: str = ""


@dataclass
class OutputSection:
    dir: str = "experiments/{condition}/runs/{name}"
    write_candidates: bool = True


@dataclass
class ThesisConfig:
    name: str = "thesis-run"
    seed: int = 42
    #: Path to the RAG2 configuration governing reproduction-critical settings.
    #: Referenced, not duplicated.
    rag2_config: str = "rag2/configs/thesis_corpus.yaml"
    corpus: CorpusSection = field(default_factory=CorpusSection)
    queries: QueriesSection = field(default_factory=QueriesSection)
    retrieval: RetrievalSection = field(default_factory=RetrievalSection)
    recency: RecencySection = field(default_factory=RecencySection)
    condition: ConditionSection = field(default_factory=ConditionSection)
    evaluation: EvaluationSection = field(default_factory=EvaluationSection)
    output: OutputSection = field(default_factory=OutputSection)

    # -- helpers ----------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def fingerprint(self) -> str:
        """Stable short hash of the whole configuration, for the run manifest."""
        from .provenance import stable_hash

        return stable_hash(self.to_dict())

    def resolved_output_dir(self) -> str:
        return self.output.dir.format(condition=self.condition.name, name=self.name)

    def path(self, relative: str) -> str:
        """Resolve a repo-relative config path against the repository root."""
        if not relative:
            return ""
        return relative if os.path.isabs(relative) else os.path.join(REPO_ROOT, relative)

    def load_rag2_config(self):
        """Parse the referenced RAG2 config with RAG2's own loader.

        Returns ``None`` when no RAG2 config is referenced, which is valid for a
        retrieval-only condition that never calls the baseline.
        """
        if not self.rag2_config:
            return None
        module = rag2_config_module()
        return module.load_config(self.path(self.rag2_config))


# ---------------------------------------------------------------------------
# Loading -- same shape as rag2.config, reusing its YAML and merge helpers
# ---------------------------------------------------------------------------
def _from_mapping(cls: Type[T], payload: Mapping[str, Any], path: str = "") -> T:
    if not is_dataclass(cls):
        return payload  # type: ignore[return-value]
    known = {f.name: f for f in fields(cls)}
    unknown = set(payload) - set(known)
    if unknown:
        where = path or cls.__name__
        raise ValueError(f"unknown config key(s) under {where}: {sorted(unknown)}")
    hints = get_type_hints(cls)
    kwargs: Dict[str, Any] = {}
    for name, value in payload.items():
        hint = hints.get(name)
        sub = f"{path}.{name}" if path else name
        if is_dataclass(hint) and isinstance(value, Mapping):
            kwargs[name] = _from_mapping(hint, value, sub)
        else:
            kwargs[name] = value
    return cls(**kwargs)  # type: ignore[arg-type]


def load_config(
    path: Optional[str] = None, overrides: Optional[Mapping[str, Any]] = None
) -> ThesisConfig:
    """Load a thesis config, following its ``base:`` chain, then apply overrides.

    Both the YAML reader and the deep-merge come from ``rag2.config`` so the two
    layers cannot drift in how they interpret a config file.
    """
    module = rag2_config_module()
    payload: Dict[str, Any] = {}
    if path:
        payload = _load_with_base(path, module, seen=set())
    if overrides:
        payload = module._deep_merge(payload, overrides)
    return _from_mapping(ThesisConfig, payload)


def _load_with_base(path: str, module, seen: set) -> Dict[str, Any]:
    real = os.path.realpath(path)
    if real in seen:
        raise ValueError(f"circular config base chain at {path}")
    seen.add(real)
    payload = module.load_yaml(path)
    base = payload.pop("base", None)
    if not base:
        return payload
    base_path = base if os.path.isabs(base) else os.path.join(os.path.dirname(real), base)
    return module._deep_merge(_load_with_base(base_path, module, seen), payload)


def merge_overrides(items: List[str]) -> Dict[str, Any]:
    """Parse ``section.key=value`` CLI overrides, using RAG2's parser."""
    module = rag2_config_module()
    merged: Dict[str, Any] = {}
    for item in items:
        merged = module._deep_merge(merged, module.parse_override(item))
    return merged
