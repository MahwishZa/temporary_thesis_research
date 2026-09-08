#!/usr/bin/env python3
"""The single source of truth for where things live on disk.

Before the repository was reorganised every stage module recomputed the
repository root for itself (``Path(__file__).resolve().parent.parent``) and then
spelled out its own data paths. Ten copies of the same two lines meant the
layout could only be changed in ten places at once. This module computes it
once; every stage imports the constant it needs.

Resolution order, most specific first:

1. ``THESIS_REPO_ROOT`` in the environment, when set. This is the escape hatch
   for running the code against a checkout other than the installed one --
   notably on the GPU machine, where the corpus and the code may not sit
   together.
2. The nearest ancestor of this file containing ``pyproject.toml``. This is what
   happens in a normal checkout, whether or not the package is installed in
   editable mode.
3. Two levels above the ``thesis`` package (``src/thesis`` -> ``src`` -> repo).
   The last-resort fallback, correct for the layout in this repository.

Nothing here creates directories or touches the filesystem beyond the walk in
step 2: importing ``thesis.paths`` on a machine with no corpus is legal, and the
stage that needs a missing input is the one that must say so.
"""

from __future__ import annotations

import os
from pathlib import Path

_MARKER = "pyproject.toml"


def _discover_repo_root() -> Path:
    override = os.environ.get("THESIS_REPO_ROOT")
    if override:
        return Path(override).expanduser().resolve()
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / _MARKER).is_file():
            return parent
    return here.parents[2]


REPO_ROOT = _discover_repo_root()

# Source, tests, configuration, documentation.
SRC_ROOT = REPO_ROOT / "src"
TESTS_ROOT = REPO_ROOT / "tests"
CONFIGS_ROOT = REPO_ROOT / "configs"
DOCS_ROOT = REPO_ROOT / "docs"
EXPERIMENTS_ROOT = REPO_ROOT / "experiments"

# The reproduced original RAG2 baseline, vendored whole.
RAG2_ROOT = REPO_ROOT / "rag2"

# Research data. Committed where it is evidence, gitignored where it is a large
# derived artifact that a documented command rebuilds deterministically.
DATA_ROOT = REPO_ROOT / "data"

PUBMED_DIR = DATA_ROOT / "pubmed"
PUBMED_QUERIES = PUBMED_DIR / "search_queries.txt"
PUBMED_RESULTS_CSV = PUBMED_DIR / "pubmed_results.csv"
PUBMED_RESULTS_JSON = PUBMED_DIR / "pubmed_results.json"
PUBMED_SEARCH_LOG = PUBMED_DIR / "search_log.csv"

PMC_DIR = DATA_ROOT / "pmc"
PMC_INVENTORY_DIR = PMC_DIR / "inventory"
PMC_INVENTORY_CSV = PMC_INVENTORY_DIR / "pmc_oa_inventory.csv"
PMC_INVENTORY_FAILURES_CSV = PMC_INVENTORY_DIR / "pmc_oa_failures.csv"

PMC_FULLTEXT_DIR = PMC_DIR / "fulltext"
PMC_XML_DIR = PMC_FULLTEXT_DIR / "xml"
PMC_MANIFEST_CSV = PMC_FULLTEXT_DIR / "manifest.csv"

PMC_PARSED_DIR = PMC_DIR / "parsed"
PMC_ARTICLES_JSONL = PMC_PARSED_DIR / "articles.jsonl"

PMC_METADATA_DIR = PMC_DIR / "metadata"
PMC_CURRENCY_PACK_DIR = PMC_DIR / "currency_pack"
PMC_CURRENCY_PACK_PARSED = PMC_CURRENCY_PACK_DIR / "parsed" / "PMC13082890.json"

PMC_CHUNKS_DIR = PMC_DIR / "chunks"
PMC_CHUNKS_JSONL = PMC_CHUNKS_DIR / "chunks.jsonl"
PMC_CHUNK_STATS = PMC_CHUNKS_DIR / "chunk_stats.json"

PMC_INDEX_DIR = PMC_DIR / "index"
PMC_CANDIDATES_DIR = PMC_DIR / "candidates"

# Generated QC reports are documentation, not data: they are read by a human.
QC_REPORTS_DIR = DOCS_ROOT / "corpus" / "qc"
PMC_QC_REPORT_POSTFIX = QC_REPORTS_DIR / "pmc_qc_report_2026-09-03-postfix.md"

__all__ = [name for name in dir() if name.isupper() and not name.startswith("_")]
