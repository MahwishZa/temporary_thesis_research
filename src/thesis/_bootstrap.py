#!/usr/bin/env python3
"""Import shim for the vendored RAG2 package.

The reproduced baseline lives at ``rag2/rag2/`` -- that is, the importable
package ``rag2`` sits inside the directory ``rag2/``. It is deliberately not
moved into ``src/``: ``docs/rag2_reproduction_audit.md`` certifies that tree
against the authors' release, and its own ``rag2/tests`` and
``rag2/scripts/smoke_test.py`` are part of what was certified. Rather than
disturb that or duplicate its code, this module puts ``<repo>/rag2`` on
``sys.path`` the first time it is needed.

Imports are deferred behind functions so that ``import thesis`` costs nothing
and works even where RAG2's own optional dependencies are absent.
"""

from __future__ import annotations

import sys
from types import ModuleType

from .paths import RAG2_ROOT, REPO_ROOT

__all__ = [
    "REPO_ROOT",
    "RAG2_ROOT",
    "ensure_rag2_importable",
    "rag2_available",
    "rag2_config_module",
    "pmc_retrieve_module",
]


def ensure_rag2_importable() -> str:
    """Put ``<repo>/rag2`` on sys.path. Idempotent. Returns the path added."""
    root = str(RAG2_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    return root


def rag2_config_module() -> ModuleType:
    ensure_rag2_importable()
    from rag2 import config

    return config


def rag2_available() -> bool:
    """Whether the RAG2 package imports here. False on a minimal install."""
    try:
        rag2_config_module()
        return True
    except Exception:
        return False


def pmc_retrieve_module() -> ModuleType:
    """The approved retrieval implementation.

    Kept as a function rather than a plain import so the retrieval facade has a
    single, greppable seam onto the indexing stage -- and so the name the rest
    of the architecture uses does not change if that stage is ever relocated
    again.
    """
    from .corpus_build.indexing import retrieve

    return retrieve
