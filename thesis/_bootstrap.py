#!/usr/bin/env python3
"""Import shim for the vendored RAG2 package.

The reproduced baseline lives at ``rag2/rag2/`` -- that is, the importable
package ``rag2`` sits inside the directory ``rag2/``. Rather than move it (the
audit's "unmodified" claim covers that tree) or duplicate its code, this module
puts ``<repo>/rag2`` on ``sys.path`` the first time it is needed.

Imports are deferred behind functions so that ``import thesis`` costs nothing
and works even where RAG2's own optional dependencies are absent.
"""

from __future__ import annotations

import os
import sys
from types import ModuleType

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAG2_ROOT = os.path.join(REPO_ROOT, "rag2")


def ensure_rag2_importable() -> str:
    """Put ``<repo>/rag2`` on sys.path. Idempotent. Returns the path added."""
    if RAG2_ROOT not in sys.path:
        sys.path.insert(0, RAG2_ROOT)
    return RAG2_ROOT


def ensure_repo_importable() -> str:
    """Put the repository root on sys.path so ``pmc`` and ``thesis`` import."""
    if REPO_ROOT not in sys.path:
        sys.path.insert(0, REPO_ROOT)
    return REPO_ROOT


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
    """The approved retrieval implementation (``pmc/retrieve.py``)."""
    ensure_repo_importable()
    from pmc import retrieve

    return retrieve
