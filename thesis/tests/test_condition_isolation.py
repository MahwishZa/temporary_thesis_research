#!/usr/bin/env python3
"""Enforces the architecture's central invariant: the baseline is temporally blind.

The thesis measures what the *original* RAG2 filter does with evidence of
different ages. If the baseline path ever learned to read publication dates, the
thing being measured would no longer exist and every recency finding would be
uninterpretable. ``rag2/tests/test_metadata_isolation.py`` already guards
``rag2/rag2/**``; this file guards the layer above it.

Three separate guarantees, because each can fail independently:

1. **Static** -- no baseline module contains executable code that names a
   temporal field. Comments and docstrings may discuss them; code may not.
2. **Behavioural** -- stripping or permuting dates changes no baseline decision.
3. **Structural** -- the reproduced RAG2 tree is not modified by this layer.
"""

from __future__ import annotations

import ast
import copy
import io
import os
import re
import shutil
import subprocess
import sys
import tempfile
import tokenize
import unittest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from thesis.conditions.base import build_condition  # noqa: E402
from thesis.queries import in_memory_query_set  # noqa: E402
from thesis.retrieval import RetrievalService  # noqa: E402
from thesis.smoke import HashEncoder, build_fixture, smoke_config  # noqa: E402

#: Modules on the baseline path. A temporal reference in executable code here is
#: a defect. thesis/recency.py and thesis/conditions/recency_aware.py are the
#: declared exceptions: reading dates is their entire purpose.
BASELINE_MODULES = (
    "thesis/conditions/base.py",
    "thesis/conditions/retrieval_only.py",
    "thesis/conditions/rag2_condition.py",
    "thesis/retrieval.py",
    "thesis/pipeline.py",
    "thesis/evaluation.py",
)

#: Date *values* the baseline must never read. Deliberately excludes the word
#: "recency" on its own: baseline modules legitimately name the policy interface
#: (``config.recency.policy``, ``from ..recency import TemporalPolicy``) because
#: declaring the boundary is how the invariant is kept. What must not appear is
#: code that reaches into a date field.
TEMPORAL_PATTERN = re.compile(
    r"canonical_date|date_precision|date_source|split_june|publication_date|"
    r"currency_score|freshness|age_days|supersed|pub_year|days_old",
    re.IGNORECASE,
)


def _docstring_spans(source: str):
    spans = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        body = getattr(node, "body", None)
        if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
                and isinstance(body[0].value.value, str):
            spans.append((body[0].lineno, body[0].end_lineno or body[0].lineno))
    return spans


def executable_temporal_references(path: str):
    """Temporal references in executable tokens only (not comments/docstrings)."""
    with open(path, "r", encoding="utf-8") as handle:
        source = handle.read()
    spans = _docstring_spans(source)
    hits = []
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type == tokenize.COMMENT:
            continue
        if token.type == tokenize.STRING and any(lo <= token.start[0] <= hi for lo, hi in spans):
            continue
        if TEMPORAL_PATTERN.search(token.string):
            hits.append(f"{os.path.basename(path)}:{token.start[0]}: {token.line.strip()[:80]}")
    return hits


class TestStaticIsolation(unittest.TestCase):
    def test_no_baseline_module_reads_a_temporal_field(self):
        offenders = []
        for relative in BASELINE_MODULES:
            offenders.extend(executable_temporal_references(os.path.join(REPO_ROOT, relative)))
        self.assertEqual(
            offenders, [],
            "baseline-path code references temporal fields:\n" + "\n".join(offenders),
        )

    def test_the_scanner_actually_fires(self):
        """A guard that cannot fail proves nothing."""
        directory = tempfile.mkdtemp()
        try:
            path = os.path.join(directory, "violation.py")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(
                    '"""A docstring mentioning canonical_date must NOT trip it."""\n'
                    "# nor must a comment mentioning recency\n"
                    "def f(evidence):\n"
                    "    return evidence.metadata['canonical_date']\n"
                )
            hits = executable_temporal_references(path)
            self.assertEqual(len(hits), 1, hits)
            self.assertIn("canonical_date", hits[0])
        finally:
            shutil.rmtree(directory, ignore_errors=True)

    def test_recency_modules_are_the_declared_exception(self):
        """The recency layer is *supposed* to read dates; confirm it is separate."""
        for relative in ("thesis/recency.py", "thesis/conditions/recency_aware.py"):
            self.assertNotIn(relative, BASELINE_MODULES)


class TestBehaviouralIsolation(unittest.TestCase):
    def setUp(self) -> None:
        self.root = tempfile.mkdtemp(prefix="thesis-iso-")
        self.fixture = build_fixture(self.root)
        self.config = smoke_config(self.root, self.fixture)
        self.queries = in_memory_query_set([{"query_id": "T1", "query": "amyloid"}])

    def tearDown(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)

    def _admitted(self, mutate=None):
        service = RetrievalService(self.config, encoder=HashEncoder())
        retrieved = service.retrieve(self.queries.queries[0])
        if mutate:
            retrieved.candidates = [mutate(dict(c)) for c in retrieved.candidates]
        result = build_condition(self.config).run(self.queries.queries[0], retrieved)
        return [(e.chunk_id, e.rank) for e in result.admitted]

    def test_stripping_dates_changes_no_baseline_decision(self):
        def strip(candidate):
            for field in ("canonical_date", "date_precision", "split_june_2024"):
                candidate.pop(field, None)
            return candidate

        self.assertEqual(self._admitted(), self._admitted(strip))

    def test_permuting_dates_changes_no_baseline_decision(self):
        def scramble(candidate):
            if candidate.get("canonical_date"):
                candidate["canonical_date"] = "1901-01-01"
            return candidate

        self.assertEqual(self._admitted(), self._admitted(scramble))

    def test_dates_still_reach_the_output(self):
        """Carried, not consulted: removing them must not be how isolation is achieved."""
        service = RetrievalService(self.config, encoder=HashEncoder())
        retrieved = service.retrieve(self.queries.queries[0])
        result = build_condition(self.config).run(self.queries.queries[0], retrieved)
        self.assertTrue(result.admitted[0].metadata.get("canonical_date"))


class TestRag2TreeUntouched(unittest.TestCase):
    def test_thesis_layer_does_not_modify_the_reproduction(self):
        """The architecture calls RAG2; it must not edit it."""
        try:
            changed = subprocess.check_output(
                ["git", "diff", "--name-only", "origin/main", "--", "rag2/"],
                cwd=REPO_ROOT, stderr=subprocess.DEVNULL, text=True,
            ).strip()
        except Exception:
            self.skipTest("origin/main not available in this checkout")
        self.assertEqual(
            changed, "",
            f"the thesis architecture modified the reproduced RAG2 tree:\n{changed}",
        )


if __name__ == "__main__":
    unittest.main()
