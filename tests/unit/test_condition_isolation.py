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
SRC_ROOT = os.path.join(REPO_ROOT, "src")
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

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


def _blank_string_constants(node: ast.AST) -> None:
    for child in ast.walk(node):
        if isinstance(child, ast.Constant) and isinstance(child.value, str):
            child.value = ""


def _code_shape(source: str) -> str:
    """The module's executable content, with its prose erased.

    Two kinds of string are treated as prose and blanked: docstrings, and the
    message a ``raise`` builds. Both exist to be read by a person, and a
    repository reorganisation legitimately moves the paths they name.

    Everything else survives -- control flow, calls, comparisons, and every
    other string constant. That matters: field names like ``chunk_id`` and
    config keys like ``chunks_path`` are string constants outside ``raise``
    statements, so renaming one still fails this comparison loudly.
    """
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Raise):
            _blank_string_constants(node)
            continue
        if not isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                                 ast.AsyncFunctionDef)):
            continue
        body = node.body
        if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
                and isinstance(body[0].value.value, str):
            body[0].value.value = ""
    return ast.dump(tree)


class TestStaticIsolation(unittest.TestCase):
    def test_no_baseline_module_reads_a_temporal_field(self):
        offenders = []
        for relative in BASELINE_MODULES:
            offenders.extend(executable_temporal_references(os.path.join(SRC_ROOT, relative)))
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
    """The architecture calls RAG2; it must not edit it.

    Two files under ``rag2/`` were written by this thesis rather than by the
    RAG2 authors, and both must live there for mechanical reasons: the loader is
    discovered through RAG2's corpus registry, and the config is loaded by RAG2's
    own test. They are therefore the only permitted exceptions -- and even for
    them only *documentation and corpus paths* may move, never behaviour, which
    the second assertion below proves by comparing the parsed module with
    docstrings removed.
    """

    #: Thesis-authored files inside the reproduction tree. Nothing else may change.
    THESIS_AUTHORED = {
        "rag2/configs/thesis_corpus.yaml",
        "rag2/docs/rag2_reproduction.md",
        "rag2/rag2/corpora/thesis_chunks.py",
        "rag2/tests/test_thesis_chunk_corpus.py",
    }

    #: Of those, the Python files whose executable content must still match.
    CODE_MUST_MATCH = {
        "rag2/rag2/corpora/thesis_chunks.py",
        "rag2/tests/test_thesis_chunk_corpus.py",
    }

    def _changed_since_main(self):
        try:
            out = subprocess.check_output(
                ["git", "diff", "--name-only", "origin/main", "--", "rag2/"],
                cwd=REPO_ROOT, stderr=subprocess.DEVNULL, text=True,
            )
        except Exception:
            self.skipTest("origin/main not available in this checkout")
        return {line for line in out.split("\n") if line.strip()}

    def test_only_thesis_authored_files_differ(self):
        unexpected = sorted(self._changed_since_main() - self.THESIS_AUTHORED)
        self.assertEqual(
            unexpected, [],
            "the thesis modified files it does not own inside the reproduced "
            "RAG2 tree:\n" + "\n".join(unexpected),
        )

    def test_permitted_files_changed_documentation_only(self):
        changed = self._changed_since_main()
        for relative in sorted(changed & self.CODE_MUST_MATCH):
            before = subprocess.check_output(
                ["git", "show", f"origin/main:{relative}"],
                cwd=REPO_ROOT, text=True,
            )
            after = open(os.path.join(REPO_ROOT, relative), encoding="utf-8").read()
            self.assertEqual(
                _code_shape(before), _code_shape(after),
                f"{relative} changed beyond its docstrings; the reproduction's "
                "behaviour must not move during a repository reorganisation.",
            )

    def test_the_prose_comparison_actually_fires(self):
        """A guard that cannot fail proves nothing."""
        base = "def f():\n    'doc'\n    raise ValueError('build with pmc/x.py')\n"
        # Prose moves freely: a docstring, and the text of a raised message.
        self.assertEqual(
            _code_shape(base),
            _code_shape("def f():\n    'other'\n    raise ValueError('build with a/b.py')\n"),
        )
        # Behaviour does not: a different exception type, or a renamed field.
        self.assertNotEqual(
            _code_shape(base),
            _code_shape("def f():\n    'doc'\n    raise KeyError('build with pmc/x.py')\n"),
        )
        self.assertNotEqual(
            _code_shape("d['chunk_id']\n"), _code_shape("d['chunkid']\n"),
        )


if __name__ == "__main__":
    unittest.main()
