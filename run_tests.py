#!/usr/bin/env python3
"""Run every offline test suite in this repository.

    python3 run_tests.py            # all four suites
    python3 run_tests.py rag2 scaf  # only the named ones

Each suite runs in its **own subprocess**, and that is a correctness
requirement, not a convenience:

* ``import scaf`` registers the ``scaf`` filter with the baseline's filter
  registry. ``architecture/rag2/tests/test_filter_scoring.py`` asserts that the
  baseline registry does *not* know ``scaf`` -- the guard that pins the one-way
  ``scaf -> rag2`` dependency. Run both suites in one interpreter and SCAF's
  import satisfies the guard it is meant to fail against.
* The ``preprocessing/pmc`` modules import each other by bare name
  (``build_corpus_metadata`` imports ``parse_pmc_xml``), so their tests need
  that directory on ``sys.path`` -- which is what running from inside it does.
* ``architecture/rag2`` carries its own ``pytest.ini`` and ``conftest.py``.

A single root ``pytest.ini`` cannot honour all three at once. Separate
processes can, and cost about three seconds.

Nothing here needs a GPU, a network, torch, transformers, faiss, the production
index, or a model checkpoint. Suites that would need them skip or assert on the
interface instead. A green run therefore says the software is consistent -- it
does not say any scientific result has been reproduced.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# name -> (working directory, argv)
SUITES: dict[str, tuple[Path, list[str]]] = {
    "pmc": (
        ROOT / "preprocessing" / "pmc",
        [sys.executable, "-m", "unittest", "discover", "-s", ".", "-p", "test_*.py"],
    ),
    "pubmed": (
        ROOT / "preprocessing" / "pubmed",
        [sys.executable, "-m", "unittest", "discover", "-s", ".", "-p", "test_*.py"],
    ),
    "rag2": (
        ROOT / "architecture" / "rag2",
        [sys.executable, "-m", "pytest"],
    ),
    "scaf": (
        ROOT,
        [sys.executable, "-m", "pytest", "architecture/scaf/tests"],
    ),
}


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    unknown = [a for a in argv if a not in SUITES]
    if unknown:
        print(f"unknown suite(s): {', '.join(unknown)}", file=sys.stderr)
        print(f"available: {', '.join(SUITES)}", file=sys.stderr)
        return 2

    selected = argv or list(SUITES)
    failed: list[str] = []
    for name in selected:
        cwd, cmd = SUITES[name]
        print(f"\n=== {name} ({cwd.relative_to(ROOT)}) ".ljust(72, "="))
        rc = subprocess.run(cmd, cwd=cwd).returncode
        if rc != 0:
            failed.append(name)

    print("\n" + "=" * 72)
    if failed:
        print(f"FAILED: {', '.join(failed)}")
        return 1
    print(f"OK: {', '.join(selected)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
