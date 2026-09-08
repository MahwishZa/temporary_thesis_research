#!/usr/bin/env python3
"""The end-to-end wiring check, run as part of the ordinary test suite.

``python -m thesis.run --smoke`` builds a synthetic corpus and index in the
on-disk shape the corpus-build stages produce, then drives every stage of the
architecture over it with a hash encoder -- no model weights, no network, no
production corpus. It is the one check that fails when two components stop
fitting together, which unit tests scoped to a single module cannot see.

It lives here as well as behind the CLI flag so that a reorganisation, a
renamed module or a moved data path cannot pass the suite while leaving the
pipeline unrunnable.
"""

from __future__ import annotations

import unittest

from thesis.smoke import run_smoke


class ArchitectureSmokeTest(unittest.TestCase):
    def test_pipeline_runs_end_to_end_on_a_synthetic_corpus(self) -> None:
        self.assertEqual(run_smoke(verbose=False), 0)


if __name__ == "__main__":
    unittest.main()
