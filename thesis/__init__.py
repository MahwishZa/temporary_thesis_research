#!/usr/bin/env python3
"""Thesis research architecture: the layer that composes the existing components.

This package implements no research method of its own. It is the wiring that
turns the repository's already-validated parts into one executable pipeline:

    research query
      -> query normalisation/representation      thesis.queries
      -> retrieval over the frozen PMC corpus    thesis.retrieval  -> pmc/retrieve.py
      -> candidate evidence (provenance intact)  thesis.corpus
      -> experimental condition                  thesis.conditions
           baseline        retrieval only, no filter
           rag2            the reproduced RAG2 filter + generation (called, never edited)
           recency_aware   a temporal policy applied at a declared boundary
      -> generation / answer construction        (delegated to rag2)
      -> evidence + provenance record            thesis.provenance
      -> evaluation                              thesis.evaluation

Design rules this package is required to keep, and which its tests enforce:

* **Nothing here modifies ``rag2/rag2/**``.** The reproduced baseline is called
  through its public interfaces. ``rag2/tests/test_metadata_isolation.py``
  guards that tree; ``thesis/tests/test_condition_isolation.py`` guards this one.
* **The baseline never reads publication dates.** Provenance is carried through
  every stage and read only by the recency condition. The thesis measures what
  the original filter does with evidence of different ages; if the baseline
  learned about dates, the thing being measured would cease to exist.
* **One corpus, one query set, one evaluation protocol across conditions.**
  Conditions differ in admission policy, never in the evidence population they
  see. ``pmc.retrieve``'s fingerprinted candidate replay is what makes that
  checkable rather than merely intended (validity control V3).
* **No invented methods.** Where the thesis has not yet fixed an algorithm --
  the recency policy, SCAF -- this package defines the interface and the
  configuration point and stops there, loudly.
"""

__version__ = "0.1.0"
