# `architecture/` — the computational system under study

This is the research contribution. Two systems live here, and keeping them apart
is what makes the thesis's comparison meaningful.

```
architecture/
├── rag2/     the REPRODUCED ORIGINAL RAG² system  — the baseline
└── scaf/     the PROPOSED SCAF admission policy   — the thesis extension
```

## The dependency rule

```
  scaf/  ──imports──▶  rag2/          ALLOWED
  rag2/  ──imports──▶  scaf/          FORBIDDEN
```

The thesis measures what the *original, untouched* RAG² filter does. If baseline
code ever learned about SCAF — or about publication dates, recency or currency —
the thing being measured would no longer exist.

This is machine-checked, not merely intended.
`architecture/rag2/tests/test_metadata_isolation.py` scans every module under
`architecture/rag2/rag2/` for executable references to date, recency or currency
fields and fails the build if one appears; a companion test proves the scanner
actually fires by feeding it real violations. A second test in
`architecture/scaf/tests/` asserts no baseline file imports `scaf`.

## Reading `architecture/rag2/` — three different things, similar names

This trips people up, so it is worth being explicit:

| Path | What it is |
| --- | --- |
| `architecture/rag2/` | the **container**: the RAG² research/reproduction project as a whole — its package, the authors' code, configs, scripts, tests and docs |
| `architecture/rag2/rag2/` | the **Python package** — the code that `import rag2` actually loads. Its modules are `rag2.config`, `rag2.schema`, `rag2.datasets`, `rag2.retrieval`, `rag2.filtering`, … |
| `architecture/rag2/retriever/` and `architecture/rag2/classifier/` | the **RAG² authors' released code**, vendored **byte-identical** and never edited, so it stays citable as published |

The doubled name is not an accident and is not redundancy. `architecture/rag2/`
must be on `sys.path` for `import rag2` to find `architecture/rag2/rag2/`, and
the authors' `retriever/`/`classifier/` must sit beside the package rather than
inside it so they remain unmodified. Flattening it would either break the import
or break the byte-identical claim in `docs/reproduction/`.

**Path convention.** Everywhere outside `architecture/rag2/`, paths in this
repository are written **relative to the repository root** — so the package
module `rag2.datasets` is written `architecture/rag2/rag2/datasets/`.

*Inside* `architecture/rag2/`, a path to **code** stays relative to that
container (`rag2/filtering/rag2_filter.py`, `scripts/02_retrieve.py`), because
the container is a self-contained reproduction unit and the move changed no
relationship within it. The one exception is references to **documentation**:
the reproduction documents moved out to the repository-level `docs/reproduction/`
tree, so the container names them root-relative and links them with `../../`.

## `architecture/scaf/`

A normal Python package — `import scaf`. It sits outside `architecture/rag2/`
precisely because of the guard described above: SCAF exists to read publication
dates and currency, which baseline code may not. Its experiment configuration
and run scripts are not here; they live in `experiments/`, because *what the
system does* and *how we tested it* are different questions.
