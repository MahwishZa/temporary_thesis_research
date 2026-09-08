# Architecture map

Where every piece lives, what may depend on what, and why the boundaries are
where they are. This is the document to read before moving anything.

---

## 1. The lifecycle

The top-level directories are the stages of the research, in order. Each stage
consumes the previous stage's output and nothing else.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  data/            WHAT WE OBSERVED                                          │
│                                                                             │
│    corpora/pubmed/    approved search strategy · results (LFS) · audit log  │
│    corpora/pmc/       OA inventory · download manifest · metadata overlays  │
│                       · currency pack · chunk statistics                    │
│    datasets/          the 30 fixed Alzheimer questions                      │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  preprocessing/   HOW IT BECOMES A CORPUS                                   │
│                                                                             │
│    pubmed/   fetch_pubmed.py                                                │
│    pmc/      inventory → download → parse → QC → policy → chunk → index     │
│                                                                             │
│    ACQUIRE → PARSE → QUALITY CONTROL → POLICY → CHUNK → INDEX               │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  indexes/production/   MedCPT vectors · index_manifest.jsonl · index_meta   │
│                        (gitignored — machine-specific, deterministic)       │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
                  experiments/scripts/freeze_candidates.py
                       CANDIDATE SET FROZEN ONCE, DIGESTED
                                     │
                        ┌────────────┴────────────┐
                        ▼                         ▼
┌───────────────────────────────┐   ┌───────────────────────────────┐
│  architecture/rag2/           │   │  architecture/scaf/           │
│  THE BASELINE                 │   │  THE THESIS EXTENSION         │
│                               │   │                               │
│  rationale → balanced         │   │  A(s) = w_σ·σ + w_γ·γ         │
│  retrieval → rerank →         │◀──│         + w_ρ·ρ + w_τ·τ       │
│  ΔPPL filter → answer         │   │                               │
│                               │   │  imports rag2, never the      │
│  never imports scaf           │   │  other way round              │
└───────────────────────────────┘   └───────────────────────────────┘
                        └────────────┬────────────┘
                                     ▼
                   experiments/scripts/run_comparison.py
                    SAME GENERATOR · SAME DECODING · SAME
                    CANDIDATES · ONLY ADMISSION DIFFERS
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  experiments/runs/   per_question.jsonl · manifest.json  (gitignored)       │
│  docs/experiments/   the written result                                     │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. The dependency rule

```
  scaf  ──imports──▶  rag2          ALLOWED
  rag2  ──imports──▶  scaf          FORBIDDEN
  rag2  ──imports──▶  preprocessing FORBIDDEN
  preprocessing ──imports──▶ either FORBIDDEN
```

The thesis measures what the *original, untouched* RAG² filter does with
evidence of different ages. If baseline code ever learned about SCAF — or about
publication dates, recency or currency — the thing being measured would no
longer exist.

### How each direction is enforced

| Rule | Enforced by |
| --- | --- |
| No baseline module reads date/recency/currency fields | `architecture/rag2/tests/test_metadata_isolation.py` scans every module under `architecture/rag2/rag2/` for executable references and fails the build; a companion test feeds it real violations to prove the scanner fires |
| No baseline file imports `scaf` | a test in `architecture/scaf/tests/` |
| The baseline filter registry does not know `scaf` | `architecture/rag2/tests/test_filter_scoring.py` asserts `build_filter(kind="scaf")` raises `KeyError` |
| Both arms score the same evidence | `architecture/scaf/frozen.py` digests the candidate set over identity **and order**; `run_comparison.py` verifies the replay |

The third rule has a consequence that surprises people: **the baseline and SCAF
test suites must run in separate processes.** Importing `scaf` registers the
`scaf` filter with the baseline registry, so co-running them in one interpreter
makes SCAF's import silently satisfy the guard it exists to fail against.
`run_tests.py` runs each suite as its own subprocess for exactly this reason —
that is a correctness requirement, not a convenience.

### Why the baseline carries provenance it may not read

Every chunk carries publication date, date precision, source category, authority
tier, guideline family, currency-pack membership and retraction flags all the way
through retrieval into `Evidence.metadata`. No baseline component reads any of
it. It is carried so the recency-bias probe can correlate filter decisions with
date **without re-running retrieval** — which would break the frozen-candidate
guarantee. Carrying-but-not-reading is the design; the metadata-isolation test is
what keeps "not reading" true.

---

## 3. Three things with similar names

| Path | What it is |
| --- | --- |
| `architecture/rag2/` | the **container** — the RAG² research/reproduction project as a whole |
| `architecture/rag2/rag2/` | the **Python package** that `import rag2` loads (`rag2.config`, `rag2.retrieval`, `rag2.filtering`, …) |
| `architecture/rag2/retriever/`, `architecture/rag2/classifier/` | the **authors' released code**, vendored byte-identical and never edited |

The doubled name is deliberate. `architecture/rag2/` must be on `sys.path` for
`import rag2` to resolve to `architecture/rag2/rag2/`, and the authors' code must
sit *beside* the package rather than inside it so it stays unmodified and
citable. Flattening the tree would either break the import or break the
byte-identical claim the audit depends on.

Verified empirically before the migration: having both `architecture/` and
`architecture/rag2/` on `sys.path` does **not** make `import rag2` resolve to the
container as a namespace package. A regular package wins over a namespace portion
regardless of path order.

### Path convention

Paths are written **relative to the repository root** everywhere except inside
`architecture/rag2/`, where references to *code* stay relative to the container
(`rag2/filtering/rag2_filter.py`, `scripts/02_retrieve.py`). References to
*documentation* are root-relative even there, because the reproduction documents
now live in `docs/reproduction/` rather than inside the container.

---

## 4. Why `preprocessing/pmc/` is one flat directory

The modules import each other by bare name: `build_corpus_metadata` imports
`parse_pmc_xml`, `embed_chunks` imports `build_chunks`, `retrieve` imports
`embed_chunks`. Python resolves those only when the modules are **siblings on
`sys.path`**, which is what running from inside the directory provides.

Splitting them into `acquisition/`, `chunking/` and `indexing/` subdirectories
would look tidier in a tree diagram and break every one of those imports. The
stages are therefore expressed in `preprocessing/README.md`'s table rather than
in directories — a deliberate trade of cosmetic structure for working code.

---

## 5. Code and data are separated, deliberately

Research data does not live inside source directories, and source code does not
live inside data directories. The reason is not tidiness: it is that a reader
must be able to tell, from the path alone, whether a file is *an observation
about the world* or *an instruction to a computer*. Those two kinds of file have
different review standards, different immutability rules, and different reasons
to change.

Two consequences worth knowing:

- **A new corpus is a sibling, not a redesign.** Clinical notes, drug labels or
  institutional guidelines go under `data/corpora/` beside `pmc/` and `pubmed/`,
  with their own README recording provenance and licence. Nothing in
  `preprocessing/`, `architecture/` or `experiments/` needs to move.
- **Manifests live beside their artifact.** `chunk_stats.json` sits next to
  `chunks.jsonl`; `manifest.csv` next to `fulltext/xml/`; `index_meta.json`
  inside `indexes/production/`. A manifest filed away from its artifact goes
  stale silently when the artifact is rebuilt.

---

## 6. What the boundaries do *not* guarantee

The structure enforces that the baseline is untouched and that both arms see the
same evidence. It does not, and cannot, establish that any scientific result has
been produced. A green test run means the software is internally consistent —
nothing more. Whether a number is reportable is decided by the 9 fairness checks
and 16 scientific preconditions recorded in a run's manifest, not by the
directory layout.
