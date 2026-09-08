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
`parse_pmc_xml`, and `embed_chunks` imports `compose_embed_text` from
`build_chunks` so the encoder sees exactly the text the chunker composed. Python
resolves bare names only when the modules are **siblings on `sys.path`**, which is
what running from inside the directory provides.

Splitting them into `acquisition/`, `chunking/` and `indexing/` subdirectories
would look tidier in a tree diagram and break every one of those imports. The
stages are therefore expressed in `preprocessing/README.md`'s table rather than
in directories — a deliberate trade of cosmetic structure for working code.

---

## 5. One canonical implementation, and what was retired

There is exactly **one** experimental path. This section exists because there
were briefly two, and anyone reading older commits or documents will meet the
other one.

| Question | Canonical answer |
| --- | --- |
| Where is the RAG² baseline? | `architecture/rag2/` — package in `rag2/rag2/`, authors' release in `retriever/` + `classifier/` |
| Where is SCAF? | `architecture/scaf/` — `policy.py` (admission), `frozen.py` (candidate replay), `compare.py` (arms, fairness, preconditions), `generation.py` (the shared generator) |
| What do they share? | `architecture/scaf/` calls the baseline's own interfaces — `rag2.filtering`, `rag2.llm`, `rag2.generation`, `rag2.evaluation`. There is no third "shared" package, deliberately: a shared layer between the thing under study and its extension is where the two quietly start influencing each other |
| Where are experiments run? | `experiments/scripts/freeze_candidates.py`, then `experiments/scripts/run_comparison.py`, configured by `experiments/configs/preliminary_experiment.yaml` |
| Where do results go? | `experiments/runs/` (gitignored), written up in `docs/experiments/` |

### Retired, and why

| Retired | Superseded by | Why |
| --- | --- | --- |
| `thesis/` (20 files) | `architecture/scaf/` + `experiments/` | It was an orchestration scaffold whose temporal policies all raised `TemporalPolicyError`, including `currency_three_state` — the very term SCAF implements. Its own header called it "a seam, not a method" |
| `configs/thesis/` | `experiments/configs/` | configuration for the above |
| `docs/architecture.md` | this document | described `thesis/`; its path also collided with the `docs/architecture/` directory |
| `preprocessing/pmc/retrieve.py` | `architecture/rag2/rag2/retrieval/balanced.py` and `architecture/scaf/frozen.py` | a pre-reproduction prototype of balanced retrieval and candidate replay; its last consumer was `thesis/retrieval.py` |

What was **kept** from `thesis/` before it went: the dirty-tree reportability
gate, the carried-dates precondition, per-source admission counts, the
answer-metrics hook, and two tracked package versions. Each is named in an
attribution comment in `architecture/scaf/compare.py`. See
[`reorganization_2026-09-08.md`](reorganization_2026-09-08.md) §13 for the full
comparison and the evidence behind the choice.

---

## 6. Code and data are separated, deliberately

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

## 7. What the boundaries do *not* guarantee

The structure enforces that the baseline is untouched and that both arms see the
same evidence. It does not, and cannot, establish that any scientific result has
been produced. A green test run means the software is internally consistent —
nothing more. Whether a number is reportable is decided by the 9 fairness checks
and 18 scientific preconditions recorded in a run's manifest, not by the
directory layout.
