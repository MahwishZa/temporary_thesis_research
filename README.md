# thesis_research — Alzheimer's evidence corpus

Research repository for an MS thesis on **recency bias in confidence-derived
evidence-utility signals for retrieval-augmented Alzheimer's clinical reasoning**,
extending the RAG² framework (Sohn et al., NAACL 2025).

It holds four things, kept deliberately apart:

| | Where | What it is |
| --- | --- | --- |
| **Source** | `src/thesis/` | one installable package: the corpus-build stages and the research architecture |
| **Data** | `data/` | the corpus and its provenance record — evidence, never code |
| **Baseline** | `rag2/` | the reproduced **original** RAG² system, vendored whole |
| **Experiments** | `experiments/` | the three-layer experiment boundary; none has been run |

Start with [`docs/architecture.md`](docs/architecture.md): it states which
components are implemented and validated, which are baselines, which are declared
interfaces awaiting a method, and which are future work. No experiment has been
run and no result exists.

## Quick start

```bash
pip install -e .                  # src-layout package; no heavy dependencies
python -m thesis.run --list       # conditions and temporal policies
python -m thesis.run --smoke      # offline end-to-end wiring check, ~1s
python -m pytest                  # every suite: 657 passed, 2 skipped
```

The package installs from `src/`, so every command above works from any working
directory. Without an install, prefix commands with `PYTHONPATH=src`.

## Pipeline stages

| Stage | Code | Data it produces |
| --- | --- | --- |
| 1. PubMed acquisition | `corpus_build/acquisition/fetch_pubmed.py` | `data/pubmed/` — 43,409 records from the approved query bank |
| 2. PMC open-access inventory | `corpus_build/acquisition/inventory_pmc_oa.py` | `data/pmc/inventory/` — 27,508 candidates, licence + availability |
| 3. PMC full-text download | `corpus_build/acquisition/download_pmc_xml.py` | `data/pmc/fulltext/` — 25,742 MD5-verified JATS XML + manifest |
| 4. XML parsing | `corpus_build/parsing/parse_pmc_xml.py` | `data/pmc/parsed/` — one structured JSON record per article |
| 5. Quality control | `corpus_build/qc/` | `docs/corpus/qc/` — full-corpus QC reports and integrity gates |
| 6. Corpus policy metadata (M1–M4) | `corpus_build/metadata/build_corpus_metadata.py` | `data/pmc/metadata/`, `data/pmc/currency_pack/` |
| 7. Retrieval-ready chunks | `corpus_build/chunking/build_chunks.py` | `data/pmc/chunks/` — deterministic chunks with full provenance |
| 8. MedCPT index + retrieval | `corpus_build/embedding/`, `corpus_build/indexing/` | `data/pmc/index/` — exact inner-product search, candidate replay |
| 9. Original RAG² baseline | `rag2/` | rationale → balanced retrieval → filter → answer |
| 10. Research architecture | `src/thesis/` (top level) | query → retrieval → condition → result + provenance |

Stages 1–8 are one package, one stage per subdirectory, and a stage never
imports a later one. Each is runnable on its own:

```bash
python -m thesis.corpus_build.chunking.build_chunks --help
```

## Layout

```
pyproject.toml             package + tooling configuration
conftest.py                puts src/ on sys.path so tests run without installing

src/thesis/                THE SOURCE. One package, installed from src/.
  paths.py                   the single source of truth for every on-disk location
  config.py                  typed config with YAML inheritance
  queries.py                 query normalisation                       [1]
  retrieval.py               retrieval facade over the indexing stage  [2]
  corpus.py                  corpus handle + digest verification       [3]
  recency.py                 the temporal-policy boundary              [4]
  conditions/                baseline | rag2 | recency_aware           [5]
  provenance.py              corpus/model stamps, run records          [7]
  evaluation.py              per-condition metrics and comparison      [8]
  pipeline.py                the orchestrator
  run.py                     the CLI entry point  (`python -m thesis.run`)
  smoke.py                   the offline end-to-end wiring check
  _bootstrap.py              the one seam onto the vendored rag2/ package
  corpus_build/              THE CORPUS PIPELINE, one stage per directory
    acquisition/               PubMed search, PMC inventory, XML download
    parsing/                   JATS XML -> structured records
    qc/                        QC detectors + the chunk and index integrity gates
    metadata/                  the frozen M1-M4 corpus policy overlays
    chunking/                  256-word / 32-overlap retrieval units
    embedding/                 MedCPT article-encoder vectors + row manifest
    indexing/                  exact search, balanced retrieval, candidate replay

data/                      RESEARCH DATA. Evidence, not code.
  pubmed/                    query bank, retrieved records (Git LFS), audit log
  pmc/
    inventory/                 OA inventory + reconciliation snapshot (immutable)
    fulltext/                  manifest.csv, failures.csv   (xml/ gitignored)
    parsed/                    parsed records               (gitignored, regenerable)
    metadata/                  M1-M4 overlays + registries  (see its README)
    currency_pack/             externally ingested currency-pack documents
    chunks/                    chunk_stats.json committed; chunks.jsonl gitignored
    index/, candidates/        (gitignored; rebuilt on the GPU machine)

tests/                     ONE SUITE, three levels
  unit/                      per-module tests
  integration/               stages composed: chunk -> embed -> retrieve
  smoke/                     the architecture end to end on a synthetic corpus

configs/thesis/            architecture.yaml + one file per experimental condition
docs/                      architecture, the RAG2 audit, corpus documentation + QC reports
experiments/               baseline | recency_bias | scaf   (see experiments/README.md)

rag2/                      reproduced ORIGINAL RAG2 baseline  (see below)
  rag2/                      the reproduction package
  configs/, scripts/, tests/, docs/
  retriever/, classifier/    the RAG2 authors' released code, UNMODIFIED
```

`rag2/` deliberately keeps its own layout, tests and pytest.ini. It is a vendored
unit certified as a whole by [`docs/rag2_reproduction_audit.md`](docs/rag2_reproduction_audit.md);
restructuring it would invalidate that certificate for no architectural gain.

## Chunking (stage 7)

Strategy is taken from the thesis proposal (§5.1), not invented: **256-token
sliding windows with 32-token overlap**, sized against the article encoder's
512-token limit with headroom for a prepended title and section header, plus
exact content-hash deduplication.

Two implementation decisions follow from that text:

- **Windows never cross a section boundary.** The proposal requires that a
  recommendation is never separated from its qualifying conditions; windowing
  inside sections also keeps section provenance exact for every chunk.
- **Windows are measured in whitespace words** — deterministic and dependency
  free (this repository is standard-library only). A 256-word window is always
  fewer than 512 sub-word tokens, so it stays inside the encoder limit with
  headroom. `--window/--overlap` can later be set in sub-word tokens without
  changing any other logic.

Title and section heading are stored as separate fields, not baked into the
text; `build_chunks.compose_embed_text()` is the single shared rule for
composing what the encoder sees — the embedding stage imports it rather than
restating it, so what is indexed and what is scored cannot drift apart.

Frozen policy is enforced, never re-decided: records whose M4
`eligibility_status` is `excluded` are not chunked; everything else carries its
frozen status through so retrieval can filter. Exact-duplicate text is
**flagged** via `duplicate_of`, never deleted — distinct versions and source
types must survive, because recency is an experimental variable.

```bash
python -m thesis.corpus_build.chunking.build_chunks   # -> data/pmc/chunks/
python -m thesis.corpus_build.qc.validate_chunks      # integrity gate; non-zero on failure
```

Both are deterministic: the same frozen inputs produce byte-identical output.
`validate_chunks` prints a content digest for cross-run comparison.

## Retrieval infrastructure (stage 8)

Models are fixed by the base paper and the proposal (§5.3), not chosen for
convenience — the retriever is deliberately frozen, and that is the thesis's
central internal-validity guarantee:

| Role | Model |
| --- | --- |
| Document encoder | `ncbi/MedCPT-Article-Encoder` |
| Query encoder | `ncbi/MedCPT-Query-Encoder` |
| Reranker | `ncbi/MedCPT-Cross-Encoder` |

**Exact flat search, not ANN.** Validity control V3 requires the candidate set
to be replayed byte-identically to every experimental arm; an approximate index
introduces run-to-run variation. Search is an exact inner-product scan over
unit-normalized vectors, and ties break on `chunk_id` so ordering is total.
FAISS/numpy are used when present purely for speed and give identical results.

**Balanced retrieval** draws an equal quota per `source_category` before
merging (base paper §3.4). Without it a PubMed-trained dense retriever drowns
the small but decisive CPG and currency-pack corpora.

**Candidate-set replay (V3).** `save_candidates()` serialises the candidate list
with a digest over identity *and order*; `replay_candidates()` reloads and
verifies it; `verify_replay()` proves a later arm scored the same population.

### Installing torch with CUDA

**`pip install torch` gives you the CPU-only build on Windows.** That build makes
`torch.cuda.is_available()` return `False`, and the embedding run then executes
on the CPU — for ~780k chunks that is the difference between hours and days. The
GPU build must be installed from the PyTorch CUDA index explicitly:

```bash
pip uninstall -y torch
pip install torch==2.4.1+cu121 --index-url https://download.pytorch.org/whl/cu121
pip install transformers                  # unchanged; numpy optional, for speed

python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

The CUDA runtime ships inside the wheel — no system CUDA toolkit is needed. Pick
the `cuXXX` suffix your driver supports; a driver new enough for CUDA 12.1 or
later runs the `cu121` build, and newer drivers stay backward compatible.

### Building the index

```bash
python -m thesis.corpus_build.embedding.embed_chunks --device cuda --batch-size 8
python -m thesis.corpus_build.qc.verify_index                    # integrity gate
python -m thesis.corpus_build.indexing.retrieve --query "..." --query-id q1
python -m thesis.corpus_build.indexing.retrieve --replay data/pmc/candidates/q1.json
```

Three properties matter for a run this long:

- **`--device cuda` is a requirement, not a preference.** It fails with an
  actionable message rather than silently falling back to the CPU. `--device
  auto` keeps the old fallback but says which device it chose.
- **The run resumes by default.** Output is flushed every batch, and a restart
  picks up at the last complete row — a partial vector or half-written manifest
  line is trimmed first. A resumed index is byte-identical to an uninterrupted
  one, `content_digest` included. `--restart` starts over.
- **CUDA OOM halves the batch and continues**, and stays reduced. It never falls
  back to the CPU mid-run, which would make the index internally inconsistent.

`--batch-size 8` is sized for a 4 GB card at 512 tokens; raise it on a larger
GPU. `--limit N` embeds only the first N chunks for a smoke test and stamps the
result `partial_index_limit` so it cannot be mistaken for the production index.

`verify_index` is the gate to run before anything retrieves: it checks row
count against the chunk layer, dimension, row-for-row alignment, absence of
NaN/Inf, L2 normalisation, and that `content_digest` recomputes to the recorded
value.

A deterministic stub encoder exists for offline testing only. It refuses to
write an index without `--allow-stub` and stamps `production=false`, so a stub
index can never be mistaken for a real one.

## Original RAG² baseline (stage 9)

`rag2/` holds a reproduction of the **original** RAG² system — the thesis
baseline, not a thesis contribution. Read
**[`docs/rag2_reproduction_audit.md`](docs/rag2_reproduction_audit.md)** first:
it records, component by component, what was reproduced, how it was verified,
and what remains unverified. The reproduction's own specification (every
assumption, every place the paper and the authors' code disagree) is
[`rag2/docs/rag2_reproduction.md`](rag2/docs/rag2_reproduction.md).

Inside `rag2/`, two things are kept apart on purpose:

| Path | What it is |
| --- | --- |
| `rag2/retriever/`, `rag2/classifier/` | the RAG² authors' released code, **unmodified** |
| `rag2/rag2/`, `configs/`, `scripts/`, `tests/` | the reproduction |

Exactly two files under `rag2/` were written by this thesis rather than by the
authors — `rag2/rag2/corpora/thesis_chunks.py` (the corpus loader, which must
live in the package to be registered) and `rag2/configs/thesis_corpus.yaml`
(loaded by RAG²'s own test). `tests/unit/test_condition_isolation.py` names
those two, fails on any other change under `rag2/`, and proves the loader's
executable content still matches `origin/main` docstring-for-docstring.

**Models required** (none are downloaded by the tests or the smoke test):

| Role | Model | Needed for |
| --- | --- | --- |
| Query encoder | `ncbi/MedCPT-Query-Encoder` | retrieval |
| Article encoder | `ncbi/MedCPT-Article-Encoder` | building `data/pmc/index/` |
| Reranker | `ncbi/MedCPT-Cross-Encoder` | reranking |
| Filter | `google/flan-t5-large` + a trained checkpoint | filtering |
| Backbone LLM | `meta-llama/Meta-Llama-3-8B-Instruct` | rationales, answers |

The paper's trained filter checkpoint **is not distributed by its authors** and
must be retrained (`rag2/scripts/03_build_filter_labels.py`, then `04_train_filter.py`).

### Running it

```bash
cd rag2
pip install -r requirements.txt
python3 scripts/smoke_test.py     # offline wiring check: no GPU, no downloads, ~2s
python3 -m pytest                 # the baseline suite on its own
```

Configure it against this repository's corpus with
`rag2/configs/thesis_corpus.yaml`, which points the baseline at `data/pmc/index/`
and `data/pmc/chunks/chunks.jsonl` through the `thesis_chunks` corpus loader. Run
the stage scripts from the repository root so those relative paths resolve. Stage
commands are in [`experiments/baseline/README.md`](experiments/baseline/README.md).

### What is verified, and what is not

**Verified here:** all stages wired end to end (the smoke test executes
question → rationale → balanced retrieval → rerank → cache → ΔPPL labeling →
filter → answer → evaluation); the filter prompt and option format round-trip
byte-identically against the authors' released training data; Figure 2's
labeling tree matches the paper path by path; provenance never reaches a model
input.

**Not verified:** anything requiring model weights. This container has no torch,
transformers, faiss or GPU, so MedCPT encoding, Flan-T5 filtering, ΔPPL under a
real LLM and answer generation were checked by code reading and stub-driven
execution, not by running a real model. **No accuracy has been measured** —
`rag2/docs/reproduction_results.md` is deliberately blank. See audit §10.

## Where thesis experiments belong

`experiments/` establishes a one-directional boundary: a layer may call the layer
below and may not modify it.

1. `experiments/baseline/` — original RAG² runs. Runnable now.
2. `experiments/recency_bias/` — the thesis probe. **Not started.**
3. `experiments/scaf/` — the SCAF extension. **Not started.**

Nothing in this repository implements SCAF, recency weighting, authority
weighting, currency scoring, supersession or abstention. The baseline measures
the *untouched* original, so `rag2/tests/test_metadata_isolation.py` fails the
build if any baseline module starts reading publication dates, and
`tests/unit/test_condition_isolation.py` enforces the same rule on the
architecture layer.

## Running the tests

```bash
python -m pytest                  # everything: tests/ and rag2/tests
python -m pytest tests/unit       # or any single level
cd rag2 && python -m pytest       # the baseline suite standalone
```

All suites are offline — no network, no corpus files, no model weights required.
Last measured: **657 passed, 2 skipped** in total (both skips are torch-gated
modules in `rag2/`).

## Regenerating derived data

Raw XML (`data/pmc/fulltext/xml/`) and parsed records (`data/pmc/parsed/`) are
gitignored: they are research data, reconstructible from the inventory plus the
manifest's MD5s. The corpus policy overlays are committed as research evidence
and are regenerated with:

```bash
python -m thesis.corpus_build.metadata.build_corpus_metadata --no-fetch
```

Run this on the machine holding the **complete** parsed corpus: it upgrades
canonical dates from PubMed fallback to JATS-primary for all PMC records. See
[`data/pmc/metadata/README.md`](data/pmc/metadata/README.md).

## Current status

Acquisition, parsing, QC and corpus-policy metadata (M1–M4) are complete. The chunk layer
and the retrieval stack are built. The original RAG² baseline is integrated and audited.

Outstanding, in order:

1. Full-corpus canonical-date regeneration (above), on the machine with the complete
   parsed corpus. `data/pmc/chunks/chunk_stats.json` should be refreshed from that run —
   the committed copy records a partial container run, not the production one.
2. Build the MedCPT index on the GPU machine:
   `python -m thesis.corpus_build.embedding.embed_chunks --device cuda --batch-size 8`,
   then `python -m thesis.corpus_build.qc.verify_index`. Resumable — re-run the same
   command after any interruption. Install the CUDA torch build first (see stage 8);
   a `+cpu` build silently runs this on the CPU.
3. Train the RAG² filter, then run the baseline and record results in
   `rag2/docs/reproduction_results.md`. **No accuracy has been measured yet.**
4. Only then: the recency-bias probe.

## Provenance notes

- `data/pmc/inventory/pmc_oa_inventory.csv` is immutable — the authoritative
  acquisition record.
- `data/pmc/currency_pack/xml/PMC13082890.xml` is an exact pinned snapshot (MD5
  `dcb1ac4eaa24b75ab3202f2315c6b2e4`). The PMC object is revised in place, so re-fetching
  yields a different hash; this snapshot must not be replaced. Licensed CC BY-NC —
  redistribution is non-commercial, with attribution to the Cochrane review it contains.
- Document identity is **PMID-primary**, PMCID for the full-text join; DOI is a consistency
  check only, never an identity key.
- The QC reports under `docs/corpus/qc/` quote absolute Windows paths from the runs
  that produced them. Those are the record of what was executed and are left as
  written; they are not live references.
