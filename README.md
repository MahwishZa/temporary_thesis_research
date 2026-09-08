# thesis_research

MS thesis research repository: **does confidence-based evidence admission in
retrieval-augmented medical QA prefer the past?**

The repository is organised around the research lifecycle, so the directory you
are in tells you what kind of question you are asking:

```
data/  →  preprocessing/  →  architecture/  →  experiments/  →  docs/
what we    how it becomes     the system        how we           what we
observed   a corpus           under study       tested it        concluded
```

---

## 1. Purpose

Retrieval-augmented generation for clinical questions depends on a filter that
decides which retrieved snippets reach the answer model. RAG² (Sohn et al.,
NAACL 2025) makes that decision with a small model trained on perplexity-based
labels — a *confidence* signal. This repository exists to ask what that
confidence signal does with the **age** of evidence, and to test an alternative
admission policy that scores age explicitly instead of implicitly.

It contains everything needed to run that comparison: the corpus, the pipeline
that built it, a reproduction of the original RAG² baseline, the proposed SCAF
extension, and the experimental harness that holds everything but the admission
policy constant.

## 2. Research question

> Does a confidence-derived evidence-utility signal systematically prefer older
> evidence over newer evidence, and does an explicit currency-aware admission
> policy (SCAF) change what reaches the answer?

Two arms, one controlled comparison. **Everything upstream of admission is
frozen and shared**: the same corpus, the same MedCPT retrieval, the same
reranking, the same candidate set replayed byte-identically, the same generator,
the same decoding parameters. Only the admission policy differs.

- **Arm A** — the original RAG² perplexity filter (Flan-T5-large, `[HELPFUL]` /
  `[NOT_HELPFUL]`, τ at the top 25% of ΔPPL).
- **Arm B** — SCAF: `A(s) = w_σ·σ + w_γ·γ + w_ρ·ρ + w_τ·τ` over support,
  currency, corroboration and authority.

**§12 records a corpus-design finding that currently blocks this question.**
Read it before planning a run.

## 3. Architecture

```
                     data/corpora/            SOURCES → CORPORA
                            │
                            ▼
                     preprocessing/           PARSE · QC · POLICY · CHUNK · INDEX
                            │
                            ▼
                     indexes/production/      MedCPT vectors + manifest
                            │
                            ▼
        experiments/scripts/freeze_candidates.py
                            │                 CANDIDATES FROZEN ONCE
                  ┌─────────┴─────────┐
                  ▼                   ▼
        architecture/rag2/      architecture/scaf/
          BASELINE                EXTENSION
          (admission by ΔPPL)     (admission by A(s))
                  └─────────┬─────────┘
                            ▼
        experiments/scripts/run_comparison.py
                            │                 SAME GENERATOR, SAME DECODING
                            ▼
              experiments/runs/  +  docs/experiments/
```

The one rule that makes the comparison mean anything:

```
  scaf  ──imports──▶  rag2          ALLOWED
  rag2  ──imports──▶  scaf          FORBIDDEN
```

This is machine-checked, not merely intended.
`architecture/rag2/tests/test_metadata_isolation.py` scans every baseline module
for executable references to date, recency or currency fields and fails the
build if one appears; a companion test proves the scanner fires by feeding it
real violations. See [`architecture/README.md`](architecture/README.md).

## 4. Repository map

| Path | What lives here | Read |
| --- | --- | --- |
| `data/` | corpora, question sets, provenance records | [`data/README.md`](data/README.md) |
| `preprocessing/` | acquisition → parsing → QC → policy → chunking → indexing | [`preprocessing/README.md`](preprocessing/README.md) |
| `architecture/` | the system under study: RAG² baseline + SCAF extension | [`architecture/README.md`](architecture/README.md) |
| `experiments/` | configs, run scripts, the three experimental layers | [`experiments/README.md`](experiments/README.md) |
| `indexes/` | built MedCPT indexes (gitignored) | [`indexes/README.md`](indexes/README.md) |
| `docs/` | audit, reproduction spec, results, runbooks, QC | [`docs/README.md`](docs/README.md) |
| `run_tests.py` | every offline suite, each in its own process | §10 |

**Path convention.** Paths are written relative to the repository root, except
inside `architecture/rag2/`, which is a self-contained reproduction unit whose
internal code references stay relative to that container.
[`architecture/README.md`](architecture/README.md) explains why, and why
`architecture/rag2/rag2/` is doubled rather than redundant.

## 5. Data organisation

```
data/
├── corpora/          the evidence the system retrieves from
│   ├── pubmed/         approved search strategy, results (Git LFS), audit log
│   └── pmc/            OA inventory, download manifest, metadata overlays,
│                       currency pack, chunk statistics
└── datasets/         the questions the system is asked
    └── thesis_questions/   30 fixed Alzheimer development questions
```

Corpus scale, as recorded in `data/corpora/pmc/chunks/chunk_stats.json`:

| | |
| --- | --- |
| PubMed records acquired | 43,409 |
| Documents in the chunk layer | 42,964 |
| Chunks | 781,563 (773,183 unique; 8,380 exact duplicates flagged, not deleted) |
| Source categories | `pubmed-abstract` 22,583 · `pmc-fulltext` 758,867 · `currency-pack` 113 |

A future study can add clinical notes, drug labels or another database as a
sibling under `data/corpora/` without any architectural change.

**Immutable by instruction:** `data/corpora/pmc/pmc_oa_inventory.csv` is the
authoritative acquisition record. `data/corpora/pubmed/search_queries.txt` is the
approved search strategy. `data/corpora/pmc/currency_pack/xml/PMC13082890.xml` is
a byte-exact snapshot pinned by MD5 `dcb1ac4eaa24b75ab3202f2315c6b2e4` — the PMC
object is revised in place, so re-fetching yields a different file and this
snapshot cannot be regenerated. Licensed CC BY-NC.

Document identity is **PMID-primary**, PMCID for the full-text join; DOI is a
consistency check only, never an identity key.

## 6. The RAG² baseline

`architecture/rag2/` holds a reproduction of the **original** RAG² system — the
comparison point, not a thesis contribution. It sits beside the authors' own
released `retriever/` and `classifier/`, vendored byte-identically so they stay
citable as published.

Read [`docs/reproduction/rag2_reproduction_audit.md`](docs/reproduction/rag2_reproduction_audit.md)
first: it records component by component what was reproduced, how it was
verified, and what remains unverified. The reproduction's own specification —
every assumption, every place the paper and the authors' code disagree — is
[`docs/reproduction/rag2_reproduction.md`](docs/reproduction/rag2_reproduction.md).

**Models required** (none are downloaded by any test):

| Role | Model | Needed for |
| --- | --- | --- |
| Query encoder | `ncbi/MedCPT-Query-Encoder` | retrieval |
| Article encoder | `ncbi/MedCPT-Article-Encoder` | building `indexes/production/` |
| Reranker | `ncbi/MedCPT-Cross-Encoder` | reranking |
| Filter | `google/flan-t5-large` + a trained checkpoint | filtering |
| Backbone LLM | `meta-llama/Meta-Llama-3-8B-Instruct` (gated) | rationales, answers |

The paper's trained filter checkpoint **is not distributed by its authors** and
must be retrained — `architecture/rag2/scripts/03_build_filter_labels.py`, then
`04_train_filter.py`. See §11.

## 7. The SCAF extension

`architecture/scaf/` implements the thesis's proposed admission policy:

```
A(s) = w_σ·σ(support) + w_γ·γ(currency) + w_ρ·ρ(corroboration) + w_τ·τ(authority)
```

`γ` is three-state: 0 if the document is retracted, 1 if the question is not
time-sensitive, `δ·2^(−age/H)` if superseded, `2^(−age/H)` otherwise. Retraction
is the only hard rejection; everything else is a weighted score against an
a-priori threshold that is **not** tuned on the evaluation set.

**Status: prototype, and the report says so.** Two components are deliberately
incomplete and must not be presented as finished:

| Component | State |
| --- | --- |
| σ support | lexical coverage + corpus IDF (`is_entailment: false`) — a stand-in for entailment |
| ρ corroboration | **not implemented**; its weight is 0.0 |
| γ currency, τ authority | implemented |

SCAF imports the baseline and is never imported by it (§3). It lives outside
`architecture/rag2/` precisely because it reads the publication-date fields the
baseline is forbidden to touch.

## 8. Experimental workflow

`experiments/` establishes a one-directional boundary: a layer may call the layer
below and may not modify it.

1. `experiments/baseline/` — original RAG² runs. **Runnable, not yet run.**
2. `experiments/recency_bias/` — the thesis probe. **Not started.**
3. `experiments/scaf/` — the SCAF comparison. **Development runs only.**

A reportable run must pass two gates, both recorded in its manifest:

- **9 fairness checks** — both arms saw the same candidates in the same order,
  the same generator object, the same decoding parameters.
- **16 scientific preconditions** — Arm A is the trained perplexity filter and
  not passthrough, retrieval was production MedCPT and not a development
  stand-in, a real generator is configured, the checkpoint exists and looks
  trained, and so on.

`run_comparison.py --scientific` **aborts** if any precondition fails. Without
`--scientific`, the manifest is stamped `DEVELOPMENT RUN -- NOT A SCIENTIFIC
RESULT` with `reportable: false`. There is no path that quietly downgrades to a
mock generator, lexical retrieval, or a passthrough filter.

## 9. Where production data lives

Production-scale artifacts are gitignored and exist only on the machine that
built them. Each is deterministic and reconstructible from what *is* committed;
committing them would bloat the repository without adding reproducibility,
because the manifests already pin their content digests.

| Artifact | Path | Size | Rebuild with |
| --- | --- | --- | --- |
| Raw PMC XML | `data/corpora/pmc/fulltext/xml/` | ~22 GB | `preprocessing/pmc/download_pmc_xml.py` |
| Parsed records | `data/corpora/pmc/parsed/` | — | `preprocessing/pmc/parse_pmc_xml.py` |
| Chunk layer | `data/corpora/pmc/chunks/chunks.jsonl` | 152 MB | `preprocessing/pmc/build_chunks.py` |
| MedCPT index | `indexes/production/` | ~2.4 GB | `preprocessing/pmc/embed_chunks.py` |
| Run outputs | `experiments/runs/` | — | the two scripts in `experiments/scripts/` |

`data/corpora/pmc/chunks/chunk_stats.json` **is** committed: it is the evidence
of the production chunk build, and it lives beside `chunks.jsonl` so that a
rebuild rewrites both together rather than leaving a stale record behind.

## 10. Running the tests

```bash
python3 run_tests.py              # all four suites
python3 run_tests.py rag2 scaf    # a subset
```

Last measured, all green:

| Suite | Tests |
| --- | --- |
| `preprocessing/pmc` | 355 |
| `preprocessing/pubmed` | 51 |
| `architecture/rag2` | 194 passed, 2 skipped (torch-gated) |
| `architecture/scaf` | 132 |

Each suite runs in **its own process**, which is a correctness requirement
rather than a convenience: importing `scaf` registers the `scaf` filter with the
baseline's filter registry, and a baseline test asserts that registry does *not*
know `scaf` — the guard that pins the one-way dependency. Run both in one
interpreter and SCAF's import silently satisfies the guard it exists to fail
against.

Every suite is offline: no network, no corpus files, no model weights, no GPU.
**A green run says the software is internally consistent. It does not say any
scientific result has been reproduced.**

## 11. Running the experiments

The full procedure — every command checked against the actual scripts, with the
CUDA install, the VRAM constraint, and what each precondition needs — is
[`docs/runbooks/windows_experiment_runbook.md`](docs/runbooks/windows_experiment_runbook.md).
In outline, from the repository root on the GPU machine:

```bash
python preprocessing/pmc/verify_index.py                    # 1. 16/16, record the digest
python architecture/rag2/scripts/02_retrieve.py \
    -c architecture/rag2/configs/thesis_corpus.yaml          # 2. real MedCPT retrieval
python experiments/scripts/freeze_candidates.py \
    --source medcpt --cache cache/candidates/<name>.jsonl    # 3. freeze, record the digest
python architecture/rag2/scripts/03_build_filter_labels.py \
    -c architecture/rag2/configs/thesis_corpus.yaml          # 4. ΔPPL labels (expensive)
python architecture/rag2/scripts/04_train_filter.py \
    -c architecture/rag2/configs/thesis_corpus.yaml --select # 5. train the filter
python experiments/scripts/run_comparison.py --scientific \
    --rag2-filter rag2_perplexity \
    --rag2-checkpoint runs/filter-medqa/best \
    --generator huggingface                                  # 6. the comparison
```

Nothing in this sequence runs in a CPU-only environment, and that is by design:
the guards refuse rather than downgrade.

## 12. Limitations

**The corpus cannot currently answer the research question.** The approved
PubMed strategy uses a five-year window
(`"2021/08/30"[Date - Publication] : "2026/08/30"[Date - Publication]`), so **7
of 43,409 documents predate 2020** — 0.02%. A recency comparison needs an older
stratum and this corpus has none; FRB-PAIRS cannot be constructed from it. This
is a corpus-design finding requiring a supervisor decision, not a software
defect. See [`docs/experiments/preliminary_rag2_vs_scaf.md`](docs/experiments/preliminary_rag2_vs_scaf.md) §0.

**No accuracy has been measured.** `docs/reproduction/reproduction_results.md` is
deliberately blank. Every result table in this repository is either empty or
stamped as a development run.

**σ is not entailment.** SCAF's support scorer is a lexical prototype (§7).
Any thesis claim about support must wait for a real entailment model.

**ρ is not implemented.** Corroboration carries weight 0.0.

**Model-dependent components are unverified.** The development container has no
torch, transformers, faiss or GPU, so MedCPT encoding, Flan-T5 filtering, ΔPPL
under a real LLM, and answer generation were checked by code reading and
stub-driven execution — not by running a real model.

**SCAF support scores are not bit-reproducible across processes.**
`SupportScorer.score` sums corpus IDF over a Python `set` of query tokens
(`architecture/scaf/policy.py:201`), so float summation order follows the string
hash seed and scores vary by one ULP (~2.2e-16) between runs. Measured: 56 of
600 scores differ between `PYTHONHASHSEED=1` and `=2`. **No admission decision
changed** in any run observed, and the σ *value* is mathematically identical —
only the rounding of its summation differs. The one-line fix is to sum over
`sorted(q_tokens)`; it is deliberately left for a separate commit, because
changing a scoring path is a scientific decision and not part of a
reorganisation. Until then, pin `PYTHONHASHSEED` for any run whose per-candidate
scores are quoted.

## 13. Reproducibility

- **Manifests, not memory.** Every run writes the resolved configuration, git
  commit, model revisions, seeds, prompt hashes and package versions beside its
  outputs. Record the manifest fingerprint next to any number that reaches the
  thesis.
- **Digests over identity *and* order.** The frozen candidate set is serialised
  with a digest that changes if either the members or their order change, so
  "both arms scored the same evidence" is verified rather than assumed.
- **Exact search, never approximate.** An ANN index would introduce run-to-run
  variation; search is an exact inner-product scan with ties broken on
  `chunk_id`.
- **Determinism where it is claimed.** The chunk build and the index build
  produce byte-identical output from the same frozen inputs, resumed runs
  included; `validate_chunks.py` and `verify_index.py` print content digests for
  cross-run comparison.
- **Checkpoint identity is recorded.** A run's manifest carries the resolved
  checkpoint path, file listing, sizes, mtimes and a digest over that listing —
  enough to distinguish a retrained checkpoint. Weight bytes are not hashed.
- **Failure is loud.** Guards abort with actionable messages instead of falling
  back. A stub index stamps `production=false`; a partial index stamps
  `partial_index_limit`; a non-scientific run stamps `reportable: false`.

## Current status

Acquisition, parsing, QC and corpus-policy metadata (M1–M4) are complete. The
chunk layer is built. The RAG² baseline is integrated and audited. SCAF is
implemented at prototype fidelity with the experimental harness and its guards
in place.

Outstanding, in order:

1. **Resolve the corpus date window** (§12) — this blocks the research question
   and is a supervisor decision.
2. Full-corpus canonical-date regeneration on the machine holding the complete
   parsed corpus: `python3 preprocessing/pmc/build_corpus_metadata.py --no-fetch`.
   It upgrades canonical dates from PubMed fallback to JATS-primary for all PMC
   records; refresh `chunk_stats.json` from the rebuild that follows.
   See [`data/corpora/pmc/metadata/README.md`](data/corpora/pmc/metadata/README.md).
3. Build the MedCPT index on the GPU machine (§11 step 1), then train the filter.
4. Run the baseline and record results in
   `docs/reproduction/reproduction_results.md`.
5. Only then: the recency-bias probe and the SCAF comparison.
