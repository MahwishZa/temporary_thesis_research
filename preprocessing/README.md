# `preprocessing/` — how raw sources become a retrievable corpus

The pipeline that turns `data/corpora/` into something the research architecture
can search. Read top to bottom; each stage consumes the previous stage's output.

```
ACQUIRE → PARSE → QUALITY CONTROL → POLICY → CHUNK → INDEX
```

| Stage | Module | Produces |
| --- | --- | --- |
| Acquire (PubMed) | `preprocessing/pubmed/fetch_pubmed.py` | search results + audit log |
| Acquire (PMC) | `preprocessing/pmc/inventory_pmc_oa.py`, `preprocessing/pmc/download_pmc_xml.py` | OA inventory, MD5-verified XML |
| Parse | `preprocessing/pmc/parse_pmc_xml.py` | one structured record per article |
| Quality control | `preprocessing/pmc/qc_investigate.py` | independent QC reports |
| Corpus policy | `preprocessing/pmc/build_corpus_metadata.py` | canonical dates, eligibility, authority tiers, currency pack |
| Chunk | `preprocessing/pmc/build_chunks.py`, `preprocessing/pmc/validate_chunks.py` | 256-word windows, 32-word overlap, full provenance |
| Index | `preprocessing/pmc/embed_chunks.py`, `preprocessing/pmc/verify_index.py` | MedCPT vectors + manifest, and the gate that checks them |

`retrieve.py` is **not** in that table. See "Superseded modules" below.

## Chunking strategy

Taken from the thesis proposal (§5.1), not invented: **256-word sliding windows
with 32-word overlap**, sized against the article encoder's 512-token limit with
headroom for a prepended title and section header, plus exact content-hash
deduplication. Two decisions follow from that text:

- **Windows never cross a section boundary.** The proposal requires that a
  recommendation is never separated from its qualifying conditions; windowing
  inside sections also keeps section provenance exact for every chunk.
- **Windows are measured in whitespace words** — deterministic and dependency
  free (this layer is standard-library only). A 256-word window is always fewer
  than 512 sub-word tokens, so it stays inside the encoder limit with headroom.
  `--window/--overlap` can later be set in sub-word tokens without changing any
  other logic.

Title and section heading are stored as separate fields, not baked into the
text; `build_chunks.compose_embed_text()` is the single shared rule for what the
encoder sees.

Frozen policy is enforced, never re-decided: records whose M4
`eligibility_status` is `excluded` are not chunked; everything else carries its
frozen status through so retrieval can filter. Exact-duplicate text is
**flagged** via `duplicate_of`, never deleted — distinct versions and source
types must survive, because recency is an experimental variable.

```bash
python3 preprocessing/pmc/build_chunks.py      # -> data/corpora/pmc/chunks/
python3 preprocessing/pmc/validate_chunks.py   # integrity gate; non-zero on failure
```

Both are deterministic: the same frozen inputs produce byte-identical output.
`validate_chunks.py` prints a content digest for cross-run comparison.

## Indexing

| Role | Model |
| --- | --- |
| Document encoder | `ncbi/MedCPT-Article-Encoder` |
| Query encoder | `ncbi/MedCPT-Query-Encoder` |
| Reranker | `ncbi/MedCPT-Cross-Encoder` |

Fixed by the base paper and the proposal (§5.3), not chosen for convenience. The
retriever is deliberately frozen — that is the thesis's central internal-validity
guarantee.

**Exact flat search, not ANN.** Validity control V3 requires the candidate set to
be replayed byte-identically to every experimental arm; an approximate index
introduces run-to-run variation. Search is an exact inner-product scan over
unit-normalized vectors, with ties broken on `chunk_id` so ordering is total.
FAISS/numpy are used when present purely for speed and give identical results.

```bash
python3 preprocessing/pmc/embed_chunks.py --device cuda --batch-size 8
python3 preprocessing/pmc/verify_index.py     # 16-check integrity gate
```

Three properties matter for a run this long:

- **`--device cuda` is a requirement, not a preference.** It fails with an
  actionable message rather than silently falling back to the CPU. `--device
  auto` keeps the fallback but says which device it chose. A `+cpu` torch build
  makes `torch.cuda.is_available()` return `False` and turns hours into days —
  see the runbook for the CUDA install.
- **The run resumes by default.** Output is flushed every batch and a restart
  picks up at the last complete row, trimming a partial vector or half-written
  manifest line first. A resumed index is byte-identical to an uninterrupted one,
  `content_digest` included. `--restart` starts over.
- **CUDA OOM halves the batch and continues**, and stays reduced. It never falls
  back to the CPU mid-run, which would make the index internally inconsistent.

`--batch-size 8` is sized for a 4 GB card at 512 tokens; raise it on a larger
GPU. `--limit N` embeds only the first N chunks and stamps the result
`partial_index_limit`, so a smoke-test index cannot be mistaken for production.
A deterministic stub encoder exists for offline testing only: it refuses to write
without `--allow-stub` and stamps `production=false`.

Run `verify_index.py` before anything retrieves. It checks row count against the
chunk layer, dimension, row-for-row alignment, absence of NaN/Inf, L2
normalisation, and that `content_digest` recomputes to the recorded value.

## Superseded modules

`preprocessing/pmc/retrieve.py` is a working prototype that is **no longer on
the experimental path**. It was the first implementation of balanced retrieval
and candidate-set replay, written before the RAG² reproduction existed. Both
responsibilities have since moved:

| `retrieve.py` provided | superseded by |
| --- | --- |
| balanced retrieval | `architecture/rag2/rag2/retrieval/` |
| candidate persistence and replay | `architecture/scaf/frozen.py` |

Nothing **on this branch** imports it but its own 54 tests, which still pass.
**On `origin/main` it has a live consumer**: `thesis/retrieval.py` loads it via
`thesis/_bootstrap.py`. That tree is not on this branch and was not part of this
reorganisation — see
[`docs/architecture/reorganization_2026-09-08.md`](../docs/architecture/reorganization_2026-09-08.md)
§13.

So it is retained for two reasons, not one: it is a validated,
standard-library-only reference for both mechanisms, *and* code outside this
branch depends on it. Any removal must resolve that dependency first, in its own
explicit commit.

## Why `pmc/` is one flat directory

The modules import each other by bare name — `build_corpus_metadata` imports
`parse_pmc_xml`, `embed_chunks` imports `build_chunks`, `retrieve` imports
`embed_chunks`. Python resolves those only when the modules are **siblings on
`sys.path`**. Splitting them into `acquisition/`, `chunking/` and `indexing/`
subdirectories would look tidier and break every one of those imports.

The stages are therefore expressed in the table above rather than in directories.
This is a deliberate trade of cosmetic structure for working code.

## Running the tests

Tests sit beside the modules they cover and import them by bare name, so run
them from inside the directory:

```bash
cd preprocessing/pmc    && python3 -m unittest discover -s . -p 'test_*.py'   # 355
cd preprocessing/pubmed && python3 -m unittest discover -s . -p 'test_*.py'   #  51
```

Or from the repository root, `python3 run_tests.py pmc pubmed`.

All offline: no network, no corpus files, no model weights.
