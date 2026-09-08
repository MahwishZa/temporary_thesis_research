# `data/` — research data

Everything here is **evidence**, not code. Nothing under `data/` is imported;
the code that reads and writes it lives in `src/thesis/corpus_build/`, and the
paths are named once in [`src/thesis/paths.py`](../src/thesis/paths.py).

Committed or gitignored is a deliberate distinction, not a size heuristic:

| Path | Committed? | Why |
| --- | --- | --- |
| `pubmed/search_queries.txt` | yes | the approved query bank — a read-only research input |
| `pubmed/pubmed_results.{csv,json}` | yes (Git LFS) | the acquisition record: what the search actually returned |
| `pubmed/search_log.csv` | yes | per-query audit log |
| `pmc/inventory/pmc_oa_inventory.csv` | yes | **immutable** — the authoritative acquisition record |
| `pmc/inventory/pmc_oa_failures.csv` | yes | what could not be acquired, and why |
| `pmc/inventory/*_reconciled_*.csv` | yes | the reconciliation snapshot (see `docs/corpus/`) |
| `pmc/fulltext/manifest.csv`, `failures.csv` | yes | per-file MD5s: the proof the XML is what PMC served |
| `pmc/fulltext/xml/` | **no** | reconstructible from the inventory plus those MD5s |
| `pmc/parsed/` | **no** | deterministic output of the parsing stage; each record carries its source MD5 |
| `pmc/metadata/` | yes | the frozen M1–M4 corpus policy overlays (see its README) |
| `pmc/currency_pack/` | yes | externally ingested documents, pinned by MD5 |
| `pmc/chunks/chunk_stats.json` | yes | the committed evidence of a chunk build |
| `pmc/chunks/chunks.jsonl` | **no** | deterministic rebuild from the parsed corpus + frozen overlays |
| `pmc/index/`, `pmc/candidates/` | **no** | large, machine-specific, rebuilt on the GPU machine |

A gitignored artifact is one a documented command reproduces byte-identically.
A committed one is either an input the research did not generate, or the record
of a run that cannot be repeated.

## Two files that must never be regenerated in place

- `pmc/inventory/pmc_oa_inventory.csv` — the authoritative acquisition record.
- `pmc/currency_pack/xml/PMC13082890.xml` — MD5 `dcb1ac4eaa24b75ab3202f2315c6b2e4`.
  PMC revises this object in place, so a re-fetch yields a different hash. It is
  marked `-text` in `.gitattributes` so no checkout can alter a byte of it.

## Rebuilding

```bash
python -m thesis.corpus_build.parsing.parse_pmc_xml      # -> pmc/parsed/
python -m thesis.corpus_build.chunking.build_chunks      # -> pmc/chunks/
python -m thesis.corpus_build.qc.validate_chunks         # integrity gate
python -m thesis.corpus_build.embedding.embed_chunks --device cuda
python -m thesis.corpus_build.qc.verify_index            # integrity gate
```

Run the gates. They are the reason a rebuilt artifact can be trusted to be the
same artifact.
