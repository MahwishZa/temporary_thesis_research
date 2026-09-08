# `preprocessing/` — how raw sources become a retrievable corpus

The pipeline that turns `data/corpora/` into something the research architecture
can search. Read top to bottom; each stage consumes the previous stage's output.

```
ACQUIRE → PARSE → QUALITY CONTROL → POLICY → CHUNK → INDEX
```

| Stage | Module | Produces |
| --- | --- | --- |
| Acquire (PubMed) | `pubmed/fetch_pubmed.py` | search results + audit log |
| Acquire (PMC) | `pmc/inventory_pmc_oa.py`, `pmc/download_pmc_xml.py` | OA inventory, MD5-verified XML |
| Parse | `pmc/parse_pmc_xml.py` | one structured record per article |
| Quality control | `pmc/qc_investigate.py` | independent QC reports |
| Corpus policy | `pmc/build_corpus_metadata.py` | canonical dates, eligibility, authority tiers, currency pack |
| Chunk | `pmc/build_chunks.py`, `pmc/validate_chunks.py` | 256-word windows, 32-word overlap, full provenance |
| Index | `pmc/embed_chunks.py`, `pmc/verify_index.py` | MedCPT vectors + manifest, and the gate that checks them |

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
cd preprocessing/pmc    && python3 -m unittest test_parse_pmc_xml test_qc_investigate \
                                              test_download_pmc_xml test_inventory_pmc_oa \
                                              test_build_corpus_metadata test_build_chunks \
                                              test_retrieval test_verify_index
cd preprocessing/pubmed && python3 -m unittest test_fetch_pubmed test_pipeline_integration
```

All offline: no network, no corpus files, no model weights.
