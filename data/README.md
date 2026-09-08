# `data/` — research data

Everything the research *consumes* or *produces as data*, kept apart from the
software that processes it. If a file is an observation about the world rather
than an instruction to a computer, it belongs here.

```
data/
├── corpora/     the evidence the system retrieves from
│   ├── pubmed/    acquisition results, search strategy, search log
│   └── pmc/       inventories, download manifest, metadata overlays, currency pack
└── datasets/    the questions the system is asked
    └── thesis_questions/   30 fixed Alzheimer development questions
```

Manifests — counts, digests and provenance records — live **beside the artifact
they describe**, not in a separate tree: `chunks/chunk_stats.json` next to
`chunks/chunks.jsonl`, `fulltext/manifest.csv` next to `fulltext/xml/`,
`index_meta.json` inside `indexes/production/`. A manifest separated from its
artifact goes stale silently when the artifact is rebuilt; kept together, one
run rewrites both.

## Adding a corpus later

The point of this layout is that a future study can add clinical notes, drug
labels, institutional guidelines or another biomedical database as a **sibling
under `data/corpora/`** without any architectural change. Give it its own
directory, its own README recording where the data came from and under what
licence, and a manifest written by whatever stage produces it.

## What is *not* here

Production-scale artifacts are deliberately gitignored and live only on the
machine that built them:

| Artifact | Location | Size | Rebuild with |
| --- | --- | --- | --- |
| Raw PMC XML | `data/corpora/pmc/fulltext/xml/` | ~22 GB | `preprocessing/pmc/download_pmc_xml.py` |
| Parsed article records | `data/corpora/pmc/parsed/` | — | `preprocessing/pmc/parse_pmc_xml.py` |
| Chunk layer | `data/corpora/pmc/chunks/chunks.jsonl` | 152 MB | `preprocessing/pmc/build_chunks.py` |
| MedCPT index | `indexes/production/` | ~2.4 GB | `preprocessing/pmc/embed_chunks.py` |

Each is deterministic and reconstructible from what *is* committed. Committing
them would bloat the repository without adding reproducibility, since the
manifests already pin their content digests.

## Immutability

`data/corpora/pmc/pmc_oa_inventory.csv` is the authoritative acquisition record
and must not be edited. `data/corpora/pmc/currency_pack/xml/PMC13082890.xml` is
a byte-exact snapshot pinned by MD5 (`dcb1ac4eaa24b75ab3202f2315c6b2e4`); the
PMC object is revised in place, so re-fetching yields a different file and this
snapshot cannot be regenerated.
