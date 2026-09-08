# `indexes/` — vector indexes

**Nothing here is committed.** Indexes are large, machine-specific and
deterministically rebuildable; the repository pins their content digests instead.

| Index | Path | Built by | Verified by |
| --- | --- | --- | --- |
| Production MedCPT | `indexes/production/` | `preprocessing/pmc/embed_chunks.py` | `preprocessing/pmc/verify_index.py` |

The production index is 773,183 vectors × 768 dimensions, built with
`ncbi/MedCPT-Article-Encoder`, and lives on the GPU machine. Its layout is
`embeddings.f32` (row-major float32), `index_manifest.jsonl` (one row per vector
carrying provenance) and `index_meta.json` (encoder id, dim, counts, digests).

Row order is the alignment contract: manifest line *i* describes row *i* of
`embeddings.f32`. `verify_index.py` checks that, along with dimension, NaN/Inf,
L2 normalisation and digest agreement, and refuses a stub-encoder index.

Build and verify:

```bash
python preprocessing/pmc/embed_chunks.py --device cuda --batch-size 8
python preprocessing/pmc/verify_index.py
```

The build is resumable: re-run the same command after any interruption and it
picks up at the last complete row.
