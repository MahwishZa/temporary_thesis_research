# PMC XML retry outcome — 2026-09-02

Outcome record for the targeted retry of the 71 records flagged in
`pmc_oa_inventory_reconciled_2026-09-02.csv`.

**Result: 70 of 71 resolved and MD5-verified. `PMC13082890` remains unresolved.**

## Background

The full XML acquisition on 1–2 September downloaded 25,672 of 25,743 articles.
The other 71 failed MD5 verification after four attempts each. Metadata-only
verification showed the bytes received were correct and the *inventory's*
recorded MD5 had gone out of date: PMC replaces article objects in place without
changing the version number, and the daily S3 inventory can lag the live bucket.
`pmc_oa_inventory_reconciled_2026-09-02.csv` refreshed the `?md5=` value inside
`xml_url` for exactly those 71 records, leaving
`pmc_oa_inventory.csv` untouched.

This retry re-fetched those 71 against the refreshed digests.

## What was verified

Each of the 70 successful files was re-hashed from disk and compared with the
MD5 currently embedded in that record's reconciled `xml_url`:

| Check | Result |
| --- | --- |
| Targeted records | 71 |
| Manifest status `ok` | 70 |
| Re-hashed on disk and matching the reconciled MD5 | 70 / 70 |
| Manifest `md5_verified = yes` | 70 |
| Files downloaded outside the targeted set | 0 |
| Pre-existing XML files altered or deleted | 0 |
| Manifest rows outside the targeted set written by this run | 0 |

`pmc_oa_inventory.csv`, `pmc_oa_inventory_reconciled_2026-09-02.csv` and
`download_pmc_xml.py` were confirmed byte-identical before and after the run. No
PDF, plain-text, figure or supplementary object was requested.

## The unresolved record

```
PMC13082890   pmid 41985900   attempts 8   status failed   md5_verified no
  reconciled expected  ddb1c90906a722c4473a4e4a9eb7ff9d
  received             a895267b736f98bb5ce0d021144c172e
```

This record has now produced three distinct digests across three days: the
original inventory value (31 Aug), the value seen during the first acquisition
(1 Sep), and a third value on this retry. The reconciled MD5 was read from PMC
metadata on 2 September; by the time the object was fetched it had been replaced
again.

The downloader behaved correctly: it refused bytes whose fingerprint did not
match what it had been told to expect, and verify-then-rename means nothing was
written, so no unverified file exists on disk for this PMCID.

**The retry effort for this record was stopped deliberately.** The article
appears to be undergoing revision at PMC, and each metadata read can be
superseded before a download completes. It is excluded from the resolved set
rather than forced to pass.

The reconciliation snapshot was **not** edited to make the count read 70. It
still contains all 71 rows with their 2 September evidence, including this
record's superseded digest in `original_expected_md5`. Its failure is preserved
in `fulltext/failures.csv`, and its manifest row still reads `failed`.

To pick this up later: re-read `s3://pmc-oa-opendata/metadata/PMC13082890.*.json`
for the then-current MD5, record the correction, and re-fetch. Expect further
churn until the article stabilises.

## Corpus state

| | Count |
| --- | ---: |
| XML candidates (`has_xml = yes`) | 25,743 |
| Verified | 25,742 |
| Unresolved | 1 |

## What this record attests, and what it does not

No XML is committed to this repository — that has been true throughout, and the
manifest has always been an attestation rather than a description of any
particular working copy. It records, for every article, the URL, the expected
MD5, the MD5 actually received and the byte count, so the corpus can be rebuilt
from `pmc_oa_inventory.csv` plus the reconciliation snapshot and proven
byte-identical.

A clone therefore holds the evidence, not the bytes. Anyone re-running
`download_pmc_xml.py` fetches the files themselves; matching MD5s are the proof
the corpus is the same one.

## Reproducing the retry

The 71-row scratch inventory used to scope this run is not committed — it is
derived, and regenerating it is a few lines:

```python
import csv
csv.field_size_limit(2_000_000_000)
src = "pmc/pmc_oa_inventory_reconciled_2026-09-02.csv"
keep = {"CONFIRMED_STALE", "SUPERSEDED_TWICE"}
with open(src, encoding="utf-8", newline="") as f:
    r = csv.DictReader(f)
    fields, rows = list(r.fieldnames), [x for x in r if x["reconciliation_status"] in keep]
assert len(rows) == 71, len(rows)
with open("retry71.csv", "w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
    w.writeheader()
    w.writerows(rows)
```

Then:

```bash
python3 pmc/download_pmc_xml.py --inventory retry71.csv
```

Rows are copied verbatim, so `xml_url` — and the MD5 the downloader enforces —
is exactly what the reconciliation snapshot recorded. The subset is scoped by
`--inventory`, not `--limit`, which is positional and would select the wrong
rows.

## Files changed by this retry

- `pmc/fulltext/manifest.csv` — 71 rows updated in place (70 → `ok`, 1 → `failed`).
  Row count unchanged at 25,743.
- `pmc/fulltext/failures.csv` — one row appended for `PMC13082890`.
