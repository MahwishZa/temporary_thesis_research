# Reconciled PMC inventory snapshot — 2026-09-02

**This is a DERIVED snapshot. It is not the validated inventory.**

`pmc/pmc_oa_inventory.csv` remains the immutable, authoritative record and was
not modified in any way to produce this file (sha256
`518ebf886a64b64e2e7a11cbfba7a3824c4a191709575c9fb9a445a310dfc918`, unchanged).

This documentation lives in a separate file because CSV has no comment syntax:
a `#` line inside the CSV would be read as the header row and would break the
downloader's parser.

## Why this file exists

The full XML acquisition on 1–2 September 2026 downloaded 25,672 of 25,743
articles successfully. The remaining **71 failed MD5 verification** after four
attempts each.

Metadata-only verification against the PMC Cloud Service on 2 September showed
that in **all 71 cases the file received was correct** and the *inventory's
recorded MD5 was out of date*. PMC replaces article objects in place — the
version number does not change — and the daily S3 inventory can lag the live
bucket. The original inventory was captured on 31 August; the download ran
roughly one day later.

The downloader behaved correctly throughout: it refused files whose fingerprint
did not match what it had been told to expect.

## What was and was not refreshed

**Refreshed:** the `?md5=` value inside `xml_url`, for exactly 71 records.

**Not refreshed — do not trust these for the 71 affected records:**

- `pdf_url` MD5 values
- `text_url` MD5 values

Those still carry their 31 August values and were never re-verified, because
this project downloads XML only. For the 71 reconciled records they are
presumed stale.

**Unchanged for all 27,508 records:** every other column — `pmcid`, `pmid`,
`doi_from_pmc`, `title_from_pmc`, `license_code`, `is_retracted`,
`is_manuscript`, `version`, `has_xml`, `has_pdf`, `has_text`,
`in_pmc_oa_dataset`, `is_pmc_openaccess`, `is_historical_ocr`, `media_count`,
`pmid_from_pmc`, `pdf_url`, `text_url`, `checked_at_utc`. Verified
byte-identical column by column across every row, and row order is preserved.

## Schema

The 20 original columns in their original order, followed by six audit columns:

| Column | Meaning |
| --- | --- |
| `reconciliation_status` | `UNCHANGED`, `CONFIRMED_STALE` or `SUPERSEDED_TWICE` |
| `original_expected_md5` | The immutable inventory's MD5 (changed rows only) |
| `current_pmc_md5` | MD5 read from PMC metadata on 2026-09-02 |
| `observed_downloaded_md5` | MD5 actually received during the 1–2 Sep run |
| `md5_source` | `original_inventory_2026-08-31` or `pmc_metadata_2026-09-02` |
| `reconciled_at_utc` | When this reconciliation was performed |

Nothing is silently replaced: for every changed row the superseded value stays
visible in `original_expected_md5`.

| Status | Rows | Meaning |
| --- | ---: | --- |
| `UNCHANGED` | 27,437 | Untouched; audit MD5 columns are blank |
| `CONFIRMED_STALE` | 70 | Current PMC MD5 equals what was downloaded, and differs from the inventory |
| `SUPERSEDED_TWICE` | 1 | `PMC13097789` — revised twice; see below |

## PMC13097789

This article carries three distinct MD5 values:

| When | MD5 |
| --- | --- |
| Inventory, 31 Aug | `3312c4bdf98f54d2b98a22329989acd1` |
| Downloaded, 1 Sep | `7912ee8f0c68a48717079731dcc588cf` |
| Current PMC, 2 Sep | `999477c08cb4f30521f8a2d8bad8e58a` |

`xml_url` records the **current** value. The 1 September bytes were discarded
when verification failed — verify-then-rename means nothing was kept — so no
local file has that digest, and recording it would guarantee another failure.

This record demonstrably changes fast. Re-verify its metadata immediately
before any retry and treat a further mismatch as expected churn.

## Using this file

The downloader takes its expected MD5 from `xml_url`, so no code change is
needed:

```bash
python3 pmc/download_pmc_xml.py --inventory pmc/pmc_oa_inventory_reconciled_2026-09-02.csv
```

Only `pmcid`, `has_xml` and `xml_url` are required by the parser; the audit
columns ride along harmlessly and cannot reach the manifest, whose schema is
fixed. Verified: the downloader reads 25,743 candidates from this file with
zero malformed rows, and enforces the refreshed MD5 on all 71 records.

A retry against this snapshot will skip the 25,672 already-verified files
without any network request, and attempt only the 71.

## Provenance

- Derived from `pmc/pmc_oa_inventory.csv` (27,508 records, 31 Aug 2026).
- The 71 refreshed values came from per-article metadata JSON objects at
  `s3://pmc-oa-opendata/metadata/PMC*.json`, read 2026-09-02. Metadata only —
  no XML, text, PDF or media was requested.
- Evidence of the failures is preserved in `pmc/fulltext/failures.csv` and
  `pmc/fulltext/manifest.csv` on the `acquire-pmc-xml` branch.
