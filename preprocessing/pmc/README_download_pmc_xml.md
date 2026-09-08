# PMC full-text XML downloader

`download_pmc_xml.py` reads the validated inventory `pmc/pmc_oa_inventory.csv`,
takes the records where `has_xml = yes` (25,743 of 27,508), and downloads each
article's JATS XML from the PMC Cloud Service, verifying every file against the
MD5 the inventory already carries.

## What it downloads — and what it does not

**Downloads:** XML only. One `.xml` file per article.

**Never downloads:** PDFs, plain-text renditions, figures, images, supplementary
material, or any other media. Only the `xml_url` column is ever requested, and a
test asserts that every URL the downloader touches ends in `.xml`.

**Never touches:** anything under `pubmed/`, or the inventory itself. The
inventory is opened read-only; `pubmed/` is not opened at all. A runtime guard
refuses to start if the output directory is inside `pubmed/` or beside the
inventory.

## Acquisition, not corpus selection

This tool applies **no licence policy and no retraction policy**. It copies
`license_code`, `is_retracted` and `is_manuscript` from the inventory straight
into the manifest, so corpus selection stays a separate, reversible step. In
particular, retracted articles are downloaded and flagged — never silently
dropped.

## Running it

```bash
python3 pmc/download_pmc_xml.py --list-only    # plan only; makes no network requests
python3 pmc/download_pmc_xml.py --limit 5      # small test run
python3 pmc/download_pmc_xml.py                # full run (deliberate, no flag)
python3 pmc/download_pmc_xml.py                # run again to resume or retry failures
```

The full 25,743-article run happens only when you run it **without** `--limit`.

Options: `--inventory`, `--output-dir`, `--limit N`, `--sleep SECONDS`
(default 0.10), `--timeout SECONDS` (default 60), `--max-attempts N` (default 4),
`--list-only`.

Requires only the Python standard library.

## Where things are stored

```
pmc/fulltext/
├── xml/
│   ├── PMC9277667.xml        one file per article, named by PMCID
│   └── ...
├── manifest.csv              one row per article: state and provenance
└── failures.csv              one row per failed attempt, with the reason
```

Files are named by PMCID alone, not by version — the manifest records which
version was fetched. There are no filename collisions: all 25,743 PMCIDs are
unique.

## How integrity is verified

Every `xml_url` in the inventory carries the object's MD5:

```
s3://pmc-oa-opendata/PMC9277667.1/PMC9277667.1.xml?md5=072929c9c0d1ec3a302c1cc9057dd782
```

The downloader converts this to
`https://pmc-oa-opendata.s3.amazonaws.com/PMC9277667.1/PMC9277667.1.xml` and
keeps the digest as the expected hash. Then:

1. the response is streamed to a temporary `PMC9277667.xml.part`, hashed as it
   arrives;
2. the hash is compared with the expected MD5;
3. **only on a match** is the `.part` file atomically renamed into place.

A hash mismatch is treated as a failed download: the partial file is deleted and
the article is retried. A partial or corrupted transfer can therefore never be
mistaken for a finished one, and an interrupted run never leaves a truncated
`.xml` behind.

The manifest records `expected_md5`, `actual_md5`, `md5_verified` and `bytes` for
every article.

## How it resumes

The manifest is the state. On each run the downloader:

- reads the manifest (later rows win, so a successful retry supersedes an earlier
  failure);
- for each candidate, if the target `.xml` already exists it **re-hashes it**:
  matching files are skipped without any network request; non-matching files are
  treated as incomplete and re-downloaded;
- retries anything whose last status is not `ok` or `verified_existing`.

Stop it at any point with Ctrl+C: it flushes, closes cleanly, prints that it can
be resumed, and exits with status 130. Rows are flushed every 25 articles, so an
abrupt kill loses at most 25 rows of bookkeeping — the files themselves are
already safe on disk.

Because the manifest is appended to during a run (an append can never corrupt
earlier rows) and consolidated to one row per article at the end, an interrupted
run leaves a readable manifest either way.

## Failures

Every failure is recorded in `failures.csv` with the PMCID, PMID, resolved URL,
HTTP status where available, reason, attempt count and timestamp. Nothing is
silently lost, and failed articles are picked up automatically on the next run.

HTTP handling:

| Condition | Treatment |
| --- | --- |
| 408, 425, 429, 500–504 | transient — retried with exponential backoff and jitter |
| Timeouts, connection errors | transient — retried |
| MD5 mismatch | transient — retried, never accepted |
| 404, 403, other 4xx | permanent — recorded immediately, not retried |
| Malformed `xml_url` in the inventory | recorded as a failure, never downloaded |

Attempts are bounded by `--max-attempts`; attempt counts accumulate across runs
so a persistently failing article is visible in the manifest.

## Manifest columns

`pmcid`, `pmid`, `doi`, `title`, `license_code`, `is_retracted`, `is_manuscript`,
`version`, `source_xml_url`, `resolved_url`, `expected_md5`, `actual_md5`,
`md5_verified`, `filename`, `bytes`, `status`, `http_status`, `attempts`,
`downloaded_at_utc`, `error`.

Statuses: `ok` (downloaded and verified), `verified_existing` (already present
and hash-verified), `failed` (transient, will retry), `failed_permanent`
(e.g. 404).

## Storage and Git

The downloaded XML is research data, not source code. Roughly 25,700 files at
around 100–200 KB each is on the order of **3–5 GB** — an estimate from typical
article sizes, not a measurement.

`pmc/fulltext/xml/` should **not** be committed. The manifest, failures file and
scripts are enough to reconstruct and verify the corpus exactly: every file's MD5
is recorded, so anyone can rebuild it and prove it byte-identical. That is a
stronger reproducibility claim than committing the bytes, and it avoids
redistributing content whose licence may not permit it.

*No `.gitignore` change has been made — that needs explicit approval.*

## Tests

`test_download_pmc_xml.py` — 32 tests, none of which touch the network.
`urlopen` is replaced by a fake serving in-memory bytes, so no real PMC article
is ever requested.

```bash
python3 -m unittest discover -s pmc -v
```

Coverage: inventory reading and candidate selection, malformed rows, URL
conversion, MD5 verification, successful download, skipping an already-verified
file, detecting and replacing a corrupt file, failure recording, retry and
backoff behaviour, permanent-vs-transient classification, interruption and
resumability, manifest consolidation, and guards preventing any write under
`pubmed/` or any request for a non-XML URL.
