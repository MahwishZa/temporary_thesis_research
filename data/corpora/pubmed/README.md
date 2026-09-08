# PubMed acquisition pipeline

`fetch_pubmed.py` retrieves the PubMed records defined by `search_queries.txt`
using NCBI's E-utilities (ESearch → EFetch) and writes them to
`pubmed_results.csv`, `pubmed_results.json` and `search_log.csv`.

`search_queries.txt` is read-only input. The approved queries and the approved
publication-date window are read from it at run time; nothing in the pipeline
edits it, widens the window, or applies any extra topic filtering to the records
that come back.

## Running it

```bash
cd pubmed

python3 fetch_pubmed.py --list-queries   # audit the scope without any network call
python3 fetch_pubmed.py --test           # smoke test: Q0.2, first 25 PMIDs
python3 fetch_pubmed.py                  # full run over all approved queries
```

Useful options: `--queries Q4.1,Q13.1` to run a subset, `--limit N` to cap PMIDs
per query, `--output-dir DIR` to write somewhere other than this folder.

Set `NCBI_API_KEY` to raise the rate limit from 3 to 10 requests/second (and
`NCBI_EMAIL` so NCBI can make contact about heavy usage). Both are optional; the
pipeline throttles itself to whichever limit applies.

Requires only the Python 3 standard library.

## Approved scope

`search_queries.txt` mixes two kinds of block. Some are complete searches; others
are reusable building blocks that the file's own header says to "combine with
AND". The pipeline runs the **42 self-contained blocks**, each of which already
pairs the disease axis with a reasoning or diagnosis axis:

> Q0.1–Q0.3, Q1.6, Q4.1–Q4.6, Q5.1–Q5.7, Q6.1–Q6.3, Q7.1–Q7.4, Q8.1–Q8.6,
> Q9.1–Q9.3, Q10.1–Q10.2, Q11.1–Q11.2, Q13.1–Q13.5

The bare building blocks in sections 1–3 and the filter snippets in section 12
are not run on their own: `Q3.1` (`"Diagnosis"[Mesh] OR "diagnosis"[Subheading]
…`) contains no disease term and would return millions of unrelated records, and
`Q12.x` are append-only fragments rather than searches. The list lives in
`APPROVED_QUERY_IDS` in `fetch_pubmed.py`.

Every query is scoped to the window declared in the file —
`2021/08/30`–`2026/08/30`. Blocks that already carry the filter inline have it
stripped and re-applied so the effective window is identical everywhere; the run
aborts if the file ever declares two different windows.

## How retrieval works

- **ESearch** returns the hit count first, then the PMIDs. ESearch will not
  return more than 10,000 UIDs for one search, so any query above that is split
  recursively by publication-date sub-window until every slice fits, and the
  slices are merged and deduplicated. A mismatch between the reported count and
  the PMIDs actually collected is recorded as a warning rather than passed over.
- **EFetch** pulls records in batches of 200 by POST. A batch that fails or
  returns unparseable XML is halved and retried down to single PMIDs, so one bad
  record cannot take out the other 199.
- **Rate limits and retries**: requests are throttled to NCBI's published limit;
  429/5xx responses back off exponentially with jitter and honour `Retry-After`.
- **Resumability**: fetched records are cached outside the repository
  (`PUBMED_CACHE_DIR`, default under the system temp dir), so an interrupted run
  resumes without re-fetching. `--no-cache` disables it.

## Output

`pubmed_results.csv` and `pubmed_results.json` hold the same records in the same
order — one row/object per unique PMID, sorted by PMID. Columns:

| column | notes |
| --- | --- |
| `pmid` | unique; the deduplication key |
| `title`, `abstract`, `journal` | `abstract` is empty when PubMed has none |
| `publication_date` | normalised `YYYY`, `YYYY-MM` or `YYYY-MM-DD` |
| `publication_year` | derived from `publication_date` |
| `authors` | `Last, First`, `; `-separated (CSV) or a list (JSON) |
| `doi`, `pmcid` | empty when absent |
| `mesh_terms` | `Descriptor (Qualifier)`, `*` marks a major topic |
| `publication_types` | e.g. `Journal Article; Review` |
| `query_ids` | **every** approved query that retrieved this record |
| `record_status` | `ok`, or `metadata_unavailable` for a PMID ESearch returned but EFetch could not supply |

Records are never dropped for a missing abstract, DOI or PMCID. A PMID that
ESearch found but EFetch could not return is kept as an ID-only row flagged
`metadata_unavailable` and reported in the log, so nothing is lost silently.

`search_log.csv` records one row per query: the query ID and label, the exact
term as executed, the UTC search timestamp, the reported hit count, how many
PMIDs were retrieved and parsed, how many were new, and any errors or NCBI
warnings (unmatched phrases, date-window splits, fetch failures).

If `pubmed_results.csv`/`.json`/`search_log.csv` already contain data, each is
copied to `<name>_backup_<UTC timestamp>.<ext>` before being replaced. Files that
are absent or empty are not backed up.

## Verification

Every run re-reads what it wrote and checks that PMIDs are unique; that CSV and
JSON contain the same records field for field; that each query's retrieved PMIDs
are attributed to exactly those records and no record lacks provenance; that no
searched PMID is missing from the output and no saved PMID was never searched;
and that publication dates parse and fall inside the approved window. Failures
are printed and the process exits non-zero.

## Tests

Two suites cover the pipeline offline, with no network access required:

- `test_fetch_pubmed.py` — query-file parsing, XML parsing (journal articles,
  records with no abstract/DOI/PMCID, `MedlineDate` values, book chapters), date
  normalisation, deduplication with provenance, backup behaviour, CSV/JSON
  generation, and the verification checks themselves.
- `test_pipeline_integration.py` — whole-run behaviour against a fake
  E-utilities server: date-window splitting past the UID ceiling (and that the
  sub-windows neither overlap nor escape the approved range), EFetch batching
  and bisection around an unparseable record, cache reuse, provenance across
  overlapping queries, a failing query that must not stop the run, and the
  retry/backoff rules.

```bash
python3 -m unittest discover -s pubmed -v
```
