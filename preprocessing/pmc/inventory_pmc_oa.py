#!/usr/bin/env python3
"""Read-only PMC Open Access / reuse inventory.

For every PMCID in pubmed/pubmed_results.csv this asks the PMC Cloud Service
whether the article is in the PMC open-access dataset and, if so, under what
licence and where its files live. It writes pmc/pmc_oa_inventory.csv.

It does NOT download any full text -- only the small per-article metadata
record (roughly 1.5 KB each). It never writes to anything under pubmed/.

Why this service: the old PMC OA Web Service API (pmc/utils/oa/oa.fcgi) was
retired on or after 24 August 2026, along with the FTP Service and the legacy
Cloud Service files. Its replacement is the PMC Cloud Service on AWS Open Data
-- the public bucket "pmc-oa-opendata" -- documented at
https://pmc.ncbi.nlm.nih.gov/tools/cloud/ and
https://pmc.ncbi.nlm.nih.gov/tools/pmcaws/. The bucket is public, so no AWS
account, credentials or extra libraries are needed: plain HTTPS is enough.

Usage:
    python3 pmc/inventory_pmc_oa.py --limit 25    # small test first
    python3 pmc/inventory_pmc_oa.py               # the whole set
    python3 pmc/inventory_pmc_oa.py               # run again to resume

Requires only the Python standard library.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Settings. Everything here can be overridden from the command line.
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = REPO_ROOT / "pubmed" / "pubmed_results.csv"
DEFAULT_OUTPUT = REPO_ROOT / "pmc" / "pmc_oa_inventory.csv"
DEFAULT_FAILURES = REPO_ROOT / "pmc" / "pmc_oa_failures.csv"

# The public PMC Cloud Service bucket, reachable over ordinary HTTPS.
BUCKET_URL = "https://pmc-oa-opendata.s3.amazonaws.com/"

# How long to pause between requests. This is an AWS Open Data bucket, not
# NCBI E-utilities, so the 3-requests-per-second E-utilities rule does not
# apply here -- but staying gentle is good manners and costs little.
DEFAULT_SLEEP = 0.10

# How many times to retry a request that failed for a temporary reason.
MAX_RETRIES = 5
REQUEST_TIMEOUT = 60

# Save progress to disk every N articles, so a crash loses at most this many.
CHECKPOINT_EVERY = 50

# The columns of pmc_oa_inventory.csv.
INVENTORY_FIELDS = [
    "pmcid",                # from our PubMed data
    "pmid",                 # from our PubMed data
    "in_pmc_oa_dataset",    # yes / no -- is the article in the bucket at all?
    "is_pmc_openaccess",    # PMC's own open-access flag
    "license_code",         # e.g. CC BY, CC BY-NC, CC0
    "is_manuscript",        # author manuscript rather than publisher version
    "is_historical_ocr",    # scanned/OCR'd historical article
    "is_retracted",
    "version",              # article version we inspected (usually 1)
    "pmid_from_pmc",        # PMC's own PMID, to cross-check ours
    "doi_from_pmc",
    "title_from_pmc",
    "has_xml",
    "has_pdf",
    "has_text",
    "media_count",
    "xml_url",              # download identifiers (s3:// URLs with md5)
    "pdf_url",
    "text_url",
    "checked_at_utc",
]

FAILURE_FIELDS = ["pmcid", "pmid", "stage", "error", "checked_at_utc"]

# Python's csv module refuses any single field longer than 131,072 characters,
# and PubMed abstracts are long, so we raise that ceiling before reading.
#
# It must NOT be raised with sys.maxsize. csv.field_size_limit() hands its
# argument to C as a "long". On Linux and macOS a long is 64-bit so sys.maxsize
# fits, but on Windows a long is 32-bit even in 64-bit Python, so the largest
# accepted value is 2,147,483,647 and sys.maxsize raises OverflowError.
#
# 64 MB is far more than any PubMed field needs -- the longest field in our
# dataset is a ~15,000-character abstract -- and fits in a 32-bit long.
CSV_FIELD_LIMIT = 64 * 1024 * 1024


def widen_csv_field_limit(target: int = CSV_FIELD_LIMIT) -> int:
    """Raise the csv field-size ceiling as far as this platform allows.

    Asks for `target`; if the platform rejects it, halves the request and tries
    again. Never lowers the limit below whatever is already in effect. Returns
    the limit actually in force.
    """
    current = csv.field_size_limit()
    limit = target
    while limit > current:
        try:
            csv.field_size_limit(limit)
            return limit
        except OverflowError:
            limit //= 2
    return current


# ---------------------------------------------------------------------------
# Step 1: read the PMCIDs out of our PubMed results (read-only).
# ---------------------------------------------------------------------------


def looks_like_lfs_pointer(path: Path) -> bool:
    """Detect a Git LFS placeholder instead of the real CSV.

    If git-lfs is not installed, pubmed_results.csv is a ~134-byte text stub
    beginning 'version https://git-lfs.github.com/spec/v1'. Reading it would
    silently yield zero rows, which is a confusing way to fail.
    """
    try:
        with path.open("rb") as handle:
            return handle.read(60).startswith(b"version https://git-lfs.github.com/spec/v1")
    except OSError:
        return False


def read_pmcids(path: Path) -> list[tuple[str, str]]:
    """Return [(pmcid, pmid), ...] for every row that has a non-empty PMCID."""
    if not path.exists():
        raise SystemExit(f"ERROR: cannot find {path}")
    if looks_like_lfs_pointer(path):
        raise SystemExit(
            f"ERROR: {path} is a Git LFS pointer, not the real data.\n"
            "Fix it with:  git lfs install && git lfs pull"
        )

    # Abstracts are long, so lift the CSV field-size ceiling before reading.
    widen_csv_field_limit()

    pairs: list[tuple[str, str]] = []
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "pmcid" not in reader.fieldnames:
            raise SystemExit(f"ERROR: {path} has no 'pmcid' column")
        for row in reader:
            pmcid = (row.get("pmcid") or "").strip()
            if pmcid:
                pairs.append((pmcid, (row.get("pmid") or "").strip()))
    return pairs


# ---------------------------------------------------------------------------
# Step 2: talk to the PMC Cloud Service, retrying temporary failures.
# ---------------------------------------------------------------------------


class NotFound(Exception):
    """The object does not exist. A real answer, not a failure."""


def http_get(url: str, sleep: float) -> bytes:
    """Fetch a URL, retrying briefly on temporary problems.

    A 404 means 'this article is not in the dataset' -- a genuine result, so it
    is raised as NotFound and never retried. Server errors (5xx), rate limiting
    (429) and dropped connections are temporary, so those wait a little longer
    each time before trying again ('exponential backoff').
    """
    last_error: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        time.sleep(sleep)
        try:
            request = urllib.request.Request(url, method="GET")
            request.add_header("User-Agent", "thesis_research_pmc_oa_inventory/1.0")
            with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT) as response:
                return response.read()
        except urllib.error.HTTPError as exc:
            if exc.code in (403, 404):
                raise NotFound(f"HTTP {exc.code}") from exc
            last_error = RuntimeError(f"HTTP {exc.code}")
            if exc.code not in (408, 429, 500, 502, 503, 504) or attempt == MAX_RETRIES:
                raise last_error from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_error = RuntimeError(f"network error: {exc}")
            if attempt == MAX_RETRIES:
                raise last_error from exc
        # Wait 2s, then 4s, 8s... plus a little randomness so repeated failures
        # do not all retry at the same instant.
        time.sleep(min(60.0, 2**attempt + random.uniform(0, 0.5)))
    raise last_error or RuntimeError("request failed")


def list_versions(pmcid: str, sleep: float) -> list[str]:
    """Ask the bucket which versions of this article's metadata exist.

    Keys look like 'metadata/PMC9277667.1.json'. An article can be revised, so
    there may be a .2, .3 and so on. An empty list means the article is not in
    the PMC open-access dataset at all.
    """
    query = urllib.parse.urlencode(
        {"list-type": "2", "max-keys": "100", "prefix": f"metadata/{pmcid}."}
    )
    body = http_get(BUCKET_URL + "?" + query, sleep).decode("utf-8", "replace")
    return re.findall(r"<Key>([^<]+)</Key>", body)


def newest_version_key(keys: list[str]) -> str:
    """Pick the highest-numbered version from a list of metadata keys."""
    def version_number(key: str) -> int:
        match = re.search(r"\.(\d+)\.json$", key)
        return int(match.group(1)) if match else 0
    return max(keys, key=version_number)


def fetch_metadata(key: str, sleep: float) -> dict[str, Any]:
    raw = http_get(BUCKET_URL + urllib.parse.quote(key), sleep)
    return json.loads(raw.decode("utf-8"))


# ---------------------------------------------------------------------------
# Step 3: turn one article's metadata into one row of our inventory.
# ---------------------------------------------------------------------------


def yes_no(value: Any) -> str:
    if value is None:
        return ""
    return "yes" if value else "no"


def build_row(pmcid: str, pmid: str, meta: dict[str, Any] | None) -> dict[str, str]:
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    if meta is None:
        # Not in the dataset. Recorded explicitly rather than left out, so the
        # inventory has one row for every PMCID we started with.
        return {
            "pmcid": pmcid, "pmid": pmid, "in_pmc_oa_dataset": "no",
            "checked_at_utc": now,
            **{f: "" for f in INVENTORY_FIELDS
               if f not in {"pmcid", "pmid", "in_pmc_oa_dataset", "checked_at_utc"}},
        }

    media = meta.get("media_urls") or []
    return {
        "pmcid": pmcid,
        "pmid": pmid,
        "in_pmc_oa_dataset": "yes",
        "is_pmc_openaccess": yes_no(meta.get("is_pmc_openaccess")),
        "license_code": meta.get("license_code") or "",
        "is_manuscript": yes_no(meta.get("is_manuscript")),
        "is_historical_ocr": yes_no(meta.get("is_historical_ocr")),
        "is_retracted": yes_no(meta.get("is_retracted")),
        "version": str(meta.get("version") or ""),
        "pmid_from_pmc": str(meta.get("pmid") or ""),
        "doi_from_pmc": meta.get("doi") or "",
        "title_from_pmc": meta.get("title") or "",
        # bool(...) so a missing or null URL reads "no" rather than blank.
        "has_xml": yes_no(bool(meta.get("xml_url"))),
        "has_pdf": yes_no(bool(meta.get("pdf_url"))),
        "has_text": yes_no(bool(meta.get("text_url"))),
        "media_count": str(len(media)),
        "xml_url": meta.get("xml_url") or "",
        "pdf_url": meta.get("pdf_url") or "",
        "text_url": meta.get("text_url") or "",
        "checked_at_utc": now,
    }


# ---------------------------------------------------------------------------
# Step 4: resuming. We append as we go and skip what is already done.
# ---------------------------------------------------------------------------


def already_done(path: Path) -> set[str]:
    """PMCIDs already present in the output file from an earlier run."""
    if not path.exists():
        return set()
    widen_csv_field_limit()
    done: set[str] = set()
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            pmcid = (row.get("pmcid") or "").strip()
            if pmcid:
                done.add(pmcid)
    return done


def open_appending(path: Path, fields: list[str]):
    """Open a CSV for appending, writing the header only if the file is new."""
    path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not path.exists() or path.stat().st_size == 0
    handle = path.open("a", encoding="utf-8", newline="")
    writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
    if is_new:
        writer.writeheader()
        handle.flush()
    return handle, writer


# ---------------------------------------------------------------------------
# Step 5: the main loop.
# ---------------------------------------------------------------------------


def run(args: argparse.Namespace) -> int:
    pairs = read_pmcids(args.input)
    print(f"Found {len(pairs):,} PMCIDs in {args.input}")

    done = already_done(args.output)
    if done:
        print(f"Resuming: {len(done):,} already inventoried, skipping those")

    todo = [(pmcid, pmid) for pmcid, pmid in pairs if pmcid not in done]
    if args.limit:
        todo = todo[: args.limit]
        print(f"TEST MODE: only the first {len(todo)} of the remaining articles")
    if not todo:
        print("Nothing left to do. The inventory is complete.")
        return 0

    print(f"To do now: {len(todo):,}   (about {len(todo) * 2:,} small requests)")
    print(f"Writing to {args.output}")
    print(f"Failures to {args.failures}\n")

    out_handle, out_writer = open_appending(args.output, INVENTORY_FIELDS)
    fail_handle, fail_writer = open_appending(args.failures, FAILURE_FIELDS)

    counts = {"in_oa": 0, "not_in_oa": 0, "failed": 0}
    started = time.monotonic()

    try:
        for index, (pmcid, pmid) in enumerate(todo, start=1):
            stage = "list"
            try:
                keys = list_versions(pmcid, args.sleep)
                if keys:
                    stage = "metadata"
                    meta = fetch_metadata(newest_version_key(keys), args.sleep)
                    out_writer.writerow(build_row(pmcid, pmid, meta))
                    counts["in_oa"] += 1
                else:
                    out_writer.writerow(build_row(pmcid, pmid, None))
                    counts["not_in_oa"] += 1
            except NotFound:
                # The listing said a version existed but the object had gone,
                # or access was refused. Treat as "not available", not a crash.
                out_writer.writerow(build_row(pmcid, pmid, None))
                counts["not_in_oa"] += 1
            except Exception as exc:  # noqa: BLE001 - recorded, never dropped
                counts["failed"] += 1
                fail_writer.writerow({
                    "pmcid": pmcid,
                    "pmid": pmid,
                    "stage": stage,
                    "error": f"{type(exc).__name__}: {exc}",
                    "checked_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                })
                print(f"  ! {pmcid} failed at {stage}: {exc}", flush=True)

            if index % CHECKPOINT_EVERY == 0 or index == len(todo):
                out_handle.flush()
                fail_handle.flush()
                elapsed = time.monotonic() - started
                rate = index / elapsed if elapsed else 0
                remaining = (len(todo) - index) / rate if rate else 0
                print(
                    f"  {index:,}/{len(todo):,}  "
                    f"in OA: {counts['in_oa']:,}  not in OA: {counts['not_in_oa']:,}  "
                    f"failed: {counts['failed']:,}  "
                    f"({rate:.1f}/s, ~{remaining/60:.0f} min left)",
                    flush=True,
                )
    except KeyboardInterrupt:
        print("\nStopped by user. Progress is saved -- run again to resume.")
    finally:
        out_handle.close()
        fail_handle.close()

    print("\n" + "=" * 60)
    print("PMC OA inventory summary (this run)")
    print("=" * 60)
    print(f"Checked:          {sum(counts.values()):,}")
    print(f"In PMC OA set:    {counts['in_oa']:,}")
    print(f"Not in PMC OA:    {counts['not_in_oa']:,}")
    print(f"Failed:           {counts['failed']:,}")
    if counts["failed"]:
        print(f"\nFailures are listed in {args.failures}.")
        print("Run this script again to retry them -- failed PMCIDs are not")
        print("written to the inventory, so a re-run picks them up.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT,
                        help="PubMed results CSV to read PMCIDs from (read-only)")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help="inventory CSV to create or append to")
    parser.add_argument("--failures", type=Path, default=DEFAULT_FAILURES,
                        help="CSV of articles that could not be checked")
    parser.add_argument("--limit", type=int,
                        help="only process this many articles (use for a test run)")
    parser.add_argument("--sleep", type=float, default=DEFAULT_SLEEP,
                        help=f"pause between requests in seconds (default {DEFAULT_SLEEP})")
    return parser


if __name__ == "__main__":
    sys.exit(run(build_parser().parse_args()))
