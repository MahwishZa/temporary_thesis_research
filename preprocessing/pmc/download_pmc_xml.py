#!/usr/bin/env python3
"""Safe, resumable downloader for PMC full-text XML.

Reads the validated inventory at pmc/pmc_oa_inventory.csv, selects the records
where has_xml == "yes", and downloads each article's JATS XML from the PMC Cloud
Service. Every file is verified against the MD5 that the inventory already
carries, so a download either matches the expected bytes or is treated as a
failure.

WHAT THIS DOWNLOADS
    XML only -- one .xml file per article.

WHAT THIS NEVER DOWNLOADS
    PDFs, plain-text renditions, figures, images, supplementary material or any
    other media. Only the xml_url column is ever requested.

WHAT THIS NEVER TOUCHES
    Anything under pubmed/, and the inventory itself. Both are opened read-only
    (the inventory) or not at all (pubmed/), and a runtime guard refuses to write
    into either.

ACQUISITION, NOT SELECTION
    This is an acquisition tool. It applies no licence policy and no retraction
    policy: it copies license_code, is_retracted and is_manuscript from the
    inventory into the manifest so that corpus selection can happen later, as a
    separate and reversible step.

Usage:
    python3 pmc/download_pmc_xml.py --list-only      # plan, no network at all
    python3 pmc/download_pmc_xml.py --limit 5        # small test run
    python3 pmc/download_pmc_xml.py                  # full run (deliberate)
    python3 pmc/download_pmc_xml.py                  # run again to resume

Requires only the Python standard library.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import random
import re
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INVENTORY = REPO_ROOT / "pmc" / "pmc_oa_inventory.csv"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "pmc" / "fulltext"

# The PMC Cloud Service bucket. The inventory stores s3:// URLs; the same object
# is readable over plain HTTPS at this prefix, so no credentials are needed.
S3_PREFIX = "s3://pmc-oa-opendata/"
HTTPS_PREFIX = "https://pmc-oa-opendata.s3.amazonaws.com/"

# s3://pmc-oa-opendata/PMC9277667.1/PMC9277667.1.xml?md5=072929c9c0d1ec...
XML_URL_RE = re.compile(
    r"^s3://pmc-oa-opendata/(?P<key>PMC\d+\.\d+/PMC\d+\.\d+\.xml)\?md5=(?P<md5>[0-9a-f]{32})$"
)

DEFAULT_SLEEP = 0.10
DEFAULT_TIMEOUT = 60
DEFAULT_MAX_ATTEMPTS = 4
CHECKPOINT_EVERY = 25
DOWNLOAD_CHUNK = 64 * 1024

# HTTP codes worth trying again; anything else is permanent for this run.
TRANSIENT_HTTP = {408, 425, 429, 500, 502, 503, 504}

# Statuses that mean "this article is done"; anything else is retried next run.
DONE_STATUSES = {"ok", "verified_existing"}

MANIFEST_FIELDS = [
    "pmcid", "pmid", "doi", "title",
    "license_code", "is_retracted", "is_manuscript", "version",
    "source_xml_url", "resolved_url",
    "expected_md5", "actual_md5", "md5_verified",
    "filename", "bytes",
    "status", "http_status", "attempts", "downloaded_at_utc", "error",
]

FAILURE_FIELDS = [
    "pmcid", "pmid", "url", "http_status", "reason", "attempts", "timestamp_utc",
]


class PermanentError(Exception):
    """A failure that retrying will not fix (404, 403, bad URL, hash mismatch)."""

    def __init__(self, message: str, http_status: int | None = None) -> None:
        super().__init__(message)
        self.http_status = http_status


class TransientError(Exception):
    """A failure that may succeed on a later attempt."""

    def __init__(self, message: str, http_status: int | None = None) -> None:
        super().__init__(message)
        self.http_status = http_status


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# Reading the inventory (read-only) and selecting candidates
# ---------------------------------------------------------------------------


def widen_csv_field_limit(target: int = 64 * 1024 * 1024) -> int:
    """Raise the csv field-size ceiling portably.

    Deliberately not sys.maxsize: csv.field_size_limit() takes a C long, which is
    32-bit on Windows even in 64-bit Python, so sys.maxsize raises OverflowError
    there. Kept local rather than imported so this script stays standalone, in
    line with the other scripts in this repository.
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


def looks_like_lfs_pointer(path: Path) -> bool:
    try:
        with path.open("rb") as handle:
            return handle.read(60).startswith(b"version https://git-lfs.github.com/spec/v1")
    except OSError:
        return False


def split_xml_url(url: str) -> tuple[str, str]:
    """Turn an inventory s3:// XML URL into (https url, expected md5).

    Raises ValueError if the URL is not the exact shape the inventory produces,
    so a malformed row is reported rather than silently downloaded from
    somewhere unexpected.
    """
    match = XML_URL_RE.fullmatch((url or "").strip())
    if not match:
        raise ValueError(f"unrecognised xml_url: {url!r}")
    return HTTPS_PREFIX + match.group("key"), match.group("md5")


def read_candidates(path: Path) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Return (candidates, malformed rows) from the inventory.

    A candidate is a row with has_xml == "yes" and a well-formed xml_url. Rows
    that claim XML but carry an unusable URL are returned separately so they are
    reported instead of dropped.
    """
    if not path.exists():
        raise SystemExit(f"ERROR: inventory not found: {path}")
    if looks_like_lfs_pointer(path):
        raise SystemExit(
            f"ERROR: {path} is a Git LFS pointer, not the real inventory.\n"
            "Fix it with:  git lfs install && git lfs pull"
        )
    widen_csv_field_limit()

    candidates: list[dict[str, str]] = []
    malformed: list[dict[str, str]] = []
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"pmcid", "has_xml", "xml_url"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise SystemExit(f"ERROR: {path} is missing columns: {sorted(missing)}")
        for row in reader:
            if (row.get("has_xml") or "").strip() != "yes":
                continue
            pmcid = (row.get("pmcid") or "").strip()
            if not pmcid:
                malformed.append({**row, "_reason": "blank pmcid"})
                continue
            try:
                resolved, md5 = split_xml_url(row.get("xml_url", ""))
            except ValueError as exc:
                malformed.append({**row, "_reason": str(exc)})
                continue
            candidates.append({**row, "_resolved_url": resolved, "_expected_md5": md5})
    return candidates, malformed


# ---------------------------------------------------------------------------
# Output layout, and the guard that keeps us out of pubmed/
# ---------------------------------------------------------------------------


def assert_safe_output_dir(output_dir: Path, inventory: Path) -> None:
    """Refuse to write anywhere that could damage validated source data."""
    resolved = output_dir.resolve()
    protected = (REPO_ROOT / "pubmed").resolve()
    if resolved == protected or protected in resolved.parents:
        raise SystemExit(
            f"REFUSING TO RUN: output directory {resolved} is inside {protected}.\n"
            "This tool must never write under pubmed/."
        )
    if resolved == inventory.resolve().parent and output_dir.name != "fulltext":
        # Writing straight into pmc/ risks clobbering the inventory beside it.
        raise SystemExit(
            f"REFUSING TO RUN: output directory {resolved} holds the inventory.\n"
            "Use a subdirectory such as pmc/fulltext."
        )


def md5_of_file(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(DOWNLOAD_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# The manifest: append-only during a run, consolidated at the end
# ---------------------------------------------------------------------------


def load_manifest(path: Path) -> dict[str, dict[str, str]]:
    """Current state per PMCID. Later rows win, so a retry supersedes a failure.

    Rows are appended as the run proceeds (an append can never corrupt earlier
    rows), which means one PMCID may appear more than once after an interrupted
    run. Reading last-wins gives the true state, and consolidate_manifest()
    collapses the history at the end of a clean run.
    """
    if not path.exists():
        return {}
    widen_csv_field_limit()
    state: dict[str, dict[str, str]] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            pmcid = (row.get("pmcid") or "").strip()
            if pmcid:
                state[pmcid] = row
    return state


def consolidate_manifest(path: Path, fields: list[str]) -> None:
    """Rewrite the manifest with one row per PMCID, newest state last.

    Written to a temporary file and renamed atomically, so an interrupted
    consolidation leaves the original intact.
    """
    if not path.exists():
        return
    state = load_manifest(path)
    if not state:
        return
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for pmcid in sorted(state, key=lambda p: int(p[3:]) if p[3:].isdigit() else 0):
            writer.writerow(state[pmcid])
    os.replace(tmp, path)


def open_appending(path: Path, fields: list[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not path.exists() or path.stat().st_size == 0
    handle = path.open("a", encoding="utf-8", newline="")
    writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
    if is_new:
        writer.writeheader()
        handle.flush()
    return handle, writer


# ---------------------------------------------------------------------------
# Downloading one article
# ---------------------------------------------------------------------------


def http_download(url: str, destination: Path, timeout: int) -> tuple[str, int]:
    """Stream one URL to disk, returning (md5 hex, bytes written).

    Writes to the given path (a .part file) and hashes as it goes, so the file
    never has to be re-read to be verified.
    """
    request = urllib.request.Request(url, method="GET")
    request.add_header("User-Agent", "thesis_research_pmc_xml_downloader/1.0")
    digest = hashlib.md5()
    total = 0
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            with destination.open("wb") as handle:
                while True:
                    chunk = response.read(DOWNLOAD_CHUNK)
                    if not chunk:
                        break
                    handle.write(chunk)
                    digest.update(chunk)
                    total += len(chunk)
    except urllib.error.HTTPError as exc:
        if exc.code in TRANSIENT_HTTP:
            raise TransientError(f"HTTP {exc.code}", exc.code) from exc
        raise PermanentError(f"HTTP {exc.code}", exc.code) from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise TransientError(f"network error: {exc}") from exc
    return digest.hexdigest(), total


def download_verified(
    url: str, final_path: Path, expected_md5: str, timeout: int,
    max_attempts: int, sleep: float,
) -> tuple[str, int, int]:
    """Download, verify against expected_md5, then atomically put in place.

    Returns (actual md5, bytes, attempts used). The temporary .part file is only
    renamed after the hash matches, so a partial or corrupted transfer can never
    be mistaken for a finished download.
    """
    part_path = final_path.with_name(final_path.name + ".part")
    final_path.parent.mkdir(parents=True, exist_ok=True)
    last: Exception | None = None

    for attempt in range(1, max_attempts + 1):
        time.sleep(sleep)
        try:
            actual, size = http_download(url, part_path, timeout)
            if actual != expected_md5:
                part_path.unlink(missing_ok=True)
                # Corruption in transit can be transient, so this is retried --
                # but it is never accepted as a success.
                raise TransientError(f"md5 mismatch (expected {expected_md5}, got {actual})")
            os.replace(part_path, final_path)   # atomic
            return actual, size, attempt
        except PermanentError:
            part_path.unlink(missing_ok=True)
            raise
        except TransientError as exc:
            last = exc
            part_path.unlink(missing_ok=True)
            if attempt == max_attempts:
                break
            time.sleep(min(60.0, 2**attempt + random.uniform(0, 0.5)))

    raise TransientError(str(last) if last else "download failed",
                         getattr(last, "http_status", None))


def manifest_row(
    record: dict[str, str], *, status: str, actual_md5: str = "", size: int | str = "",
    http_status: int | str = "", attempts: int | str = "", error: str = "",
) -> dict[str, str]:
    """Build one manifest row, copying inventory metadata through unchanged."""
    return {
        "pmcid": record["pmcid"],
        "pmid": record.get("pmid", ""),
        "doi": record.get("doi_from_pmc", ""),
        "title": record.get("title_from_pmc", ""),
        "license_code": record.get("license_code", ""),
        "is_retracted": record.get("is_retracted", ""),
        "is_manuscript": record.get("is_manuscript", ""),
        "version": record.get("version", ""),
        "source_xml_url": record.get("xml_url", ""),
        "resolved_url": record.get("_resolved_url", ""),
        "expected_md5": record.get("_expected_md5", ""),
        "actual_md5": actual_md5,
        "md5_verified": "yes" if actual_md5 and actual_md5 == record.get("_expected_md5") else "no",
        "filename": f"{record['pmcid']}.xml",
        "bytes": str(size),
        "status": status,
        "http_status": str(http_status),
        "attempts": str(attempts),
        "downloaded_at_utc": utc_now(),
        "error": error,
    }


# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------


def run(args: argparse.Namespace) -> int:
    inventory: Path = args.inventory
    output_dir: Path = args.output_dir
    assert_safe_output_dir(output_dir, inventory)

    xml_dir = output_dir / "xml"
    manifest_path = output_dir / "manifest.csv"
    failures_path = output_dir / "failures.csv"

    candidates, malformed = read_candidates(inventory)
    print(f"Inventory : {inventory}")
    print(f"Candidates: {len(candidates):,} records with has_xml=yes")
    if malformed:
        print(f"WARNING   : {len(malformed)} rows claim XML but have an unusable URL "
              f"(recorded as failures, not dropped)")
    print(f"XML output: {xml_dir}")
    print(f"Manifest  : {manifest_path}")
    print(f"Failures  : {failures_path}")
    print("This tool downloads XML only -- never PDFs, text, figures or media.")

    if args.limit:
        candidates = candidates[: args.limit]
        print(f"\nTEST MODE: limited to the first {len(candidates):,} candidates")

    if args.list_only:
        print("\n--list-only: no network access will be made. First 5 resolved URLs:")
        for record in candidates[:5]:
            print(f"  {record['pmcid']:<14} {record['_resolved_url']}")
            print(f"  {'':<14} expect md5 {record['_expected_md5']}")
        return 0

    previous = load_manifest(manifest_path)
    if previous:
        done = sum(1 for row in previous.values() if row.get("status") in DONE_STATUSES)
        print(f"\nResuming: manifest has {len(previous):,} records, {done:,} already complete")

    manifest_handle, manifest_writer = open_appending(manifest_path, MANIFEST_FIELDS)
    failure_handle, failure_writer = open_appending(failures_path, FAILURE_FIELDS)

    for row in malformed:
        failure_writer.writerow({
            "pmcid": (row.get("pmcid") or "").strip(), "pmid": row.get("pmid", ""),
            "url": row.get("xml_url", ""), "http_status": "", "reason": row.get("_reason", ""),
            "attempts": "0", "timestamp_utc": utc_now(),
        })
    failure_handle.flush()

    counts = {"downloaded": 0, "skipped": 0, "failed": 0, "repaired": 0}
    started = time.monotonic()
    interrupted = False

    try:
        for index, record in enumerate(candidates, start=1):
            pmcid = record["pmcid"]
            target = xml_dir / f"{pmcid}.xml"
            expected = record["_expected_md5"]
            prior = previous.get(pmcid, {})
            prior_attempts = int(prior.get("attempts") or 0) if prior.get("attempts", "").isdigit() else 0

            # An existing file is only trusted if its bytes hash correctly.
            if target.exists():
                actual = md5_of_file(target)
                if actual == expected:
                    if prior.get("status") not in DONE_STATUSES:
                        manifest_writer.writerow(manifest_row(
                            record, status="verified_existing", actual_md5=actual,
                            size=target.stat().st_size, attempts=prior_attempts))
                    counts["skipped"] += 1
                    continue
                # Wrong hash: not a completed download. Re-fetch it.
                counts["repaired"] += 1

            try:
                actual, size, attempts = download_verified(
                    record["_resolved_url"], target, expected,
                    args.timeout, args.max_attempts, args.sleep)
                manifest_writer.writerow(manifest_row(
                    record, status="ok", actual_md5=actual, size=size,
                    http_status=200, attempts=prior_attempts + attempts))
                counts["downloaded"] += 1
            except (PermanentError, TransientError) as exc:
                attempts = prior_attempts + (1 if isinstance(exc, PermanentError) else args.max_attempts)
                status = "failed_permanent" if isinstance(exc, PermanentError) else "failed"
                http_status = getattr(exc, "http_status", None) or ""
                manifest_writer.writerow(manifest_row(
                    record, status=status, http_status=http_status,
                    attempts=attempts, error=str(exc)))
                failure_writer.writerow({
                    "pmcid": pmcid, "pmid": record.get("pmid", ""),
                    "url": record["_resolved_url"], "http_status": http_status,
                    "reason": str(exc), "attempts": attempts, "timestamp_utc": utc_now(),
                })
                counts["failed"] += 1
                print(f"  ! {pmcid} FAILED: {exc}", flush=True)

            if index % CHECKPOINT_EVERY == 0 or index == len(candidates):
                manifest_handle.flush()
                failure_handle.flush()
                elapsed = time.monotonic() - started
                rate = index / elapsed if elapsed else 0
                remaining = (len(candidates) - index) / rate if rate else 0
                print(f"  {index:,}/{len(candidates):,}  "
                      f"downloaded {counts['downloaded']:,}  "
                      f"skipped {counts['skipped']:,}  failed {counts['failed']:,}  "
                      f"({rate:.1f}/s, ~{remaining/60:.0f} min left)", flush=True)
    except KeyboardInterrupt:
        interrupted = True
        print("\nInterrupted. Progress is saved -- run the same command again to resume.")
    finally:
        manifest_handle.close()
        failure_handle.close()
        consolidate_manifest(manifest_path, MANIFEST_FIELDS)

    print("\n" + "=" * 66)
    print("PMC XML acquisition summary")
    print("=" * 66)
    print(f"Candidates this run   : {len(candidates):,}")
    print(f"Downloaded + verified : {counts['downloaded']:,}")
    print(f"Skipped (already ok)  : {counts['skipped']:,}")
    print(f"Re-fetched (bad hash) : {counts['repaired']:,}")
    print(f"Failed                : {counts['failed']:,}")
    if malformed:
        print(f"Malformed inventory rows: {len(malformed):,} (see failures.csv)")
    print(f"Elapsed               : {time.monotonic() - started:.1f}s")
    if counts["failed"] or malformed:
        print(f"\nFailures are listed in {failures_path}.")
        print("Run the same command again to retry them.")
    if interrupted:
        return 130
    return 1 if counts["failed"] else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY,
                        help="validated PMC inventory CSV (read-only)")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                        help="where xml/, manifest.csv and failures.csv go")
    parser.add_argument("--limit", type=int,
                        help="only process the first N candidates (test runs)")
    parser.add_argument("--sleep", type=float, default=DEFAULT_SLEEP,
                        help=f"delay between requests, seconds (default {DEFAULT_SLEEP})")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT,
                        help=f"per-request timeout, seconds (default {DEFAULT_TIMEOUT})")
    parser.add_argument("--max-attempts", type=int, default=DEFAULT_MAX_ATTEMPTS,
                        help=f"attempts per article on transient errors (default {DEFAULT_MAX_ATTEMPTS})")
    parser.add_argument("--list-only", action="store_true",
                        help="show what would be downloaded and exit; makes no network requests")
    return parser


if __name__ == "__main__":
    sys.exit(run(build_parser().parse_args()))
