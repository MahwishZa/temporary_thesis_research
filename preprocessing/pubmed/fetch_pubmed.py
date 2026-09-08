#!/usr/bin/env python3
"""PubMed acquisition pipeline for the thesis literature search.

Reads the approved queries and the approved publication-date window straight out
of ``search_queries.txt``, runs each approved query through NCBI E-utilities
(ESearch to enumerate PMIDs, EFetch to pull the PubMed records), parses the
returned XML, deduplicates by PMID while keeping every query ID that retrieved a
record, and writes ``pubmed_results.csv`` / ``pubmed_results.json`` plus a
per-query ``search_log.csv``.

``search_queries.txt`` is read-only here: the file defines the retrieval scope
and no additional topic filtering is applied to the records that come back.

Environment:
    NCBI_API_KEY  optional; raises the rate limit from 3/s to 10/s.
    NCBI_EMAIL    optional; NCBI asks for a contact address on heavy usage.
    PUBMED_CACHE_DIR
                  optional; where to keep the resumable fetch cache
                  (default: a directory under the system temp dir, outside the
                  repository).

Usage:
    python3 fetch_pubmed.py --list-queries      # audit the approved scope
    python3 fetch_pubmed.py --test              # one query, capped, scratch output
    python3 fetch_pubmed.py                     # full run into the repo files
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import shutil
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator

BASE_DIR = Path(__file__).resolve().parent
QUERIES_PATH = BASE_DIR / "search_queries.txt"

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
ESEARCH_URL = f"{EUTILS}/esearch.fcgi"
EFETCH_URL = f"{EUTILS}/efetch.fcgi"

TOOL_NAME = "thesis_research_pubmed_acquisition"
API_KEY = os.environ.get("NCBI_API_KEY", "").strip()
CONTACT_EMAIL = os.environ.get("NCBI_EMAIL", "").strip()

# NCBI allows 3 requests/second without an API key and 10/second with one.
# Stay comfortably under either ceiling.
MIN_REQUEST_INTERVAL = 0.11 if API_KEY else 0.35

# ESearch will not return more than 10,000 UIDs for one search, so any query
# above this threshold is retrieved by splitting its publication-date window.
ESEARCH_ID_CAP = 9_998
EFETCH_BATCH_SIZE = 200
MAX_RETRIES = 5
REQUEST_TIMEOUT = 180

# Blocks of search_queries.txt that are complete, self-contained searches: each
# one already pairs the disease axis (Alzheimer disease / dementia) with a
# reasoning or diagnosis axis. The bare building blocks in sections 1-3 and the
# filter snippets in section 12 are deliberately not run on their own -- the file
# states they are meant to be combined with AND, and running e.g. Q3.1
# ('"Diagnosis"[Mesh] OR "diagnosis"[Subheading] ...') alone would return
# millions of records with no disease term at all.
APPROVED_QUERY_IDS = [
    "Q0.1", "Q0.2", "Q0.3",
    "Q1.6",
    "Q4.1", "Q4.2", "Q4.3", "Q4.4", "Q4.5", "Q4.6",
    "Q5.1", "Q5.2", "Q5.3", "Q5.4", "Q5.5", "Q5.6", "Q5.7",
    "Q6.1", "Q6.2", "Q6.3",
    "Q7.1", "Q7.2", "Q7.3", "Q7.4",
    "Q8.1", "Q8.2", "Q8.3", "Q8.4", "Q8.5", "Q8.6",
    "Q9.1", "Q9.2", "Q9.3",
    "Q10.1", "Q10.2",
    "Q11.1", "Q11.2",
    "Q13.1", "Q13.2", "Q13.3", "Q13.4", "Q13.5",
]

TEST_QUERY_ID = "Q0.2"
TEST_PMID_LIMIT = 25

CSV_FIELDS = [
    "pmid",
    "title",
    "abstract",
    "publication_date",
    "publication_year",
    "journal",
    "authors",
    "doi",
    "pmcid",
    "mesh_terms",
    "publication_types",
    "query_ids",
    "record_status",
]

LIST_FIELDS = ("authors", "mesh_terms", "publication_types", "query_ids")
LIST_SEPARATOR = "; "

LOG_FIELDS = [
    "query_id",
    "query_label",
    "query_term",
    "search_timestamp_utc",
    "result_count",
    "retrieved_pmid_count",
    "fetched_record_count",
    "new_unique_pmids",
    "errors_warnings",
]

DATE_FILTER_RE = re.compile(
    r'\(\s*"(\d{4}/\d{2}/\d{2})"\s*\[Date\s*-\s*Publication\]\s*:\s*'
    r'"(\d{4}/\d{2}/\d{2})"\s*\[Date\s*-\s*Publication\]\s*\)',
    re.IGNORECASE,
)
QUERY_HEADER_RE = re.compile(r"^#\s+(Q\d+\.\d+)\b\s*(.*)$")
SECTION_RULE_RE = re.compile(r"^#\s*=+\s*$")

MONTHS = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}


class EutilsError(RuntimeError):
    """An error reported by, or while talking to, NCBI E-utilities."""


# --------------------------------------------------------------------------
# HTTP plumbing: rate limiting, retries, error surfacing
# --------------------------------------------------------------------------


class RateLimiter:
    def __init__(self, min_interval: float) -> None:
        self.min_interval = min_interval
        self._last = 0.0

    def wait(self) -> None:
        gap = self.min_interval - (time.monotonic() - self._last)
        if gap > 0:
            time.sleep(gap)
        self._last = time.monotonic()


LIMITER = RateLimiter(MIN_REQUEST_INTERVAL)


def common_params() -> dict[str, str]:
    params = {"tool": TOOL_NAME}
    if CONTACT_EMAIL:
        params["email"] = CONTACT_EMAIL
    if API_KEY:
        params["api_key"] = API_KEY
    return params


def backoff_seconds(attempt: int) -> float:
    return min(60.0, (2**attempt) + random.uniform(0, 0.5))


def eutils_request(url: str, params: dict[str, Any]) -> bytes:
    """POST to an E-utility, retrying transient failures with backoff."""
    payload = {k: v for k, v in params.items() if v is not None and v != ""}
    body = urllib.parse.urlencode(payload, doseq=True).encode("utf-8")
    last_error: Exception | None = None

    for attempt in range(1, MAX_RETRIES + 1):
        request = urllib.request.Request(url, data=body, method="POST")
        request.add_header("User-Agent", f"{TOOL_NAME}/1.0")
        request.add_header("Content-Type", "application/x-www-form-urlencoded")
        LIMITER.wait()
        try:
            with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT) as response:
                return response.read()
        except urllib.error.HTTPError as exc:
            detail = b""
            try:
                detail = exc.read()[:500]
            except Exception:  # pragma: no cover - best-effort diagnostics
                pass
            last_error = EutilsError(f"HTTP {exc.code} from NCBI: {detail!r}")
            if exc.code not in {408, 429, 500, 502, 503, 504} or attempt == MAX_RETRIES:
                raise last_error from exc
            delay = backoff_seconds(attempt)
            retry_after = exc.headers.get("Retry-After") if exc.headers else None
            if retry_after:
                try:
                    delay = max(delay, float(retry_after))
                except ValueError:
                    pass
            time.sleep(delay)
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_error = EutilsError(f"Network error calling NCBI: {exc}")
            if attempt == MAX_RETRIES:
                raise last_error from exc
            time.sleep(backoff_seconds(attempt))

    raise last_error or EutilsError("E-utilities request failed")


# --------------------------------------------------------------------------
# search_queries.txt parsing (read-only)
# --------------------------------------------------------------------------


def parse_date_ymd(value: str) -> date:
    return datetime.strptime(value, "%Y/%m/%d").date()


def format_date_ymd(value: date) -> str:
    return value.strftime("%Y/%m/%d")


def date_filter_term(start: date, end: date) -> str:
    return (
        f'("{format_date_ymd(start)}"[Date - Publication] : '
        f'"{format_date_ymd(end)}"[Date - Publication])'
    )


def strip_date_filter(lines: list[str]) -> list[str]:
    """Drop the approved date filter (and its dangling AND) from a query block.

    The window is re-applied by :func:`scoped_term`, which lets the same query be
    re-run over narrower sub-windows when a search exceeds ESearch's 10,000-UID
    ceiling. The window itself always comes from the file and is never widened.
    """
    kept: list[str] = []
    for line in lines:
        without = DATE_FILTER_RE.sub("", line).strip()
        if not without:
            # The whole line was the date filter; drop a bare AND in front of it.
            if kept and kept[-1].strip().upper() == "AND":
                kept.pop()
            continue
        kept.append(without)
    return kept


def normalize_term(lines: list[str]) -> str:
    term = re.sub(r"\s+", " ", " ".join(lines)).strip()
    term = re.sub(r"^(?:AND|OR)\s+", "", term, flags=re.IGNORECASE)
    term = re.sub(r"\s+(?:AND|OR)$", "", term, flags=re.IGNORECASE)
    return re.sub(r"\s+AND\s+AND\s+", " AND ", term, flags=re.IGNORECASE).strip()


def load_queries_file(path: Path) -> tuple[date, date, list[dict[str, str]]]:
    """Return the approved (start, end) window and every query block in the file."""
    text = path.read_text(encoding="utf-8")

    windows = {(m[0], m[1]) for m in DATE_FILTER_RE.findall(text)}
    if not windows:
        raise ValueError(f"No [Date - Publication] window found in {path}")
    if len(windows) > 1:
        raise ValueError(
            f"{path} declares conflicting publication-date windows: {sorted(windows)}"
        )
    raw_start, raw_end = windows.pop()
    start, end = parse_date_ymd(raw_start), parse_date_ymd(raw_end)
    if start > end:
        raise ValueError(f"Publication-date window in {path} is inverted")

    queries: list[dict[str, str]] = []
    current_id: str | None = None
    current_label = ""
    buffer: list[str] = []

    def flush() -> None:
        nonlocal current_id, current_label, buffer
        if current_id is not None:
            content = [ln.rstrip() for ln in buffer
                       if ln.strip() and not ln.lstrip().startswith("#")]
            as_written = normalize_term(content)
            base = normalize_term(strip_date_filter(content))
            if as_written:
                queries.append(
                    {
                        "id": current_id,
                        "label": current_label,
                        "term_as_written": as_written,
                        "base_term": base,
                    }
                )
        current_id, current_label, buffer = None, "", []

    for raw_line in text.splitlines():
        header = QUERY_HEADER_RE.match(raw_line)
        if header:
            flush()
            current_id, current_label = header.group(1), header.group(2).strip(" -")
            continue
        if SECTION_RULE_RE.match(raw_line):
            flush()
            continue
        if current_id is not None:
            buffer.append(raw_line)
    flush()

    return start, end, queries


def select_queries(
    all_queries: list[dict[str, str]], wanted_ids: list[str]
) -> list[dict[str, str]]:
    by_id = {q["id"]: q for q in all_queries}
    missing = [qid for qid in wanted_ids if qid not in by_id]
    if missing:
        raise ValueError(
            f"search_queries.txt does not contain: {', '.join(missing)}"
        )
    selected = [by_id[qid] for qid in wanted_ids]
    empty = [q["id"] for q in selected if not q["base_term"]]
    if empty:
        raise ValueError(
            "These blocks have no searchable terms once the date filter is "
            f"removed: {', '.join(empty)}"
        )
    return selected


def scoped_term(base_term: str, start: date, end: date) -> str:
    return f"({base_term}) AND {date_filter_term(start, end)}"


# --------------------------------------------------------------------------
# ESearch: enumerate every PMID for a query inside the approved window
# --------------------------------------------------------------------------


def esearch(term: str, retmax: int) -> tuple[int, list[str], list[str]]:
    """Return (total count, up to retmax PMIDs, NCBI warnings)."""
    raw = eutils_request(
        ESEARCH_URL,
        {
            **common_params(),
            "db": "pubmed",
            "term": term,
            "retmode": "json",
            "retmax": retmax,
            "retstart": 0,
        },
    )
    try:
        payload = json.loads(raw.decode("utf-8", errors="replace"))
    except json.JSONDecodeError as exc:
        raise EutilsError(f"ESearch returned non-JSON: {raw[:300]!r}") from exc

    result = payload.get("esearchresult") or {}
    if "ERROR" in result:
        raise EutilsError(f"ESearch error: {result['ERROR']}")

    notes: list[str] = []
    for bucket in ("errorlist", "warninglist"):
        for key, value in (result.get(bucket) or {}).items():
            if not value:
                continue
            items = value if isinstance(value, list) else [value]
            notes.append(f"{key}: {', '.join(str(v) for v in items)}")

    count = int(result.get("count") or 0)
    ids = [str(uid) for uid in (result.get("idlist") or [])]
    return count, ids, notes


def search_window(
    base_term: str, start: date, end: date, warnings: list[str]
) -> tuple[int, list[str]]:
    """All PMIDs for base_term in [start, end], splitting the window past 10k hits."""
    term = scoped_term(base_term, start, end)
    window_label = f"{format_date_ymd(start)}:{format_date_ymd(end)}"

    count, _ids, notes = esearch(term, retmax=0)
    warnings.extend(f"{note} [{window_label}]" for note in notes)
    if count == 0:
        return 0, []

    if count <= ESEARCH_ID_CAP:
        _count, ids, more_notes = esearch(term, retmax=count)
        warnings.extend(f"{note} [{window_label}]" for note in more_notes)
        if len(ids) != count:
            warnings.append(
                f"ESearch returned {len(ids)} of {count} PMIDs for {window_label}"
            )
        return count, ids

    if start >= end:
        # A single day above the cap: retrieve what ESearch will give us and say so.
        _count, ids, more_notes = esearch(term, retmax=ESEARCH_ID_CAP)
        warnings.extend(f"{note} [{window_label}]" for note in more_notes)
        warnings.append(
            f"ESearch 10,000-UID cap reached on the single day {window_label}; "
            f"kept {len(ids)} of {count}"
        )
        return count, ids

    midpoint = start + timedelta(days=(end - start).days // 2)
    _left_count, left = search_window(base_term, start, midpoint, warnings)
    _right_count, right = search_window(
        base_term, midpoint + timedelta(days=1), end, warnings
    )

    merged = dedupe_preserving_order(left + right)
    if len(merged) != count:
        warnings.append(
            f"date-window split for {window_label} collected {len(merged)} PMIDs "
            f"for a reported count of {count}"
        )
    return count, merged


def dedupe_preserving_order(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            out.append(value)
    return out


# --------------------------------------------------------------------------
# EFetch + PubMed XML parsing
# --------------------------------------------------------------------------


def element_text(element: ET.Element | None) -> str:
    if element is None:
        return ""
    return re.sub(r"\s+", " ", "".join(element.itertext())).strip()


def first_text(parent: ET.Element, paths: Iterable[str]) -> str:
    for path in paths:
        text = element_text(parent.find(path))
        if text:
            return text
    return ""


def month_number(value: str) -> int | None:
    value = value.strip()
    if not value:
        return None
    if value.isdigit():
        number = int(value)
        return number if 1 <= number <= 12 else None
    return MONTHS.get(value[:3].lower())


def compose_date(year: str, month: str, day: str) -> str:
    """Build the most precise YYYY[-MM[-DD]] string the parts support."""
    if not year or not year.isdigit():
        return ""
    parts = [f"{int(year):04d}"]
    month_num = month_number(month)
    if month_num:
        parts.append(f"{month_num:02d}")
        if day.strip().isdigit() and 1 <= int(day) <= 31:
            parts.append(f"{int(day):02d}")
    return "-".join(parts)


def parse_medline_date(text: str) -> str:
    """Best-effort normalisation of free-text MedlineDate values.

    Handles the shapes PubMed actually emits: '2021 Nov-Dec', '2022 Winter',
    '2021-2022', '2023 Jan 15'.
    """
    if not text:
        return ""
    year_match = re.search(r"\b(\d{4})\b", text)
    if not year_match:
        return ""
    year = year_match.group(1)
    tail = text[year_match.end():]
    month_match = re.search(r"[A-Za-z]{3,}", tail)
    month = month_match.group(0) if month_match else ""
    day = ""
    if month_match:
        day_match = re.search(r"\b(\d{1,2})\b", tail[month_match.end():])
        if day_match:
            day = day_match.group(1)
    return compose_date(year, month, day)


def parse_date_element(element: ET.Element | None) -> str:
    if element is None:
        return ""
    medline = element_text(element.find("MedlineDate"))
    composed = compose_date(
        element_text(element.find("Year")),
        element_text(element.find("Month")),
        element_text(element.find("Day")),
    )
    return composed or parse_medline_date(medline)


def parse_publication_date(article: ET.Element, book: bool) -> str:
    """Publication date, preferring the journal/book issue date.

    Falls back to the electronic ArticleDate and finally to the Entrez history,
    so a record is never dropped just because one of them is absent.
    """
    if book:
        candidates = [
            article.find("./BookDocument/Book/PubDate"),
            article.find("./BookDocument/Book/BeginningDate"),
        ]
    else:
        candidates = [
            article.find(
                "./MedlineCitation/Article/Journal/JournalIssue/PubDate"
            ),
            article.find("./MedlineCitation/Article/ArticleDate"),
        ]
    for status in ("pubmed", "entrez", "medline"):
        candidates.append(
            article.find(f'./PubmedData/History/PubMedPubDate[@PubStatus="{status}"]')
        )
        candidates.append(
            article.find(
                f'./PubmedBookData/History/PubMedPubDate[@PubStatus="{status}"]'
            )
        )

    for candidate in candidates:
        parsed = parse_date_element(candidate)
        if parsed:
            return parsed
    return ""


def parse_authors(article: ET.Element, book: bool) -> list[str]:
    root = "./BookDocument/AuthorList/Author" if book else (
        "./MedlineCitation/Article/AuthorList/Author"
    )
    authors: list[str] = []
    for author in article.findall(root):
        collective = element_text(author.find("CollectiveName"))
        if collective:
            authors.append(collective)
            continue
        last = element_text(author.find("LastName"))
        given = element_text(author.find("ForeName")) or element_text(
            author.find("Initials")
        )
        if last and given:
            authors.append(f"{last}, {given}")
        elif last:
            authors.append(last)
        elif given:
            authors.append(given)
    return authors


def parse_abstract(article: ET.Element, book: bool) -> str:
    root = "./BookDocument/Abstract" if book else "./MedlineCitation/Article/Abstract"
    sections: list[str] = []
    abstract = article.find(root)
    if abstract is not None:
        for node in abstract.findall("AbstractText"):
            body = element_text(node)
            if not body:
                continue
            label = (node.get("Label") or "").strip()
            sections.append(f"{label}: {body}" if label else body)
    return "\n".join(sections)


def parse_identifiers(article: ET.Element, book: bool) -> tuple[str, str, str]:
    pmid = element_text(
        article.find("./BookDocument/PMID" if book else "./MedlineCitation/PMID")
    )
    doi = ""
    pmcid = ""

    for eloc in article.findall("./MedlineCitation/Article/ELocationID"):
        if (eloc.get("EIdType") or "").lower() == "doi" and element_text(eloc):
            doi = element_text(eloc)

    id_lists = ["./PubmedData/ArticleIdList/ArticleId",
                "./PubmedBookData/ArticleIdList/ArticleId",
                "./BookDocument/ArticleIdList/ArticleId"]
    for path in id_lists:
        for article_id in article.findall(path):
            id_type = (article_id.get("IdType") or "").lower()
            value = element_text(article_id)
            if not value:
                continue
            if id_type == "doi":
                doi = value
            elif id_type in {"pmc", "pmcid"}:
                pmcid = value if value.upper().startswith("PMC") else f"PMC{value}"
            elif id_type == "pubmed" and not pmid:
                pmid = value
    return pmid, doi, pmcid


def parse_mesh_terms(article: ET.Element) -> list[str]:
    """MeSH headings as 'Descriptor (Qualifier)', with '*' marking major topics."""
    terms: list[str] = []
    for path in ("./MedlineCitation/MeshHeadingList/MeshHeading",
                 "./BookDocument/MeshHeadingList/MeshHeading"):
        for heading in article.findall(path):
            descriptor = heading.find("DescriptorName")
            name = element_text(descriptor)
            if not name:
                continue
            if descriptor is not None and descriptor.get("MajorTopicYN") == "Y":
                name += "*"
            qualifiers = []
            for qualifier in heading.findall("QualifierName"):
                qualifier_name = element_text(qualifier)
                if qualifier_name:
                    if qualifier.get("MajorTopicYN") == "Y":
                        qualifier_name += "*"
                    qualifiers.append(qualifier_name)
            terms.append(f"{name} ({', '.join(qualifiers)})" if qualifiers else name)
    return terms


def parse_publication_types(article: ET.Element) -> list[str]:
    types: list[str] = []
    for path in ("./MedlineCitation/Article/PublicationTypeList/PublicationType",
                 "./BookDocument/PublicationTypeList/PublicationType"):
        for node in article.findall(path):
            value = element_text(node)
            if value and value not in types:
                types.append(value)
    return types


def parse_article(article: ET.Element, book: bool) -> dict[str, Any] | None:
    pmid, doi, pmcid = parse_identifiers(article, book)
    if not pmid:
        return None

    if book:
        title = first_text(
            article,
            ["./BookDocument/ArticleTitle", "./BookDocument/Book/BookTitle"],
        )
        journal = first_text(article, ["./BookDocument/Book/BookTitle"])
    else:
        title = first_text(
            article,
            [
                "./MedlineCitation/Article/ArticleTitle",
                "./MedlineCitation/Article/VernacularTitle",
            ],
        )
        journal = first_text(
            article,
            [
                "./MedlineCitation/Article/Journal/Title",
                "./MedlineCitation/Article/Journal/ISOAbbreviation",
                "./MedlineCitation/MedlineJournalInfo/MedlineTA",
            ],
        )

    publication_date = parse_publication_date(article, book)
    return {
        "pmid": pmid,
        "title": title,
        "abstract": parse_abstract(article, book),
        "publication_date": publication_date,
        "publication_year": publication_date[:4],
        "journal": journal,
        "authors": parse_authors(article, book),
        "doi": doi,
        "pmcid": pmcid,
        "mesh_terms": parse_mesh_terms(article),
        "publication_types": parse_publication_types(article),
        "record_status": "ok",
    }


def parse_pubmed_xml(xml_bytes: bytes) -> list[dict[str, Any]]:
    """Parse an EFetch PubmedArticleSet, including book/chapter records."""
    root = ET.fromstring(xml_bytes)
    records: list[dict[str, Any]] = []
    for article in root.iter("PubmedArticle"):
        record = parse_article(article, book=False)
        if record:
            records.append(record)
    for article in root.iter("PubmedBookArticle"):
        record = parse_article(article, book=True)
        if record:
            records.append(record)
    return records


def chunked(values: list[str], size: int) -> Iterator[list[str]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def efetch_batch(pmids: list[str]) -> list[dict[str, Any]]:
    raw = eutils_request(
        EFETCH_URL,
        {
            **common_params(),
            "db": "pubmed",
            "id": ",".join(pmids),
            "retmode": "xml",
        },
    )
    try:
        return parse_pubmed_xml(raw)
    except ET.ParseError as exc:
        raise EutilsError(f"Could not parse EFetch XML: {exc}") from exc


def efetch_with_bisect(
    pmids: list[str], errors: list[str]
) -> list[dict[str, Any]]:
    """Fetch a batch, halving it on failure so one bad PMID cannot sink the rest."""
    try:
        return efetch_batch(pmids)
    except Exception as exc:  # noqa: BLE001 - recorded, never swallowed
        if len(pmids) == 1:
            errors.append(f"EFetch failed for PMID {pmids[0]}: {exc}")
            return []
        half = len(pmids) // 2
        return efetch_with_bisect(pmids[:half], errors) + efetch_with_bisect(
            pmids[half:], errors
        )


class RecordCache:
    """Resumable PMID -> record cache, kept outside the repository."""

    def __init__(self, directory: Path | None) -> None:
        self.path = directory / "records.jsonl" if directory else None
        self.records: dict[str, dict[str, Any]] = {}
        if self.path and self.path.exists():
            with self.path.open(encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if record.get("pmid"):
                        self.records[str(record["pmid"])] = record

    def __contains__(self, pmid: str) -> bool:
        return pmid in self.records

    def get(self, pmid: str) -> dict[str, Any] | None:
        return self.records.get(pmid)

    def add(self, records: list[dict[str, Any]]) -> None:
        fresh = [r for r in records if r["pmid"] not in self.records]
        for record in fresh:
            self.records[record["pmid"]] = record
        if self.path and fresh:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                for record in fresh:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def fetch_records(
    pmids: list[str], cache: RecordCache, errors: list[str], progress_prefix: str
) -> list[dict[str, Any]]:
    outstanding = [pmid for pmid in pmids if pmid not in cache]
    total = len(outstanding)
    done = 0
    for batch in chunked(outstanding, EFETCH_BATCH_SIZE):
        cache.add(efetch_with_bisect(batch, errors))
        done += len(batch)
        print(f"    {progress_prefix} EFetch {done}/{total}", flush=True)

    records: list[dict[str, Any]] = []
    for pmid in pmids:
        record = cache.get(pmid)
        if record is None:
            # ESearch found it but EFetch returned nothing (deleted, suppressed,
            # or a persistent fetch error). Keep the PMID rather than lose it.
            records.append(empty_record(pmid))
        else:
            records.append(record)
    return records


def empty_record(pmid: str) -> dict[str, Any]:
    return {
        "pmid": pmid,
        "title": "",
        "abstract": "",
        "publication_date": "",
        "publication_year": "",
        "journal": "",
        "authors": [],
        "doi": "",
        "pmcid": "",
        "mesh_terms": [],
        "publication_types": [],
        "record_status": "metadata_unavailable",
    }


# --------------------------------------------------------------------------
# Deduplication with query provenance
# --------------------------------------------------------------------------


def query_sort_key(query_id: str) -> tuple[int, int]:
    match = re.match(r"Q(\d+)\.(\d+)$", query_id)
    return (int(match.group(1)), int(match.group(2))) if match else (999, 999)


def merge_record(
    store: dict[str, dict[str, Any]], record: dict[str, Any], query_id: str
) -> bool:
    """Merge one fetched record into the store. Returns True if the PMID is new."""
    pmid = record["pmid"]
    existing = store.get(pmid)
    if existing is None:
        store[pmid] = {**record, "query_ids": [query_id]}
        return True

    if query_id not in existing["query_ids"]:
        existing["query_ids"].append(query_id)
    # A later query may return a fuller record (e.g. the first pass hit a fetch
    # error); fill gaps but never overwrite data that is already present.
    for field in (
        "title", "abstract", "publication_date", "publication_year",
        "journal", "doi", "pmcid", "authors", "mesh_terms", "publication_types",
    ):
        if not existing.get(field) and record.get(field):
            existing[field] = record[field]
    if existing.get("record_status") != "ok" and record.get("record_status") == "ok":
        existing["record_status"] = "ok"
    return False


# --------------------------------------------------------------------------
# Output: backup, CSV, JSON, search log
# --------------------------------------------------------------------------


def has_data(path: Path) -> bool:
    """True if an existing output file holds records worth backing up."""
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        if path.suffix == ".json":
            payload = json.loads(path.read_text(encoding="utf-8") or "null")
            if isinstance(payload, list):
                return len(payload) > 0
            if isinstance(payload, dict):
                return len(payload) > 0
            return payload is not None
        with path.open(encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle)
            next(reader, None)  # header
            return next(reader, None) is not None
    except (json.JSONDecodeError, UnicodeDecodeError, csv.Error):
        # Unreadable but non-empty: treat as data so it is preserved, not lost.
        return True


def backup_existing(paths: Iterable[Path], stamp: str) -> list[Path]:
    made: list[Path] = []
    for path in paths:
        if has_data(path):
            backup = path.with_name(f"{path.stem}_backup_{stamp}{path.suffix}")
            shutil.copy2(path, backup)
            made.append(backup)
    return made


def csv_value(record: dict[str, Any], field: str) -> str:
    value = record.get(field)
    if field in LIST_FIELDS:
        return LIST_SEPARATOR.join(value or [])
    return "" if value is None else str(value)


def write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(CSV_FIELDS)
        for record in records:
            writer.writerow([csv_value(record, field) for field in CSV_FIELDS])


def write_json(path: Path, records: list[dict[str, Any]]) -> None:
    """One record per line inside a real JSON array: valid, greppable, compact."""
    with path.open("w", encoding="utf-8") as handle:
        handle.write("[\n")
        for index, record in enumerate(records):
            ordered = {field: record.get(field) for field in CSV_FIELDS}
            for field in LIST_FIELDS:
                ordered[field] = list(ordered.get(field) or [])
            handle.write("  " + json.dumps(ordered, ensure_ascii=False))
            handle.write(",\n" if index < len(records) - 1 else "\n")
        handle.write("]\n")


def write_log(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=LOG_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


# --------------------------------------------------------------------------
# Verification
# --------------------------------------------------------------------------


def date_bounds(value: str) -> tuple[date, date] | None:
    """Earliest and latest day a partial YYYY[-MM[-DD]] string could denote."""
    if not value:
        return None
    parts = value.split("-")
    try:
        year = int(parts[0])
        if len(parts) == 1:
            return date(year, 1, 1), date(year, 12, 31)
        month = int(parts[1])
        if len(parts) == 2:
            last = date(year + month // 12, month % 12 + 1, 1) - timedelta(days=1)
            return date(year, month, 1), last
        day = int(parts[2])
        return date(year, month, day), date(year, month, day)
    except ValueError:
        return None


def verify_outputs(
    csv_path: Path,
    json_path: Path,
    records: list[dict[str, Any]],
    per_query_pmids: dict[str, list[str]],
    window: tuple[date, date],
) -> tuple[list[str], dict[str, Any]]:
    """Re-read the written files and check every guarantee the pipeline claims."""
    problems: list[str] = []

    with csv_path.open(encoding="utf-8", newline="") as handle:
        csv_rows = list(csv.DictReader(handle))
    json_rows = json.loads(json_path.read_text(encoding="utf-8"))

    csv_pmids = [row["pmid"] for row in csv_rows]
    json_pmids = [str(row.get("pmid") or "") for row in json_rows]

    # 1. PMIDs are unique.
    if len(csv_pmids) != len(set(csv_pmids)):
        problems.append("CSV contains duplicate PMIDs")
    if len(json_pmids) != len(set(json_pmids)):
        problems.append("JSON contains duplicate PMIDs")

    # 2. CSV and JSON hold the same records, field for field.
    if csv_pmids != json_pmids:
        only_csv = set(csv_pmids) - set(json_pmids)
        only_json = set(json_pmids) - set(csv_pmids)
        problems.append(
            f"CSV and JSON differ: {len(only_csv)} CSV-only, "
            f"{len(only_json)} JSON-only, order match={csv_pmids == json_pmids}"
        )
    else:
        mismatched = 0
        for csv_row, json_row in zip(csv_rows, json_rows):
            for field in CSV_FIELDS:
                json_value = json_row.get(field)
                if field in LIST_FIELDS:
                    json_value = LIST_SEPARATOR.join(json_value or [])
                if (csv_row.get(field) or "") != ("" if json_value is None
                                                  else str(json_value)):
                    mismatched += 1
                    break
        if mismatched:
            problems.append(f"{mismatched} records differ in content between CSV and JSON")

    # 3. Query provenance survives deduplication, in both directions.
    provenance = {row["pmid"]: set(
        p for p in (row.get("query_ids") or "").split(LIST_SEPARATOR) if p
    ) for row in csv_rows}
    for query_id, pmids in per_query_pmids.items():
        expected = set(pmids)
        actual = {pmid for pmid, ids in provenance.items() if query_id in ids}
        if expected != actual:
            problems.append(
                f"{query_id}: provenance mismatch "
                f"({len(expected - actual)} retrieved PMIDs not attributed, "
                f"{len(actual - expected)} attributed but not retrieved)"
            )
    unattributed = [pmid for pmid, ids in provenance.items() if not ids]
    if unattributed:
        problems.append(f"{len(unattributed)} records carry no query ID")

    # 4. Nothing was silently lost between search and save.
    searched = {pmid for pmids in per_query_pmids.values() for pmid in pmids}
    saved = set(csv_pmids)
    if searched - saved:
        problems.append(f"{len(searched - saved)} searched PMIDs are missing from the output")
    if saved - searched:
        problems.append(f"{len(saved - searched)} saved PMIDs were never returned by a search")

    # 5. Publication dates parse and sit inside the approved window.
    start, end = window
    undated = [r["pmid"] for r in records if not r.get("publication_date")]
    out_of_range: list[str] = []
    unparseable: list[str] = []
    for record in records:
        raw = record.get("publication_date") or ""
        if not raw:
            continue
        bounds = date_bounds(raw)
        if bounds is None:
            unparseable.append(record["pmid"])
        elif bounds[1] < start or bounds[0] > end:
            out_of_range.append(record["pmid"])
    if unparseable:
        problems.append(
            f"{len(unparseable)} records have an unparseable publication date "
            f"(e.g. {', '.join(unparseable[:5])})"
        )

    stats = {
        "csv_rows": len(csv_rows),
        "json_rows": len(json_rows),
        "unique_pmids": len(set(csv_pmids)),
        "undated": undated,
        "out_of_range": out_of_range,
    }
    return problems, stats


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------


def run_queries(
    queries: list[dict[str, str]],
    window: tuple[date, date],
    cache: RecordCache,
    pmid_limit: int | None,
) -> tuple[list[dict[str, Any]], list[dict[str, str]], dict[str, list[str]], int, list[str]]:
    start, end = window
    store: dict[str, dict[str, Any]] = {}
    log_rows: list[dict[str, str]] = []
    per_query_pmids: dict[str, list[str]] = {}
    fatal_errors: list[str] = []
    total_hits = 0

    for position, query in enumerate(queries, start=1):
        query_id = query["id"]
        timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
        notes: list[str] = []
        executed_term = scoped_term(query["base_term"], start, end)
        before = len(store)
        print(f"[{position}/{len(queries)}] {query_id} searching...", flush=True)

        try:
            count, pmids = search_window(query["base_term"], start, end, notes)
            if pmid_limit is not None and len(pmids) > pmid_limit:
                pmids = pmids[:pmid_limit]
                notes.append(f"test mode: capped at the first {pmid_limit} PMIDs")

            fetch_errors: list[str] = []
            records = fetch_records(pmids, cache, fetch_errors, query_id)
            notes.extend(fetch_errors)

            unavailable = [r["pmid"] for r in records
                           if r.get("record_status") != "ok"]
            if unavailable:
                notes.append(
                    f"EFetch returned no metadata for {len(unavailable)} PMIDs, "
                    f"kept as ID-only rows: {', '.join(unavailable[:20])}"
                    + ("..." if len(unavailable) > 20 else "")
                )

            for record in records:
                merge_record(store, record, query_id)

            per_query_pmids[query_id] = pmids
            total_hits += len(pmids)
            log_rows.append(
                {
                    "query_id": query_id,
                    "query_label": query["label"],
                    "query_term": executed_term,
                    "search_timestamp_utc": timestamp,
                    "result_count": str(count),
                    "retrieved_pmid_count": str(len(pmids)),
                    "fetched_record_count": str(len(records) - len(unavailable)),
                    "new_unique_pmids": str(len(store) - before),
                    "errors_warnings": " | ".join(notes),
                }
            )
            print(
                f"    {query_id}: hits={count} retrieved={len(pmids)} "
                f"new={len(store) - before} unique_total={len(store)}",
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001 - logged and reported, not hidden
            message = f"{type(exc).__name__}: {exc}"
            fatal_errors.append(f"{query_id}: {message}")
            log_rows.append(
                {
                    "query_id": query_id,
                    "query_label": query["label"],
                    "query_term": executed_term,
                    "search_timestamp_utc": timestamp,
                    "result_count": "",
                    "retrieved_pmid_count": "0",
                    "fetched_record_count": "0",
                    "new_unique_pmids": "0",
                    "errors_warnings": " | ".join([*notes, message]),
                }
            )
            print(f"    {query_id}: FAILED {message}", file=sys.stderr, flush=True)

    records = [store[pmid] for pmid in sorted(store, key=int)]
    for record in records:
        record["query_ids"] = sorted(record["query_ids"], key=query_sort_key)
    return records, log_rows, per_query_pmids, total_hits, fatal_errors


def default_cache_dir() -> Path:
    override = os.environ.get("PUBMED_CACHE_DIR", "").strip()
    if override:
        return Path(override)
    return Path(tempfile.gettempdir()) / "thesis_research_pubmed_cache"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--test",
        action="store_true",
        help=f"pipeline smoke test: {TEST_QUERY_ID} capped at {TEST_PMID_LIMIT} PMIDs",
    )
    parser.add_argument(
        "--queries",
        help="comma-separated query IDs to run (default: all approved blocks)",
    )
    parser.add_argument("--limit", type=int, help="cap PMIDs retrieved per query")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=BASE_DIR,
        help="where to write the results and log (default: alongside this script)",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=default_cache_dir(),
        help="resumable EFetch cache directory (outside the repository)",
    )
    parser.add_argument(
        "--no-cache", action="store_true", help="do not read or write the fetch cache"
    )
    parser.add_argument(
        "--list-queries",
        action="store_true",
        help="print the approved queries as they will be executed, then exit",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    start, end, all_queries = load_queries_file(QUERIES_PATH)
    print(f"search_queries.txt: {len(all_queries)} query blocks parsed")
    print(f"Approved publication-date window: {start.isoformat()} to {end.isoformat()}")

    if args.test:
        wanted = [TEST_QUERY_ID]
        pmid_limit = args.limit or TEST_PMID_LIMIT
    else:
        wanted = (
            [q.strip() for q in args.queries.split(",") if q.strip()]
            if args.queries
            else list(APPROVED_QUERY_IDS)
        )
        pmid_limit = args.limit

    queries = select_queries(all_queries, wanted)
    print(f"Running {len(queries)} approved queries: {', '.join(q['id'] for q in queries)}")

    if args.list_queries:
        for query in queries:
            print(f"\n--- {query['id']} {query['label']}")
            print(scoped_term(query["base_term"], start, end))
        return 0

    if not API_KEY:
        print(
            "note: NCBI_API_KEY is not set; limiting to ~3 requests/second. "
            "Set NCBI_API_KEY (and NCBI_EMAIL) to run faster.",
            flush=True,
        )

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "pubmed_results.csv"
    json_path = output_dir / "pubmed_results.json"
    log_path = output_dir / "search_log.csv"

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backups = backup_existing([csv_path, json_path, log_path], stamp)
    if backups:
        print("Backed up existing data: " + ", ".join(b.name for b in backups))
    else:
        print("No existing data to back up (output files are absent or empty)")

    cache = RecordCache(None if args.no_cache else args.cache_dir)
    if cache.records:
        print(f"Fetch cache: {len(cache.records)} records available for reuse")

    started = time.monotonic()
    records, log_rows, per_query_pmids, total_hits, errors = run_queries(
        queries, (start, end), cache, pmid_limit
    )

    write_csv(csv_path, records)
    write_json(json_path, records)
    write_log(log_path, log_rows)

    problems, stats = verify_outputs(
        csv_path, json_path, records, per_query_pmids, (start, end)
    )

    parsed_ok = sum(1 for r in records if r.get("record_status") == "ok")
    missing_abstract = sum(1 for r in records if not r.get("abstract"))
    missing_doi = sum(1 for r in records if not r.get("doi"))
    missing_pmcid = sum(1 for r in records if not r.get("pmcid"))

    print("\n" + "=" * 70)
    print("PubMed acquisition summary")
    print("=" * 70)
    print(f"Queries executed:              {len(log_rows)}")
    print(f"Records retrieved (with dups): {total_hits}")
    print(f"Unique records saved:          {stats['unique_pmids']}")
    print(f"Successfully parsed records:   {parsed_ok}")
    print(f"ID-only records (no metadata): {len(records) - parsed_ok}")
    print(f"Missing abstract:              {missing_abstract}")
    print(f"Missing DOI:                   {missing_doi}")
    print(f"Missing PMCID:                 {missing_pmcid}")
    print(f"Missing publication date:      {len(stats['undated'])}")
    print(f"Publication date outside window: {len(stats['out_of_range'])}")
    print(f"Elapsed:                       {time.monotonic() - started:.1f}s")
    print(f"Written: {csv_path.name}, {json_path.name}, {log_path.name}")

    if errors:
        print(f"\nQuery-level errors ({len(errors)}):")
        for error in errors:
            print(f"  {error}")
    else:
        print("\nQuery-level errors: none")

    warning_rows = [row for row in log_rows if row["errors_warnings"]]
    if warning_rows:
        print(f"Queries with warnings ({len(warning_rows)}): see {log_path.name}")

    if problems:
        print(f"\nVERIFICATION FAILED ({len(problems)}):")
        for problem in problems:
            print(f"  {problem}")
        return 1

    print("\nVerification passed: PMIDs unique, CSV and JSON identical, "
          "query provenance complete, no records lost.")
    return 0 if not errors else 1


if __name__ == "__main__":
    sys.exit(main())
