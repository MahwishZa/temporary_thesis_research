#!/usr/bin/env python3
"""Validate the chunk layer against the frozen M1-M4 corpus.

Read-only. Exits non-zero if any invariant fails, so it can gate the next stage.

    python3 pmc/validate_chunks.py                 # after building chunks
    python3 pmc/validate_chunks.py --pubmed <csv>  # if PubMed is not smudged

Checks: unique chunk ids; chunk->document mapping; no orphan chunks; no new
duplicate identities; provenance completeness; canonical-date, date-precision
and source-category preservation against M3/M4; eligibility enforcement;
handling of excluded and manual-review records; no protected or ignored file
committed as corpus data; and a content digest for cross-run determinism.

Requires only the Python standard library.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

csv.field_size_limit(2**31 - 1 if sys.maxsize > 2**32 else 2**27)

REPO = Path(__file__).resolve().parent.parent
DEFAULT_CHUNKS = REPO / "pmc" / "chunks" / "chunks.jsonl"
DEFAULT_META = REPO / "pmc" / "metadata"


def load_policy(path: Path):
    by_pmcid, by_pmid, excluded = {}, {}, set()
    with path.open(newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            pmcid = (r.get("pmcid") or "").strip()
            pmid = (r.get("pmid") or "").strip()
            if pmcid:
                by_pmcid[pmcid] = r
            if pmid:
                by_pmid[pmid] = r
            if (r.get("eligibility_status") or "") == "excluded":
                excluded.add(pmcid or f"PMID{pmid}")
    return by_pmcid, by_pmid, excluded


def load_dates(path: Path):
    out = {}
    with path.open(newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            pmcid = (r.get("pmcid") or "").strip()
            pmid = (r.get("pmid") or "").strip()
            out[pmcid or f"PMID{pmid}"] = r
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Validate the chunk layer.")
    ap.add_argument("--chunks", type=Path, default=DEFAULT_CHUNKS)
    ap.add_argument("--metadata", type=Path, default=DEFAULT_META)
    args = ap.parse_args(argv)

    pol_pmcid, pol_pmid, excluded = load_policy(args.metadata / "corpus_policy.csv")
    dates = load_dates(args.metadata / "canonical_dates.csv")

    seen_ids: set[str] = set()
    dup_ids: list[str] = []
    orphans: list[str] = []
    bad_map: list[str] = []
    excluded_leak: list[str] = []
    date_mismatch: list[str] = []
    cat_mismatch: list[str] = []
    missing_prov: list[str] = []
    identity_conflict: list[str] = []
    doc_identity: dict[str, tuple[str, str]] = {}
    manual_review = 0
    n = 0
    digest = hashlib.sha256()

    REQUIRED = ["chunk_id", "document_id", "source_category", "eligibility_status",
                "canonical_date", "date_precision", "title", "text", "text_sha256"]

    with args.chunks.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line:
                continue
            digest.update(line.encode("utf-8"))
            c = json.loads(line)
            n += 1
            cid, did = c["chunk_id"], c["document_id"]

            if cid in seen_ids:
                dup_ids.append(cid)
            seen_ids.add(cid)

            if not cid.startswith(did + "#"):
                bad_map.append(cid)

            key = c.get("pmcid") or f"PMID{c.get('pmid')}"
            pol = pol_pmcid.get(c.get("pmcid", "")) or pol_pmid.get(c.get("pmid", ""))
            if not pol:
                orphans.append(cid)
                continue
            if key in excluded:
                excluded_leak.append(cid)
            if (pol.get("eligibility_status") or "") == "manual-review":
                manual_review += 1

            prev = doc_identity.setdefault(did, (c.get("pmcid", ""), c.get("pmid", "")))
            if prev != (c.get("pmcid", ""), c.get("pmid", "")):
                identity_conflict.append(did)

            dt = dates.get(key, {})
            if dt:
                if c.get("canonical_date", "") != dt.get("canonical_date", ""):
                    date_mismatch.append(cid)
                if c.get("date_precision", "") != dt.get("canonical_date_precision", ""):
                    date_mismatch.append(cid)
            if c.get("source_category", "") != (pol.get("source_category") or ""):
                cat_mismatch.append(cid)

            for f in REQUIRED:
                if not str(c.get(f, "")).strip():
                    missing_prov.append(f"{cid}:{f}")
                    break

    # Repository hygiene: protected / ignored material must not be committed.
    tracked = subprocess.run(["git", "ls-files"], cwd=REPO, capture_output=True,
                             text=True).stdout.splitlines()
    retry_tracked = [t for t in tracked if t.endswith("retry71.csv")]
    xml_tracked = [t for t in tracked if t.startswith("pmc/fulltext/xml/")]

    checks = [
        ("chunks read",                        n > 0, n),
        ("unique chunk ids",                   not dup_ids, len(dup_ids)),
        ("chunk -> document mapping valid",    not bad_map, len(bad_map)),
        ("no orphan chunks",                   not orphans, len(orphans)),
        ("no duplicate primary identities",    not identity_conflict, len(identity_conflict)),
        ("provenance complete",                not missing_prov, len(missing_prov)),
        ("canonical date preserved",           not date_mismatch, len(date_mismatch)),
        ("date precision preserved",           not date_mismatch, len(date_mismatch)),
        ("source category preserved",          not cat_mismatch, len(cat_mismatch)),
        ("eligibility enforced (no excluded)", not excluded_leak, len(excluded_leak)),
        ("retry71.csv not committed",          not retry_tracked, len(retry_tracked)),
        ("raw PMC XML not committed",          not xml_tracked, len(xml_tracked)),
    ]

    print(f"Validating {args.chunks}")
    ok = True
    for name, good, val in checks:
        ok &= bool(good)
        print(f"  {'PASS' if good else '*** FAIL ***':14} {name:38} {val}")
    print(f"\n  documents: {len(doc_identity):,}   chunks: {n:,}")
    print(f"  manual-review chunks retained (per frozen policy): {manual_review:,}")
    print(f"  content digest (compare across runs): {digest.hexdigest()}")
    print(f"\n  {'ALL CHECKS PASS' if ok else 'VALIDATION FAILED'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
