#!/usr/bin/env python3
"""Integrity gate for the MedCPT embedding index.

Run this after pmc/embed_chunks.py finishes, before anything retrieves over the
result. It answers one question: does pmc/index/ actually correspond, row for
row, to the frozen chunk layer it claims to be built from?

    python3 pmc/verify_index.py

Checks, each reported pass/fail with the evidence:

  1. all three artifacts exist
  2. embeddings.f32 is a whole number of rows of the declared dimension
  3. vector count == manifest rows == the chunks that should have been embedded
  4. dimension is MedCPT's 768 (a production index; a stub index is flagged)
  5. row alignment -- manifest row i names the i-th chunk in sorted chunk_id
     order, which is the order embed_chunks writes and the order retrieval
     assumes; a mismatch here silently returns the wrong document
  6. no NaN or Inf components
  7. vectors are L2-normalised (MedCPT output is; inner product == cosine)
  8. content_digest recomputes to the value in index_meta.json
  9. the manifest carries the provenance the recency-bias study needs

Exits non-zero if any check fails, so it can gate a pipeline.

Standard library only. Reads the vector file in bounded blocks, so a multi-GB
index does not have to fit in memory.
"""

from __future__ import annotations

import argparse
import array
import hashlib
import json
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DEFAULT_INDEX = REPO / "pmc" / "index"
DEFAULT_CHUNKS = REPO / "pmc" / "chunks" / "chunks.jsonl"

MEDCPT_DIM = 768
NORM_TOLERANCE = 1e-3

# Provenance without which the later recency-bias experiments cannot run.
REQUIRED_MANIFEST_FIELDS = [
    "chunk_id", "document_id", "source_category", "canonical_date",
    "date_precision", "split_june_2024", "eligibility_status", "retracted",
]


class Report:
    def __init__(self) -> None:
        self.failures = 0
        self.checks = 0

    def check(self, ok: bool, label: str, detail: str = "") -> bool:
        self.checks += 1
        if not ok:
            self.failures += 1
        mark = "PASS" if ok else "FAIL"
        print(f"  [{mark}] {label}" + (f" -- {detail}" if detail else ""))
        return ok

    def note(self, label: str, detail: str) -> None:
        print(f"  [info] {label} -- {detail}")


def expected_chunk_ids(chunks_path: Path, skip_duplicates: bool) -> list[str]:
    """chunk_ids in the order embed_chunks writes them: sorted, duplicates dropped."""
    ids: list[str] = []
    with chunks_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            chunk = json.loads(line)
            if skip_duplicates and chunk.get("duplicate_of"):
                continue
            ids.append(chunk["chunk_id"])
    ids.sort()
    return ids


def read_manifest(man_path: Path) -> list[dict]:
    rows: list[dict] = []
    with man_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def scan_vectors(vec_path: Path, dim: int, manifest: list[dict], report: Report) -> dict:
    """One streaming pass: digest, finiteness, and L2 norms."""
    row_bytes = dim * 4
    digest = hashlib.sha256()
    non_finite = []
    bad_norm = []
    rows = 0
    min_norm, max_norm = float("inf"), 0.0

    with vec_path.open("rb") as fh:
        while True:
            raw = fh.read(row_bytes)
            if not raw:
                break
            if len(raw) != row_bytes:
                report.check(False, "embeddings.f32 ends on a whole row",
                             f"trailing {len(raw)} bytes after row {rows}")
                break
            buf = array.array("f")
            buf.frombytes(raw)
            if sys.byteorder != "little":                # pragma: no cover
                buf.byteswap()

            total = 0.0
            finite = True
            for value in buf:
                if not math.isfinite(value):
                    finite = False
                    break
                total += value * value
            if not finite:
                if len(non_finite) < 5:
                    non_finite.append(rows)
            else:
                norm = math.sqrt(total)
                min_norm = min(min_norm, norm)
                max_norm = max(max_norm, norm)
                if abs(norm - 1.0) > NORM_TOLERANCE and len(bad_norm) < 5:
                    bad_norm.append((rows, round(norm, 6)))

            if rows < len(manifest):
                digest.update(str(manifest[rows]["chunk_id"]).encode("utf-8"))
                digest.update(raw)
            rows += 1

    return {
        "rows": rows,
        "digest": digest.hexdigest(),
        "non_finite": non_finite,
        "bad_norm": bad_norm,
        "min_norm": min_norm if rows else 0.0,
        "max_norm": max_norm,
    }


def verify(index_dir: Path, chunks_path: Path, skip_duplicates: bool = True) -> int:
    report = Report()
    print(f"Verifying {index_dir}")

    vec_path = index_dir / "embeddings.f32"
    man_path = index_dir / "index_manifest.jsonl"
    meta_path = index_dir / "index_meta.json"

    for path in (vec_path, man_path, meta_path):
        if not report.check(path.exists(), f"{path.name} exists", str(path)):
            print("\nFAILED: the index is incomplete. Run pmc/embed_chunks.py first.")
            return 1

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    dim = int(meta.get("dim", 0))
    report.note("encoder", f"{meta.get('encoder')} (production={meta.get('production')})")
    if meta.get("device"):
        report.note("device", str(meta["device"]))

    report.check(bool(meta.get("production")), "index is a production index",
                 "stub indexes are not valid for thesis results"
                 if not meta.get("production") else "")
    report.check("partial_index_limit" not in meta, "index is not a --limit smoke test",
                 f"partial_index_limit={meta.get('partial_index_limit')}"
                 if "partial_index_limit" in meta else "")
    report.check(dim == MEDCPT_DIM, "vector dimension is MedCPT's 768", f"dim={dim}")
    if dim <= 0:
        print("\nFAILED: index_meta.json declares no usable dimension.")
        return 1

    size = vec_path.stat().st_size
    row_bytes = dim * 4
    report.check(size % row_bytes == 0, "embeddings.f32 is a whole number of rows",
                 f"{size} bytes / {row_bytes} bytes per row")

    manifest = read_manifest(man_path)
    scan = scan_vectors(vec_path, dim, manifest, report)

    report.check(scan["rows"] == len(manifest),
                 "vector count == manifest rows",
                 f"{scan['rows']} vectors, {len(manifest)} manifest rows")
    report.check(int(meta.get("vectors", -1)) == scan["rows"],
                 "index_meta vector count matches the file",
                 f"meta={meta.get('vectors')}, file={scan['rows']}")

    if chunks_path.exists():
        expected = expected_chunk_ids(chunks_path, skip_duplicates)
        report.check(len(expected) == scan["rows"],
                     "vector count == embeddable chunks in the chunk layer",
                     f"{scan['rows']} vectors, {len(expected)} chunks")
        limit = min(len(expected), len(manifest))
        misaligned = [
            i for i in range(limit)
            if str(manifest[i].get("chunk_id")) != expected[i]
        ]
        report.check(
            not misaligned,
            "row alignment: manifest row i is the i-th chunk in sorted order",
            "" if not misaligned else
            f"{len(misaligned)} mismatched, first at row {misaligned[0]}: "
            f"manifest={manifest[misaligned[0]].get('chunk_id')!r} "
            f"expected={expected[misaligned[0]]!r}",
        )
    else:
        report.note("chunk layer", f"{chunks_path} not present -- alignment not checked")

    rows_declare_position = [
        i for i, row in enumerate(manifest) if int(row.get("row", -1)) != i
    ]
    report.check(not rows_declare_position,
                 "every manifest row declares its own position",
                 "" if not rows_declare_position
                 else f"first offender at line {rows_declare_position[0] + 1}")

    report.check(not scan["non_finite"], "no NaN or Inf components",
                 "" if not scan["non_finite"] else f"rows {scan['non_finite']}")
    report.check(not scan["bad_norm"], "vectors are L2-normalised",
                 "" if not scan["bad_norm"]
                 else f"(row, norm) {scan['bad_norm']}")
    if scan["rows"]:
        report.note("norm range",
                    f"{scan['min_norm']:.6f} .. {scan['max_norm']:.6f}")

    report.check(scan["digest"] == meta.get("content_digest"),
                 "content_digest recomputes to the recorded value",
                 f"recorded={meta.get('content_digest')} recomputed={scan['digest']}")

    missing_fields = sorted({
        field
        for row in manifest[:1000]
        for field in REQUIRED_MANIFEST_FIELDS
        if field not in row
    })
    report.check(not missing_fields,
                 "manifest carries the provenance the thesis needs",
                 "" if not missing_fields else f"missing {missing_fields}")

    print(f"\n{report.checks - report.failures}/{report.checks} checks passed")
    if report.failures:
        print(f"FAILED: {report.failures} check(s) failed.")
        return 1
    print(f"OK: {scan['rows']:,} vectors x {dim} dims, digest {scan['digest']}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Verify the MedCPT embedding index.")
    ap.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    ap.add_argument("--chunks", type=Path, default=DEFAULT_CHUNKS)
    ap.add_argument("--include-duplicates", action="store_true",
                    help="the index was built with --include-duplicates")
    args = ap.parse_args(argv)
    return verify(args.index, args.chunks, skip_duplicates=not args.include_duplicates)


if __name__ == "__main__":
    raise SystemExit(main())
