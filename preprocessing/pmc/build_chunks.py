#!/usr/bin/env python3
"""Build the retrieval-ready chunk layer from the frozen M1-M4 corpus.

Reads only frozen inputs -- the parsed records, the PubMed abstracts and the
M1-M4 overlays -- and writes an additive chunk layer. Nothing upstream is
modified: not the manifest, the raw XML, the parsed records, or any M1-M4
policy file.

Chunking strategy (thesis proposal 5.1, not invented here):
    "Sliding-window chunking at 256 tokens with 32-token overlap, sized against
     the article encoder's 512-token limit with headroom for a prepended title
     and section header. Exact deduplication by content hash."

Two implementation decisions follow from that text:

* Windows never cross a section boundary. The proposal requires that a
  recommendation is never separated from its qualifying conditions ("ARIA
  monitoring cadence severed from the APOE stratification it applies to"), and
  windowing inside sections also keeps section provenance exact for every chunk.
* Windows are measured in whitespace words, which is deterministic and needs no
  tokenizer dependency (this repository is standard-library only). A 256-word
  window is always fewer than 512 sub-word tokens, so it stays inside the
  encoder limit with headroom for the prepended title and section heading.
  --window/--overlap can later be set in sub-word tokens without changing any
  other logic.

The title and section heading are NOT baked into the stored text. They are kept
as separate fields and composed at embed time by compose_embed_text(), so the
stored chunk stays the exact source span and the composition rule is one
function shared by every downstream stage.

Eligibility is enforced, never re-decided: records whose frozen M4
eligibility_status is "excluded" are not chunked. Everything else is chunked
with its frozen status carried through, so the retrieval stage can filter.
Exact-duplicate text is flagged via duplicate_of, never deleted -- distinct
versions and source types must survive for the recency study.

Requires only the Python standard library.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Iterator

csv.field_size_limit(2**31 - 1 if sys.maxsize > 2**32 else 2**27)

REPO = Path(__file__).resolve().parent.parent
DEFAULT_ARTICLES = REPO / "pmc" / "parsed" / "articles.jsonl"
DEFAULT_COCHRANE = REPO / "pmc" / "currency_pack" / "parsed" / "PMC13082890.json"
DEFAULT_PUBMED = REPO / "pubmed" / "pubmed_results.csv"
DEFAULT_META = REPO / "pmc" / "metadata"
DEFAULT_OUT = REPO / "pmc" / "chunks"

WINDOW_WORDS = 256
OVERLAP_WORDS = 32
MIN_CHUNK_WORDS = 20          # below this a trailing window carries no evidence


# ---------------------------------------------------------------------------
# Loading the frozen layers
# ---------------------------------------------------------------------------
def load_csv_index(path: Path, key: str) -> dict[str, dict]:
    out: dict[str, dict] = {}
    if not path.exists():
        return out
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            k = (row.get(key) or "").strip()
            if k:
                out[k] = row
    return out


def load_dates(path: Path) -> dict[str, dict]:
    """M3 rows keyed by pmcid when present, else PMID<pmid>."""
    out: dict[str, dict] = {}
    if not path.exists():
        return out
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            pmcid = (row.get("pmcid") or "").strip()
            pmid = (row.get("pmid") or "").strip()
            out[pmcid or f"PMID{pmid}"] = row
    return out


def load_policy(path: Path) -> tuple[dict[str, dict], dict[str, dict]]:
    """M4 rows indexed by pmcid and by pmid."""
    by_pmcid: dict[str, dict] = {}
    by_pmid: dict[str, dict] = {}
    if not path.exists():
        return by_pmcid, by_pmid
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            pmcid = (row.get("pmcid") or "").strip()
            pmid = (row.get("pmid") or "").strip()
            if pmcid:
                by_pmcid[pmcid] = row
            if pmid:
                by_pmid[pmid] = row
    return by_pmcid, by_pmid


def load_pubmed(path: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    if not path.exists():
        return out
    with path.open(newline="", encoding="utf-8") as fh:
        head = fh.readline()
        if head.startswith("version https://git-lfs"):
            raise SystemExit(
                f"ERROR: {path} is an unsmudged Git-LFS pointer, not the PubMed data.\n"
                "Run `git lfs pull` (or pass --pubmed <resolved object>) before chunking."
            )
        fh.seek(0)
        for row in csv.DictReader(fh):
            pmid = (row.get("pmid") or "").strip()
            if pmid:
                out[pmid] = row
    return out


def iter_parsed(path: Path) -> Iterator[dict]:
    if not path.exists():
        return
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


# ---------------------------------------------------------------------------
# Windowing
# ---------------------------------------------------------------------------
def window_words(text: str, size: int, overlap: int) -> list[tuple[int, str]]:
    """Sliding windows over whitespace words. Returns (start_word_index, text).

    Deterministic: identical input always yields identical windows. The final
    window is kept only if it carries enough words to be evidence on its own,
    otherwise its content is already covered by the previous window's overlap.
    """
    words = text.split()
    if not words:
        return []
    if len(words) <= size:
        return [(0, " ".join(words))]
    step = size - overlap
    if step <= 0:
        raise ValueError("overlap must be smaller than window")
    out: list[tuple[int, str]] = []
    start = 0
    while start < len(words):
        piece = words[start:start + size]
        if start > 0 and len(piece) < MIN_CHUNK_WORDS:
            break
        out.append((start, " ".join(piece)))
        if start + size >= len(words):
            break
        start += step
    return out


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def compose_embed_text(title: str, section_heading: str, text: str) -> str:
    """The exact string handed to the encoder. One rule, shared by every stage."""
    parts = [p.strip() for p in (title, section_heading, text) if p and p.strip()]
    return " | ".join(parts)


# ---------------------------------------------------------------------------
# Section walking
# ---------------------------------------------------------------------------
def walk_sections(sections: list[dict]) -> Iterator[dict]:
    """Every section, depth-first, in document order."""
    for sec in sections:
        yield sec
        yield from walk_sections(sec.get("subsections") or [])


def section_text(sec: dict) -> tuple[str, list[str]]:
    """A section's own paragraph text (not its subsections') and paragraph ids."""
    paras = sec.get("paragraphs") or []
    text = " ".join((p.get("text") or "").strip() for p in paras if (p.get("text") or "").strip())
    ids = [p.get("paragraph_id", "") for p in paras]
    return text.strip(), ids


# ---------------------------------------------------------------------------
# Chunk construction
# ---------------------------------------------------------------------------
def base_fields(doc_id: str, meta: dict) -> dict:
    """Provenance and thesis-critical metadata carried onto every chunk."""
    return {
        "document_id": doc_id,
        "pmcid": meta.get("pmcid", ""),
        "pmcid_versioned": meta.get("pmcid_versioned", ""),
        "pmid": meta.get("pmid", ""),
        "doi": meta.get("doi", ""),
        "title": meta.get("title", ""),
        "journal": meta.get("journal", ""),
        "source_category": meta.get("source_category", ""),
        "eligibility_status": meta.get("eligibility_status", ""),
        "fulltext_eligible": meta.get("fulltext_eligible", ""),
        "document_type": meta.get("document_type", ""),
        "canonical_date": meta.get("canonical_date", ""),
        "date_precision": meta.get("date_precision", ""),
        "date_source": meta.get("date_source", ""),
        "split_june_2024": meta.get("split_june_2024", ""),
        "authority_tier_label": meta.get("authority_tier_label", ""),
        "guideline_family": meta.get("guideline_family", ""),
        "organization": meta.get("organization", ""),
        "in_currency_pack": meta.get("in_currency_pack", "no"),
        "claim_class": meta.get("claim_class", ""),
        "license_code": meta.get("license_code", ""),
        "license_band": meta.get("license_band", ""),
        "retracted": meta.get("retracted", ""),
        "flags": meta.get("flags", ""),
        "source_xml_md5": meta.get("source_xml_md5", ""),
    }


def chunks_for_text(doc_id: str, meta: dict, loc: str, location_id: str,
                    heading: str, imrad: str, text: str, para_ids: list[str],
                    window: int, overlap: int) -> list[dict]:
    out = []
    for i, (start, piece) in enumerate(window_words(text, window, overlap), 1):
        rec = base_fields(doc_id, meta)
        rec.update({
            "chunk_id": f"{doc_id}#{location_id}.w{i}",
            "location": loc,
            "section_id": location_id,
            "section_heading": heading,
            "imrad": imrad,
            "paragraph_id_first": para_ids[0] if para_ids else "",
            "paragraph_id_last": para_ids[-1] if para_ids else "",
            "chunk_index": i,
            "window_start_word": start,
            "word_count": len(piece.split()),
            "text": piece,
            "text_sha256": sha256(piece),
            "duplicate_of": "",
        })
        out.append(rec)
    return out


def chunks_for_document(meta: dict, parsed: dict | None, abstract_text: str,
                        window: int, overlap: int) -> list[dict]:
    """All chunks for one document: abstract first, then body sections."""
    doc_id = meta["document_id"]
    out: list[dict] = []

    if abstract_text.strip():
        out += chunks_for_text(doc_id, meta, "abstract", "abs", meta.get("title", ""),
                               "abstract", abstract_text, [], window, overlap)

    # Body only when the frozen policy says the full text is usable.
    if parsed and meta.get("fulltext_eligible") == "yes":
        for sec in walk_sections(parsed.get("sections") or []):
            text, para_ids = section_text(sec)
            if not text:
                continue
            heading = sec.get("title_raw") or " > ".join(
                str(x) for x in (sec.get("section_title_path") or []))
            loc_id = (sec.get("section_id") or "").split("#", 1)[-1] or "s?"
            out += chunks_for_text(doc_id, meta, "body", loc_id, heading,
                                   sec.get("imrad", ""), text, para_ids, window, overlap)
    return out


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def build(articles: Path, cochrane: Path, pubmed_path: Path, meta_dir: Path,
          out_dir: Path, window: int, overlap: int) -> dict:
    dates = load_dates(meta_dir / "canonical_dates.csv")
    pol_pmcid, pol_pmid = load_policy(meta_dir / "corpus_policy.csv")
    cpg = load_csv_index(meta_dir / "cpg_registry.csv", "pmcid")
    curr = load_csv_index(meta_dir / "currency_pack.csv", "pmcid")
    pubmed = load_pubmed(pubmed_path)

    parsed_by_pmcid: dict[str, dict] = {}
    for rec in iter_parsed(articles):
        parsed_by_pmcid[rec["pmcid"]] = rec
    if cochrane.exists():
        rec = json.loads(cochrane.read_text(encoding="utf-8"))
        parsed_by_pmcid.setdefault(rec["pmcid"], rec)

    def meta_for(pmcid: str, pmid: str) -> dict | None:
        pol = pol_pmcid.get(pmcid) or pol_pmid.get(pmid)
        if not pol:
            return None
        if (pol.get("eligibility_status") or "") == "excluded":
            return None                      # frozen policy: not retrievable
        key = pmcid or f"PMID{pmid}"
        dt = dates.get(key, {})
        pm = pubmed.get(pmid, {})
        parsed = parsed_by_pmcid.get(pmcid)
        g = cpg.get(pmcid, {})
        c = curr.get(pmcid, {})
        doc_id = (parsed or {}).get("pmcid_versioned") or pmcid or f"PMID{pmid}"
        return {
            "document_id": doc_id,
            "pmcid": pmcid, "pmcid_versioned": (parsed or {}).get("pmcid_versioned", ""),
            "pmid": pmid, "doi": (parsed or {}).get("doi") or pm.get("doi", ""),
            "title": (parsed or {}).get("title") or pm.get("title", ""),
            "journal": (parsed or {}).get("journal") or pm.get("journal", ""),
            "source_category": pol.get("source_category", ""),
            "eligibility_status": pol.get("eligibility_status", ""),
            "fulltext_eligible": pol.get("fulltext_eligible", ""),
            "document_type": g.get("document_type") or pm.get("publication_types", ""),
            "canonical_date": dt.get("canonical_date", ""),
            "date_precision": dt.get("canonical_date_precision", ""),
            "date_source": dt.get("date_source", ""),
            "split_june_2024": dt.get("split_june_2024", ""),
            "authority_tier_label": g.get("authority_tier_label", ""),
            "guideline_family": g.get("guideline_family", ""),
            "organization": g.get("organization", ""),
            "in_currency_pack": "yes" if c else "no",
            "claim_class": c.get("claim_class", ""),
            "license_code": pol.get("license_code", ""),
            "license_band": pol.get("license_band", ""),
            "retracted": pol.get("retracted", ""),
            "flags": pol.get("flags", ""),
            "source_xml_md5": (parsed or {}).get("provenance", {}).get("xml_md5", ""),
        }

    # Deterministic document order: PMC records by pmcid, then abstract-only by pmid.
    order: list[tuple[str, str]] = sorted(
        ((p, (pol_pmcid[p].get("pmid") or "")) for p in pol_pmcid), key=lambda t: t[0])
    seen_pmids = {pmid for _, pmid in order if pmid}
    order += sorted(((("", pmid)) for pmid in pol_pmid if pmid not in seen_pmids),
                    key=lambda t: t[1])

    out_dir.mkdir(parents=True, exist_ok=True)
    chunks_path = out_dir / "chunks.jsonl"
    first_by_hash: dict[str, str] = {}
    docs = n_chunks = n_dupes = 0
    docs_with_body = docs_abstract_only = 0
    by_category: dict[str, int] = {}
    by_location: dict[str, int] = {}

    with chunks_path.open("w", encoding="utf-8", newline="\n") as fh:
        for pmcid, pmid in order:
            meta = meta_for(pmcid, pmid)
            if meta is None:
                continue
            parsed = parsed_by_pmcid.get(pmcid)
            abstract = ""
            if parsed:
                abstract = ((parsed.get("abstract") or {}).get("text") or "").strip()
            if not abstract:
                abstract = (pubmed.get(pmid, {}).get("abstract") or "").strip()
            recs = chunks_for_document(meta, parsed, abstract, window, overlap)
            if not recs:
                continue
            docs += 1
            if any(r["location"] == "body" for r in recs):
                docs_with_body += 1
            else:
                docs_abstract_only += 1
            for r in recs:
                prior = first_by_hash.get(r["text_sha256"])
                if prior:
                    r["duplicate_of"] = prior
                    n_dupes += 1
                else:
                    first_by_hash[r["text_sha256"]] = r["chunk_id"]
                by_category[r["source_category"]] = by_category.get(r["source_category"], 0) + 1
                by_location[r["location"]] = by_location.get(r["location"], 0) + 1
                fh.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")
                n_chunks += 1

    stats = {
        "window_words": window, "overlap_words": overlap,
        "documents_chunked": docs,
        "documents_with_body": docs_with_body,
        "documents_abstract_only": docs_abstract_only,
        "chunks": n_chunks,
        "exact_duplicate_chunks_flagged": n_dupes,
        "unique_chunk_texts": len(first_by_hash),
        "chunks_by_source_category": dict(sorted(by_category.items())),
        "chunks_by_location": dict(sorted(by_location.items())),
        "parsed_records_available": len(parsed_by_pmcid),
        "policy_records": len(pol_pmcid) + sum(1 for p in pol_pmid if p not in seen_pmids),
    }
    (out_dir / "chunk_stats.json").write_text(
        json.dumps(stats, indent=2, sort_keys=True) + "\n", encoding="utf-8",
        newline="\n")
    return stats


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Build the retrieval-ready chunk layer.")
    ap.add_argument("--articles", type=Path, default=DEFAULT_ARTICLES)
    ap.add_argument("--cochrane", type=Path, default=DEFAULT_COCHRANE)
    ap.add_argument("--pubmed", type=Path, default=DEFAULT_PUBMED)
    ap.add_argument("--metadata", type=Path, default=DEFAULT_META)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--window", type=int, default=WINDOW_WORDS)
    ap.add_argument("--overlap", type=int, default=OVERLAP_WORDS)
    args = ap.parse_args(argv)

    stats = build(args.articles, args.cochrane, args.pubmed, args.metadata,
                  args.out, args.window, args.overlap)
    print(f"Chunks written to {args.out}")
    for k, v in stats.items():
        print(f"  {k:34} {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
