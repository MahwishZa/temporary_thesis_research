#!/usr/bin/env python3
"""Parse downloaded PMC JATS XML into one structured JSON record per article.

Reads the XML files in pmc/fulltext/xml/, joins acquisition provenance from
pmc/fulltext/manifest.csv, and writes JSONL -- one article per line.

The raw XML is opened read-only and never modified. No network access. No
relevance judgement of any kind is applied: every article given to this parser
comes out the other side, flagged where something looks wrong rather than
dropped.

Usage:
    python3 pmc/parse_pmc_xml.py --pmcids PMC9277667,PMC11868538   # sample
    python3 pmc/parse_pmc_xml.py --limit 10                        # first N
    python3 pmc/parse_pmc_xml.py                                   # everything

Requires only the Python standard library.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

PARSER_VERSION = "1.0.0"
SCHEMA_VERSION = "1.0"

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_XML_DIR = REPO_ROOT / "pmc" / "fulltext" / "xml"
DEFAULT_MANIFEST = REPO_ROOT / "pmc" / "fulltext" / "manifest.csv"
DEFAULT_OUTPUT = REPO_ROOT / "pmc" / "parsed" / "articles.jsonl"

# Below this many body words an article is almost certainly a placeholder
# rather than a paper. PMC13405269 -- a preprint whose licence blocks PMC from
# hosting the text -- has 29 body words against a corpus median of ~5,500.
STUB_WORD_THRESHOLD = 250

REF_PLACEHOLDER = "[REF]"

# Elements whose text must never leak into paragraph prose: figure and table
# captions live inside these, and flattening them would splice caption text
# into the middle of a sentence.
SKIP_IN_TEXT = {
    "fig", "table-wrap", "table", "graphic", "media", "alternatives",
    "supplementary-material", "disp-formula-group",
}

# Inline formatting that is flattened away. Counted so a reader can tell why a
# paragraph reads oddly.
INLINE_MARKUP = {
    "italic", "bold", "sup", "sub", "underline", "sc", "monospace",
    "ext-link", "inline-formula", "inline-graphic", "styled-content",
    "named-content", "disp-formula", "list", "email", "uri",
}

# Section titles that are not article prose. Matched case-insensitively against
# the normalised title, and against the sec-type attribute.
NON_CONTENT_TITLE_PATTERNS = [
    r"^supplementary", r"^supporting information", r"^supplemental",
    r"conflict of interest", r"^competing interest", r"^consent statement",
    r"^ethics statement", r"^data availability", r"^funding", r"^acknowledg",
    r"^author contributions", r"^declaration", r"^abbreviations",
]
NON_CONTENT_SEC_TYPES = {
    "supplementary-material", "coi-statement", "conflict", "funding",
    "data-availability", "ethics", "acknowledgments", "acknowledgements",
    "abbreviations", "author-contributions",
}

# IMRaD classification, tried in order against the normalised section title.
# Longest / most specific patterns first.
IMRAD_TITLE_PATTERNS: list[tuple[str, str]] = [
    ("methods", r"^(materials?\s*(and|&|\|)\s*methods?|patients?\s*(and|&)\s*methods?"
                r"|methods?\s*(and|&)\s*materials?|methodology|methods?|"
                r"experimental\s+(section|procedures?|design)|study design)\b"),
    ("results", r"^(results?|findings)\b"),
    ("discussion", r"^discussion\b"),
    ("conclusion", r"^(conclusions?|summary and conclusions?)\b"),
    ("introduction", r"^(introduction|background|rationale)\b"),
]
IMRAD_SEC_TYPES = {
    "intro": "introduction", "introduction": "introduction",
    "background": "introduction",
    "methods": "methods", "materials|methods": "methods",
    "materials-and-methods": "methods", "subjects|methods": "methods",
    "results": "results", "results|discussion": "results",
    "discussion": "discussion", "conclusions": "conclusion",
    "conclusion": "conclusion",
}

# Licence normalisation. Checked in order: the NC/ND compounds must be tested
# before the plainer codes they contain as substrings.
LICENCE_PATTERNS: list[tuple[str, str]] = [
    ("CC BY-NC-SA", r"by[-_ ]?nc[-_ ]?sa"),
    ("CC BY-NC-ND", r"by[-_ ]?nc[-_ ]?nd"),
    ("CC BY-NC", r"by[-_ ]?nc(?![-_ ]?(nd|sa))"),
    ("CC BY-SA", r"by[-_ ]?sa"),
    ("CC BY-ND", r"by[-_ ]?nd"),
    # publicdomain/mark and the content-type token "cc0license" both appear in
    # PMC. \bcc0\b cannot match "cc0license" -- there is no word boundary
    # between "0" and "l" -- so the token is spelled out. Public Domain Mark is
    # not literally CC0, but PMC labels it cc0license, and this field records
    # what the source declares.
    ("CC0", r"publicdomain/(zero|mark)|\bcc0\b|cc0[-_ ]?licen[cs]e|zerolicense"),
    ("CC BY", r"by(?![-_ ]?(nc|nd|sa))"),
]

# Documents nested inside an article: a peer review, an author response. Their
# content is their own, so article-level traversals must not descend into them.
NESTED_DOCUMENT_TAGS = {"sub-article", "response"}

# Which pub-date wins. "preprint" is deliberately last: PMC9277667 carries a
# preprint date four years before its actual publication date.
DATE_TYPE_PRECEDENCE = ["epub", "pub", "ppub", "collection", "epub-ppub"]

MONTH_NAMES = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}


# ---------------------------------------------------------------------------
# Small XML helpers. PMC's article elements carry no namespace, but ali: and
# xlink: attributes do, so everything matches on the local name.
# ---------------------------------------------------------------------------


def local(tag: Any) -> str:
    if not isinstance(tag, str):
        return ""
    return tag.rsplit("}", 1)[-1]


def children(elem: ET.Element | None, name: str) -> list[ET.Element]:
    if elem is None:
        return []
    return [c for c in elem if local(c.tag) == name]


def child(elem: ET.Element | None, name: str) -> ET.Element | None:
    found = children(elem, name)
    return found[0] if found else None


def descendants(elem: ET.Element | None, name: str) -> Iterator[ET.Element]:
    if elem is None:
        return
    for node in elem.iter():
        if local(node.tag) == name and node is not elem:
            yield node


def path_child(root: ET.Element | None, *names: str) -> ET.Element | None:
    node = root
    for name in names:
        node = child(node, name)
        if node is None:
            return None
    return node


def squash(text: str) -> str:
    return " ".join(text.split())


def flatten(elem: ET.Element | None, stripped: Counter[str] | None = None) -> str:
    """Readable text for one element.

    Citation cross-references become [REF]; figure and table blocks are skipped
    entirely so captions cannot be mistaken for prose; other inline markup is
    unwrapped. Every removal is counted in `stripped`.
    """
    if elem is None:
        return ""
    if stripped is None:
        stripped = Counter()

    parts: list[str] = []
    if elem.text:
        parts.append(elem.text)

    for node in elem:
        tag = local(node.tag)
        if tag in SKIP_IN_TEXT:
            stripped[tag] += 1
        elif tag == "xref":
            stripped["xref"] += 1
            if (node.get("ref-type") or "") == "bibr":
                parts.append(REF_PLACEHOLDER)
            else:
                # A figure or table pointer: its own text ("Fig. 1") reads fine.
                parts.append(flatten(node, stripped))
        else:
            if tag in INLINE_MARKUP:
                stripped[tag] += 1
            parts.append(flatten(node, stripped))
        if node.tail:
            parts.append(node.tail)

    return "".join(parts)


def flat_text(elem: ET.Element | None, stripped: Counter[str] | None = None) -> str:
    return squash(flatten(elem, stripped))


# ---------------------------------------------------------------------------
# Front matter
# ---------------------------------------------------------------------------


def parse_identifiers(article_meta: ET.Element | None) -> dict[str, str]:
    ids: dict[str, str] = {}
    for node in children(article_meta, "article-id"):
        key = node.get("pub-id-type") or ""
        if key and node.text:
            ids[key] = node.text.strip()
    return ids


def parse_dates(article_meta: ET.Element | None) -> dict[str, Any]:
    """Every pub-date variant, plus the one selected and how precise it is."""
    found: dict[str, dict[str, Any]] = {}
    for node in children(article_meta, "pub-date"):
        key = node.get("pub-type") or node.get("date-type") or "unspecified"
        year = flat_text(child(node, "year"))
        if not year:
            continue
        month_raw = flat_text(child(node, "month"))
        day_raw = flat_text(child(node, "day"))

        month = None
        if month_raw:
            if month_raw.isdigit():
                month = int(month_raw)
            else:
                month = MONTH_NAMES.get(month_raw[:3].lower())
        day = int(day_raw) if day_raw.isdigit() else None

        if month and day:
            value, precision = f"{year}-{month:02d}-{day:02d}", "day"
        elif month:
            value, precision = f"{year}-{month:02d}", "month"
        else:
            value, precision = year, "year"
        # Keep the first occurrence of a given type; later duplicates are rare
        # and the first is the one the publisher listed as primary.
        found.setdefault(key, {"date": value, "precision": precision})

    order = DATE_TYPE_PRECEDENCE + sorted(k for k in found if k not in DATE_TYPE_PRECEDENCE)
    for key in order:
        if key in found:
            return {
                "publication_date": found[key]["date"],
                "publication_date_type": key,
                "publication_date_precision": found[key]["precision"],
                "publication_dates_all": {k: v["date"] for k, v in found.items()},
            }
    return {
        "publication_date": "",
        "publication_date_type": "",
        "publication_date_precision": "",
        "publication_dates_all": {},
    }


def parse_affiliations(front: ET.Element | None) -> list[dict[str, str]]:
    affiliations = []
    for node in descendants(front, "aff"):
        text = flat_text(node)
        label = child(node, "label")
        if label is not None and label.text:
            # Drop a leading "1" / "a" marker so the text starts at the institution.
            marker = label.text.strip()
            if text.startswith(marker):
                text = text[len(marker):].strip()
        affiliations.append({"id": node.get("id") or "", "text": text})
    return affiliations


def parse_authors(
    article_meta: ET.Element | None, aff_ids: set[str]
) -> tuple[list[dict[str, Any]], bool]:
    """Authors only. Returns (authors, editors_present)."""
    authors: list[dict[str, Any]] = []
    editors_present = False
    position = 0

    for group in children(article_meta, "contrib-group"):
        for contrib in children(group, "contrib"):
            kind = contrib.get("contrib-type") or ""
            if kind and kind != "author":
                if kind == "editor":
                    editors_present = True
                continue
            if not kind:
                # No contrib-type at all: treat as author, but only when a name
                # is present, so stray contribs do not become phantom authors.
                if child(contrib, "name") is None and child(contrib, "collab") is None:
                    continue

            name = child(contrib, "name")
            collab = child(contrib, "collab")
            surname = flat_text(child(name, "surname")) if name is not None else ""
            given = flat_text(child(name, "given-names")) if name is not None else ""

            orcid = ""
            for cid in children(contrib, "contrib-id"):
                if (cid.get("contrib-id-type") or "") == "orcid" and cid.text:
                    orcid = cid.text.strip().rsplit("/", 1)[-1]

            rids: list[str] = []
            for xref in children(contrib, "xref"):
                rid = xref.get("rid") or ""
                ref_type = xref.get("ref-type") or ""
                for candidate in rid.split():
                    if ref_type == "aff" or candidate in aff_ids:
                        rids.append(candidate)

            position += 1
            authors.append({
                "position": position,
                "surname": surname,
                "given_names": given,
                "collab": flat_text(collab) if collab is not None else None,
                "orcid": orcid,
                "affiliation_ids": rids,
            })

    return authors, editors_present


def parse_abstracts(article_meta: ET.Element | None) -> tuple[dict[str, Any], list[dict[str, str]], int]:
    """(scientific abstract, other abstracts, total abstract elements)."""
    nodes = children(article_meta, "abstract")
    if not nodes:
        return {"text": "", "is_structured": False, "sections": []}, [], 0

    # The scientific abstract is the one with no abstract-type; a labelled
    # plain-language or web summary must not be concatenated into it.
    preferred = next((n for n in nodes if not n.get("abstract-type")), nodes[0])
    primary = preferred
    if not flat_text(primary):
        for node in nodes:
            if node is not preferred and flat_text(node):
                primary = node
                break

    others = [
        {"type": n.get("abstract-type") or "untyped", "text": flat_text(n)}
        for n in nodes if n is not primary
    ]

    sections = []
    for sec in children(primary, "sec"):
        sections.append({
            "label": flat_text(child(sec, "title")),
            "text": squash(" ".join(flat_text(p) for p in children(sec, "p"))),
        })

    return (
        {"text": flat_text(primary), "is_structured": bool(sections), "sections": sections},
        others,
        len(nodes),
    )


def _licence_fields(licence: ET.Element) -> tuple[dict[str, str], bool]:
    """Parse one <license> element. The bool is True when ali:license_ref yielded a code."""
    ref_url = ""
    content_type = ""
    found_license_ref = False
    for node in licence.iter():
        if local(node.tag) == "license_ref":
            found_license_ref = True
            ref_url = (node.text or "").strip()
            content_type = node.get("content-type") or ""
            break

    statement = squash(" ".join(flat_text(p) for p in children(licence, "license-p")))
    if not statement:
        statement = flat_text(licence)

    if not ref_url:
        for node in licence.iter():
            if local(node.tag) == "ext-link":
                href = next((v for k, v in node.attrib.items() if local(k) == "href"), "")
                if "creativecommons.org" in href:
                    ref_url = href
                    break

    haystack = " ".join([content_type, ref_url]).lower()
    code = ""
    for name, pattern in LICENCE_PATTERNS:
        if re.search(pattern, haystack):
            code = name
            break
    from_license_ref = bool(found_license_ref and code)
    if not code and re.search(r"text\s*mining", statement, re.I):
        code = "TDM"

    return (
        {
            "license_ref_xml": ref_url,
            "license_content_type_xml": content_type,
            "license_statement_xml": statement,
            "license_code_xml": code,
        },
        from_license_ref,
    )


def parse_licence(article_meta: ET.Element | None) -> dict[str, str]:
    """Licence as recorded inside the XML.

    The machine-readable value lives in <ali:license_ref>, not in attributes on
    <license> -- across the inspected sample, <license> carried neither
    license-type nor xlink:href. When several <license> elements are present,
    prefer one whose ali:license_ref yields a licence code.
    """
    empty = {"license_ref_xml": "", "license_content_type_xml": "",
             "license_statement_xml": "", "license_code_xml": ""}
    licences = children(child(article_meta, "permissions"), "license")
    if not licences:
        return empty

    parsed = [_licence_fields(node) for node in licences]
    for fields, from_license_ref in parsed:
        if from_license_ref:
            return fields
    return parsed[0][0]


def main_article_elements(node: ET.Element | None) -> Iterator[ET.Element]:
    """Every element of the main article, never entering a nested document.

    A <sub-article> (peer review) or <response> (author reply) carries its own
    front-stub, body and back. Its content belongs to that document, so no
    traversal of the article may descend into one.
    """
    if node is None:
        return
    yield node
    for kid in node:
        if local(kid.tag) in NESTED_DOCUMENT_TAGS:
            continue
        yield from main_article_elements(kid)


def count_references(root: ET.Element | None) -> int:
    """Unique <ref> elements under any <ref-list> belonging to the main article.

    Publishers place the bibliography in several positions: directly under
    <back>, inside a <back><sec>, inside an appendix, repeated as siblings, and
    -- unusually but legally -- inside the <body> itself. All of those are the
    article's own references and all are counted.

    A sub-article's or response's references are not: they cite a different
    document. That distinction is why this walks the article rather than
    searching the whole tree.
    """
    unique: set[int] = set()
    for node in main_article_elements(root):
        if local(node.tag) != "ref-list":
            continue
        for ref in main_article_elements(node):
            if ref is not node and local(ref.tag) == "ref":
                unique.add(id(ref))
    return len(unique)


# ---------------------------------------------------------------------------
# Body
# ---------------------------------------------------------------------------


def normalise_title(title: str) -> str:
    return squash(title).strip(" .:").lower()


def classify_section(title: str, sec_type: str | None) -> tuple[str, str]:
    """(imrad, imrad_source). Title text is tried first, as agreed."""
    normalised = normalise_title(title)
    if normalised:
        for label, pattern in IMRAD_TITLE_PATTERNS:
            if re.search(pattern, normalised):
                return label, "title_match"
    if sec_type:
        mapped = IMRAD_SEC_TYPES.get(sec_type.strip().lower())
        if mapped:
            return mapped, "sec_type"
    return "other", "unknown"


def is_content_section(title: str, sec_type: str | None) -> bool:
    if sec_type and sec_type.strip().lower() in NON_CONTENT_SEC_TYPES:
        return False
    normalised = normalise_title(title)
    return not any(re.search(p, normalised) for p in NON_CONTENT_TITLE_PATTERNS)


def parse_section(
    sec: ET.Element, prefix: str, path: list[int], counter: dict[str, int],
    inherited_imrad: str = "",
) -> dict[str, Any]:
    title_el = child(sec, "title")
    title_raw = flat_text(title_el)
    sec_type = sec.get("sec-type")

    imrad, imrad_source = classify_section(title_raw, sec_type)
    if imrad == "other" and inherited_imrad:
        imrad, imrad_source = inherited_imrad, "inherited"

    section_path = ".".join(str(i) for i in path)
    section_id = f"{prefix}#s{section_path}"
    content = is_content_section(title_raw, sec_type)

    paragraphs = []
    # Direct <p> children only. Captions are <p> nested inside <fig>/<table-wrap>,
    # so a descendant search would pull them in as body prose.
    for index, node in enumerate(children(sec, "p"), start=1):
        stripped: Counter[str] = Counter()
        text = flat_text(node, stripped)
        if not text:
            continue
        counter["ordinal"] += 1
        words = len(text.split())
        counter["words"] += words
        paragraphs.append({
            "paragraph_id": f"{section_id}.p{index}",
            "ordinal_in_article": counter["ordinal"],
            "ordinal_in_section": index,
            "section_path": list(path),
            "imrad": imrad,
            "text": text,
            "word_count": words,
            "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "inline_stripped": dict(stripped),
            "contains_float_reference": any(
                k in stripped for k in ("fig", "table-wrap", "table")
            ),
        })

    subsections = [
        parse_section(sub, prefix, path + [i], counter, imrad)
        for i, sub in enumerate(children(sec, "sec"), start=1)
    ]

    return {
        "section_id": section_id,
        "path": list(path),
        "depth": len(path),
        "title_raw": title_raw,
        "title_normalized": normalise_title(title_raw),
        "imrad": imrad,
        "imrad_source": imrad_source,
        "sec_type_attr": sec_type,
        "xml_id_attr": sec.get("id"),
        "is_content": content,
        "paragraphs": paragraphs,
        "subsections": subsections,
    }


def walk_sections(sections: list[dict[str, Any]]) -> Iterator[dict[str, Any]]:
    for section in sections:
        yield section
        yield from walk_sections(section["subsections"])


def add_title_paths(sections: list[dict[str, Any]], trail: list[str]) -> None:
    """Denormalise the section title trail onto each paragraph.

    A retrieved chunk then carries its own context ("Methods -> Participants")
    without a walk back up the tree.
    """
    for section in sections:
        here = trail + [section["title_raw"]]
        for paragraph in section["paragraphs"]:
            paragraph["section_title_path"] = list(here)
        add_title_paths(section["subsections"], here)


# ---------------------------------------------------------------------------
# One article
# ---------------------------------------------------------------------------


def parse_article(path: Path, manifest_row: dict[str, str], stub_threshold: int) -> dict[str, Any]:
    raw = path.read_bytes()
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    base: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "pmcid": path.stem,
        "source_xml_filename": path.name,
    }

    try:
        root = ET.fromstring(raw)
    except ET.ParseError as exc:
        base.update({
            "qc": {"status": "parse_error", "flags": ["xml_parse_error"],
                   "error": str(exc), "parser_version": PARSER_VERSION,
                   "parsed_at_utc": now},
            "provenance": provenance_block(manifest_row, len(raw)),
        })
        return base

    front = child(root, "front")
    article_meta = child(front, "article-meta")
    journal_meta = child(front, "journal-meta")
    body = child(root, "body")
    back = child(root, "back")
    floats = child(root, "floats-group")

    ids = parse_identifiers(article_meta)
    affiliations = parse_affiliations(front)
    authors, editors_present = parse_authors(article_meta, {a["id"] for a in affiliations})
    abstract, abstract_other, abstract_count = parse_abstracts(article_meta)
    licence = parse_licence(article_meta)
    dates = parse_dates(article_meta)

    title_el = path_child(article_meta, "title-group", "article-title")
    journal_title = (
        flat_text(path_child(journal_meta, "journal-title-group", "journal-title"))
        or flat_text(child(journal_meta, "journal-title"))
    )

    pmcid = ids.get("pmcid") or path.stem
    pmcid_versioned = ids.get("pmcid-ver") or pmcid
    prefix = pmcid_versioned

    counter = {"ordinal": 0, "words": 0}
    sections = [
        parse_section(sec, prefix, [i], counter)
        for i, sec in enumerate(children(body, "sec"), start=1)
    ]

    # Some articles put paragraphs straight on <body> with no <sec> wrapper.
    flat_paragraphs = children(body, "p")
    if flat_paragraphs:
        pseudo = ET.Element("sec")
        for node in flat_paragraphs:
            pseudo.append(node)
        sections.append(parse_section(pseudo, prefix, [len(sections) + 1], counter))

    add_title_paths(sections, [])

    all_sections = list(walk_sections(sections))
    paragraph_count = sum(len(s["paragraphs"]) for s in all_sections)
    body_word_count = counter["words"]

    figure_count = len(list(descendants(body, "fig"))) + len(list(descendants(floats, "fig")))
    table_count = (len(list(descendants(body, "table-wrap")))
                   + len(list(descendants(floats, "table-wrap"))))
    reference_count = count_references(root)

    # ---- QC ----
    flags: list[str] = []
    if body is None:
        status = "no_body"
    elif body_word_count < stub_threshold:
        status = "stub"
    elif not abstract["text"]:
        status = "no_abstract"
    else:
        status = "ok"

    if not all_sections:
        flags.append("no_sections")
    if flat_paragraphs:
        flags.append("flat_sections")
    if reference_count == 0:
        flags.append("no_references")
    if floats is not None:
        flags.append("floats_group_present")
    if abstract_count > 1:
        flags.append("multiple_abstracts")
    if not affiliations:
        flags.append("no_affiliations")
    if editors_present:
        flags.append("editors_present")
    if not authors:
        flags.append("no_authors")
    if dates["publication_date_precision"] in {"month", "year"}:
        flags.append("partial_date")
    if not dates["publication_date"]:
        flags.append("no_date")
    manifest_code = (manifest_row.get("license_code") or "").strip()
    if manifest_code and licence["license_code_xml"] and manifest_code != licence["license_code_xml"]:
        flags.append("license_disagreement")
    if manifest_code and not licence["license_code_xml"]:
        flags.append("license_absent_in_xml")
    if not manifest_code and licence["license_code_xml"]:
        flags.append("license_recovered_from_xml")
    if not ids.get("pmid"):
        flags.append("no_pmid")
    if not ids.get("doi"):
        flags.append("no_doi")

    base.update({
        "pmcid": pmcid,
        "pmcid_versioned": pmcid_versioned,
        "pmid": ids.get("pmid", ""),
        "doi": ids.get("doi", ""),
        "journal": journal_title,
        "title": flat_text(title_el),
        "title_had_inline_markup": title_el is not None and len(list(title_el)) > 0,
        "authors": authors,
        "author_count": len(authors),
        "affiliations": affiliations,
        **dates,
        "abstract": abstract,
        "abstract_other": abstract_other,
        "sections": sections,
        "provenance": provenance_block(manifest_row, len(raw), ids),
        "body_word_count": body_word_count,
        "paragraph_count": paragraph_count,
        "section_count": len(all_sections),
        "reference_count": reference_count,
        "figure_count": figure_count,
        "table_count": table_count,
        "qc": {"status": status, "flags": flags, "error": "",
               "parser_version": PARSER_VERSION, "parsed_at_utc": now},
    })
    base["provenance"].update({k: v for k, v in licence.items()})
    return base


def provenance_block(
    row: dict[str, str], xml_bytes: int, ids: dict[str, str] | None = None
) -> dict[str, Any]:
    return {
        "xml_md5": row.get("actual_md5") or row.get("expected_md5", ""),
        "xml_bytes": xml_bytes,
        "manifest_bytes": row.get("bytes", ""),
        "pmc_version": row.get("version", ""),
        "is_manuscript": row.get("is_manuscript", ""),
        "is_retracted": row.get("is_retracted", ""),
        "license_code_manifest": row.get("license_code", ""),
        "source_xml_url": row.get("source_xml_url", ""),
        "downloaded_at_utc": row.get("downloaded_at_utc", ""),
        "acquisition_status": row.get("status", ""),
        "manuscript_id": (ids or {}).get("manuscript-id", ""),
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def widen_csv_field_limit(target: int = 64 * 1024 * 1024) -> None:
    """Portable: never sys.maxsize, which overflows a 32-bit C long on Windows."""
    limit = target
    while limit > csv.field_size_limit():
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 2


def load_manifest(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        print(f"WARNING: no manifest at {path}; provenance will be sparse", file=sys.stderr)
        return {}
    widen_csv_field_limit()
    with path.open(encoding="utf-8", newline="") as handle:
        return {row["pmcid"]: row for row in csv.DictReader(handle)}


def assert_safe_output(output: Path, xml_dir: Path) -> None:
    resolved = output.resolve()
    if (REPO_ROOT / "pubmed").resolve() in resolved.parents:
        raise SystemExit("ERROR: this tool must never write under pubmed/.")
    if xml_dir.resolve() in resolved.parents or resolved == xml_dir.resolve():
        raise SystemExit("ERROR: refusing to write inside the raw XML directory.")


def select_files(xml_dir: Path, pmcids: list[str] | None, limit: int | None) -> list[Path]:
    if pmcids:
        chosen = []
        for pmcid in pmcids:
            candidate = xml_dir / f"{pmcid}.xml"
            if not candidate.exists():
                raise SystemExit(f"ERROR: {candidate} does not exist")
            chosen.append(candidate)
        return chosen
    files = sorted(xml_dir.glob("*.xml"), key=lambda p: int(p.stem[3:]) if p.stem[3:].isdigit() else 0)
    return files[:limit] if limit else files


def run(args: argparse.Namespace) -> int:
    xml_dir: Path = args.xml_dir
    output: Path = args.output
    assert_safe_output(output, xml_dir)

    if not xml_dir.exists():
        raise SystemExit(f"ERROR: no XML directory at {xml_dir}")

    manifest = load_manifest(args.manifest)
    files = select_files(xml_dir, args.pmcids.split(",") if args.pmcids else None, args.limit)

    print(f"XML source : {xml_dir}")
    print(f"Manifest   : {args.manifest} ({len(manifest):,} rows)")
    print(f"Output     : {output}")
    print(f"Articles   : {len(files):,}")
    print(f"Stub cutoff: {args.stub_threshold} body words")
    print("The raw XML is opened read-only and never modified.\n")

    output.parent.mkdir(parents=True, exist_ok=True)
    statuses: Counter[str] = Counter()
    flags: Counter[str] = Counter()
    word_counts: list[tuple[str, int]] = []
    missing = Counter()

    with output.open("w", encoding="utf-8") as handle:
        for index, path in enumerate(files, start=1):
            record = parse_article(path, manifest.get(path.stem, {}), args.stub_threshold)
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

            statuses[record["qc"]["status"]] += 1
            flags.update(record["qc"]["flags"])
            if "body_word_count" in record:
                word_counts.append((record["pmcid"], record["body_word_count"]))
            for field in ("pmcid", "pmid", "doi"):
                if not record.get(field):
                    missing[field] += 1
            if index % 500 == 0:
                print(f"  {index:,}/{len(files):,}", flush=True)

    print("=" * 66)
    print("Parse summary")
    print("=" * 66)
    print(f"Records written : {sum(statuses.values()):,}  -> {output}")
    print(f"QC status       : {dict(statuses)}")
    print(f"QC flags        : {dict(flags.most_common())}")
    if word_counts:
        low = min(word_counts, key=lambda x: x[1])
        high = max(word_counts, key=lambda x: x[1])
        ordered = sorted(w for _, w in word_counts)
        print(f"Body words      : min {low[1]:,} ({low[0]})  "
              f"median {ordered[len(ordered)//2]:,}  max {high[1]:,} ({high[0]})")
    print(f"Missing ids     : {dict(missing) or 'none -- pmcid, pmid and doi captured for all'}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--xml-dir", type=Path, default=DEFAULT_XML_DIR)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--pmcids", type=str, default="",
                        help="comma-separated PMCIDs to parse (for sample runs)")
    parser.add_argument("--limit", type=int, help="parse only the first N files")
    parser.add_argument("--stub-threshold", type=int, default=STUB_WORD_THRESHOLD)
    return parser


if __name__ == "__main__":
    sys.exit(run(build_parser().parse_args()))
