#!/usr/bin/env python3
"""Materialize the approved Corpus Decision Spec (M1-M4) as additive metadata.

This script does NOT modify the raw acquisition corpus, the PMC XML, the parsed
records, the manifest, the inventory, the PubMed pipeline, or search_queries.txt.
It reads them and writes derived overlay files under pmc/metadata/ and the
externally-added currency-pack layer under pmc/currency_pack/.

    M1  currency pack        -> pmc/metadata/currency_pack.csv
                                pmc/currency_pack/{xml,parsed}/PMC13082890.*
    M2  CPG-AD layer (seed)  -> pmc/metadata/cpg_registry.csv   (see the STOP note)
    M3  canonical dates      -> pmc/metadata/canonical_dates.csv
    M4  corpus policy        -> pmc/metadata/corpus_policy.csv

Every overlay is keyed by pmcid (M1/M2/M3/M4 on the PMC layer) or pmid (the
PubMed abstract layer in M3/M4) and joins back to the existing records without
touching them. Re-running on the machine that holds the full parsed corpus
upgrades M3 dates from PubMed fallback to JATS-primary automatically.

Design decisions are fixed by the Corpus Decision Spec; this script implements
them, it does not make new corpus-policy decisions.

Requires only the Python standard library. Network is used once, for M1, to
fetch the single OA Cochrane object if it is not already staged locally.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

csv.field_size_limit(2**31 - 1 if sys.maxsize > 2**32 else 2**27)

REPO = Path(__file__).resolve().parent.parent
DEFAULT_MANIFEST = REPO / "pmc" / "fulltext" / "manifest.csv"
DEFAULT_PUBMED = REPO / "pubmed" / "pubmed_results.csv"
DEFAULT_ARTICLES = REPO / "pmc" / "parsed" / "articles.jsonl"
DEFAULT_REPORT = REPO / "pmc" / "pmc_qc_report_2026-09-03-postfix.md"
DEFAULT_OUT = REPO / "pmc" / "metadata"
DEFAULT_CURRENCY_DIR = REPO / "pmc" / "currency_pack"

# The counterfactual index boundary (proposal 6.4): the June-2024 criteria
# revision. Records earlier than this month are "pre", the rest "post".
JUNE_2024 = "2024-06"

# JATS pub-date types that represent a real publication event, in the order the
# parser already trusts. "preprint" is deliberately excluded: a preprint date
# can predate actual publication by years (parser note, PMC9277667).
REAL_PUB_TYPES = ["epub", "pub", "ppub", "collection", "epub-ppub"]

# ---------------------------------------------------------------------------
# The four currency-pack anchors named by the proposal (1.3, 4.4, 5.1).
# in_raw = already downloaded in the OA corpus; manual = externally ingested.
# ---------------------------------------------------------------------------
CURRENCY_ANCHORS = [
    {
        "pmcid": "PMC11350039", "pmid": "38934362", "doi": "10.1002/alz.13859",
        "acquisition": "pmc-oa", "claim_class": "diagnostic-criteria",
        "rationale": "2024 revised NIA-AA/AA criteria; supersedes the 2011 criteria; "
                     "core changed-knowledge axis and a supersession source.",
    },
    {
        "pmcid": "PMC10313141", "pmid": "37357276", "doi": "10.14283/jpad.2023.30",
        "acquisition": "pmc-oa", "claim_class": "eligibility/aria/pharmacotherapy",
        "rationale": "Lecanemab Appropriate Use Recommendations; introduces eligibility, "
                     "APOE genotyping and ARIA monitoring; high-authority source.",
    },
    {
        "pmcid": "PMC12180672", "pmid": "40155270", "doi": "10.1016/j.tjpad.2025.100150",
        "acquisition": "pmc-oa", "claim_class": "eligibility/aria/pharmacotherapy",
        "rationale": "Donanemab Appropriate Use Recommendations; eligibility and ARIA "
                     "schedule; high-authority source.",
    },
    {
        "pmcid": "PMC13082890", "pmid": "41985900", "doi": "10.1002/14651858.CD016297",
        "acquisition": "manual-oa-fetch", "claim_class": "pharmacotherapy/contested",
        "rationale": "CONTESTED-STATE ANCHOR. April-2026 Cochrane review of anti-amyloid "
                     "monoclonal antibodies; the CONTESTED-AD set is defined around it. "
                     "In PMC-OA but its object is revised in place, so the inventory MD5 "
                     "is stale; pinned by observed MD5 (manual ingestion).",
    },
    # Recency anchors added to complete the small high-value pack (all in the raw OA
    # corpus already; verified metadata, no fabrication). Each also serves the CPG
    # layer -- the dual role is recorded, not duplicated (in_cpg_layer flag).
    {
        "pmcid": "PMC11251480", "pmid": "38939962", "doi": "10.1002/acn3.52042",
        "acquisition": "pmc-oa", "claim_class": "aria",
        "rationale": "Dedicated ARIA diagnosis/management guidance (2024); the ARIA "
                     "safety axis the appropriate-use recommendations introduced.",
    },
    {
        "pmcid": "PMC12306682", "pmid": "40729527", "doi": "10.1002/alz.70535",
        "acquisition": "pmc-oa", "claim_class": "plasma-biomarkers",
        "rationale": "Alzheimer's Association clinical practice guideline on blood-based "
                     "biomarkers (2025); the major recent biomarker guidance.",
    },
    {
        "pmcid": "PMC11772739", "pmid": "39776249", "doi": "10.1002/alz.14338",
        "acquisition": "pmc-oa", "claim_class": "amyloid-imaging",
        "rationale": "Updated appropriate-use criteria for amyloid and tau PET (2025); "
                     "major recent diagnostic-imaging guidance.",
    },
]

# Currency-pack members that also belong to the CPG layer (dual role, not duplicated).
CURRENCY_ALSO_CPG = {"PMC11350039", "PMC10313141", "PMC12180672",
                     "PMC11251480", "PMC12306682", "PMC11772739"}

# ---------------------------------------------------------------------------
# M2 -- curated Alzheimer/dementia CPG/guidance layer. Every entry is a record
# VERIFIED to be in the raw OA corpus; the curation fields (organization,
# document_type, authority tier, guideline family, claim classes) are assigned
# from the document's own title/type. Dates, licences and identifiers are joined
# from the manifest/PubMed at build time -- never fabricated here. Cochrane and
# other systematic reviews are NOT listed here (they are not CPGs); the PET AUC
# is an appropriate-use-criteria report and is tagged as such with its PubMed
# type noted. Off-domain guideline-typed records (e.g. pain, brain-heart) are
# deliberately excluded.
#   authority_tier_label is a LABEL only -- no ordering/weight is implied
#   (authority is a tested thesis variable, ablation A12).
# ---------------------------------------------------------------------------
TIER_CPG = "clinical-practice-guideline"
TIER_AUC = "appropriate-use-criteria"
TIER_CONSENSUS = "consensus-recommendation"
TIER_CRITERIA = "diagnostic-criteria"

CURATED_CPG: list[dict] = [
    # criteria / appropriate-use (also currency)
    dict(pmcid="PMC11350039", organization="NIA-AA / Alzheimer's Association Workgroup",
         document_type="diagnostic & staging criteria", tier=TIER_CRITERIA,
         family="NIA-AA-criteria", claim_classes="criteria;biomarkers;amyloid;tau;plasma;staging",
         supersession_note="revised criteria; supersedes 2011/2018 NIA-AA criteria (not in corpus)"),
    dict(pmcid="PMC10313141", organization="Alzheimer's Association appropriate-use workgroup",
         document_type="appropriate-use recommendations", tier=TIER_AUC,
         family="lecanemab-AUR", claim_classes="eligibility;aria;apoe;pharmacotherapy",
         supersession_note="drug-specific; may be updated"),
    dict(pmcid="PMC12180672", organization="Alzheimer's Association appropriate-use workgroup",
         document_type="appropriate-use recommendations", tier=TIER_AUC,
         family="donanemab-AUR", claim_classes="eligibility;aria;apoe;pharmacotherapy",
         supersession_note="drug-specific; may be updated"),
    dict(pmcid="PMC11772739", organization="Alzheimer's Association / SNMMI",
         document_type="appropriate-use criteria (report; PubMed type=Systematic Review)",
         tier=TIER_AUC, family="amyloid-tau-PET-AUC",
         claim_classes="amyloid-imaging;tau;biomarkers",
         supersession_note="updated AUC; supersedes prior amyloid-PET AUC (not in corpus)"),
    # Alzheimer's Association clinical practice guidelines
    dict(pmcid="PMC12306682", organization="Alzheimer's Association",
         document_type="clinical practice guideline", tier=TIER_CPG,
         family="AA-blood-biomarker-CPG", claim_classes="plasma;biomarkers;diagnosis",
         supersession_note="new (2025); no prior version"),
    dict(pmcid="PMC12173843", organization="Alzheimer's Association",
         document_type="clinical practice guideline", tier=TIER_CPG,
         family="AA-DETeCD-ADRD", claim_classes="diagnosis;differential-diagnosis;criteria",
         supersession_note="DETeCD-ADRD family; part/version relationship within family not verified"),
    dict(pmcid="PMC11772712", organization="Alzheimer's Association",
         document_type="clinical practice guideline", tier=TIER_CPG,
         family="AA-DETeCD-ADRD", claim_classes="diagnosis;differential-diagnosis",
         supersession_note="DETeCD-ADRD family; part/version relationship within family not verified"),
    dict(pmcid="PMC11772716", organization="Alzheimer's Association",
         document_type="clinical practice guideline", tier=TIER_CPG,
         family="AA-DETeCD-ADRD", claim_classes="diagnosis;differential-diagnosis",
         supersession_note="DETeCD-ADRD family; part/version relationship within family not verified"),
    # ARIA / imaging guidance
    dict(pmcid="PMC11251480", organization="multi-society (ARIA working group)",
         document_type="ARIA diagnosis/management guidance (PubMed type=Review)",
         tier=TIER_CONSENSUS, family="ARIA-guidance", claim_classes="aria;pharmacotherapy;amyloid-imaging",
         supersession_note="none verified"),
    dict(pmcid="PMC12772267", organization="Japanese neuroradiology (MRMS)",
         document_type="clinical practice guideline (MRI for anti-Aβ)", tier=TIER_CPG,
         family="anti-Ab-MRI-guidelines", claim_classes="aria;amyloid-imaging;pharmacotherapy",
         supersession_note="related to PMC12287243 (same guidance, different journal); relationship recorded"),
    dict(pmcid="PMC12287243", organization="Japanese neuroradiology (Jpn J Radiol)",
         document_type="clinical practice guideline (MRI for anti-Aβ)", tier=TIER_CPG,
         family="anti-Ab-MRI-guidelines", claim_classes="aria;amyloid-imaging;pharmacotherapy",
         supersession_note="related to PMC12772267 (same guidance, different journal); relationship recorded"),
    dict(pmcid="PMC11592784", organization="Thai society of nuclear medicine",
         document_type="clinical practice guideline (nuclear medicine)", tier=TIER_CPG,
         family="Thai-NM-guideline", claim_classes="amyloid-imaging;biomarkers",
         supersession_note="none verified"),
    dict(pmcid="PMC12893748", organization="Swiss Society for Neuroradiology",
         document_type="clinical practice recommendations (neuroimaging)", tier=TIER_CPG,
         family="Swiss-neuroradiology", claim_classes="amyloid-imaging;differential-diagnosis",
         supersession_note="none verified"),
    # national / society dementia diagnosis & management guidelines (differential dx, care)
    dict(pmcid="PMC11564805", organization="Italian intersocietal (ISS)",
         document_type="national clinical practice guideline", tier=TIER_CPG,
         family="Italian-dementia-guideline", claim_classes="diagnosis;differential-diagnosis;pharmacotherapy",
         supersession_note="none verified"),
    dict(pmcid="PMC11227111", organization="SIGN (Scotland)",
         document_type="national clinical guideline (summary)", tier=TIER_CPG,
         family="SIGN-dementia", claim_classes="diagnosis;differential-diagnosis;care",
         supersession_note="summary of a national guideline"),
    dict(pmcid="PMC13016819", organization="British Geriatrics Society / partners",
         document_type="best-practice guideline (cognitive disorders)", tier=TIER_CPG,
         family="cognitive-disorders-BPG", claim_classes="diagnosis;differential-diagnosis",
         supersession_note="none verified"),
    dict(pmcid="PMC11772713", organization="Canadian Stroke consortium",
         document_type="best-practice recommendations (vascular cognitive impairment)", tier=TIER_CPG,
         family="Canadian-stroke-VCI", claim_classes="differential-diagnosis;vascular",
         supersession_note="7th edition"),
    dict(pmcid="PMC10587099", organization="European geriatric medicine society",
         document_type="clinical practice guideline (physical activity)", tier=TIER_CPG,
         family="EuGMS-activity", claim_classes="management;prevention",
         supersession_note="none verified"),
    dict(pmcid="PMC10072340", organization="international expert group",
         document_type="practice recommendations (hearing/vision in dementia)", tier=TIER_CONSENSUS,
         family="hearing-vision-dementia", claim_classes="differential-diagnosis;comorbidity;care",
         supersession_note="none verified"),
]

# Named authoritative families NOT found in the OA corpus. Recorded as targets for
# manual acquisition with licensing verification -- NOT fabricated. No dates,
# versions, or licence terms are asserted for these.
EXTERNAL_CPG_TARGETS: list[dict] = [
    dict(organization="American Academy of Neurology (AAN)",
         document_type="clinical practice guideline / practice parameter",
         claim_classes="diagnosis;MCI;differential-diagnosis",
         status="manual-acquisition-required",
         note="0 AD guideline records found in the OA corpus; acquire from AAN; verify licence."),
    dict(organization="NICE (UK)",
         document_type="national clinical guideline (e.g. dementia NG-series)",
         claim_classes="diagnosis;management;pharmacotherapy",
         status="manual-acquisition-required",
         note="0 records in the OA corpus; NICE reuse terms must be verified before ingestion."),
    dict(organization="International Working Group (IWG)",
         document_type="diagnostic criteria (alternative framework)",
         claim_classes="criteria;biomarkers",
         status="manual-acquisition-required",
         note="IWG criteria not confirmed present in the OA corpus; acquire from source; verify version."),
    dict(organization="European Academy of Neurology (EAN) / EU consortia (AD-specific)",
         document_type="guideline / consensus statement",
         claim_classes="diagnosis;biomarkers;imaging",
         status="manual-acquisition-required",
         note="Only a non-AD-central EAN joint guideline found; AD-specific EAN guidance to acquire."),
]
COCHRANE_PMCID = "PMC13082890"
COCHRANE_OA_URL = "https://pmc-oa-opendata.s3.amazonaws.com/PMC13082890.1/PMC13082890.1.xml"

# Guideline-class publication types (PubMed) that seed the CPG-AD layer.
CPG_PUB_TYPES = {"Practice Guideline", "Guideline", "Consensus Development Conference",
                 "Consensus Development Conference, NIH"}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_manifest(path: Path) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    with path.open(newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            rows[r["pmcid"]] = r
    return rows


def load_pubmed(path: Path) -> dict[str, dict]:
    """Index PubMed rows by PMID. Path may be the working-tree CSV or, on a
    machine without git-lfs smudge, the resolved LFS object."""
    rows: dict[str, dict] = {}
    with path.open(newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            pmid = (r.get("pmid") or "").strip()
            if pmid:
                rows[pmid] = r
    return rows


def load_articles(path: Path) -> dict[str, dict]:
    """Index parsed records by pmcid. Present in full only on the machine that
    holds the parsed corpus; here it is the local sample."""
    out: dict[str, dict] = {}
    if not path.exists():
        return out
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            out[rec["pmcid"]] = rec
    return out


def load_report_groups(path: Path, groups: Iterable[str]) -> dict[str, set[str]]:
    """Extract the full PMCID list of each named QC group from the report."""
    want = set(groups)
    out: dict[str, set[str]] = {g: set() for g in want}
    if not path.exists():
        return out
    text = path.read_text(encoding="utf-8")
    for g in want:
        marker = f"## Group: `{g}`"
        if marker not in text:
            continue
        section = text.split(marker, 1)[1].split("## Group:", 1)[0]
        for line in section.splitlines():
            if line.startswith("| PMC"):
                out[g].add(line.strip("|").split("|")[0].strip())
    return out


# ---------------------------------------------------------------------------
# Date helpers
# ---------------------------------------------------------------------------
def precision_of(value: str) -> str:
    v = (value or "").strip()
    if len(v) == 10 and v[4] == "-" and v[7] == "-":
        return "day"
    if len(v) == 7 and v[4] == "-":
        return "month"
    if len(v) == 4 and v.isdigit():
        return "year"
    return ""


def canonical_from_dates_all(dates_all: dict[str, str]) -> tuple[str, str, str]:
    """Pick the canonical date from a JATS pub-date map by the trusted
    precedence (electronic first), excluding preprint. Returns
    (date, precision, jats_type) or ("","","")."""
    order = REAL_PUB_TYPES + sorted(k for k in dates_all if k not in REAL_PUB_TYPES
                                    and k != "preprint")
    for t in order:
        d = (dates_all.get(t) or "").strip()
        if d:
            return d, precision_of(d), t
    return "", "", ""


def month_key(value: str) -> str:
    """The YYYY-MM prefix if the value has month-or-finer precision, else ''."""
    p = precision_of(value)
    if p in ("day", "month"):
        return value[:7]
    return ""


def split_side(month: str) -> str:
    return "pre" if month < JUNE_2024 else "post"


def recover_month(candidates: Iterable[str]) -> str:
    """First month-precision-or-finer 2024 date among candidates, for placing a
    year-only-2024 record. Never fabricates: only returns a real recovered
    YYYY-MM that already exists in an authoritative source."""
    for c in candidates:
        mk = month_key(c)
        if mk.startswith("2024-"):
            return mk
    return ""


# ---------------------------------------------------------------------------
# M3 -- canonical dates
# ---------------------------------------------------------------------------
def m3_row(pmcid: str, pmid: str, article: dict | None, pm: dict | None) -> dict:
    """One canonical-date overlay row. JATS-primary when the parsed record is
    available; PubMed fallback otherwise. Precision is never upgraded."""
    date = precision = jtype = ""
    source = ""
    jats_dates: dict[str, str] = {}
    if article:
        jats_dates = article.get("publication_dates_all") or {}
        date, precision, jtype = canonical_from_dates_all(jats_dates)
        if date:
            source = f"jats:{jtype}"
    pubmed_date = (pm.get("publication_date") if pm else "") or ""
    if not date and pubmed_date:
        date, precision, source = pubmed_date, precision_of(pubmed_date), "pubmed"

    recovered = recovered_source = ""
    if not date:
        side = "unknown"
    elif precision in ("day", "month"):
        side = split_side(date[:7])
    else:  # year precision
        year = date[:4]
        if year <= "2023":
            side = "pre"
        elif year >= "2025":
            side = "post"
        else:  # exactly 2024, year-only -> try to recover a real month
            alt = list(jats_dates.values()) + ([pubmed_date] if pubmed_date else [])
            recovered = recover_month(alt)
            if recovered:
                recovered_source = "jats" if any(month_key(v) == recovered
                                                 for v in jats_dates.values()) else "pubmed"
                side = split_side(recovered)
            else:
                side = "unknown"
    return {
        "pmcid": pmcid, "pmid": pmid,
        "canonical_date": date, "canonical_date_precision": precision,
        "canonical_date_type": jtype, "date_source": source,
        "recovered_month": recovered, "recovered_month_source": recovered_source,
        "split_june_2024": side,
    }


# ---------------------------------------------------------------------------
# M4 -- corpus policy
# ---------------------------------------------------------------------------
LICENSE_BANDS = {
    "CC BY": "open", "CC0": "open",
    "CC BY-NC": "cc-restrictive", "CC BY-NC-ND": "cc-restrictive",
    "CC BY-NC-SA": "cc-restrictive", "CC BY-ND": "cc-restrictive", "CC BY-SA": "cc-restrictive",
    "TDM": "tdm",
}


def license_band(code: str) -> str:
    code = (code or "").strip()
    if not code:
        return "none"
    return LICENSE_BANDS.get(code, "other")


def pub_types_of(pm: dict | None) -> set[str]:
    if not pm:
        return set()
    return {t.strip() for t in (pm.get("publication_types") or "").split(";") if t.strip()}


def m4_row(pmcid: str, pmid: str, mf: dict | None, pm: dict | None, *,
           in_fulltext: bool, is_currency: bool, stub: bool, no_body: bool,
           doi_disagree: bool, license_disagree: bool) -> dict:
    types = pub_types_of(pm)
    code = (mf.get("license_code") if mf else "") or ""
    retracted = bool(mf and (mf.get("is_retracted") == "yes"))
    manuscript = bool(mf and (mf.get("is_manuscript") == "yes"))

    if is_currency and pmcid == COCHRANE_PMCID:
        source_category = "currency-pack"
    elif in_fulltext:
        source_category = "pmc-fulltext"
    else:
        source_category = "pubmed-abstract"

    flags: list[str] = []
    if is_currency:
        flags.append("currency-pack")
    if retracted:
        flags.append("retracted")
    if manuscript:
        flags.append("author-manuscript")
    if "Preprint" in types:
        flags.append("preprint")
    if "Published Erratum" in types:
        flags.append("erratum")
    if "Retraction of Publication" in types:
        flags.append("retraction-notice")
    if "Expression of Concern" in types:
        flags.append("expression-of-concern")
    if types & {"Editorial", "Comment", "Letter", "News"}:
        flags.append("opinion")
    if types & CPG_PUB_TYPES:
        flags.append("guideline")
    if not code and source_category != "pubmed-abstract":
        flags.append("no-license")
    if doi_disagree:
        flags.append("doi-disagreement")
    if license_disagree:
        flags.append("license-disagreement")
    if stub:
        flags.append("stub")
    if no_body:
        flags.append("no-body")

    # eligibility_status (5-tier). Deletion never happens; this is a flag.
    if "erratum" in flags or "retraction-notice" in flags:
        status, reason = "excluded", "correction/retraction notice; metadata, not standalone evidence"
    elif doi_disagree or license_disagree or ("no-license" in flags):
        status, reason = "manual-review", "identifier/licence needs adjudication before release"
    else:
        status, reason = "eligible", "admissible; retrieval-time gates (currency/authority) apply"
    if is_currency:
        status = "externally-added" if pmcid == COCHRANE_PMCID else "eligible"
        reason = "currency-pack anchor" if status == "externally-added" else \
                 "currency-pack anchor (also in raw OA corpus)"

    # full-text usable? stubs / body-less are abstract-only. The externally
    # ingested Cochrane record carries full text even though it is not in the
    # downloaded OA corpus (manifest status is "failed").
    has_fulltext = in_fulltext or (is_currency and pmcid == COCHRANE_PMCID)
    fulltext_eligible = has_fulltext and not (stub or no_body)

    return {
        "pmcid": pmcid, "pmid": pmid,
        "source_category": source_category,
        "eligibility_status": status,
        "fulltext_eligible": "yes" if fulltext_eligible else "no",
        "retracted": "yes" if retracted else "no",
        "is_manuscript": "yes" if manuscript else "no",
        "license_code": code,
        "license_band": license_band(code),
        "flags": ";".join(flags),
        "eligibility_reason": reason,
    }


# ---------------------------------------------------------------------------
# M1 -- currency pack ingestion
# ---------------------------------------------------------------------------
def sha_md5(path: Path) -> str:
    h = hashlib.md5()
    h.update(path.read_bytes())
    return h.hexdigest()


def ingest_cochrane(currency_dir: Path, *, allow_fetch: bool) -> dict:
    """Ensure the Cochrane OA XML is staged, parse it, record provenance.
    Uses an already-staged file if present; only fetches when permitted."""
    xml_dir = currency_dir / "xml"
    parsed_dir = currency_dir / "parsed"
    xml_dir.mkdir(parents=True, exist_ok=True)
    parsed_dir.mkdir(parents=True, exist_ok=True)
    xml_path = xml_dir / f"{COCHRANE_PMCID}.xml"

    fetched_at = ""
    if not xml_path.exists():
        if not allow_fetch:
            return {"pmcid": COCHRANE_PMCID, "staged": False,
                    "note": "not staged and fetch disabled"}
        with urllib.request.urlopen(COCHRANE_OA_URL, timeout=120) as resp:
            xml_path.write_bytes(resp.read())
        fetched_at = datetime.now(timezone.utc).isoformat()

    observed = sha_md5(xml_path)
    # Parse with the project parser so the record matches the corpus schema.
    sys.path.insert(0, str(REPO / "pmc"))
    import parse_pmc_xml as pp  # noqa: E402
    record = pp.parse_article(xml_path, {
        "pmcid": COCHRANE_PMCID, "pmid": "41985900",
        "doi": "10.1002/14651858.CD016297", "license_code": "CC BY-NC",
        "is_retracted": "no", "is_manuscript": "no", "version": "1",
    }, 250)
    (parsed_dir / f"{COCHRANE_PMCID}.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8",
        newline="\n")

    provenance = {
        "pmcid": COCHRANE_PMCID, "doi": "10.1002/14651858.CD016297",
        "source_url": COCHRANE_OA_URL, "acquisition": "manual-oa-fetch",
        "observed_md5": observed,
        "fetched_at_utc": fetched_at or "pre-staged",
        "inventory_expected_md5_stale": "b20dcfe941aba9ea27b37557ffd485f8",
        "note": "Object is revised in place; distinct MD5s observed across attempts "
                "(inventory expected, reconciled, and this fetch all differ). Pinned "
                "by observed MD5. Licence CC BY-NC: local text-mining permitted, "
                "redistribution non-commercial.",
        "parsed_ok": record["qc"]["status"] != "parse_error",
        "body_word_count": record["body_word_count"],
        "reference_count": record["reference_count"],
        "canonical_date": record["publication_date"],
        "license_code_xml": record["provenance"].get("license_code_xml", ""),
    }
    (currency_dir / f"{COCHRANE_PMCID}.provenance.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8",
        newline="\n")
    return {"pmcid": COCHRANE_PMCID, "staged": True, "observed_md5": observed,
            "provenance": provenance, "record": record}


def build_currency_pack(manifest, pubmed, articles, cochrane_info) -> list[dict]:
    rows = []
    for a in CURRENCY_ANCHORS:
        pmcid = a["pmcid"]
        mf = manifest.get(pmcid, {})
        pm = pubmed.get(a["pmid"], {})
        art = articles.get(pmcid)
        if pmcid == COCHRANE_PMCID and cochrane_info.get("record"):
            art = cochrane_info["record"]
        m3 = m3_row(pmcid, a["pmid"], art, pm)
        in_raw = bool(mf) and mf.get("status") == "ok"
        full_text = in_raw or (pmcid == COCHRANE_PMCID and cochrane_info.get("staged"))
        rows.append({
            "pmcid": pmcid, "pmid": a["pmid"], "doi": a["doi"],
            "title": (mf.get("title") or (pm.get("title") or "")).strip()[:160],
            "journal": (pm.get("journal") or "").strip(),
            "canonical_date": m3["canonical_date"],
            "date_precision": m3["canonical_date_precision"],
            "license_code": (mf.get("license_code") or "CC BY-NC").strip(),
            "acquisition": a["acquisition"],
            "in_raw_corpus": "yes" if in_raw else "no",
            "full_text_available": "yes" if full_text else "no",
            "observed_md5": cochrane_info.get("observed_md5", "") if pmcid == COCHRANE_PMCID else "",
            "in_cpg_layer": "yes" if pmcid in CURRENCY_ALSO_CPG else "no",
            "claim_class": a["claim_class"],
            "rationale": a["rationale"],
        })
    return rows


# ---------------------------------------------------------------------------
# M2 -- curated Alzheimer/dementia CPG/guidance layer (enriched schema).
# ---------------------------------------------------------------------------
def build_cpg_layer(manifest, pubmed, articles) -> list[dict]:
    """Materialize the curated CPG layer. Every record is verified in the OA
    corpus; curation fields come from CURATED_CPG, factual fields from the
    manifest/PubMed/M3. Nothing is fabricated. Records not found in the manifest
    are reported (they should all be present)."""
    rows = []
    for c in CURATED_CPG:
        pmcid = c["pmcid"]
        mf = manifest.get(pmcid)
        if not mf:
            rows.append({"pmcid": pmcid, "status": "MISSING-from-manifest"})
            continue
        pm = pubmed.get(mf.get("pmid", ""), {})
        m3 = m3_row(pmcid, mf.get("pmid", ""), articles.get(pmcid), pm)
        code = (mf.get("license_code") or "").strip()
        rows.append({
            "cpg_id": pmcid, "pmcid": pmcid, "pmid": mf.get("pmid", ""),
            "doi": mf.get("doi", ""),
            "title": (mf.get("title") or "").strip()[:160],
            "organization": c["organization"],
            "document_type": c["document_type"],
            "source_category": "cpg",
            "guideline_family": c["family"],
            "version_note": "as-published",
            "issue_date": m3["canonical_date"],
            "date_precision": m3["canonical_date_precision"],
            "date_source": m3["date_source"],
            "supersession_note": c["supersession_note"],
            "authority_tier_label": c["tier"],
            "claim_classes": c["claim_classes"],
            "license_code": code,
            "license_source": "PMC manifest (from inventory)",
            "acquisition": "pmc-oa",
            "acquisition_timestamp": mf.get("downloaded_at_utc", ""),
            "content_hash_md5": mf.get("actual_md5", ""),
            "in_currency_pack": "yes" if pmcid in {a["pmcid"] for a in CURRENCY_ANCHORS} else "no",
            "eligibility_status": "eligible",
        })
    return rows


def build_cpg_external_targets() -> list[dict]:
    """External authoritative families not in the OA corpus. Recorded as
    acquisition targets with uncertainty; no dates/versions/licences asserted."""
    return [dict(cpg_id=f"EXTERNAL-{i+1}", **t) for i, t in enumerate(EXTERNAL_CPG_TARGETS)]


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def build(manifest_path, pubmed_path, articles_path, report_path, out_dir,
          currency_dir, allow_fetch) -> dict:
    manifest = load_manifest(manifest_path)
    pubmed = load_pubmed(pubmed_path)
    articles = load_articles(articles_path)
    groups = load_report_groups(report_path, ["stub", "no_body", "license_disagreement"])
    stubs, no_bodies = groups["stub"], groups["no_body"]
    license_dis = groups["license_disagreement"]

    # DOI disagreements: manifest DOI vs PubMed DOI (both present, normalised).
    def norm(d):
        return (d or "").strip().lower().removeprefix("https://doi.org/")
    doi_dis = set()
    for pmcid, mf in manifest.items():
        pm = pubmed.get(mf.get("pmid", ""), {})
        a, b = norm(mf.get("doi", "")), norm(pm.get("doi", ""))
        if a and b and a != b:
            doi_dis.add(pmcid)

    # M1
    cochrane = ingest_cochrane(currency_dir, allow_fetch=allow_fetch)
    currency_rows = build_currency_pack(manifest, pubmed, articles, cochrane)
    currency_pmcids = {r["pmcid"] for r in currency_rows}

    # Which PMC records reached full text (downloaded ok).
    fulltext_pmcids = {p for p, m in manifest.items() if m.get("status") == "ok"}

    # PMID -> pmcid for the PMC layer (to skip abstract-layer duplication).
    pmc_pmids = {m.get("pmid", "") for m in manifest.values() if m.get("pmid")}

    # M3 + M4 over the PMC layer (manifest) and the PubMed abstract layer.
    m3_rows: list[dict] = []
    m4_rows: list[dict] = []

    for pmcid, mf in manifest.items():
        pmid = mf.get("pmid", "")
        pm = pubmed.get(pmid, {})
        art = articles.get(pmcid)
        if pmcid == COCHRANE_PMCID and cochrane.get("record"):
            art = cochrane["record"]
        m3_rows.append(m3_row(pmcid, pmid, art, pm))
        m4_rows.append(m4_row(
            pmcid, pmid, mf, pm,
            in_fulltext=(pmcid in fulltext_pmcids),
            is_currency=(pmcid in currency_pmcids),
            stub=(pmcid in stubs), no_body=(pmcid in no_bodies),
            doi_disagree=(pmcid in doi_dis),
            license_disagree=(pmcid in license_dis),
        ))

    # PubMed abstract layer: records with no PMC full text.
    for pmid, pm in pubmed.items():
        if pmid in pmc_pmids:
            continue
        m3_rows.append(m3_row("", pmid, None, pm))
        m4_rows.append(m4_row(
            "", pmid, None, pm,
            in_fulltext=False, is_currency=False, stub=False, no_body=False,
            doi_disagree=False, license_disagree=False,
        ))

    # M2 curated CPG layer + external targets
    cpg_rows = build_cpg_layer(manifest, pubmed, articles)
    cpg_external = build_cpg_external_targets()

    # Write overlays.
    write_csv(out_dir / "canonical_dates.csv", m3_rows,
              ["pmcid", "pmid", "canonical_date", "canonical_date_precision",
               "canonical_date_type", "date_source", "recovered_month",
               "recovered_month_source", "split_june_2024"])
    write_csv(out_dir / "corpus_policy.csv", m4_rows,
              ["pmcid", "pmid", "source_category", "eligibility_status",
               "fulltext_eligible", "retracted", "is_manuscript", "license_code",
               "license_band", "flags", "eligibility_reason"])
    write_csv(out_dir / "currency_pack.csv", currency_rows,
              ["pmcid", "pmid", "doi", "title", "journal", "canonical_date",
               "date_precision", "license_code", "acquisition", "in_raw_corpus",
               "full_text_available", "observed_md5", "in_cpg_layer",
               "claim_class", "rationale"])
    write_csv(out_dir / "cpg_registry.csv", cpg_rows,
              ["cpg_id", "pmcid", "pmid", "doi", "title", "organization",
               "document_type", "source_category", "guideline_family", "version_note",
               "issue_date", "date_precision", "date_source", "supersession_note",
               "authority_tier_label", "claim_classes", "license_code", "license_source",
               "acquisition", "acquisition_timestamp", "content_hash_md5",
               "in_currency_pack", "eligibility_status"])
    write_csv(out_dir / "cpg_external_targets.csv", cpg_external,
              ["cpg_id", "organization", "document_type", "claim_classes",
               "status", "note"])

    return {
        "counts": {
            "manifest": len(manifest), "pubmed": len(pubmed),
            "articles_local": len(articles),
            "m3_rows": len(m3_rows), "m4_rows": len(m4_rows),
            "currency": len(currency_rows),
            "cpg_layer": len(cpg_rows), "cpg_external": len(cpg_external),
            "fulltext": len(fulltext_pmcids),
            "doi_disagreements": len(doi_dis),
            "license_disagreements": len(license_dis),
            "stubs": len(stubs), "no_bodies": len(no_bodies),
        },
        "cochrane": {k: v for k, v in cochrane.items() if k != "record"},
        "m3_rows": m3_rows, "m4_rows": m4_rows,
        "currency_rows": currency_rows, "cpg_rows": cpg_rows,
        "cpg_external": cpg_external,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Build M1-M4 corpus metadata overlays.")
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument("--pubmed", type=Path, default=DEFAULT_PUBMED)
    ap.add_argument("--articles", type=Path, default=DEFAULT_ARTICLES)
    ap.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--currency-dir", type=Path, default=DEFAULT_CURRENCY_DIR)
    ap.add_argument("--no-fetch", action="store_true",
                    help="Do not fetch the Cochrane OA object; use only a staged file.")
    args = ap.parse_args(argv)

    result = build(args.manifest, args.pubmed, args.articles, args.report,
                   args.out, args.currency_dir, allow_fetch=not args.no_fetch)
    c = result["counts"]
    print("M1-M4 overlays written to", args.out)
    for k, v in c.items():
        print(f"  {k:22} {v:,}")
    print("Cochrane:", json.dumps(result["cochrane"].get("observed_md5", "n/a")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
