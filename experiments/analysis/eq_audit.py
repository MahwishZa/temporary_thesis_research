"""Read-only integrity audit of the completed evidence-quality study.

Every check here answers a question an examiner could reasonably ask, and every
one is answered against the saved artifacts rather than against a claim made
about them. Nothing in this module writes, moves or deletes anything: it opens
files, compares them, and returns findings.

The checks fall into four groups:

``annotations``  the 120 labels are complete, valid, unique, and each one
                 corresponds to a real decision in the completed comparison,
                 with the question and passage the annotator actually saw
``provenance``   the sample came from the production MedCPT frozen set and the
                 scientific comparison run, not from the lexical-development
                 set that also exists in this workspace
``sampling``     the sample matches what the predeclared rule produces when it
                 is re-run from the 600 decisions -- recomputed here, not read
                 from the manifest that claims it
``anchoring``    how far the labels track the suggestion the annotator was shown

A finding is one of:

    PASS      verified against the artifact
    FAIL      verified to be wrong; the study is affected
    WARN      not wrong, but a limitation a reader must be told about
    SKIP      the artifact needed for this check is not present
"""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence

from experiments.analysis.evidence_quality import (
    VALID_LABELS,
    load_decisions,
    stratified_sample,
    suggestion_anchoring,
    tertile_edges,
)

#: The shape of the completed study, as reported. These are what the audit
#: checks the artifacts *against*; they are parameters rather than literals
#: buried in the checks so that the checks themselves can be exercised on a
#: smaller study in the tests, and so a reader can see what is being assumed.
EXPECTED_DISTRIBUTION = {2: 75, 1: 41, 0: 4}
EXPECTED_ROWS = 120
EXPECTED_POPULATION = 600
EXPECTED_QUESTIONS = 30


class Finding:
    __slots__ = ("section", "check", "status", "detail")

    def __init__(self, section: str, check: str, status: str, detail: str = "") -> None:
        self.section, self.check, self.status, self.detail = section, check, status, detail

    def as_dict(self) -> Dict[str, str]:
        return {"section": self.section, "check": self.check,
                "status": self.status, "detail": self.detail}

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<{self.status} {self.section}/{self.check}>"


def _ok(cond: bool, yes: str = "", no: str = "") -> str:
    return "PASS" if cond else "FAIL"


def sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(65536), b""):
            digest.update(block)
    return digest.hexdigest()


def label_fingerprint(rows: Sequence[Dict[str, Any]]) -> str:
    """A digest of (annotation_id, human_label) alone.

    The file digest changes if a single space moves. This changes only if a
    label changes, so it is the one to compare across a cleanup.
    """
    payload = "\n".join(f"{r.get('annotation_id')}={r.get('human_label')}"
                        for r in sorted(rows, key=lambda r: str(r.get("annotation_id"))))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


# --------------------------------------------------------------------------
def audit_annotations(sheet_rows: Sequence[Dict[str, Any]],
                      decisions: Sequence[Dict[str, Any]],
                      expected: Optional[Dict[int, int]] = None,
                      expected_rows: int = EXPECTED_ROWS) -> List[Finding]:
    """The labels themselves: complete, valid, unique, and really annotated."""
    expected = EXPECTED_DISTRIBUTION if expected is None else expected
    found: List[Finding] = []
    add = lambda c, s, d="": found.append(Finding("annotations", c, s, d))  # noqa: E731

    add(f"row count is {expected_rows}", _ok(len(sheet_rows) == expected_rows),
        f"{len(sheet_rows)} rows")

    raw = [str(r.get("human_label", "")).strip() for r in sheet_rows]
    missing = [r.get("annotation_id") for r, v in zip(sheet_rows, raw) if not v]
    invalid = [(r.get("annotation_id"), v) for r, v in zip(sheet_rows, raw)
               if v and not (v.isdigit() and int(v) in VALID_LABELS)]
    add("no missing labels", _ok(not missing), f"{len(missing)} missing")
    add("no invalid labels", _ok(not invalid), f"{len(invalid)} invalid: {invalid[:5]}")

    labels = [int(v) for v in raw if v.isdigit() and int(v) in VALID_LABELS]
    distribution = dict(sorted(Counter(labels).items()))
    add("label distribution matches the reported "
        + "/".join(str(expected[k]) for k in sorted(expected, reverse=True)),
        _ok(distribution == dict(sorted(expected.items()))),
        f"found {distribution}, expected {dict(sorted(expected.items()))}")

    ids = Counter(str(r.get("annotation_id")) for r in sheet_rows)
    duplicates = [i for i, n in ids.items() if n > 1]
    add("annotation ids are unique", _ok(not duplicates), f"duplicates: {duplicates[:5]}")

    # Every annotated row must be a real (question, candidate) decision.
    real = {(d["qid"], d["chunk_id"]) for d in decisions}
    by_pair = {(d["qid"], d["chunk_id"]): d for d in decisions}
    unreal = [r.get("annotation_id") for r in sheet_rows
              if (r.get("qid"), r.get("chunk_id")) not in real]
    add("every row is a real evaluation decision", _ok(not unreal),
        f"{len(unreal)} row(s) match no decision: {unreal[:5]}")

    altered = [r.get("annotation_id") for r in sheet_rows
               if (r.get("qid"), r.get("chunk_id")) in by_pair
               and str(r.get("question", "")).strip()
               != str(by_pair[(r["qid"], r["chunk_id"])].get("question", "")).strip()]
    add("question text is unchanged from the comparison", _ok(not altered),
        f"{len(altered)} row(s) differ: {altered[:5]}")

    empty = [r.get("annotation_id") for r in sheet_rows
             if not str(r.get("candidate_text", "")).strip()]
    add("every row carries passage text", _ok(not empty),
        f"{len(empty)} row(s) have no candidate_text")

    # The training questions are alz-train-*; the evaluation questions are not.
    training = [r.get("annotation_id") for r in sheet_rows
                if str(r.get("qid", "")).startswith("alz-train")]
    add("no filter-training examples in the sample", _ok(not training),
        f"{len(training)} training row(s): {training[:5]}")

    # human_label must not be a copy of the suggestion on every row.
    both = [(int(str(r["human_label"]).strip()), int(str(r["ai_suggested_label"]).strip()))
            for r in sheet_rows
            if str(r.get("human_label", "")).strip().isdigit()
            and str(r.get("ai_suggested_label", "")).strip().isdigit()]
    if not both:
        add("human_label is not a copy of ai_suggested_label", "SKIP",
            "no row records both fields")
    elif all(h == a for h, a in both):
        add("human_label is not a copy of ai_suggested_label", "FAIL",
            f"all {len(both)} rows are identical to the suggestion; this is what an "
            "automatic copy looks like and must be ruled out by hand")
    else:
        differ = sum(1 for h, a in both if h != a)
        add("human_label is not a copy of ai_suggested_label", "PASS",
            f"{differ} of {len(both)} rows differ from the suggestion")

    return found


def audit_provenance(manifest: Dict[str, Any], comparison_manifest: Dict[str, Any],
                     frozen_meta: Optional[Dict[str, Any]]) -> List[Finding]:
    """Which retrieval and which comparison run this sample actually came from."""
    found: List[Finding] = []
    add = lambda c, s, d="": found.append(Finding("provenance", c, s, d))  # noqa: E731

    sample_digest = manifest.get("frozen_set_digest")
    run_digest = comparison_manifest.get("frozen_set_digest")
    add("sample was built from the scientific run's frozen set",
        _ok(bool(sample_digest) and sample_digest == run_digest),
        f"sample={sample_digest}  scientific_run={run_digest}")

    is_medcpt = manifest.get("retrieval_is_medcpt")
    if is_medcpt is None:
        add("retrieval is production MedCPT", "SKIP",
            "sample_manifest.json records no retrieval_is_medcpt")
    else:
        add("retrieval is production MedCPT", _ok(is_medcpt is True),
            f"retrieval_is_medcpt={is_medcpt}"
            + ("" if is_medcpt else "  <- this is the lexical-development set"))

    source = os.path.basename(os.path.dirname(str(manifest.get("source_per_question", ""))))
    add("sample was built from comparison_scientific/",
        _ok(source == "comparison_scientific"),
        f"source_per_question is under '{source or '(unrecorded)'}'")

    if frozen_meta is None:
        add("local frozen file matches the scientific run", "SKIP",
            "frozen_candidates.meta.json not present next to the frozen file")
    else:
        local = frozen_meta.get("frozen_set_digest")
        provenance = frozen_meta.get("provenance") or {}
        status = "PASS" if local == run_digest else "WARN"
        add("local frozen file matches the scientific run", status,
            f"local={local} ({provenance.get('source', 'unrecorded')}); "
            f"scientific_run={run_digest}"
            + ("" if status == "PASS" else
               "  <- the frozen file on this disk is NOT the one the comparison ran "
               "against; the sample is judged by the manifest digest above, not by "
               "this file, but re-running export here would use the wrong text"))
    return found


def audit_sampling(manifest: Dict[str, Any], sheet_rows: Sequence[Dict[str, Any]],
                   decisions: Sequence[Dict[str, Any]],
                   expected_rows: int = EXPECTED_ROWS,
                   expected_population: int = EXPECTED_POPULATION,
                   expected_questions: int = EXPECTED_QUESTIONS) -> List[Finding]:
    """Re-run the predeclared rule and compare, rather than trusting the manifest."""
    found: List[Finding] = []
    add = lambda c, s, d="": found.append(Finding("sampling", c, s, d))  # noqa: E731

    add(f"population is the {expected_population} evaluation decisions",
        _ok(manifest.get("population") == expected_population
            and len(decisions) == expected_population),
        f"manifest={manifest.get('population')}  decisions on disk={len(decisions)}")
    add(f"sample size is {expected_rows}",
        _ok(manifest.get("sampled") == expected_rows),
        f"manifest={manifest.get('sampled')}")

    forced = sum(1 for d in decisions if d["rag2_admitted"])
    add("every RAG2 admission was forced into the sample",
        _ok(manifest.get("forced_rag2_admitted") == forced),
        f"manifest={manifest.get('forced_rag2_admitted')}  admissions in population={forced}")

    # Tertile edges must come from all 600, not from the sample.
    for arm in ("scaf", "rag2"):
        recomputed = [round(e, 6) for e in tertile_edges([d[f"{arm}_score"] for d in decisions])]
        recorded = [round(float(e), 6) for e in (manifest.get(f"{arm}_tertile_edges") or [])]
        add(f"{arm} tertile edges recompute from all {expected_population} decisions",
            _ok(recomputed == recorded), f"recomputed={recomputed}  recorded={recorded}")

    # The whole sample is deterministic: re-run it and compare row for row.
    seed = manifest.get("seed", 42)
    rebuilt, _ = stratified_sample([dict(d) for d in decisions],
                                   size=int(manifest.get("requested_size", 120)),
                                   seed=int(seed))
    rebuilt_pairs = {r["annotation_id"]: (r["qid"], r["chunk_id"]) for r in rebuilt}
    sheet_pairs = {str(r.get("annotation_id")): (r.get("qid"), r.get("chunk_id"))
                   for r in sheet_rows}
    add("the sample reproduces exactly from the predeclared rule and seed",
        _ok(rebuilt_pairs == sheet_pairs),
        f"{sum(1 for k, v in sheet_pairs.items() if rebuilt_pairs.get(k) != v)} "
        f"row(s) of {len(sheet_pairs)} differ from a re-run with seed {seed}")

    per_question = Counter(str(r.get("qid")) for r in sheet_rows)
    add("sample spans the question set",
        "WARN" if len(per_question) < expected_questions else "PASS",
        f"{len(per_question)} of {expected_questions} questions represented; "
        f"up to {max(per_question.values()) if per_question else 0} passages come "
        f"from a single question, so rows are not independent observations")

    add("forcing all RAG2 admissions makes the sample non-self-weighting", "WARN",
        f"{forced} of {len(sheet_rows)} rows were included by rule rather than by "
        f"chance. Those {forced} are a complete census of RAG2's admissions, not a "
        "sample of them, so the RAG2 admitted/rejected contrast is descriptive of "
        f"exactly those {forced} passages, and the sample is not a simple random "
        f"sample of the {len(decisions)}")
    return found


def audit_anchoring(sheet_rows: Sequence[Dict[str, Any]]) -> List[Finding]:
    """The suggestion the annotator saw, against the label they chose."""
    found: List[Finding] = []
    add = lambda c, s, d="": found.append(Finding("anchoring", c, s, d))  # noqa: E731

    labelled = [r for r in sheet_rows
                if str(r.get("human_label", "")).strip().isdigit()
                and int(str(r["human_label"]).strip()) in VALID_LABELS]
    human = [int(str(r["human_label"]).strip()) for r in labelled]
    stats = suggestion_anchoring(labelled, human)

    if not stats.get("rows_with_a_recorded_suggestion"):
        add("anchoring can be measured", "SKIP", str(stats.get("note", "")))
        return found

    shown = (stats.get("shown") or {}).get("n", 0)
    hidden = (stats.get("hidden") or {}).get("n", 0)
    unknown = (stats.get("unknown") or {}).get("n", 0)
    add("suggestion visibility is recorded per row",
        "PASS" if not unknown else "WARN",
        f"shown={shown}  hidden={hidden}  unknown={unknown}")

    add("an unanchored subset exists for comparison",
        "PASS" if hidden >= 10 else "WARN",
        f"{hidden} row(s) were annotated with the suggestion hidden. Without such "
        "rows there is no within-study estimate of how much the suggestion moved "
        "the labels")

    overall = stats.get("all") or {}
    rate = overall.get("agreement_rate")
    if rate is None:
        add("labels are distinguishable from the suggestion", "SKIP")
    else:
        status = "PASS" if rate < 0.85 else ("WARN" if rate < 0.95 else "FAIL")
        add("labels are distinguishable from the suggestion", status,
            f"agreement {rate} over n={overall.get('n')}; "
            f"human above suggestion on {overall.get('human_above_suggestion')} rows, "
            f"below on {overall.get('human_below_suggestion')}. The suggestion is "
            "lexical and SCAF's sigma is lexical, so agreement inflates the "
            "sigma-versus-human correlation")
    found.append(Finding("anchoring", "_stats", "INFO", json.dumps(stats, sort_keys=True)))
    return found


# --------------------------------------------------------------------------
def run_audit(sheet_path: str, per_question_path: str, manifest_path: str,
              comparison_manifest_path: str, key_path: str = "",
              frozen_meta_path: str = "",
              expected: Optional[Dict[int, int]] = None,
              expected_rows: int = EXPECTED_ROWS,
              expected_population: int = EXPECTED_POPULATION,
              expected_questions: int = EXPECTED_QUESTIONS) -> Dict[str, Any]:
    """Every check, against files only. Returns findings plus integrity digests."""
    sheet_rows = _read_jsonl(sheet_path)
    decisions = load_decisions(per_question_path)
    with open(manifest_path, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    with open(comparison_manifest_path, "r", encoding="utf-8") as handle:
        comparison_manifest = json.load(handle)

    frozen_meta = None
    if frozen_meta_path and os.path.isfile(frozen_meta_path):
        with open(frozen_meta_path, "r", encoding="utf-8") as handle:
            frozen_meta = json.load(handle)

    findings: List[Finding] = []
    findings += audit_annotations(sheet_rows, decisions, expected, expected_rows)
    findings += audit_provenance(manifest, comparison_manifest, frozen_meta)
    findings += audit_sampling(manifest, sheet_rows, decisions, expected_rows,
                               expected_population, expected_questions)
    findings += audit_anchoring(sheet_rows)

    if key_path and os.path.isfile(key_path):
        key_ids = {str(r.get("annotation_id")) for r in _read_jsonl(key_path)}
        sheet_ids = {str(r.get("annotation_id")) for r in sheet_rows}
        findings.append(Finding(
            "annotations", "every label joins back to a machine score",
            _ok(sheet_ids <= key_ids),
            f"{len(sheet_ids - key_ids)} sheet id(s) absent from the key"))

    counts = Counter(f.status for f in findings if f.status != "INFO")
    return {
        "findings": [f.as_dict() for f in findings],
        "counts": dict(sorted(counts.items())),
        "verdict": ("FAIL" if counts.get("FAIL") else
                    "PASS WITH LIMITATIONS" if counts.get("WARN") else "PASS"),
        "integrity": {
            "annotation_sheet_sha256": sha256(sheet_path),
            "label_fingerprint": label_fingerprint(sheet_rows),
            "rows": len(sheet_rows),
            "label_distribution": dict(sorted(Counter(
                int(str(r["human_label"]).strip()) for r in sheet_rows
                if str(r.get("human_label", "")).strip().isdigit()
                and int(str(r["human_label"]).strip()) in VALID_LABELS).items())),
        },
    }
