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

#: The shape of the study's SAMPLE. These are properties of the sampling design,
#: so they hold for every annotation pass over that sample: the same 120 rows
#: drawn from the same 600 decisions across the same 30 questions.
EXPECTED_ROWS = 120
EXPECTED_POPULATION = 600
EXPECTED_QUESTIONS = 30

#: The label distribution of the RETIRED ANCHORED PILOT, and of nothing else.
#:
#: A label distribution is not a property of the sample -- it is the outcome of
#: one particular annotation pass. Two independent human passes over identical
#: passages will produce different distributions, and requiring the second to
#: reproduce the first would be requiring it not to be independent. Treating this
#: as a global expectation was a real bug: it failed the corrected human-only
#: pass ({0: 14, 1: 46, 2: 60}) for the sole reason that it disagreed with the
#: pilot it was created to replace.
#:
#: It survives only so the preserved pilot can still be checked against its own
#: recorded outcome. Pass it explicitly, or let ``run_audit`` read the pilot's
#: own ``pilot_integrity.json``; never apply it to a corrected pass.
PILOT_DISTRIBUTION = {2: 75, 1: 41, 0: 4}

#: Rows written by a corrected human-only pass carry this in ``annotation_pass``.
CORRECTED_PASS_LABEL = "corrected-human-only-v1"


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
                      expected_rows: int = EXPECTED_ROWS,
                      independent: bool = False) -> List[Finding]:
    """The labels themselves: complete, valid, unique, and really annotated.

    ``expected`` is the label distribution this pass is required to reproduce.
    Pass it only when auditing a pass whose outcome is already on record -- the
    retired pilot against its own ``pilot_integrity.json``. Leave it ``None`` for
    an independent pass: its distribution is the result being collected, not a
    target to hit, and checking it against another pass's outcome would test
    agreement rather than validity.
    """
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
    if expected is None and independent:
        add("label distribution is recorded (independent pass -- no target)", "PASS",
            f"{distribution}. This is the outcome of an independent human pass, "
            "not a value it was required to produce. It is NOT compared against "
            "the retired anchored pilot's distribution, and it is not expected "
            "to match it: two independent readings of the same passages "
            "legitimately differ, and requiring agreement would defeat the "
            "purpose of re-annotating.")
    elif expected is None:
        add("label distribution is recorded (no outcome on record)", "PASS",
            f"{distribution}. No recorded outcome was supplied for this pass, so "
            "the distribution is reported rather than checked.")
    else:
        add("label distribution matches the recorded outcome for this pass "
            + "/".join(str(expected[k]) for k in sorted(expected, reverse=True)),
            _ok(distribution == dict(sorted(expected.items()))),
            f"found {distribution}, on record {dict(sorted(expected.items()))}")

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

    # human_label must not be a copy of the suggestion. Agreement alone is not
    # the fault -- a genuine judgement may legitimately match a suggestion, and
    # on an easy row usually will. What is fatal is TOTAL agreement combined
    # with TOTAL visibility, because then nothing distinguishes judgement from
    # transcription and no unanchored subset exists to calibrate against.
    both = [(int(str(r["human_label"]).strip()),
             int(str(r["ai_suggested_label"]).strip()),
             r.get("ai_suggestion_shown"))
            for r in sheet_rows
            if str(r.get("human_label", "")).strip().isdigit()
            and str(r.get("ai_suggested_label", "")).strip().isdigit()]
    if not both:
        add("human_label is not a copy of ai_suggested_label", "PASS",
            "no row carries a machine suggestion at all, so no label can be a "
            "copy of one")
    else:
        differ = sum(1 for h, a, _ in both if h != a)
        unshown = sum(1 for _, _, shown in both if shown is False)
        if differ == 0 and unshown == 0:
            add("human_label is not a copy of ai_suggested_label", "FAIL",
                f"all {len(both)} rows match the suggestion AND all {len(both)} were "
                "shown it. Perfect agreement with an always-visible suggestion "
                "records agreement with the rule, not an independent judgement, and "
                "no unanchored rows exist to estimate the effect from")
        elif differ == 0:
            add("human_label is not a copy of ai_suggested_label", "WARN",
                f"all {len(both)} rows match the suggestion, but {unshown} were "
                "annotated without seeing it; agreement on hidden rows is evidence "
                "the rule is right, not that the labels were copied")
        else:
            add("human_label is not a copy of ai_suggested_label", "PASS",
                f"{differ} of {len(both)} rows differ from the suggestion")

    return found


def audit_corrected_pass(corrected_rows: Sequence[Dict[str, Any]],
                         original_rows: Sequence[Dict[str, Any]]) -> List[Finding]:
    """The corrected pass covers the pilot's rows, and saw no suggestion.

    Two things must hold together. The sample must be *identical* -- same ids,
    same questions, same passages -- or the corrected pass answers a different
    question than the one already asked. And no suggestion may have been shown,
    or the correction has reproduced the fault it exists to remove.
    """
    found: List[Finding] = []
    add = lambda c, s, d="": found.append(Finding("corrected", c, s, d))  # noqa: E731

    original_ids = [str(r.get("annotation_id")) for r in original_rows]
    corrected_ids = [str(r.get("annotation_id")) for r in corrected_rows]
    missing = sorted(set(original_ids) - set(corrected_ids))
    added = sorted(set(corrected_ids) - set(original_ids))
    add("same annotation ids as the pilot sample", _ok(not missing and not added),
        f"{len(missing)} missing, {len(added)} added"
        + (f"; missing {missing[:5]}" if missing else "")
        + (f"; added {added[:5]}" if added else ""))
    add("same row order as the pilot sample", _ok(original_ids == corrected_ids),
        "ids match position for position" if original_ids == corrected_ids
        else "same set, different order" if set(original_ids) == set(corrected_ids)
        else "sets differ")

    by_id = {str(r.get("annotation_id")): r for r in original_rows}
    for field, name in (("question", "question text"), ("candidate_text", "passage text")):
        changed = [str(r.get("annotation_id")) for r in corrected_rows
                   if str(r.get("annotation_id")) in by_id
                   and str(r.get(field, "")) != str(by_id[str(r["annotation_id"])].get(field, ""))]
        add(f"{name} unchanged from the pilot sheet", _ok(not changed),
            f"{len(changed)} row(s) differ: {changed[:5]}")

    labelled = [r for r in corrected_rows
                if str(r.get("human_label", "")).strip().isdigit()
                and int(str(r["human_label"]).strip()) in VALID_LABELS]
    shown = [r for r in labelled if r.get("ai_suggestion_shown") is True]
    generated = [r for r in labelled
                 if str(r.get("ai_suggested_label", "")).strip().isdigit()]
    add("no suggestion was displayed during the corrected pass", _ok(not shown),
        f"{len(shown)} of {len(labelled)} labelled row(s) recorded "
        "ai_suggestion_shown=true")
    add("no suggestion was generated for the corrected pass", _ok(not generated),
        f"{len(generated)} of {len(labelled)} labelled row(s) carry an "
        "ai_suggested_label")

    timed = [r for r in labelled if str(r.get("annotated_utc", "")).strip()]
    add("labels were entered through the interface", _ok(len(timed) == len(labelled)),
        f"{len(timed)} of {len(labelled)} labelled row(s) carry an annotated_utc "
        "timestamp, which only a button press writes")

    marked = [r for r in corrected_rows if r.get("annotation_pass")]
    add("rows are marked as the corrected pass",
        "PASS" if len(marked) == len(corrected_rows) else "WARN",
        f"{len(marked)} of {len(corrected_rows)} row(s) carry annotation_pass")
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

    # The manifest records whatever path the machine that ran `export` used, and
    # that machine is usually Windows while an audit may run anywhere. os.path
    # on POSIX does not split backslashes, so ntpath separators must be
    # normalised first or the directory reads as empty and a correct manifest
    # fails. Split on both, always.
    recorded = str(manifest.get("source_per_question", "")).replace("\\", "/")
    parts = [p for p in recorded.split("/") if p]
    source = parts[-2] if len(parts) >= 2 else ""
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
        # No suggestion at all is the goal state, not a missing measurement --
        # but only when the rows say so explicitly.
        deliberate = stats.get("suggestions_generated") is False
        add("free of suggestion anchoring", "PASS" if deliberate else "SKIP",
            str(stats.get("note", "")))
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
              expected_questions: int = EXPECTED_QUESTIONS,
              original_sheet_path: str = "") -> Dict[str, Any]:
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

    # Which pass is this? An independent pass carries its own marker and has no
    # distribution to reproduce; only the retired pilot is checked against a
    # recorded outcome, and that outcome comes from the pilot's own integrity
    # record rather than from a constant in this file.
    corrected = bool(sheet_rows) and all(
        r.get("annotation_pass") == CORRECTED_PASS_LABEL for r in sheet_rows)
    if corrected:
        expected = None
    elif expected is None:
        # The pilot's recorded outcome lives beside the pilot sheet -- whether
        # that is the sheet under audit or the one it is being compared against.
        for neighbour in (sheet_path, original_sheet_path):
            if not neighbour:
                continue
            integrity = os.path.join(os.path.dirname(os.path.abspath(neighbour)),
                                     "pilot_integrity.json")
            if os.path.isfile(integrity):
                with open(integrity, "r", encoding="utf-8") as handle:
                    recorded = ((json.load(handle).get("completeness") or {})
                                .get("label_distribution") or {})
                if recorded:
                    expected = {int(k): int(v) for k, v in recorded.items()}
                    break

    findings: List[Finding] = []
    findings.append(Finding(
        "annotations", "annotation pass identified", "PASS",
        f"{CORRECTED_PASS_LABEL}: an independent human-only pass. Its label "
        "distribution is its own result and is not required to match the "
        "retired anchored pilot." if corrected else
        "no corrected-pass marker: treated as the original pass, checked "
        "against its recorded outcome where one exists."))
    findings += audit_annotations(sheet_rows, decisions, expected, expected_rows,
                                  independent=corrected)
    findings += audit_provenance(manifest, comparison_manifest, frozen_meta)
    findings += audit_sampling(manifest, sheet_rows, decisions, expected_rows,
                               expected_population, expected_questions)
    findings += audit_anchoring(sheet_rows)

    if original_sheet_path and os.path.isfile(original_sheet_path):
        findings += audit_corrected_pass(sheet_rows, _read_jsonl(original_sheet_path))

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
