"""The audit's own failure modes.

An audit that passes everything is worthless, so most of these tests break
something specific and assert that the audit notices. The cases chosen are the
ones that would actually have happened: a stale frozen set, a sheet and key from
different exports, a sample re-drawn after annotation, labels that are a copy of
the machine suggestion.
"""

import copy
import json
import os
import sys

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from experiments.analysis.evidence_quality import (  # noqa: E402
    analyse,
    load_decisions,
    split_blind_and_key,
    stratified_sample,
    suggestion_anchoring,
    question_clustering,
)
from experiments.analysis.eq_audit import (  # noqa: E402
    audit_annotations,
    audit_anchoring,
    audit_provenance,
    audit_sampling,
    label_fingerprint,
    run_audit,
)

SCIENTIFIC = os.path.join(_ROOT, "experiments", "results", "rag2_vs_scaf_alzheimer",
                          "comparison_scientific")
DIGEST = "316260f04c1720fbc20c1b584ea9f0a093dedbae9fd46476d8a7f45380e86aad"


def status(findings, check):
    return next(f.status for f in findings if f.check == check)


# --------------------------------------------------------------------------
# A synthetic study, shaped exactly like the real one
# --------------------------------------------------------------------------
def _decision(qid, i, rag2_score, scaf_score, rag2_keep, scaf_keep):
    common = {"chunk_id": f"{qid}-c{i:02d}", "rerank_rank": i + 1,
              "source_category": "pmc-fulltext", "canonical_date": "2024-01-01",
              "authority_tier_label": ""}
    rag2 = {**common, "score": rag2_score, "keep": rag2_keep, "label": ""}
    scaf = {**common, "score": scaf_score, "keep": scaf_keep, "label": "admit",
            "detail": {"sigma_support": scaf_score * 0.8, "gamma_currency": 0.9,
                       "tau_authority": 0.45, "rho_corroboration": 0.0,
                       "scaf_score": scaf_score, "threshold": 0.45, "gate": ""}}
    return rag2, scaf


@pytest.fixture()
def study(tmp_path):
    """Ten questions x 20 candidates = 200 decisions, sampled down to 40."""
    run_dir = tmp_path / "comparison_scientific"      # the real layout
    run_dir.mkdir()
    per_question = run_dir / "per_question.jsonl"
    with open(per_question, "w", encoding="utf-8") as fh:
        for q in range(10):
            qid = f"alz-{q:03d}"
            rag2s, scafs = [], []
            for i in range(20):
                r, s = _decision(qid, i, 0.20 + 0.01 * i, 0.30 + 0.03 * i,
                                 rag2_keep=(q == 0 and i == 19),
                                 scaf_keep=(0.30 + 0.03 * i) >= 0.45)
                rag2s.append(r); scafs.append(s)
            fh.write(json.dumps({
                "qid": qid, "question": f"Question {q}?", "time_sensitive": True,
                "num_candidates": 20,
                "rag2": {"decisions": rag2s,
                         "admitted_chunk_ids": [d["chunk_id"] for d in rag2s if d["keep"]]},
                "scaf": {"decisions": scafs,
                         "admitted_chunk_ids": [d["chunk_id"] for d in scafs if d["keep"]]},
            }) + "\n")

    comparison_manifest = run_dir / "manifest.json"
    comparison_manifest.write_text(json.dumps({"frozen_set_digest": DIGEST}),
                                   encoding="utf-8")

    decisions = load_decisions(str(per_question))
    enriched = [{**d, "candidate_text": f"passage for {d['chunk_id']}",
                 "title": "T", "pmid": "", "pmcid": ""} for d in decisions]
    sample, provenance = stratified_sample(enriched, size=40, seed=42)
    blind, key = split_blind_and_key(sample)

    for index, row in enumerate(blind):          # a plausible label pattern
        row["human_label"] = [2, 1, 2, 0, 2][index % 5]
        row["ai_suggested_label"] = [2, 2, 2, 1, 1][index % 5]
        row["ai_suggestion_shown"] = True

    sheet = tmp_path / "annotation_sheet.jsonl"
    with open(sheet, "w", encoding="utf-8") as fh:
        for row in blind:
            fh.write(json.dumps(row, sort_keys=True) + "\n")
    key_path = tmp_path / "annotation_key.jsonl"
    with open(key_path, "w", encoding="utf-8") as fh:
        for row in key:
            fh.write(json.dumps(row, sort_keys=True) + "\n")

    manifest = tmp_path / "sample_manifest.json"
    manifest.write_text(json.dumps({
        **provenance, "frozen_set_digest": DIGEST, "retrieval_is_medcpt": True,
        "source_per_question": str(per_question),
    }), encoding="utf-8")

    return {"dir": tmp_path, "sheet": str(sheet), "key": str(key_path),
            "manifest": str(manifest), "per_question": str(per_question),
            "comparison_manifest": str(comparison_manifest),
            "blind": blind, "decisions": decisions,
            "expected": {2: 24, 1: 8, 0: 8}}


def _run(study, **kw):
    kw.setdefault("expected_rows", 40)
    kw.setdefault("expected_population", 200)
    kw.setdefault("expected_questions", 10)
    return run_audit(study["sheet"], study["per_question"], study["manifest"],
                     study["comparison_manifest"], key_path=study["key"],
                     expected=study["expected"], **kw)


def _annotations(study, rows):
    from experiments.analysis.eq_audit import audit_annotations as _a
    return _a(rows, study["decisions"], study["expected"], expected_rows=40)


def _rewrite_sheet(study, rows):
    with open(study["sheet"], "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True) + "\n")


# --------------------------------------------------------------------------
# The clean case
# --------------------------------------------------------------------------
def test_a_sound_study_produces_no_failures(study):
    report = _run(study)
    failures = [f for f in report["findings"] if f["status"] == "FAIL"]
    assert failures == [], failures
    assert report["verdict"] in ("PASS", "PASS WITH LIMITATIONS")


def test_the_audit_writes_nothing(study):
    before = {p: os.path.getmtime(os.path.join(study["dir"], p))
              for p in os.listdir(study["dir"])}
    _run(study)
    after = {p: os.path.getmtime(os.path.join(study["dir"], p))
             for p in os.listdir(study["dir"])}
    assert before == after


def test_integrity_block_reports_the_labels(study):
    integrity = _run(study)["integrity"]
    assert integrity["rows"] == 40
    assert integrity["label_distribution"] == study["expected"]
    assert len(integrity["annotation_sheet_sha256"]) == 64
    assert len(integrity["label_fingerprint"]) == 64


# --------------------------------------------------------------------------
# Things the audit must catch
# --------------------------------------------------------------------------
def test_catches_a_missing_label(study):
    rows = copy.deepcopy(study["blind"])
    rows[0]["human_label"] = ""
    _rewrite_sheet(study, rows)
    assert status(_annotations(study, rows),
                  "no missing labels") == "FAIL"


def test_catches_an_invalid_label(study):
    rows = copy.deepcopy(study["blind"])
    rows[0]["human_label"] = 7
    assert status(_annotations(study, rows),
                  "no invalid labels") == "FAIL"


def test_catches_a_changed_label_distribution(study):
    rows = copy.deepcopy(study["blind"])
    rows[0]["human_label"] = 1 if rows[0]["human_label"] != 1 else 2
    assert status(_annotations(study, rows),
                  "label distribution matches the reported 24/8/8") == "FAIL"


def test_catches_duplicate_annotation_ids(study):
    rows = copy.deepcopy(study["blind"])
    rows[1]["annotation_id"] = rows[0]["annotation_id"]
    assert status(_annotations(study, rows),
                  "annotation ids are unique") == "FAIL"


def test_catches_a_row_that_is_not_a_real_decision(study):
    """A fabricated annotation row is the thing that must never pass."""
    rows = copy.deepcopy(study["blind"])
    rows[0]["chunk_id"] = "invented-chunk"
    assert status(_annotations(study, rows),
                  "every row is a real evaluation decision") == "FAIL"


def test_catches_edited_question_text(study):
    rows = copy.deepcopy(study["blind"])
    rows[0]["question"] = "a different question entirely?"
    assert status(_annotations(study, rows),
                  "question text is unchanged from the comparison") == "FAIL"


def test_catches_an_empty_passage(study):
    rows = copy.deepcopy(study["blind"])
    rows[0]["candidate_text"] = ""
    assert status(_annotations(study, rows),
                  "every row carries passage text") == "FAIL"


def test_catches_a_filter_training_question_in_the_sample(study):
    rows = copy.deepcopy(study["blind"])
    rows[0]["qid"] = "alz-train-0001"
    assert status(_annotations(study, rows),
                  "no filter-training examples in the sample") == "FAIL"


def test_catches_labels_copied_wholesale_from_the_suggestion(study):
    rows = copy.deepcopy(study["blind"])
    for row in rows:
        row["human_label"] = row["ai_suggested_label"]
    assert status(_annotations(study, rows),
                  "human_label is not a copy of ai_suggested_label") == "FAIL"


def test_accepts_labels_that_differ_from_the_suggestion(study):
    assert status(_annotations(study, study["blind"]),
                  "human_label is not a copy of ai_suggested_label") == "PASS"


# --------------------------------------------------------------------------
# Provenance
# --------------------------------------------------------------------------
def test_catches_a_frozen_digest_that_is_not_the_scientific_run(study):
    findings = audit_provenance({"frozen_set_digest": "deadbeef",
                                 "retrieval_is_medcpt": True,
                                 "source_per_question": study["per_question"]},
                                {"frozen_set_digest": DIGEST}, None)
    assert status(findings, "sample was built from the scientific run's frozen set") == "FAIL"


def test_catches_the_lexical_development_retrieval(study):
    """The exact stale-set hazard: retrieval_is_medcpt false must never pass."""
    findings = audit_provenance({"frozen_set_digest": DIGEST,
                                 "retrieval_is_medcpt": False,
                                 "source_per_question": study["per_question"]},
                                {"frozen_set_digest": DIGEST}, None)
    assert status(findings, "retrieval is production MedCPT") == "FAIL"


def test_warns_when_the_local_frozen_file_is_the_stale_one(study):
    findings = audit_provenance(
        {"frozen_set_digest": DIGEST, "retrieval_is_medcpt": True,
         "source_per_question": study["per_question"]},
        {"frozen_set_digest": DIGEST},
        {"frozen_set_digest": "151b7d54", "provenance": {"source": "lexical-dev"}})
    assert status(findings, "local frozen file matches the scientific run") == "WARN"


def test_flags_a_sample_built_from_the_wrong_comparison_directory(study):
    findings = audit_provenance(
        {"frozen_set_digest": DIGEST, "retrieval_is_medcpt": True,
         "source_per_question": "/x/comparison/per_question.jsonl"},
        {"frozen_set_digest": DIGEST}, None)
    assert status(findings, "sample was built from comparison_scientific/") == "FAIL"


# --------------------------------------------------------------------------
# Sampling
# --------------------------------------------------------------------------
def test_sample_reproduces_from_the_rule_and_seed(study):
    with open(study["manifest"], encoding="utf-8") as fh:
        manifest = json.load(fh)
    findings = audit_sampling(manifest, study["blind"], study["decisions"], 40, 200, 10)
    assert status(findings, "the sample reproduces exactly from the predeclared "
                            "rule and seed") == "PASS"
    assert status(findings, "population is the 200 evaluation decisions") == "PASS"
    assert status(findings, "sample size is 40") == "PASS"


def test_catches_a_sample_redrawn_after_annotation(study):
    """Swapping which rows were annotated must not survive the re-run check."""
    with open(study["manifest"], encoding="utf-8") as fh:
        manifest = json.load(fh)
    rows = copy.deepcopy(study["blind"])
    rows[0]["chunk_id"] = rows[1]["chunk_id"]
    rows[0]["qid"] = rows[1]["qid"]
    findings = audit_sampling(manifest, rows, study["decisions"], 40, 200, 10)
    assert status(findings, "the sample reproduces exactly from the predeclared "
                            "rule and seed") == "FAIL"


def test_catches_tertile_edges_that_were_not_computed_from_the_population(study):
    with open(study["manifest"], encoding="utf-8") as fh:
        manifest = json.load(fh)
    manifest["scaf_tertile_edges"] = [0.1, 0.2]
    findings = audit_sampling(manifest, study["blind"], study["decisions"], 40, 200, 10)
    assert status(findings, "scaf tertile edges recompute from all 200 decisions") == "FAIL"


def test_forced_rag2_admissions_are_always_reported_as_a_limitation(study):
    with open(study["manifest"], encoding="utf-8") as fh:
        manifest = json.load(fh)
    findings = audit_sampling(manifest, study["blind"], study["decisions"], 40, 200, 10)
    assert status(findings, "forcing all RAG2 admissions makes the sample "
                            "non-self-weighting") == "WARN"


# --------------------------------------------------------------------------
# Anchoring
# --------------------------------------------------------------------------
def test_anchoring_is_measured_and_reported(study):
    findings = audit_anchoring(study["blind"])
    assert status(findings, "labels are distinguishable from the suggestion") == "PASS"
    stats = json.loads(next(f.detail for f in findings if f.check == "_stats"))
    assert stats["all"]["n"] == 40
    assert 0.0 <= stats["all"]["agreement_rate"] <= 1.0


def test_total_agreement_with_the_suggestion_is_a_failure(study):
    rows = copy.deepcopy(study["blind"])
    for row in rows:
        row["human_label"] = row["ai_suggested_label"]
    assert status(audit_anchoring(rows),
                  "labels are distinguishable from the suggestion") == "FAIL"


def test_no_unanchored_subset_is_a_warning(study):
    assert status(audit_anchoring(study["blind"]),
                  "an unanchored subset exists for comparison") == "WARN"


def test_a_hidden_suggestion_subset_satisfies_the_baseline_check(study):
    rows = copy.deepcopy(study["blind"])
    for row in rows[:12]:
        row["ai_suggestion_shown"] = False
    assert status(audit_anchoring(rows),
                  "an unanchored subset exists for comparison") == "PASS"


def test_anchoring_splits_shown_from_hidden(study):
    rows = copy.deepcopy(study["blind"])
    for row in rows[:12]:
        row["ai_suggestion_shown"] = False
    human = [int(r["human_label"]) for r in rows]
    stats = suggestion_anchoring(rows, human)
    assert stats["shown"]["n"] == 28
    assert stats["hidden"]["n"] == 12
    assert stats["all"]["n"] == 40


def test_anchoring_skips_cleanly_when_no_suggestion_was_recorded(study):
    rows = copy.deepcopy(study["blind"])
    for row in rows:
        row.pop("ai_suggested_label", None)
        row.pop("ai_suggestion_shown", None)
    stats = suggestion_anchoring(rows, [int(r["human_label"]) for r in rows])
    assert stats["rows_with_a_recorded_suggestion"] == 0
    assert "note" in stats
    assert "outside the interface" in stats["note"]
    assert status(audit_anchoring(rows), "free of suggestion anchoring") == "SKIP"


def test_analyse_now_carries_the_anchoring_and_clustering_blocks(study):
    with open(study["key"], encoding="utf-8") as fh:
        key_by_id = {json.loads(l)["annotation_id"]: json.loads(l) for l in fh if l.strip()}
    merged = [{**key_by_id[r["annotation_id"]], **r} for r in study["blind"]]
    result = analyse(merged)
    assert result["suggestion_anchoring"]["all"]["n"] == 40
    assert result["question_clustering"]["questions_represented"] <= 10
    # the pre-existing numbers must be untouched by the addition
    assert result["labelled_rows"] == 40
    assert set(result["rank_association_with_human_label"]) == {
        "scaf_score", "scaf_sigma_support", "scaf_gamma_currency",
        "scaf_tau_authority", "rag2_score"}


def test_question_clustering_counts_rows_per_question():
    rows = [{"qid": "a"}, {"qid": "a"}, {"qid": "a"}, {"qid": "b"}]
    stats = question_clustering(rows)
    assert stats["questions_represented"] == 2
    assert stats["rows_per_question"]["max"] == 3
    assert stats["rows_per_question"]["min"] == 1


# --------------------------------------------------------------------------
# Label fingerprint
# --------------------------------------------------------------------------
def test_fingerprint_ignores_formatting_but_not_labels(study):
    rows = copy.deepcopy(study["blind"])
    baseline = label_fingerprint(rows)

    reordered = list(reversed(copy.deepcopy(rows)))
    for row in reordered:
        row["human_notes"] = "reformatted"
    assert label_fingerprint(reordered) == baseline

    changed = copy.deepcopy(rows)
    changed[0]["human_label"] = 0 if changed[0]["human_label"] != 0 else 2
    assert label_fingerprint(changed) != baseline


# --------------------------------------------------------------------------
# The corrected, human-only second pass
#
# The pilot failed in one specific way: every row was shown a lexical
# suggestion and every label matched it. These pin the correction -- the same
# rows, no suggestion, and an audit that can tell the two passes apart.
# --------------------------------------------------------------------------
from experiments.analysis.evidence_quality import (  # noqa: E402
    SECOND_PASS_LABEL,
    blank_sheet_from,
    compare_passes,
)
from experiments.analysis.eq_audit import audit_corrected_pass  # noqa: E402


def test_blank_sheet_keeps_the_sample_and_clears_the_judgements(study):
    fresh = blank_sheet_from(study["blind"])
    assert [r["annotation_id"] for r in fresh] == \
           [r["annotation_id"] for r in study["blind"]]
    for new, old in zip(fresh, study["blind"]):
        assert new["question"] == old["question"]
        assert new["candidate_text"] == old["candidate_text"]
        assert new["human_label"] == ""
        assert new["human_notes"] == ""
        assert new["annotation_pass"] == SECOND_PASS_LABEL


def test_blank_sheet_removes_every_trace_of_the_suggestion(study):
    """A corrected row must carry no suggestion, so 'the annotator agreed with
    the suggestion' is not even expressible for it."""
    fresh = blank_sheet_from(study["blind"])
    for row in fresh:
        for field in ("ai_suggested_label", "ai_explanation", "ai_rule_version",
                      "ai_coverage", "ai_suggestion_shown", "ai_suggestion_generated",
                      "annotated_utc"):
            assert field not in row, field


def test_blank_sheet_does_not_mutate_the_pilot(study):
    before = copy.deepcopy(study["blind"])
    blank_sheet_from(study["blind"])
    assert study["blind"] == before


def _corrected(study, **overrides):
    rows = blank_sheet_from(study["blind"])
    for i, row in enumerate(rows):
        row["human_label"] = [2, 2, 1, 0, 2][i % 5]
        row["annotated_utc"] = "2026-09-10T12:00:00Z"
        row["ai_suggestion_shown"] = False
        row["ai_suggestion_generated"] = False
        row.update(overrides)
    return rows


def test_a_clean_corrected_pass_passes(study):
    findings = audit_corrected_pass(_corrected(study), study["blind"])
    assert [f for f in findings if f.status == "FAIL"] == []


def test_catches_a_corrected_pass_that_dropped_rows(study):
    rows = _corrected(study)[:-1]
    assert status(audit_corrected_pass(rows, study["blind"]),
                  "same annotation ids as the pilot sample") == "FAIL"


def test_catches_a_corrected_pass_that_resampled(study):
    rows = _corrected(study)
    rows[0]["annotation_id"] = "eq-9999"
    assert status(audit_corrected_pass(rows, study["blind"]),
                  "same annotation ids as the pilot sample") == "FAIL"


def test_catches_reordered_rows(study):
    rows = list(reversed(_corrected(study)))
    findings = audit_corrected_pass(rows, study["blind"])
    assert status(findings, "same annotation ids as the pilot sample") == "PASS"
    assert status(findings, "same row order as the pilot sample") == "FAIL"


def test_catches_edited_passage_text_in_the_corrected_pass(study):
    rows = _corrected(study)
    rows[0]["candidate_text"] = "a different passage"
    assert status(audit_corrected_pass(rows, study["blind"]),
                  "passage text unchanged from the pilot sheet") == "FAIL"


def test_catches_edited_question_text_in_the_corrected_pass(study):
    rows = _corrected(study)
    rows[0]["question"] = "a different question?"
    assert status(audit_corrected_pass(rows, study["blind"]),
                  "question text unchanged from the pilot sheet") == "FAIL"


def test_catches_a_corrected_pass_that_showed_suggestions(study):
    """The correction reproducing the original fault must not pass."""
    rows = _corrected(study, ai_suggestion_shown=True)
    assert status(audit_corrected_pass(rows, study["blind"]),
                  "no suggestion was displayed during the corrected pass") == "FAIL"


def test_catches_a_corrected_pass_that_generated_suggestions(study):
    rows = _corrected(study, ai_suggested_label=2)
    assert status(audit_corrected_pass(rows, study["blind"]),
                  "no suggestion was generated for the corrected pass") == "FAIL"


def test_catches_labels_that_never_went_through_the_interface(study):
    rows = _corrected(study)
    for row in rows:
        row.pop("annotated_utc")
    assert status(audit_corrected_pass(rows, study["blind"]),
                  "labels were entered through the interface") == "FAIL"


# -- the visibility-aware copy check ---------------------------------------
def test_total_agreement_with_a_visible_suggestion_fails(study):
    """The exact pilot failure: 100% agreement AND 100% visibility."""
    rows = copy.deepcopy(study["blind"])
    for row in rows:
        row["human_label"] = row["ai_suggested_label"]
        row["ai_suggestion_shown"] = True
    assert status(_annotations(study, rows),
                  "human_label is not a copy of ai_suggested_label") == "FAIL"


def test_total_agreement_with_a_hidden_suggestion_is_only_a_warning(study):
    """Section 8: agreement alone is not the fault. A genuine judgement may
    legitimately match a suggestion it never saw."""
    rows = copy.deepcopy(study["blind"])
    for row in rows:
        row["human_label"] = row["ai_suggested_label"]
        row["ai_suggestion_shown"] = False
    assert status(_annotations(study, rows),
                  "human_label is not a copy of ai_suggested_label") == "WARN"


def test_partial_agreement_with_a_visible_suggestion_passes(study):
    """Agreement on some rows is normal and must not be penalised."""
    rows = copy.deepcopy(study["blind"])
    for i, row in enumerate(rows):
        row["human_label"] = row["ai_suggested_label"] if i % 3 else 0
        row["ai_suggestion_shown"] = True
    assert status(_annotations(study, rows),
                  "human_label is not a copy of ai_suggested_label") == "PASS"


def test_a_sheet_with_no_suggestions_at_all_passes_the_copy_check(study):
    rows = _corrected(study)
    assert status(_annotations(study, rows),
                  "human_label is not a copy of ai_suggested_label") == "PASS"


# -- comparing the two passes without pooling them -------------------------
def test_compare_passes_reports_movement_without_merging(study):
    corrected = _corrected(study)
    result = compare_passes(study["blind"], corrected)
    assert result["n"] == len(corrected)
    assert 0.0 <= result["agreement_rate"] <= 1.0
    assert "guard" in result and "must not be pooled" in result["guard"]


def test_compare_passes_handles_an_unlabelled_corrected_sheet(study):
    assert compare_passes(study["blind"], blank_sheet_from(study["blind"]))["n"] == 0


def test_a_human_only_pass_is_reported_as_free_of_anchoring(study):
    """No suggestion generated is the goal state, not a missing measurement --
    and it must read that way to someone who is not a statistician."""
    rows = _corrected(study)
    stats = suggestion_anchoring(rows, [int(r["human_label"]) for r in rows])
    assert stats["suggestions_generated"] is False
    assert "human-only pass" in stats["note"]
    assert "filled in outside" not in stats["note"]
    assert status(audit_anchoring(rows), "free of suggestion anchoring") == "PASS"


def test_a_sheet_with_no_suggestion_metadata_at_all_is_not_claimed_clean(study):
    """Absence of evidence is not evidence of absence: a sheet that simply
    lacks the fields must not be reported as unanchored."""
    rows = _corrected(study)
    for row in rows:
        row.pop("ai_suggestion_generated")
        row.pop("ai_suggestion_shown")
    assert status(audit_anchoring(rows), "free of suggestion anchoring") == "SKIP"
