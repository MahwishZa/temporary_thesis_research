"""The readiness inspection: honest about absence, and never inventing a target.

The dangerous failure here is a false READY -- reporting that an answer-quality
experiment can proceed when nothing exists to grade the answers against. So the
tests drive the decision function through each state and check that the three
near-miss artifacts (evidence labels, filter training labels, machine scores)
are each named and rejected.
"""

import json
import os
import sys

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from experiments.analysis.answer_quality_readiness import (  # noqa: E402
    NOT_READY, PARTIAL, READY,
    build_readiness,
    decide,
    inspect_question_set,
    matched_k_feasibility,
)
from experiments.analysis.render_readiness import render_markdown  # noqa: E402


def _questions(tmp_path, extra=None):
    path = tmp_path / "dev_questions.jsonl"
    with open(path, "w", encoding="utf-8") as fh:
        for i in range(30):
            row = {"qid": f"alz-{i:03d}", "question": f"Q{i}?", "set": "alz-dev-v1",
                   "time_sensitive": True, "note": "DEVELOPMENT ONLY. No gold answer."}
            row.update(extra or {})
            fh.write(json.dumps(row) + "\n")
    return str(path)


# -- the question set ------------------------------------------------------
def test_reports_absence_of_gold_answers(tmp_path):
    q = inspect_question_set(_questions(tmp_path))
    assert q["questions"] == 30
    assert q["answer_target_fields"] == []
    assert q["has_gold_answers"] is False


@pytest.mark.parametrize("field", ["gold_answer", "reference_answer", "answer",
                                   "ground_truth", "answer_key", "options"])
def test_detects_a_target_field_under_any_common_name(tmp_path, field):
    q = inspect_question_set(_questions(tmp_path, {field: "x"}))
    assert q["has_gold_answers"] is True
    assert field in q["answer_target_fields"]


def test_missing_question_file_is_reported_not_crashed(tmp_path):
    q = inspect_question_set(str(tmp_path / "nope.jsonl"))
    assert q["exists"] is False


# -- the decision ----------------------------------------------------------
def test_no_target_and_working_code_is_partially_ready():
    d = decide({"has_gold_answers": False}, {"exists": True},
               {"selection_executable_now": True})
    assert d["verdict"] == PARTIAL
    assert "target does not" in d["reason"]


def test_a_target_plus_code_is_ready():
    d = decide({"has_gold_answers": True}, {"exists": True},
               {"selection_executable_now": True})
    assert d["verdict"] == READY


def test_nothing_at_all_is_not_ready():
    d = decide({"has_gold_answers": False}, {"exists": False},
               {"selection_executable_now": False})
    assert d["verdict"] == NOT_READY


def test_a_target_alone_without_code_is_not_reported_ready():
    """READY requires both halves; a gold answer with no metric is not enough."""
    d = decide({"has_gold_answers": True}, {"exists": False},
               {"selection_executable_now": False})
    assert d["verdict"] != READY


# -- matched-k -------------------------------------------------------------
def _per_question(tmp_path, candidates=20):
    path = tmp_path / "per_question.jsonl"
    with open(path, "w", encoding="utf-8") as fh:
        for i in range(30):
            arm = {"decisions": [{"chunk_id": f"c{j}"} for j in range(candidates)],
                   "answer": "an answer", "context_chars": 100, "num_admitted": 3,
                   "abstained": False, "generation_error": None}
            fh.write(json.dumps({"qid": f"alz-{i:03d}", "question": "Q?",
                                 "rag2": dict(arm), "scaf": dict(arm)}) + "\n")
    return str(path)


def test_matched_k_selection_is_executable_with_enough_candidates(tmp_path):
    f = matched_k_feasibility(_per_question(tmp_path), k=5)
    assert f["selection_executable_now"] is True
    assert f["arm_questions_with_at_least_k_candidates"] == 60


def test_matched_k_selection_blocked_when_candidates_are_too_few(tmp_path):
    f = matched_k_feasibility(_per_question(tmp_path, candidates=3), k=5)
    assert f["selection_executable_now"] is False


def test_matched_k_always_reports_the_answer_target_as_missing(tmp_path):
    f = matched_k_feasibility(_per_question(tmp_path), k=5)
    assert f["components"]["answer_quality_target"] is False
    assert "answer_quality_target" in f["blocked_on"]


# -- against the real repository -------------------------------------------
def test_the_real_repository_is_partially_ready():
    r = build_readiness(_ROOT)
    assert r["decision"]["verdict"] == PARTIAL
    assert r["question_set"]["has_gold_answers"] is False
    assert r["evaluation_code"]["exists"] is True


def test_the_three_near_miss_artifacts_are_each_rejected():
    entries = build_readiness(_ROOT)["artifacts_that_are_not_answer_targets"]
    assert "machine_scores" in entries
    for entry in entries.values():
        assert entry["suitable_as_answer_gold"] is False
        assert entry["why_not_an_answer_target"].strip()


def test_inspection_writes_nothing(tmp_path):
    results = os.path.join(_ROOT, "experiments", "results", "rag2_vs_scaf_alzheimer")
    watched = [os.path.join(results, "comparison_scientific", "per_question.jsonl"),
               os.path.join(results, "evidence_quality", "annotation_sheet_v2.jsonl")]
    before = {p: os.path.getmtime(p) for p in watched if os.path.exists(p)}
    build_readiness(_ROOT)
    after = {p: os.path.getmtime(p) for p in watched if os.path.exists(p)}
    assert before == after


# -- the report ------------------------------------------------------------
def test_markdown_states_the_decision_and_the_distinctions():
    md = render_markdown(build_readiness(_ROOT))
    assert PARTIAL in md
    for heading in ("The 30 evaluation questions", "Answers that already exist",
                    "Evaluation code that exists",
                    "Things that look like a gold standard but are not",
                    "Matched-k feasibility", "Recommended next step"):
        assert heading in md, heading
    for warning in ("Do not reuse the 120 evidence-quality labels",
                    "Do not reuse the filter training weak labels",
                    "Do not grade answers with SCAF or RAG"):
        assert warning in md, warning


def test_markdown_never_claims_a_gold_standard_exists():
    md = render_markdown(build_readiness(_ROOT))
    assert "Answer-quality target exists | **False**" in md
    assert "no gold answer, no reference answer" in md
