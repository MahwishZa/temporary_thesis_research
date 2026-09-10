"""Matched-k selection: exactly k, deterministic, and provenance-guarded.

The failure this file exists to prevent is a matched-k run built on the
lexical-development candidate set. It would produce 60 plausible answers, a
clean-looking manifest, and an entirely different experiment from the one the
thesis claims. So the provenance refusal is tested before anything else.
"""

import json
import os
import sys

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from experiments.analysis.matched_k import (  # noqa: E402
    ARMS,
    DEFAULT_K,
    SCIENTIFIC_FROZEN_DIGEST,
    build_selection,
    check_selection_invariants,
    select_top_k,
    selection_statistics,
    verify_inputs,
)

STALE = "151b7d5432b124bafb66e057e8b99c1d06c84954d5650951da3fc8c8aa8ccf3d"


def _decisions(n=20, base=0.5):
    return [{"chunk_id": f"c{i:02d}", "score": base + 0.01 * i, "keep": False}
            for i in range(n)]


def _write_study(tmp_path, questions=30, candidates=20, digest=SCIENTIFIC_FROZEN_DIGEST,
                 medcpt=True):
    run = tmp_path / "comparison_scientific"
    run.mkdir(exist_ok=True)
    pq = run / "per_question.jsonl"
    with open(pq, "w", encoding="utf-8") as fh:
        for q in range(questions):
            qid = f"alz-{q:03d}"
            rag2 = [{"chunk_id": f"{qid}-c{i:02d}", "score": 0.20 + 0.01 * i,
                     "keep": False} for i in range(candidates)]
            scaf = [{"chunk_id": f"{qid}-c{i:02d}", "score": 0.90 - 0.01 * i,
                     "keep": True, "detail": {}} for i in range(candidates)]
            fh.write(json.dumps({"qid": qid, "question": f"Question {q}?",
                                 "rag2": {"decisions": rag2},
                                 "scaf": {"decisions": scaf}}) + "\n")
    manifest = run / "manifest.json"
    manifest.write_text(json.dumps({
        "frozen_set_digest": SCIENTIFIC_FROZEN_DIGEST,
        "config": {"arm_a_generation": {"backend": "openai", "temperature": 0.0},
                   "arm_b_generation": {"backend": "openai", "temperature": 0.0}}}),
        encoding="utf-8")
    meta = tmp_path / "frozen_candidates.meta.json"
    meta.write_text(json.dumps({"frozen_set_digest": digest,
                                "provenance": {"retrieval_is_medcpt": medcpt,
                                               "source": "rag2 candidate cache"
                                               if medcpt else "lexical-dev"}}),
                    encoding="utf-8")
    questions_path = tmp_path / "dev_questions.jsonl"
    with open(questions_path, "w", encoding="utf-8") as fh:
        for q in range(questions):
            fh.write(json.dumps({"qid": f"alz-{q:03d}", "question": f"Question {q}?",
                                 "set": "alzheimer-dev-v1"}) + "\n")
    return {"per_question": str(pq), "manifest": str(manifest),
            "meta": str(meta), "questions": str(questions_path)}


def _verify(s, **kw):
    return verify_inputs(s["per_question"], s["manifest"], s["meta"],
                         s["questions"], **kw)


# --------------------------------------------------------------------------
# Provenance: the refusal that protects the experiment
# --------------------------------------------------------------------------
def test_refuses_the_stale_lexical_dev_candidate_set(tmp_path):
    s = _write_study(tmp_path, digest=STALE, medcpt=False)
    report = _verify(s)
    assert report["all_passed"] is False
    failed = [c["check"] for c in report["checks"] if not c["pass"]]
    assert "frozen candidate file on disk IS the scientific one" in failed
    assert "frozen candidate retrieval is production MedCPT" in failed


def test_accepts_the_scientific_candidate_set(tmp_path):
    assert _verify(_write_study(tmp_path))["all_passed"] is True


def test_refuses_a_run_manifest_with_the_wrong_digest(tmp_path):
    s = _write_study(tmp_path)
    payload = json.loads(open(s["manifest"], encoding="utf-8").read())
    payload["frozen_set_digest"] = STALE
    open(s["manifest"], "w", encoding="utf-8").write(json.dumps(payload))
    report = _verify(s)
    assert report["all_passed"] is False
    assert any("run digest" in c["check"] and not c["pass"] for c in report["checks"])


def test_refuses_a_wrong_sized_question_set(tmp_path):
    report = _verify(_write_study(tmp_path, questions=29))
    assert report["all_passed"] is False
    assert any("exactly 30 questions" in c["check"] and not c["pass"]
               for c in report["checks"])


def test_refuses_when_candidates_per_question_is_not_20(tmp_path):
    report = _verify(_write_study(tmp_path, candidates=10))
    assert any("exactly 20 candidates" in c["check"] and not c["pass"]
               for c in report["checks"])


def test_detects_altered_question_text(tmp_path):
    s = _write_study(tmp_path)
    rows = [json.loads(l) for l in open(s["questions"], encoding="utf-8")]
    rows[0]["question"] = "a different question?"
    with open(s["questions"], "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    report = _verify(s)
    assert any("question ids and text match" in c["check"] and not c["pass"]
               for c in report["checks"])


# --------------------------------------------------------------------------
# Selection
# --------------------------------------------------------------------------
def test_selects_exactly_k():
    assert len(select_top_k(_decisions(), "scaf", k=5)) == 5


def test_selects_the_highest_scoring_passages():
    picked = select_top_k(_decisions(), "scaf", k=3)
    assert picked == ["c19", "c18", "c17"]


def test_selection_is_deterministic_under_ties():
    """Equal scores must not make the choice depend on dict or file order."""
    tied = [{"chunk_id": f"c{i}", "score": 0.5} for i in range(10)]
    first = select_top_k(tied, "rag2", k=5)
    assert first == select_top_k(list(reversed(tied)), "rag2", k=5)
    assert first == sorted(first)


def test_unknown_arm_is_rejected():
    with pytest.raises(ValueError):
        select_top_k(_decisions(), "not-an-arm", k=5)


def test_selection_ignores_the_admission_threshold(tmp_path):
    """RAG2 admits 4 of 600; a threshold-respecting rule could not reach k=5."""
    s = _write_study(tmp_path)
    selection = build_selection(s["per_question"], k=5)
    assert all(len(e["rag2_selected_chunk_ids"]) == 5 for e in selection)


def test_every_question_yields_k_for_both_arms(tmp_path):
    s = _write_study(tmp_path)
    selection = build_selection(s["per_question"], k=DEFAULT_K)
    assert len(selection) == 30
    for entry in selection:
        for arm in ARMS:
            assert len(entry[f"{arm}_selected_chunk_ids"]) == DEFAULT_K


def test_build_selection_is_deterministic(tmp_path):
    s = _write_study(tmp_path)
    assert build_selection(s["per_question"]) == build_selection(s["per_question"])


def test_output_schema_carries_the_provenance_a_rerun_needs(tmp_path):
    entry = build_selection(_write_study(tmp_path)["per_question"])[0]
    for field in ("qid", "question", "k", "candidate_ids", "overlap_count",
                  "identical_sets", "jaccard",
                  "rag2_selected_chunk_ids", "scaf_selected_chunk_ids"):
        assert field in entry, field


# --------------------------------------------------------------------------
# Invariants
# --------------------------------------------------------------------------
def test_invariants_pass_on_a_sound_selection(tmp_path):
    s = _write_study(tmp_path)
    assert check_selection_invariants(build_selection(s["per_question"]))["all_passed"]


def test_invariants_catch_a_duplicate_passage(tmp_path):
    s = _write_study(tmp_path)
    selection = build_selection(s["per_question"])
    selection[0]["scaf_selected_chunk_ids"][1] = selection[0]["scaf_selected_chunk_ids"][0]
    result = check_selection_invariants(selection)
    assert result["all_passed"] is False
    assert any("no duplicate passage" in c["check"] and not c["pass"]
               for c in result["checks"])


def test_invariants_catch_a_passage_from_another_question(tmp_path):
    """Cross-question leakage would silently invalidate the comparison."""
    s = _write_study(tmp_path)
    selection = build_selection(s["per_question"])
    selection[0]["rag2_selected_chunk_ids"][0] = selection[1]["candidate_ids"][0]
    result = check_selection_invariants(selection)
    assert result["all_passed"] is False
    assert any("frozen candidates" in c["check"] and not c["pass"]
               for c in result["checks"])


def test_invariants_catch_the_wrong_k(tmp_path):
    s = _write_study(tmp_path)
    selection = build_selection(s["per_question"], k=5)
    selection[0]["scaf_selected_chunk_ids"].append("extra")
    assert check_selection_invariants(selection, k=5)["all_passed"] is False


# --------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------
def test_statistics_are_descriptive_and_say_so(tmp_path):
    s = _write_study(tmp_path)
    stats = selection_statistics(build_selection(s["per_question"]))
    assert stats["questions"] == 30
    assert "not answer-quality measurements" in stats["guard"]
    assert (stats["questions_with_identical_sets"]
            + stats["questions_where_evidence_differs"]) == 30


def test_statistics_detect_identical_evidence_sets(tmp_path):
    """When both arms rank the same way, every set is identical."""
    s = _write_study(tmp_path)
    rows = [json.loads(l) for l in open(s["per_question"], encoding="utf-8")]
    for row in rows:
        row["scaf"]["decisions"] = [dict(d) for d in row["rag2"]["decisions"]]
    with open(s["per_question"], "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    stats = selection_statistics(build_selection(s["per_question"]))
    assert stats["questions_with_identical_sets"] == 30
    assert stats["mean_jaccard"] == 1.0


# --------------------------------------------------------------------------
# Against the real repository
# --------------------------------------------------------------------------
def test_the_real_scientific_run_passes_every_selection_invariant():
    pq = os.path.join(_ROOT, "experiments", "results", "rag2_vs_scaf_alzheimer",
                      "comparison_scientific", "per_question.jsonl")
    selection = build_selection(pq, k=5)
    assert len(selection) == 30
    assert check_selection_invariants(selection, k=5)["all_passed"] is True


def test_this_container_cannot_supply_the_scientific_frozen_set():
    """Documents the blocker rather than letting a later run assume otherwise."""
    meta = os.path.join(_ROOT, "experiments", "runs", "frozen_candidates.meta.json")
    if not os.path.isfile(meta):
        pytest.skip("no frozen candidate metadata present")
    payload = json.loads(open(meta, encoding="utf-8").read())
    if payload.get("frozen_set_digest") == SCIENTIFIC_FROZEN_DIGEST:
        pytest.skip("the scientific frozen set IS present here")
    assert payload["provenance"]["retrieval_is_medcpt"] is False
