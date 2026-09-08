"""Tests for frozen candidate replay and the RAG2-vs-SCAF fairness controls.

These are the tests that protect the *validity* of the comparison rather than
the behaviour of either policy. If one of them fails, no number from the
experiment can be reported.
"""

import json
import os
import sys

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _p in (_ROOT, os.path.join(_ROOT, "rag2")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from rag2.config import FilterConfig
from rag2.schema import FilterDecision

import scaf
from scaf.compare import (
    ArmResult,
    build_context,
    build_manifest,
    fairness_report,
    recency_profile,
    run_arm,
    summarise,
)
from scaf.frozen import (
    FrozenCandidate,
    FrozenCandidateSet,
    candidate_digest,
    load,
    save,
    set_digest,
    validate,
)
from scaf.policy import SCAFFilter


def make_candidate(chunk_id="PMC1#abs.w1", date="2024-01-01", rank=1, **kw):
    base = dict(
        chunk_id=chunk_id, document_id=chunk_id.split("#")[0], pmcid="PMC1", pmid="111",
        text=f"amyloid evidence text for {chunk_id}", title="T",
        source_category="pmc-fulltext", canonical_date=date, date_precision="day",
        split_june_2024="post", authority_tier_label="", guideline_family="",
        in_currency_pack="no", retracted="no", eligibility_status="eligible",
        license_code="CC BY", section_heading="Intro",
        retrieval_rank=rank, retrieval_score=0.9 - rank * 0.01,
        rerank_rank=rank, rerank_score=0.8 - rank * 0.01,
    )
    base.update(kw)
    return FrozenCandidate(**base)


def make_set(qid="q1", n=4, **kw):
    return FrozenCandidateSet(
        qid=qid, question="Which biomarker is used for early Alzheimer diagnosis?",
        query="rationale text", question_metadata=kw.get("question_metadata", {}),
        candidates=[make_candidate(f"PMC{i}#abs.w1", rank=i,
                                   date=f"20{10 + i:02d}-01-01") for i in range(1, n + 1)],
    )


class AlwaysFilter:
    """A stand-in admission policy with a fixed rule, for wiring tests."""

    def __init__(self, keep=True):
        self.keep = keep

    def decide(self, question, candidates):
        return [FilterDecision(keep=self.keep, label="[X]", score=1.0 if self.keep else 0.0)
                for _ in candidates]


class TestFrozenCandidateSet:
    def test_digest_covers_identity_and_order(self):
        a = make_set()
        b = FrozenCandidateSet(qid=a.qid, question=a.question,
                               candidates=list(reversed(a.candidates)))
        assert a.digest() != b.digest(), "reordering must change the digest"

    def test_round_trips_through_disk(self, tmp_path):
        path = str(tmp_path / "frozen.jsonl")
        original = [make_set("q1"), make_set("q2", n=3)]
        meta = save(path, original, provenance={"index": "pmc/index"})
        reloaded = load(path, expected_digest=meta["frozen_set_digest"])
        assert [s.qid for s in reloaded] == ["q1", "q2"]
        assert [c.chunk_id for c in reloaded[0].candidates] == \
               [c.chunk_id for c in original[0].candidates]
        assert reloaded[0].candidates[0].canonical_date == \
               original[0].candidates[0].canonical_date

    def test_tampering_with_the_file_is_detected(self, tmp_path):
        path = str(tmp_path / "frozen.jsonl")
        save(path, [make_set("q1")])
        rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
        rows[0]["candidates"] = rows[0]["candidates"][:-1]     # drop a candidate
        with open(path, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(json.dumps(rows[0], sort_keys=True) + "\n")
        with pytest.raises(ValueError, match="digest check"):
            load(path)

    def test_file_level_digest_mismatch_is_detected(self, tmp_path):
        path = str(tmp_path / "frozen.jsonl")
        save(path, [make_set("q1")])
        with pytest.raises(ValueError, match="manifest cites"):
            load(path, expected_digest="0" * 64)

    def test_provenance_survives_into_evidence(self):
        """SCAF cannot score what the freeze did not carry."""
        candidate = make_candidate(date="2019-05-02",
                                   authority_tier_label="clinical-practice-guideline",
                                   in_currency_pack="yes", retracted="no")
        ev = candidate.to_evidence()
        assert ev.metadata["canonical_date"] == "2019-05-02"
        assert ev.metadata["authority_tier_label"] == "clinical-practice-guideline"
        assert ev.metadata["in_currency_pack"] == "yes"
        assert ev.passage_id == candidate.chunk_id

    def test_time_sensitive_flag_reaches_the_question(self):
        frozen = make_set(question_metadata={"time_sensitive": False})
        assert frozen.to_question().metadata["time_sensitive"] is False


class TestValidation:
    def test_a_clean_set_has_no_problems(self):
        assert validate([make_set("q1"), make_set("q2")]) == []

    def test_duplicate_chunk_ids_are_caught(self):
        frozen = make_set()
        frozen.candidates[1] = make_candidate(frozen.candidates[0].chunk_id, rank=2)
        problems = validate([frozen])
        assert any("duplicate chunk_ids" in p for p in problems)

    def test_empty_candidate_list_is_caught(self):
        assert any("no candidates" in p for p in validate([FrozenCandidateSet("q1", "?")]))

    def test_empty_text_is_caught(self):
        frozen = make_set()
        frozen.candidates[0].text = "   "
        assert any("empty text" in p for p in validate([frozen]))

    def test_non_finite_scores_are_caught(self):
        frozen = make_set()
        frozen.candidates[0].rerank_score = float("nan")
        assert any("not finite" in p for p in validate([frozen]))

    def test_duplicate_question_ids_are_caught(self):
        assert any("duplicate question id" in p for p in validate([make_set("q1"),
                                                                   make_set("q1")]))


class TestArms:
    def test_both_arms_see_the_identical_candidate_set(self):
        frozen = [make_set("q1"), make_set("q2")]
        a = run_arm("rag2", AlwaysFilter(True), frozen)
        b = run_arm("scaf", SCAFFilter(FilterConfig(kind="scaf")), frozen)
        assert [r.candidate_digest for r in a] == [r.candidate_digest for r in b]
        assert [r.num_candidates for r in a] == [r.num_candidates for r in b]

    def test_context_is_built_in_frozen_order(self):
        frozen = make_set(n=3)
        ids = [c.chunk_id for c in frozen.candidates]
        context = build_context(frozen, list(reversed(ids)))
        # Admission is a set; ordering comes from the frozen set, not the caller.
        assert context.index(frozen.candidates[0].text) < context.index(frozen.candidates[2].text)

    def test_a_failing_question_is_recorded_not_dropped(self):
        class Broken:
            def decide(self, question, candidates):
                raise RuntimeError("filter exploded")

        results = run_arm("broken", Broken(), [make_set("q1"), make_set("q2")])
        assert len(results) == 2
        assert all("filter exploded" in r.error for r in results)

    def test_a_filter_returning_the_wrong_count_is_an_error(self):
        class Short:
            def decide(self, question, candidates):
                return [FilterDecision(keep=True, label="x", score=1.0)]

        results = run_arm("short", Short(), [make_set("q1", n=4)])
        assert "decisions for 4 candidates" in results[0].error

    def test_admission_records_carry_dates_for_the_recency_analysis(self):
        results = run_arm("scaf", SCAFFilter(FilterConfig(kind="scaf")), [make_set()])
        assert all(d["canonical_date"] for d in results[0].decisions)


class TestFairness:
    def _arms(self, frozen):
        return (run_arm("rag2", AlwaysFilter(True), frozen),
                run_arm("scaf", SCAFFilter(FilterConfig(kind="scaf")), frozen))

    def _config(self, **overrides):
        base = {
            "arm_a_generator": "stub", "arm_b_generator": "stub",
            "arm_a_generation": {"temperature": 0.0}, "arm_b_generation": {"temperature": 0.0},
            "arm_a_index": "pmc/index", "arm_b_index": "pmc/index",
        }
        base.update(overrides)
        return base

    def test_a_correct_run_passes_every_check(self):
        frozen = [make_set("q1"), make_set("q2")]
        a, b = self._arms(frozen)
        report = fairness_report(frozen, a, b, self._config())
        assert report["all_passed"], report["failed"]

    def test_different_generators_fail(self):
        frozen = [make_set("q1")]
        a, b = self._arms(frozen)
        report = fairness_report(frozen, a, b, self._config(arm_b_generator="other"))
        assert not report["all_passed"]
        assert "same generator in both arms" in report["failed"]

    def test_different_decoding_settings_fail(self):
        frozen = [make_set("q1")]
        a, b = self._arms(frozen)
        report = fairness_report(frozen, a, b,
                                 self._config(arm_b_generation={"temperature": 0.7}))
        assert "same decoding settings in both arms" in report["failed"]

    def test_different_index_fails(self):
        frozen = [make_set("q1")]
        a, b = self._arms(frozen)
        report = fairness_report(frozen, a, b, self._config(arm_b_index="other/index"))
        assert "same corpus and index in both arms" in report["failed"]

    def test_an_arm_that_retrieved_its_own_candidates_is_caught(self):
        """The failure the whole design exists to prevent."""
        frozen = [make_set("q1")]
        a, b = self._arms(frozen)
        a[0].candidate_digest = "deadbeef"          # as if arm A had retrieved
        report = fairness_report(frozen, a, b, self._config())
        assert not report["all_passed"]
        assert "neither arm retrieved its own candidates" in report["failed"]

    def test_mismatched_candidate_sets_between_arms_are_caught(self):
        frozen = [make_set("q1")]
        a, b = self._arms(frozen)
        b[0].candidate_digest = "0" * 64
        report = fairness_report(frozen, a, b, self._config())
        assert "same candidate set and ordering in both arms" in report["failed"]

    def test_a_dropped_question_is_caught(self):
        frozen = [make_set("q1"), make_set("q2")]
        a, b = self._arms(frozen)
        report = fairness_report(frozen, a, b[:1], self._config())
        assert not report["all_passed"]


class TestSummaries:
    def test_summary_counts_failures_separately(self):
        results = [
            ArmResult("q1", "x", admitted_chunk_ids=["a"], rejected_chunk_ids=["b"]),
            ArmResult("q2", "x", error="boom"),
        ]
        summary = summarise(results)
        assert summary["questions_attempted"] == 2
        assert summary["questions_succeeded"] == 1
        assert summary["questions_failed"] == 1
        assert summary["errors"][0]["qid"] == "q2"

    def test_admission_rate_is_computed_over_successful_questions(self):
        results = [ArmResult("q1", "x", admitted_chunk_ids=["a", "b"],
                             rejected_chunk_ids=["c", "d"])]
        assert summarise(results)["admission_rate"] == pytest.approx(0.5)

    def test_recency_profile_splits_and_flags_undated(self):
        results = [ArmResult("q1", "x", decisions=[
            {"chunk_id": "a", "keep": True, "canonical_date": "2024-01-01"},
            {"chunk_id": "b", "keep": False, "canonical_date": "2005-01-01"},
            {"chunk_id": "c", "keep": True, "canonical_date": ""},
        ])]
        profile = recency_profile(results, boundary_year=2020)
        assert profile["newer"]["admitted"] == 1 and profile["newer"]["candidates"] == 1
        assert profile["older"]["admitted"] == 0 and profile["older"]["candidates"] == 1
        assert profile["undated"]["candidates"] == 1
        assert "FRB-PAIRS analysis remains pending" in profile["note"]

    def test_recency_profile_reports_no_rate_for_an_empty_bucket(self):
        profile = recency_profile([ArmResult("q1", "x", decisions=[])])
        assert profile["older"]["admission_rate"] is None

    def test_manifest_is_labelled_preliminary_and_json_serialisable(self):
        frozen = [make_set("q1")]
        a = run_arm("rag2", AlwaysFilter(True), frozen)
        b = run_arm("scaf", SCAFFilter(FilterConfig(kind="scaf")), frozen)
        manifest = build_manifest(frozen, a, b, {
            "arm_a_generator": "stub", "arm_b_generator": "stub",
            "arm_a_generation": {}, "arm_b_generation": {},
            "arm_a_index": "i", "arm_b_index": "i",
        })
        assert manifest["label"] == "PRELIMINARY / DEVELOPMENT RESULTS"
        assert manifest["fairness"]["all_passed"]
        assert manifest["frozen_set_digest"] == set_digest(frozen)
        json.dumps(manifest, default=str)
