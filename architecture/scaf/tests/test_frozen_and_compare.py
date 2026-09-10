"""Tests for frozen candidate replay and the RAG2-vs-SCAF fairness controls.

These are the tests that protect the *validity* of the comparison rather than
the behaviour of either policy. If one of them fails, no number from the
experiment can be reported.
"""

import json
import os
import sys

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
for _p in (os.path.join(_ROOT, "architecture"),
           os.path.join(_ROOT, "architecture", "rag2")):
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
        meta = save(path, original, provenance={"index": "indexes/production"})
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
            "arm_a_index": "indexes/production", "arm_b_index": "indexes/production",
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


class TestScientificPreconditions:
    """The guards that stop a prototype being reported as the scientific result.

    Each is proven to FAIL when its condition is violated -- a gate that only
    ever passes is not a gate.
    """

    def _filters(self):
        class LoadedRAG2:                      # what a real Flan-T5 filter looks like
            helpful_id, not_helpful_id = 32100, 32101
        return LoadedRAG2(), SCAFFilter(FilterConfig(
            kind="scaf", options={"document_frequency": {"amyloid": 5}, "corpus_size": 100}))

    @staticmethod
    def _trained_checkpoint(tmp_path):
        """A directory shaped like a trained HF checkpoint: config + weights."""
        directory = tmp_path / "filter-checkpoint"
        directory.mkdir(exist_ok=True)
        (directory / "config.json").write_text('{"model_type": "t5"}', encoding="utf-8")
        (directory / "model.safetensors").write_bytes(b"\x00" * 64)
        return str(directory)

    def _config(self, checkpoint=None, **overrides):
        base = {
            "arm_a_filter": "rag2_perplexity",
            "arm_a_filter_config": {"checkpoint": checkpoint},
            "retrieval_is_medcpt": True,
            "retrieval_source": "rag2 candidate cache",
            "arm_a_generator": "huggingface",
        }
        base.update(overrides)
        return base

    #: A known-good environment. The live one is used in a real run, but a test
    #: asserting "every precondition passes" must not fail merely because the
    #: developer running it has uncommitted changes.
    CLEAN_ENV = {"git_dirty": False, "git_commit": "0" * 40}

    def _run(self, tmp_path, frozen=None, env=None, **overrides):
        from scaf.compare import scientific_report
        a, b = self._filters()
        overrides.setdefault("arm_a_filter_config",
                             {"checkpoint": self._trained_checkpoint(tmp_path)})
        return scientific_report(self._config(**overrides),
                                 frozen or [make_set(f"q{i}") for i in range(20)], a, b,
                                 env=self.CLEAN_ENV if env is None else env)

    def test_a_correct_scientific_run_passes_every_precondition(self, tmp_path):
        report = self._run(tmp_path)
        assert report["all_passed"], report["failed"]

    def test_a_dirty_working_tree_fails(self, tmp_path):
        """A run whose git_commit does not describe the code that ran is not evidence."""
        report = self._run(tmp_path, env={"git_dirty": True, "git_commit": "abc123"})
        assert "working tree was clean at run time" in report["failed"]

    def test_undated_candidates_fail_the_currency_precondition(self, tmp_path):
        """Without dates SCAF's gamma is constant and the currency arm measures nothing."""
        undated = [make_set(f"q{i}") for i in range(20)]
        for fs in undated:
            for candidate in fs.candidates:
                candidate.canonical_date = ""
        report = self._run(tmp_path, frozen=undated)
        assert ("candidates carry publication dates (SCAF currency needs them)"
                in report["failed"])

    def test_dated_candidates_pass_the_currency_precondition(self, tmp_path):
        report = self._run(tmp_path)
        assert ("candidates carry publication dates (SCAF currency needs them)"
                not in report["failed"])

    def test_passthrough_arm_a_fails(self, tmp_path):
        report = self._run(tmp_path, arm_a_filter="passthrough")
        assert "Arm A is NOT passthrough" in report["failed"]
        assert "Arm A is the RAG2 perplexity filter" in report["failed"]

    def test_missing_checkpoint_fails(self, tmp_path):
        report = self._run(tmp_path, arm_a_filter_config={"checkpoint": None})
        assert "Arm A has a trained checkpoint" in report["failed"]

    def test_nonexistent_checkpoint_path_fails(self, tmp_path):
        report = self._run(tmp_path, arm_a_filter_config={"checkpoint": "/no/such/checkpoint"})
        assert "Arm A checkpoint exists on disk" in report["failed"]

    def test_a_filter_without_label_tokens_fails(self):
        """A stand-in object must not pass as the loaded Flan-T5 filter."""
        from scaf.compare import scientific_report
        _, scaf_filter = self._filters()
        report = scientific_report(self._config(), [make_set(f"q{i}") for i in range(20)],
                                   AlwaysFilter(True), scaf_filter)
        assert "Arm A filter loaded its label tokens" in report["failed"]

    def test_non_medcpt_retrieval_fails(self, tmp_path):
        report = self._run(tmp_path, retrieval_is_medcpt=False)
        assert "production MedCPT retrieval was used" in report["failed"]

    @pytest.mark.parametrize("source", [
        "lexical-dev (IDF term overlap)", "BM25 baseline", "tfidf ranking",
        "mock retrieval", "stub retriever",
    ])
    def test_development_retrieval_stand_ins_are_rejected(self, tmp_path, source):
        report = self._run(tmp_path, retrieval_source=source)
        assert "no development retrieval stand-in" in report["failed"], source

    def test_missing_generator_fails(self, tmp_path):
        assert "a generator is configured" in self._run(tmp_path, arm_a_generator="none")["failed"]
        assert "a generator is configured" in self._run(tmp_path, arm_a_generator="")["failed"]

    def test_too_few_questions_fails(self, tmp_path):
        report = self._run(tmp_path, frozen=[make_set(f"q{i}") for i in range(5)])
        assert "at least 20 questions" in report["failed"]

    def test_inactive_scaf_currency_fails(self):
        from scaf.compare import scientific_report
        a, _ = self._filters()
        dead = SCAFFilter(FilterConfig(kind="scaf", options={
            "weights": {"support": 1.0, "currency": 0.0, "corroboration": 0.0,
                        "authority": 0.0},
            "document_frequency": {"a": 1}, "corpus_size": 10}))
        report = scientific_report(self._config(), [make_set(f"q{i}") for i in range(20)],
                                   a, dead)
        assert "SCAF currency is active" in report["failed"]
        assert "SCAF authority is active" in report["failed"]

    def test_scaf_without_corpus_statistics_fails(self):
        """The v1 support collapse must not be able to reach a reported run."""
        from scaf.compare import scientific_report
        a, _ = self._filters()
        no_stats = SCAFFilter(FilterConfig(kind="scaf"))
        report = scientific_report(self._config(), [make_set(f"q{i}") for i in range(20)],
                                   a, no_stats)
        assert "SCAF support has corpus statistics" in report["failed"]

    def test_disabled_retraction_gate_fails(self):
        from scaf.compare import scientific_report
        a, _ = self._filters()
        ungated = SCAFFilter(FilterConfig(kind="scaf", options={
            "reject_retracted": False, "abstain": False,
            "document_frequency": {"a": 1}, "corpus_size": 10}))
        report = scientific_report(self._config(), [make_set(f"q{i}") for i in range(20)],
                                   a, ungated)
        assert "SCAF retraction gate is active" in report["failed"]
        assert "SCAF abstention is enabled" in report["failed"]


class TestComponentsActuallyComputed:
    def test_a_constant_component_is_reported_as_inert(self):
        from scaf.compare import components_actually_computed
        results = [ArmResult("q1", "scaf", decisions=[
            {"chunk_id": "a", "keep": True,
             "detail": {"sigma_support": 0.5, "gamma_currency": 1.0,
                        "tau_authority": 0.4, "scaf_score": 0.6}},
            {"chunk_id": "b", "keep": True,
             "detail": {"sigma_support": 0.9, "gamma_currency": 1.0,
                        "tau_authority": 0.4, "scaf_score": 0.8}},
        ])]
        out = components_actually_computed(results)
        assert out["sigma_support"]["varies"] is True
        assert out["gamma_currency"]["varies"] is False    # constant -> inert
        assert out["tau_authority"]["varies"] is False
        assert out["sigma_support"]["n"] == 2

    def test_real_scaf_output_varies_on_every_component(self):
        from scaf.compare import components_actually_computed
        frozen = [make_set("q1", n=6), make_set("q2", n=6)]
        scaf_filter = SCAFFilter(FilterConfig(kind="scaf", options={
            "document_frequency": {"amyloid": 50, "evidence": 900}, "corpus_size": 1000}))
        out = components_actually_computed(run_arm("scaf", scaf_filter, frozen))
        for key in ("sigma_support", "gamma_currency", "scaf_score"):
            assert out[key]["n"] == 12
        assert out["gamma_currency"]["varies"], "currency must respond to the dates"


class TestPairedComparison:
    """The paired view of the two arms over one frozen candidate set.

    Both arms saw the same candidates for the same question, so the difference
    per question is the admission policy and nothing else. These tests pin what
    that report may and may not claim.
    """

    @staticmethod
    def _arms(a_admitted, b_admitted, candidates=("c0", "c1", "c2", "c3")):
        arm_a, arm_b = [], []
        for i, (keep_a, keep_b) in enumerate(zip(a_admitted, b_admitted)):
            qid = f"q{i}"
            arm_a.append(ArmResult(qid=qid, arm="rag2", admitted_chunk_ids=list(keep_a),
                                   rejected_chunk_ids=[c for c in candidates if c not in keep_a]))
            arm_b.append(ArmResult(qid=qid, arm="scaf", admitted_chunk_ids=list(keep_b),
                                   rejected_chunk_ids=[c for c in candidates if c not in keep_b]))
        return arm_a, arm_b

    def test_counts_who_admits_more_per_question(self):
        from scaf.compare import paired_comparison
        arm_a, arm_b = self._arms(
            [("c0", "c1", "c2"), ("c0",), ("c0", "c1")],
            [("c0",), ("c0", "c1", "c2"), ("c0", "c1")])
        out = paired_comparison(arm_a, arm_b)
        assert out["questions_rag2_admits_more"] == 1
        assert out["questions_scaf_admits_more"] == 1
        assert out["questions_tied"] == 1
        assert out["questions_compared"] == 3

    def test_reports_median_min_and_max_not_only_the_mean(self):
        from scaf.compare import paired_comparison
        arm_a, arm_b = self._arms(
            [("c0",), ("c0", "c1"), ("c0", "c1", "c2", "c3")], [(), (), ()])
        out = paired_comparison(arm_a, arm_b)
        assert out["rag2_admitted"] == {"n": 3, "mean": 2.333, "median": 2.0, "min": 1, "max": 4}
        assert out["rag2_total_admitted"] == 7
        assert out["rag2_admission_rate"] == round(7 / 12, 6)

    def test_paired_difference_is_signed_and_also_absolute(self):
        from scaf.compare import paired_comparison
        arm_a, arm_b = self._arms([("c0", "c1"), ()], [(), ("c0", "c1")])
        out = paired_comparison(arm_a, arm_b)
        assert out["paired_difference"]["mean"] == 0.0       # +2 and -2 cancel
        assert out["paired_difference"]["mean_absolute"] == 2.0

    def test_overlap_distinguishes_same_count_from_same_evidence(self):
        """Two arms can admit equal numbers and share nothing."""
        from scaf.compare import paired_comparison
        arm_a, arm_b = self._arms([("c0", "c1")], [("c2", "c3")])
        out = paired_comparison(arm_a, arm_b)
        assert out["questions_tied"] == 1
        assert out["selected_evidence_overlap"]["mean_jaccard"] == 0.0
        assert out["selected_evidence_overlap"]["disjoint_selections"] == 1

    def test_identical_selection_scores_one(self):
        from scaf.compare import paired_comparison
        arm_a, arm_b = self._arms([("c0", "c1")], [("c1", "c0")])
        out = paired_comparison(arm_a, arm_b)
        assert out["selected_evidence_overlap"]["mean_jaccard"] == 1.0
        assert out["selected_evidence_overlap"]["identical_selections"] == 1

    def test_a_question_both_arms_reject_has_no_overlap_rather_than_zero(self):
        """Undefined is not 0.0: counting it would drag the mean down for free."""
        from scaf.compare import paired_comparison
        arm_a, arm_b = self._arms([("c0",), ()], [("c0",), ()])
        out = paired_comparison(arm_a, arm_b)
        overlap = out["selected_evidence_overlap"]
        assert overlap["measurable_questions"] == 1
        assert overlap["unmeasurable_questions"] == 1
        assert overlap["mean_jaccard"] == 1.0
        assert out["per_question"][1]["jaccard"] is None
        assert out["per_question"][1]["both_empty"] is True

    def test_a_failed_question_is_excluded_from_the_pairing(self):
        from scaf.compare import paired_comparison
        arm_a, arm_b = self._arms([("c0",), ("c1",)], [("c0",), ("c1",)])
        arm_b[1].error = "filter raised"
        out = paired_comparison(arm_a, arm_b)
        assert out["questions_compared"] == 1

    def test_per_question_rows_name_the_evidence_each_arm_admitted_alone(self):
        from scaf.compare import paired_comparison
        arm_a, arm_b = self._arms([("c0", "c1")], [("c1", "c2")])
        row = paired_comparison(arm_a, arm_b)["per_question"][0]
        assert row["rag2_only"] == ["c0"]
        assert row["scaf_only"] == ["c2"]
        assert row["intersection"] == 1 and row["union"] == 3
        assert row["jaccard"] == round(1 / 3, 4)

    def test_real_arms_produce_a_consistent_paired_report(self):
        from rag2.filtering.passthrough import PassthroughFilter
        from scaf.compare import paired_comparison, summarise
        frozen = [make_set("q1", n=6), make_set("q2", n=6)]
        scaf_filter = SCAFFilter(FilterConfig(kind="scaf", options={
            "document_frequency": {"amyloid": 50, "evidence": 900}, "corpus_size": 1000}))
        arm_a = run_arm("rag2", PassthroughFilter(), frozen)
        arm_b = run_arm("scaf", scaf_filter, frozen)
        out = paired_comparison(arm_a, arm_b)
        assert out["questions_compared"] == 2
        assert out["total_candidate_chunks"] == 12
        assert out["rag2_total_admitted"] == summarise(arm_a)["total_admitted"]
        assert out["scaf_total_admitted"] == summarise(arm_b)["total_admitted"]
        assert out["rag2_admission_rate"] == 1.0    # passthrough keeps everything


class TestMedCPTFreezeDepth:
    """Depth truncation on the MedCPT path.

    ``from_rag2_candidate_sets`` used to freeze the whole cache regardless of the
    requested depth, so a cache built at a different ``retrieval.final_top_k``
    produced a frozen set of that size while the sidecar recorded the requested
    depth. The two could disagree and the manifest's ``candidate_depth`` was then
    wrong. Truncation is of an already-ranked cache -- never a re-rank.
    """

    @staticmethod
    def _cache(n=25):
        from rag2.schema import CandidateSet, Evidence
        return [CandidateSet(
            qid="alz-001",
            rationale="a generated rationale",
            candidates=[
                Evidence(
                    text=f"evidence body {i}",
                    source="pmc-fulltext",
                    doc_id=f"PMC{i:07d}",
                    passage_id=f"c{i:02d}",
                    # ev.rank is deliberately NOT 1..n so the test can tell an
                    # preserved upstream rank from a re-numbered one.
                    rank=100 + i,
                    rerank_score=9.5 - i,
                    metadata={
                        "retrieval_score": 0.9 - i / 100,
                        "pmid": f"pmid{i}",
                        "canonical_date": "2024-01-01",
                        "carried_through": f"keep-{i}",
                    },
                )
                for i in range(n)
            ],
        )]

    @staticmethod
    def _questions():
        from rag2.schema import Question
        return {"alz-001": Question(qid="alz-001", question="Which plasma biomarker?",
                                    options={}, metadata={"time_sensitive": True})}

    def test_depth_20_keeps_exactly_20_candidates(self):
        from scaf.frozen import from_rag2_candidate_sets
        out = from_rag2_candidate_sets(self._cache(25), self._questions(), depth=20)
        assert len(out) == 1
        assert len(out[0].candidates) == 20

    def test_depth_20_keeps_the_first_20_in_upstream_order(self):
        """The MedCPT ranking is the cache's own order; [:depth] is its top-N."""
        from scaf.frozen import from_rag2_candidate_sets
        out = from_rag2_candidate_sets(self._cache(25), self._questions(), depth=20)
        assert [c.chunk_id for c in out[0].candidates] == [f"c{i:02d}" for i in range(20)]
        assert [c.text for c in out[0].candidates] == [f"evidence body {i}" for i in range(20)]
        # and nothing from beyond the cut survived
        assert "c20" not in {c.chunk_id for c in out[0].candidates}

    def test_default_depth_is_20(self):
        from scaf.frozen import from_rag2_candidate_sets
        out = from_rag2_candidate_sets(self._cache(25), self._questions())
        assert len(out[0].candidates) == 20

    def test_truncation_preserves_scores_ranks_and_metadata(self):
        from scaf.frozen import from_rag2_candidate_sets
        kept = from_rag2_candidate_sets(self._cache(25), self._questions(), depth=20)[0].candidates
        for i, candidate in enumerate(kept):
            # upstream rerank rank survives; it is NOT renumbered to 1..20
            assert candidate.rerank_rank == 100 + i
            assert candidate.rerank_score == pytest.approx(9.5 - i)
            assert candidate.retrieval_score == pytest.approx(0.9 - i / 100)
            # retrieval_rank is the position within the frozen set, as before
            assert candidate.retrieval_rank == i + 1
            assert candidate.pmid == f"pmid{i}"
            assert candidate.canonical_date == "2024-01-01"
            # leftover metadata keys are carried through untouched
            assert candidate.metadata["carried_through"] == f"keep-{i}"

    def test_a_shorter_cache_is_left_alone(self):
        """Depth is a cap, not a target: it must never pad or re-retrieve."""
        from scaf.frozen import from_rag2_candidate_sets
        out = from_rag2_candidate_sets(self._cache(8), self._questions(), depth=20)
        assert len(out[0].candidates) == 8
        assert [c.chunk_id for c in out[0].candidates] == [f"c{i:02d}" for i in range(8)]

    def test_question_fields_are_unaffected_by_truncation(self):
        from scaf.frozen import from_rag2_candidate_sets
        out = from_rag2_candidate_sets(self._cache(25), self._questions(), depth=20)[0]
        assert out.qid == "alz-001"
        assert out.question == "Which plasma biomarker?"
        assert out.query == "a generated rationale"
        assert out.question_metadata == {"time_sensitive": True}

    def test_freeze_script_passes_its_depth_through(self):
        """The CLI's --depth must reach the bridge on the medcpt path."""
        import pathlib
        source = pathlib.Path(_ROOT, "experiments", "scripts",
                              "freeze_candidates.py").read_text(encoding="utf-8")
        assert "from_rag2_cache(args.cache, questions, depth=args.depth)" in source
        assert ("from_rag2_candidate_sets(list(iter_candidates(str(cache_path))), "
                "lookup, depth=depth)") in source
