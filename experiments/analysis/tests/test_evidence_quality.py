"""Evidence-quality validation: the rules that decide what the labels can mean.

A wrong label here is invisible downstream -- no test, no manifest and no
precondition can catch it -- so the rules that select rows, validate labels and
turn them into a claim are pinned here.
"""

import json
import os
import sys

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from experiments.analysis.evidence_quality import (
    BLIND_FIELDS,
    VALID_LABELS,
    analyse,
    attach_candidate_text,
    bin_of,
    check_annotations,
    component_diagnostics,
    load_decisions,
    matched_budget,
    qualitative_cases,
    spearman,
    split_blind_and_key,
    stratified_sample,
    tertile_edges,
    time_sensitivity_split,
)


def _decision(chunk_id, rag2_score, scaf_score, rank=1, sigma=0.5, gamma=0.9, tau=0.45):
    rag2 = {"chunk_id": chunk_id, "score": rag2_score, "keep": rag2_score >= 0.5,
            "label": "[HELPFUL]" if rag2_score >= 0.5 else "[NOT_HELPFUL]",
            "rerank_rank": rank, "source_category": "pmc-fulltext",
            "canonical_date": "2024-01-01", "authority_tier_label": ""}
    scaf = {"chunk_id": chunk_id, "score": scaf_score, "keep": scaf_score >= 0.45,
            "label": "admit", "rerank_rank": rank, "source_category": "pmc-fulltext",
            "canonical_date": "2024-01-01", "authority_tier_label": "",
            "detail": {"sigma_support": sigma, "gamma_currency": gamma,
                       "tau_authority": tau, "rho_corroboration": 0.0,
                       "scaf_score": scaf_score, "threshold": 0.45, "gate": ""}}
    return rag2, scaf


def _write_per_question(path, questions=3, per=6, time_sensitive=True):
    with open(path, "w", encoding="utf-8") as fh:
        for q in range(questions):
            rag2, scaf = [], []
            for i in range(per):
                r, s = _decision(f"c{q}-{i:02d}", 0.20 + 0.08 * i, 0.30 + 0.09 * i, rank=i + 1)
                rag2.append(r); scaf.append(s)
            fh.write(json.dumps({
                "qid": f"alz-{q:03d}", "question": f"Question {q}?",
                "time_sensitive": time_sensitive, "num_candidates": per,
                "rag2": {"decisions": rag2, "admitted_chunk_ids":
                         [d["chunk_id"] for d in rag2 if d["keep"]]},
                "scaf": {"decisions": scaf, "admitted_chunk_ids":
                         [d["chunk_id"] for d in scaf if d["keep"]]},
            }) + "\n")
    return path


def _write_frozen(path, questions=3, per=6):
    with open(path, "w", encoding="utf-8") as fh:
        for q in range(questions):
            fh.write(json.dumps({
                "qid": f"alz-{q:03d}", "question": f"Question {q}?",
                "candidates": [{"chunk_id": f"c{q}-{i:02d}",
                                "text": f"passage {q}-{i} about amyloid",
                                "title": f"Title {q}-{i}", "pmid": f"pmid{q}{i}",
                                "pmcid": f"PMC{q}{i}"} for i in range(per)],
            }) + "\n")
    return path


# --------------------------------------------------------------------------
class TestLoading:
    def test_decisions_are_joined_per_candidate(self, tmp_path):
        records = load_decisions(_write_per_question(str(tmp_path / "pq.jsonl")))
        assert len(records) == 18
        first = records[0]
        for field in ("qid", "chunk_id", "rag2_score", "scaf_score",
                      "scaf_sigma_support", "scaf_gamma_currency", "scaf_tau_authority"):
            assert field in first

    def test_arms_scoring_different_candidates_is_refused(self, tmp_path):
        """A mismatch means the file is not the paired artifact it claims to be."""
        path = tmp_path / "bad.jsonl"
        rag2, scaf = _decision("a", 0.3, 0.6)
        other_rag2, other_scaf = _decision("b", 0.3, 0.6)
        path.write_text(json.dumps({
            "qid": "q", "question": "?", "rag2": {"decisions": [rag2]},
            "scaf": {"decisions": [other_scaf]}}) + "\n", encoding="utf-8")
        with pytest.raises(ValueError, match="same chunk ids"):
            load_decisions(str(path))

    def test_candidate_text_is_joined_from_the_frozen_set(self, tmp_path):
        records = load_decisions(_write_per_question(str(tmp_path / "pq.jsonl")))
        enriched, missing = attach_candidate_text(
            records, _write_frozen(str(tmp_path / "frozen.jsonl")))
        assert missing == []
        assert len(enriched) == 18
        assert enriched[0]["candidate_text"].startswith("passage ")

    def test_a_mismatched_frozen_file_reports_every_missing_id(self, tmp_path):
        """Annotating a different candidate set would answer a different question."""
        records = load_decisions(_write_per_question(str(tmp_path / "pq.jsonl")))
        enriched, missing = attach_candidate_text(
            records, _write_frozen(str(tmp_path / "other.jsonl"), questions=1))
        assert len(missing) == 12 and len(enriched) == 6


# --------------------------------------------------------------------------
class TestSampling:
    def test_tertiles_split_into_thirds(self):
        assert bin_of(1, tertile_edges(list(range(9)))) == "low"
        assert bin_of(4, tertile_edges(list(range(9)))) == "medium"
        assert bin_of(8, tertile_edges(list(range(9)))) == "high"

    def test_sample_is_deterministic_for_a_fixed_seed(self, tmp_path):
        records = load_decisions(_write_per_question(str(tmp_path / "pq.jsonl"), questions=6))
        a, _ = stratified_sample(records, size=10, seed=42)
        b, _ = stratified_sample(records, size=10, seed=42)
        assert [r["annotation_id"] for r in a] == [r["annotation_id"] for r in b]
        assert [r["chunk_id"] for r in a] == [r["chunk_id"] for r in b]

    def test_every_rag2_admission_is_forced_into_the_sample(self, tmp_path):
        """The RAG2-admits cell is tiny; without forcing it is never observed."""
        records = load_decisions(_write_per_question(str(tmp_path / "pq.jsonl"), questions=6))
        admitted = {r["chunk_id"] for r in records if r["rag2_admitted"]}
        assert admitted, "fixture should contain some RAG2 admissions"
        sample, prov = stratified_sample(records, size=10, seed=42)
        assert admitted <= {r["chunk_id"] for r in sample}
        assert prov["forced_rag2_admitted"] == len(admitted)

    def test_sample_spans_both_score_ranges(self, tmp_path):
        records = load_decisions(_write_per_question(str(tmp_path / "pq.jsonl"), questions=8))
        sample, _ = stratified_sample(records, size=24, seed=42)
        assert len({r["scaf_bin"] for r in sample}) >= 2
        assert len({r["rag2_bin"] for r in sample}) >= 2

    def test_bins_come_from_the_whole_population_not_the_sample(self, tmp_path):
        """Binning after sampling would let the bins manufacture a trend."""
        records = load_decisions(_write_per_question(str(tmp_path / "pq.jsonl"), questions=8))
        _, prov = stratified_sample(records, size=12, seed=42)
        assert prov["population"] == len(records)
        assert prov["scaf_tertile_edges"] == [round(e, 6) for e in
                                              tertile_edges([r["scaf_score"] for r in records])]


# --------------------------------------------------------------------------
class TestBlinding:
    def test_the_annotation_sheet_hides_every_machine_score(self):
        """Judging with the score in view makes the labels measure deference to it."""
        record = {"annotation_id": "eq-0001", "qid": "q", "question": "Q?",
                  "chunk_id": "c", "title": "T", "candidate_text": "text",
                  "canonical_date": "2024-01-01", "source_category": "pmc-fulltext",
                  "scaf_score": 0.7, "rag2_score": 0.3, "scaf_admitted": True,
                  "rag2_admitted": False, "scaf_sigma_support": 0.6}
        blind, key = split_blind_and_key([record])
        assert set(blind[0]) == set(BLIND_FIELDS)
        serialised = json.dumps(blind[0])
        for banned in ("scaf_score", "rag2_score", "scaf_admitted", "rag2_admitted",
                       "sigma_support"):
            assert banned not in serialised
        assert blind[0]["human_label"] == ""
        # nothing is lost -- the machine values live in the key
        assert key[0]["scaf_score"] == 0.7 and key[0]["rag2_score"] == 0.3


# --------------------------------------------------------------------------
class TestCompleteness:
    def test_missing_labels_are_counted_not_imputed(self):
        rows = [{"annotation_id": "eq-0001", "human_label": "2"},
                {"annotation_id": "eq-0002", "human_label": ""},
                {"annotation_id": "eq-0003", "human_label": "  "}]
        report = check_annotations(rows)
        assert report["completed"] == 1 and report["missing"] == 2
        assert report["ready_for_analysis"] is False

    def test_invalid_labels_are_flagged(self):
        rows = [{"annotation_id": "a", "human_label": "3"},
                {"annotation_id": "b", "human_label": "yes"},
                {"annotation_id": "c", "human_label": "-1"}]
        report = check_annotations(rows)
        assert report["invalid"] == 3 and report["completed"] == 0
        assert VALID_LABELS == frozenset({0, 1, 2})

    def test_duplicate_annotation_ids_are_flagged(self):
        rows = [{"annotation_id": "eq-0001", "human_label": "1"},
                {"annotation_id": "eq-0001", "human_label": "2"}]
        assert check_annotations(rows)["duplicate_annotation_ids"] == ["eq-0001"]

    def test_a_complete_valid_sheet_is_ready(self):
        rows = [{"annotation_id": f"eq-{i}", "human_label": str(i % 3)} for i in range(6)]
        report = check_annotations(rows)
        assert report["ready_for_analysis"] is True
        assert report["label_distribution"] == {0: 2, 1: 2, 2: 2}


# --------------------------------------------------------------------------
class TestAnalysis:
    def test_spearman_detects_monotone_agreement_and_disagreement(self):
        assert spearman([1, 2, 3, 4], [1, 2, 3, 4]) == 1.0
        assert spearman([1, 2, 3, 4], [4, 3, 2, 1]) == -1.0
        assert spearman([1, 1, 1, 1], [1, 2, 3, 4]) is None   # no variance

    def test_analysis_refuses_on_too_few_labels(self):
        assert "error" in analyse([{"human_label": "2", "scaf_score": 0.5}])

    def test_a_score_that_tracks_human_judgement_shows_positive_association(self):
        rows = [{"annotation_id": f"eq-{i}", "human_label": str(min(2, i // 3)),
                 "scaf_score": 0.1 * i, "rag2_score": 0.9 - 0.1 * i,
                 "scaf_sigma_support": 0.1 * i, "scaf_gamma_currency": 0.9,
                 "scaf_tau_authority": 0.45,
                 "scaf_admitted": i >= 4, "rag2_admitted": i < 4,
                 "scaf_bin": "low" if i < 3 else ("medium" if i < 6 else "high"),
                 "rag2_bin": "high" if i < 3 else ("medium" if i < 6 else "low")}
                for i in range(9)]
        result = analyse(rows)
        assoc = result["rank_association_with_human_label"]
        assert assoc["scaf_score"]["spearman"] > 0.8
        assert assoc["rag2_score"]["spearman"] < -0.8      # deliberately anti-correlated
        assert result["by_score_bin"]["scaf"]["high"]["mean"] >= \
               result["by_score_bin"]["scaf"]["low"]["mean"]

    def test_disagreement_cells_are_reported_in_both_directions(self):
        rows = [{"annotation_id": "a", "human_label": "2", "scaf_score": .8, "rag2_score": .2,
                 "scaf_admitted": True, "rag2_admitted": False},
                {"annotation_id": "b", "human_label": "0", "scaf_score": .2, "rag2_score": .8,
                 "scaf_admitted": False, "rag2_admitted": True},
                {"annotation_id": "c", "human_label": "1", "scaf_score": .8, "rag2_score": .8,
                 "scaf_admitted": True, "rag2_admitted": True}]
        cases = analyse(rows)["disagreement_cases"]
        assert cases["scaf_admits_rag2_rejects"]["n"] == 1
        assert cases["rag2_admits_scaf_rejects"]["n"] == 1
        assert cases["both_admit"]["n"] == 1


# --------------------------------------------------------------------------
class TestOfflineDiagnostics:
    def test_matched_budget_caps_both_arms_at_the_same_k(self, tmp_path):
        records = load_decisions(_write_per_question(str(tmp_path / "pq.jsonl"),
                                                     questions=3, per=6))
        out = matched_budget(records, ks=(1, 3))
        assert out["k"]["1"]["possible_total"] == 3      # 1 passage x 3 questions
        assert out["k"]["3"]["possible_total"] == 9
        assert 0.0 <= out["k"]["3"]["mean_jaccard"] <= 1.0

    def test_component_variants_record_their_own_score_ceiling(self, tmp_path):
        """Reduced variants cannot reach 1.0, so a fixed threshold is stricter."""
        records = load_decisions(_write_per_question(str(tmp_path / "pq.jsonl")))
        variants = component_diagnostics(records)["variants"]
        assert variants["support_only"]["max_possible_score"] == 1.0
        assert variants["support_currency"]["max_possible_score"] == 0.8
        assert variants["support_authority"]["max_possible_score"] == 0.7
        assert variants["full_scaf"]["max_possible_score"] == 1.0

    def test_component_arithmetic_matches_a_hand_computation(self, tmp_path):
        records = load_decisions(_write_per_question(str(tmp_path / "pq.jsonl")))
        # every fixture row has sigma=0.5 -> support_only score 0.5 >= 0.45
        assert component_diagnostics(records)["variants"]["support_only"]["admitted"] == len(records)

    def test_time_split_uses_the_committed_flag(self, tmp_path):
        records = load_decisions(_write_per_question(str(tmp_path / "pq.jsonl"),
                                                     time_sensitive=False))
        groups = time_sensitivity_split(records)["groups"]
        assert "False" in groups and groups["False"]["questions"] == 3

    def test_qualitative_selection_is_symmetric_between_arms(self, tmp_path):
        records = load_decisions(_write_per_question(str(tmp_path / "pq.jsonl"), questions=4))
        cases = qualitative_cases(records)
        assert "scaf_admits_rag2_rejects_highest_scaf" in cases
        assert "rag2_admits_scaf_rejects" in cases
        assert "rule" in cases
