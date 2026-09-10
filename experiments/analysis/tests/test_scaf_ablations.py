"""Tests for the counterfactual re-scoring of the completed comparison.

The failure this file exists to prevent is an ablation that quietly answers a
different question from the one it is labelled with -- by drifting away from the
policy's own arithmetic, by confusing a scale change for a term effect, or by
letting a descriptive age table be mistaken for the FRB-PAIRS probe.
"""

import json
import os
import sys

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
for _p in (_ROOT, os.path.join(_ROOT, "architecture"),
           os.path.join(_ROOT, "architecture", "rag2")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from experiments.analysis.scaf_ablations import (  # noqa: E402
    AS_OF_YEAR,
    RUN_HALF_LIFE_YEARS,
    RUN_THRESHOLD,
    RUN_WEIGHTS,
    NotTheScientificRun,
    _year_fraction,
    ablation_suite,
    admission_by_age,
    agreement_with_run,
    build_ablations,
    currency_at,
    load_records,
    rank_normalised,
    score_records,
    summarise,
    variance_decomposition,
    verify_reproduction,
)

SCIENTIFIC = os.path.join(_ROOT, "experiments", "results", "rag2_vs_scaf_alzheimer",
                          "comparison_scientific", "per_question.jsonl")


def _record(chunk="c1", sigma=0.6, gamma=1.0, tau=0.4, rank=1, date="2025-01-01",
            time_sensitive=True, admit=True, **kw):
    weights = dict(RUN_WEIGHTS)
    score = (weights["support"] * sigma + weights["currency"] * gamma
             + weights["authority"] * tau)
    return {"qid": "q1", "question": "Q?", "chunk_id": chunk,
            "time_sensitive": time_sensitive, "canonical_date": date,
            "date_precision": "day", "state": "current", "retracted": "no",
            "supersession": "unknown", "rerank_rank": rank,
            "source_category": "pubmed-abstract", "authority_tier_label": "",
            "sigma": sigma, "gamma": gamma, "tau": tau,
            "rho": rank_normalised(rank), "recorded_score": score,
            "recorded_threshold": RUN_THRESHOLD, "recorded_weights": weights,
            "recorded_gate": "", "scaf_admit": admit, "rag2_admit": False,
            "rag2_score": 0.3, **kw}


# --------------------------------------------------------------------------
# The calendar must not drift from the policy's
# --------------------------------------------------------------------------
def test_year_fraction_is_identical_to_the_policys():
    """A different calendar here would regenerate gamma against a different run."""
    from scaf.policy import _year_fraction as policy_year_fraction
    dates = [f"{y}-{m:02d}-{d:02d}"
             for y in (2019, 2021, 2024, 2026)
             for m in (1, 2, 6, 12)
             for d in (1, 15, 28, 30, 31)]
    dates += ["2024", "2024-07", "", "n/a", "not-a-date", "0001-01-01"]
    for value in dates:
        assert _year_fraction(value) == policy_year_fraction(value), value


# --------------------------------------------------------------------------
# rho
# --------------------------------------------------------------------------
def test_rank_normalisation_maps_the_ends_to_one_and_zero():
    assert rank_normalised(1, 20) == 1.0
    assert rank_normalised(20, 20) == 0.0
    assert rank_normalised(10, 20) == pytest.approx(10 / 19)


def test_a_missing_rank_is_neutral_rather_than_worst():
    assert rank_normalised(None, 20) == 0.5


# --------------------------------------------------------------------------
# Currency regeneration
# --------------------------------------------------------------------------
def test_currency_follows_the_specified_half_life_curve():
    one_half_life = currency_at(_record(date="2021-01-01"), 5.0)
    assert one_half_life == pytest.approx(0.5, abs=0.02)


def test_a_claim_that_does_not_age_is_exempt():
    assert currency_at(_record(time_sensitive=False, date="1990-01-01"), 5.0) == 1.0


def test_retraction_is_the_only_hard_zero():
    assert currency_at(_record(retracted="yes"), 5.0) == 0.0
    assert currency_at(_record(date="1960-01-01"), 5.0) > 0.0


def test_future_dated_evidence_is_not_scored_above_one():
    assert currency_at(_record(date="2030-01-01"), 5.0) == 1.0


def test_supersession_discounts_rather_than_deletes():
    plain = currency_at(_record(date="2022-01-01"), 5.0)
    superseded = currency_at(_record(date="2022-01-01", supersession="PMC9"), 5.0)
    assert superseded == pytest.approx(plain * 0.5)
    assert superseded > 0.0


def test_unknown_supersession_is_not_treated_as_superseded():
    for value in ("unknown", "n/a", "", "none"):
        assert currency_at(_record(date="2022-01-01", supersession=value), 5.0) == \
            currency_at(_record(date="2022-01-01"), 5.0)


# --------------------------------------------------------------------------
# Re-scoring semantics
# --------------------------------------------------------------------------
def test_retraction_survives_every_counterfactual_weighting():
    """No weight setting may admit retracted evidence."""
    scored = score_records([_record(retracted="yes", sigma=1.0)],
                           {"support": 1.0}, 0.0)
    assert scored[0]["admit"] is False


def test_zeroing_a_weight_and_renormalising_are_different_questions():
    """Zeroing shrinks A(s), so a fixed threshold silently gets stricter."""
    records = [_record(f"c{i}", sigma=0.5 + 0.02 * i) for i in range(10)]
    zeroed = summarise(score_records(records, {**RUN_WEIGHTS, "currency": 0.0},
                                     RUN_THRESHOLD))
    renormed = summarise(score_records(records, {**RUN_WEIGHTS, "currency": 0.0},
                                       RUN_THRESHOLD, renormalise=True))
    assert renormed["admitted"] > zeroed["admitted"]


def test_renormalisation_makes_the_weights_sum_to_one():
    records = [_record()]
    scored = score_records(records, {"support": 1.0, "currency": 1.0},
                           0.0, renormalise=True)
    assert scored[0]["score"] == pytest.approx(0.5 * records[0]["sigma"]
                                               + 0.5 * records[0]["gamma"])


def test_the_suite_reports_what_changed_rather_than_only_a_new_total():
    records = [_record(f"c{i}", sigma=0.9, admit=True) for i in range(5)]
    scored = score_records(records, RUN_WEIGHTS, 0.99)
    assert agreement_with_run(scored)["newly_rejected"] == 5


# --------------------------------------------------------------------------
# Variance decomposition
# --------------------------------------------------------------------------
def test_a_zero_weight_term_gets_zero_variance_share():
    records = [_record(f"c{i}", rank=i + 1) for i in range(20)]
    shares = variance_decomposition(records)["terms"]
    assert shares["rerank"]["weight"] == 0.0
    assert shares["rerank"]["share_of_score_variance"] == 0.0
    assert shares["rerank"]["raw_sd"] > 0        # it varies; it just cannot act


def test_a_constant_term_cannot_move_admission():
    records = [_record(f"c{i}", sigma=0.1 * i, tau=0.4) for i in range(10)]
    shares = variance_decomposition(records)["terms"]
    assert shares["authority"]["raw_sd"] == 0.0
    assert shares["authority"]["share_of_score_variance"] == 0.0


def test_shares_sum_to_one():
    records = [_record(f"c{i}", sigma=0.1 * i, gamma=1 - 0.05 * i, tau=0.4 + 0.01 * i)
               for i in range(10)]
    shares = variance_decomposition(records)["terms"]
    total = sum(b["share_of_score_variance"] for b in shares.values())
    assert total == pytest.approx(1.0, abs=1e-6)


# --------------------------------------------------------------------------
# Guards
# --------------------------------------------------------------------------
def _write_run(tmp_path, mutate=None):
    """A miniature per_question.jsonl in the scientific run's shape."""
    path = tmp_path / "per_question.jsonl"
    # gamma must be what the date implies, or the precondition rightly refuses.
    gamma_for_2024 = 2.0 ** (-(AS_OF_YEAR - _year_fraction("2024-01-01")) / 5.0)
    with open(path, "w", encoding="utf-8") as handle:
        for q in range(2):
            scaf, rag2 = [], []
            for i in range(20):
                sigma, gamma, tau = 0.30 + 0.03 * i, gamma_for_2024, 0.40
                score = 0.5 * sigma + 0.3 * gamma + 0.2 * tau
                detail = {
                    "sigma_support": sigma, "gamma_currency": gamma,
                    "tau_authority": tau, "scaf_score": score, "gate": "",
                    "threshold": RUN_THRESHOLD,
                    "weights": {"support": 0.5, "currency": 0.3,
                                "corroboration": 0.0, "authority": 0.2},
                    "currency_detail": {"canonical_date": "2024-01-01",
                                        "date_precision": "day", "state": "current",
                                        "retracted": "no", "supersession": "unknown",
                                        "time_sensitive": True,
                                        "half_life_years": 5.0}}
                if mutate:
                    mutate(detail, i)
                scaf.append({"chunk_id": f"q{q}-c{i:02d}", "keep": score >= RUN_THRESHOLD,
                             "rerank_rank": i + 1, "source_category": "pubmed-abstract",
                             "authority_tier_label": "", "canonical_date": "2024-01-01",
                             "detail": detail})
                rag2.append({"chunk_id": f"q{q}-c{i:02d}", "keep": False, "score": 0.3,
                             "rerank_rank": i + 1})
            handle.write(json.dumps({"qid": f"q{q}", "question": "Q?",
                                     "time_sensitive": True,
                                     "rag2": {"decisions": rag2},
                                     "scaf": {"decisions": scaf}}) + "\n")
    return str(path)


def test_a_sound_run_passes_the_precondition(tmp_path):
    assert build_ablations(_write_run(tmp_path))["reproduction_check"]["all_passed"]


def test_a_record_that_contradicts_its_own_sub_scores_is_refused(tmp_path):
    """Re-scoring is only meaningful if the arithmetic still reproduces the run."""
    def corrupt(detail, i):
        if i == 0:
            detail["scaf_score"] = 0.99
    with pytest.raises(NotTheScientificRun):
        build_ablations(_write_run(tmp_path, corrupt))


def test_a_gamma_that_does_not_match_its_date_is_refused(tmp_path):
    """Regenerating gamma at a new half-life needs the date to be the source."""
    def corrupt(detail, i):
        detail["gamma_currency"] = 0.10
        detail["scaf_score"] = (0.5 * detail["sigma_support"] + 0.3 * 0.10
                                + 0.2 * detail["tau_authority"])
    with pytest.raises(NotTheScientificRun):
        build_ablations(_write_run(tmp_path, corrupt))


def test_admission_by_age_refuses_to_call_itself_frb_pairs():
    table = admission_by_age([_record(f"c{i}", date=f"202{i}-01-01") for i in range(5)])
    assert table["is_frb_pairs"] is False
    assert "not matched" in table["guard"]
    assert "must not be reported as a recency-bias result" in table["guard"]


def test_the_suite_lists_what_it_cannot_answer():
    """An ablation that is impossible here must be named, not omitted."""
    missing = ablation_suite([_record()])["not_executable_here"]
    for ident in ("A1", "A2", "A3", "A6", "A8", "A9", "A10", "A13", "A14"):
        assert ident in missing, ident
    assert "answer quality" in ablation_suite([_record()])["guard"]


# --------------------------------------------------------------------------
# Against the real scientific run
# --------------------------------------------------------------------------
def test_the_scientific_run_reproduces_its_own_admissions():
    """The precondition for every ablation below it."""
    report = verify_reproduction(load_records(SCIENTIFIC))
    assert report["all_passed"] is True, [c for c in report["checks"] if not c["pass"]]
    assert report["candidates"] == 600


def test_gamma_regenerates_from_the_date_at_the_runs_half_life():
    records = load_records(SCIENTIFIC)
    worst = max(abs(currency_at(r, RUN_HALF_LIFE_YEARS) - r["gamma"])
                for r in records if r["time_sensitive"])
    assert worst < 1e-4, f"as_of={AS_OF_YEAR} no longer reproduces the run"


def test_support_dominates_the_admission_score_in_the_real_run():
    """The finding this module was written to surface, pinned as a regression."""
    shares = variance_decomposition(load_records(SCIENTIFIC))["terms"]
    assert shares["support"]["share_of_score_variance"] > 0.80
    assert shares["authority"]["share_of_score_variance"] < 0.05


def test_the_real_currency_term_has_almost_no_dynamic_range():
    """A 5-year corpus cannot exercise a 5-year half-life."""
    shares = variance_decomposition(load_records(SCIENTIFIC))["terms"]
    assert shares["currency"]["raw_min"] > 0.5     # nothing is even one half-life old
