"""Tests for the clustered statistics.

The failure this file exists to prevent is a confident interval. 600 candidates
clustered in 30 questions will happily produce a tiny interval and a tiny
p-value if the clustering is ignored, and the resulting claim would be an
artifact of the resampling unit rather than of the evidence.
"""

import json
import math
import os
import sys

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from experiments.analysis.scaf_ablations import load_records  # noqa: E402
from experiments.analysis.thesis_statistics import (  # noqa: E402
    admission_statistics,
    binomial_two_sided,
    bootstrap_p_value,
    build_statistics,
    cluster_bootstrap,
    cohens_h,
    evidence_quality_statistics,
    holm_bonferroni,
    interpret_h,
    mcnemar_exact,
    spearman,
)

RESULTS = os.path.join(_ROOT, "experiments", "results", "rag2_vs_scaf_alzheimer")
SCIENTIFIC = os.path.join(RESULTS, "comparison_scientific", "per_question.jsonl")
SHEET = os.path.join(RESULTS, "evidence_quality", "annotation_sheet_v2.jsonl")

FAST = 400          # resamples for the unit tests; the real runs use 10,000


def _candidate(qid, chunk, rag2=False, scaf=True, **kw):
    return {"qid": qid, "chunk_id": chunk, "rag2_admit": rag2, "scaf_admit": scaf,
            "rag2_score": 0.3, "recorded_score": 0.6, "sigma": 0.5, "gamma": 0.9,
            "rho": 0.5, **kw}


# --------------------------------------------------------------------------
# Exact tests
# --------------------------------------------------------------------------
def test_binomial_matches_hand_computable_cases():
    assert binomial_two_sided(0, 0) == 1.0
    assert binomial_two_sided(5, 10) == pytest.approx(1.0)
    # 10 successes in 10 fair trials: 2 * (1/1024)
    assert binomial_two_sided(10, 10) == pytest.approx(2 / 1024)


def test_mcnemar_ignores_concordant_pairs():
    """Pairs where both arms agree carry no information about a difference."""
    result = mcnemar_exact(30, 0)
    assert result["discordant_pairs"] == 30
    assert result["p_value"] < 1e-6


def test_mcnemar_is_symmetric():
    assert mcnemar_exact(7, 3)["p_value"] == mcnemar_exact(3, 7)["p_value"]


def test_mcnemar_says_it_is_anti_conservative():
    """The number a reader expects, with the reason it cannot stand alone."""
    assert "ANTI-CONSERVATIVE" in mcnemar_exact(5, 1)["independence_assumption"]


def test_cohens_h_is_zero_for_equal_proportions_and_signed():
    assert cohens_h(0.5, 0.5) == pytest.approx(0.0)
    assert cohens_h(0.9, 0.1) > 0
    assert cohens_h(0.1, 0.9) < 0


def test_cohens_h_does_not_depend_on_sample_size():
    """That is the point of an effect size, and the reason it is reported."""
    assert interpret_h(cohens_h(0.96, 0.007)) == "large"
    assert interpret_h(0.1) == "negligible"


# --------------------------------------------------------------------------
# Clustering is the whole point
# --------------------------------------------------------------------------
def test_clustered_interval_is_wider_than_ignoring_the_clustering():
    """The regression this file exists for.

    Twenty identical rows per question carry one question's worth of
    information, not twenty. Resampling rows would hide that.
    """
    clusters = [[{"v": float(q % 2)} for _ in range(20)] for q in range(30)]
    mean = lambda rows: sum(r["v"] for r in rows) / len(rows)  # noqa: E731
    clustered = cluster_bootstrap(clusters, mean, resamples=FAST)
    rows = [[row] for cluster in clusters for row in cluster]
    per_row = cluster_bootstrap(rows, mean, resamples=FAST)
    clustered_width = clustered["ci_high"] - clustered["ci_low"]
    per_row_width = per_row["ci_high"] - per_row["ci_low"]
    assert clustered_width > per_row_width * 2


def test_the_bootstrap_is_reproducible():
    clusters = [[{"v": q * 0.1}] for q in range(20)]
    mean = lambda rows: sum(r["v"] for r in rows) / len(rows)  # noqa: E731
    first = cluster_bootstrap(clusters, mean, resamples=FAST, seed=7)
    second = cluster_bootstrap(clusters, mean, resamples=FAST, seed=7)
    assert first == second


def test_a_different_seed_gives_a_different_interval():
    clusters = [[{"v": q * 0.1}] for q in range(20)]
    mean = lambda rows: sum(r["v"] for r in rows) / len(rows)  # noqa: E731
    a = cluster_bootstrap(clusters, mean, resamples=FAST, seed=1)
    b = cluster_bootstrap(clusters, mean, resamples=FAST, seed=2)
    assert (a["ci_low"], a["ci_high"]) != (b["ci_low"], b["ci_high"])
    assert a["point"] == b["point"]          # the estimate itself is not random


def test_the_interval_brackets_the_point_estimate():
    clusters = [[{"v": float(q)}] for q in range(30)]
    mean = lambda rows: sum(r["v"] for r in rows) / len(rows)  # noqa: E731
    result = cluster_bootstrap(clusters, mean, resamples=FAST)
    assert result["ci_low"] <= result["point"] <= result["ci_high"]


def test_an_undefined_statistic_returns_none_rather_than_guessing():
    result = cluster_bootstrap([[{"v": 1.0}]], lambda rows: None, resamples=FAST)
    assert result["point"] is None


def test_excludes_zero_is_reported_honestly():
    positive = cluster_bootstrap([[{"v": 5.0 + q * 0.01}] for q in range(30)],
                                 lambda r: sum(x["v"] for x in r) / len(r),
                                 resamples=FAST)
    straddling = cluster_bootstrap([[{"v": (-1.0) ** q}] for q in range(30)],
                                   lambda r: sum(x["v"] for x in r) / len(r),
                                   resamples=FAST)
    assert positive["excludes_zero"] is True
    assert straddling["excludes_zero"] is False


# --------------------------------------------------------------------------
# Multiplicity
# --------------------------------------------------------------------------
def test_bootstrap_p_value_is_never_exactly_zero():
    assert bootstrap_p_value([1.0] * 1000) > 0.0


def test_holm_is_stricter_than_the_uncorrected_threshold():
    verdicts = holm_bonferroni({"a": 0.02, "b": 0.03, "c": 0.04, "d": 0.9, "e": 0.95})
    assert verdicts["a"]["holm_threshold"] == pytest.approx(0.05 / 5)
    assert verdicts["a"]["survives_holm"] is False      # 0.02 > 0.01


def test_holm_steps_down_and_stops_at_the_first_failure():
    verdicts = holm_bonferroni({"a": 0.001, "b": 0.20, "c": 0.30})
    assert verdicts["a"]["survives_holm"] is True
    assert verdicts["b"]["survives_holm"] is False
    assert verdicts["c"]["survives_holm"] is False


# --------------------------------------------------------------------------
# Admission
# --------------------------------------------------------------------------
def test_admission_contingency_counts_the_four_cells():
    records = ([_candidate("q1", f"a{i}", rag2=True, scaf=True) for i in range(3)]
               + [_candidate("q1", f"b{i}", rag2=False, scaf=True) for i in range(5)]
               + [_candidate("q2", f"c{i}", rag2=True, scaf=False) for i in range(2)]
               + [_candidate("q2", f"d{i}", rag2=False, scaf=False) for i in range(4)])
    table = admission_statistics(records, resamples=FAST)["contingency"]
    assert table == {"both_admitted": 3, "scaf_only": 5, "rag2_only": 2, "neither": 4}


def test_admission_refuses_to_call_a_higher_rate_better():
    records = [_candidate("q1", f"c{i}") for i in range(20)]
    guard = admission_statistics(records, resamples=FAST)["interpretation_guard"]
    assert "not a better one" in guard


# --------------------------------------------------------------------------
# Evidence quality
# --------------------------------------------------------------------------
def test_unresolvable_annotations_are_listed_rather_than_dropped():
    annotations = [{"qid": "q1", "chunk_id": "missing", "human_label": 2}]
    result = evidence_quality_statistics(annotations, {}, resamples=FAST)
    assert result["rows"] == 0
    assert result["unresolved"] == ["('q1', 'missing')"]


def test_a_perfect_association_is_recovered():
    scores = {("q%d" % q, "c%d" % i): {"s": float(i)}
              for q in range(10) for i in range(3)}
    annotations = [{"qid": "q%d" % q, "chunk_id": "c%d" % i, "human_label": i}
                   for q in range(10) for i in range(3)]
    result = evidence_quality_statistics(annotations, scores, resamples=FAST)
    assert result["rank_association"]["s"]["point"] == pytest.approx(1.0)


def test_evidence_quality_states_that_correlation_is_not_validation():
    scores = {("q1", "c1"): {"s": 1.0}}
    result = evidence_quality_statistics(
        [{"qid": "q1", "chunk_id": "c1", "human_label": 1}], scores, resamples=FAST)
    assert "is not evidence that the support score measures" in result["guard"]


# --------------------------------------------------------------------------
# Against the real run
# --------------------------------------------------------------------------
def test_rag2_admissions_are_a_subset_of_scaf_admissions():
    """A structural fact of this run, and the reason 'rag2_only' is empty."""
    table = admission_statistics(load_records(SCIENTIFIC), resamples=FAST)["contingency"]
    assert table["rag2_only"] == 0
    assert table["both_admitted"] == 4


def test_the_admission_difference_is_large_and_precisely_estimated():
    stats = admission_statistics(load_records(SCIENTIFIC), resamples=2000)
    difference = stats["difference_clustered"]
    assert difference["ci_low"] > 0.85
    assert stats["effect_size"]["magnitude"] == "large"


def test_the_scaf_score_association_with_human_labels_includes_zero():
    """The finding: SCAF's admission score is not distinguishable from no signal."""
    records = load_records(SCIENTIFIC)
    annotations = [json.loads(l) for l in open(SHEET, encoding="utf-8") if l.strip()]
    stats = build_statistics(records, annotations, resamples=2000)
    scaf = stats["evidence_quality"]["rank_association"]["scaf_score"]
    assert scaf["ci_low"] < 0.0 < scaf["ci_high"]
    assert scaf["holm"]["survives_holm"] is False


def test_the_reranker_rank_is_the_signal_that_survives_correction():
    """The term the earlier implementation dropped is the only one that tracks."""
    records = load_records(SCIENTIFIC)
    annotations = [json.loads(l) for l in open(SHEET, encoding="utf-8") if l.strip()]
    stats = build_statistics(records, annotations, resamples=2000)
    rho = stats["evidence_quality"]["rank_association"]["rho_rerank"]
    assert rho["point"] > 0.15
    assert rho["ci_low"] > 0.0
    assert rho["holm"]["survives_holm"] is True


def test_every_annotation_resolves_against_the_scientific_run():
    """Provenance: the labels must attach to the run's own candidate ids."""
    records = load_records(SCIENTIFIC)
    annotations = [json.loads(l) for l in open(SHEET, encoding="utf-8") if l.strip()]
    stats = build_statistics(records, annotations, resamples=FAST)
    assert stats["evidence_quality"]["unresolved"] == []
    assert stats["evidence_quality"]["rows"] == 120


def test_untestable_hypotheses_are_named_rather_than_omitted():
    records = load_records(SCIENTIFIC)[:20]
    stats = build_statistics(records, [], resamples=FAST)
    missing = stats["not_tested_here"]
    assert any("H1" in key for key in missing)
    assert any("FRB-PAIRS" in value for value in missing.values())
