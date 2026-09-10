"""Tests for the Filter Recency-Bias Probe.

Three failures this file exists to prevent, in order of how badly each would
damage the thesis:

1. a probe built on thesis-authored pairs, which is the exact circularity
   proposal 5.2 revised itself to remove;
2. a primary result reported when the permutation control failed, which
   proposal 4.6 makes blocking;
3. an interval that looks tight because the resampling unit drifted away from
   the matched pair.

The filter is injected, so all of this is testable with no model and no GPU.
"""

import json
import os
import sys

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from experiments.recency_bias.frb_pairs import (  # noqa: E402
    DEFAULT_LENGTH_TOLERANCE,
    TARGET_PAIRS,
    ProvenanceViolation,
    UnusableDataset,
    batched_admit_fn,
    build_pairs,
    digest_pairs,
    equivalence,
    load_dataset,
    paired_bootstrap,
    permute,
    probe,
    run_probe,
)

FAST = 500


def _pair(pair_id="p1", older_year=2015, newer_year=2023, tokens=100, question="Q?"):
    text = " ".join(["word"] * tokens)
    return {"pair_id": pair_id, "question": question, "tier": "systematic-review",
            "older": {"text": text, "year": float(older_year),
                      "date": f"{older_year}-01-01", "tokens": tokens},
            "newer": {"text": text + " newer", "year": float(newer_year),
                      "date": f"{newer_year}-01-01", "tokens": tokens + 1},
            "age_gap_years": float(newer_year - older_year)}


def _write(tmp_path, rows, name="medchangeqa.jsonl"):
    path = tmp_path / name
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    return str(path)


def _row(i=0, old_year=2015, new_year=2023, old_tokens=100, new_tokens=100):
    return {"id": f"mcq-{i:03d}", "question": f"Does treatment {i} work?",
            "old_abstract": " ".join(["old"] * old_tokens),
            "old_date": f"{old_year}-03-01",
            "new_abstract": " ".join(["new"] * new_tokens),
            "new_date": f"{new_year}-06-01",
            "source_tier": "systematic-review"}


# --------------------------------------------------------------------------
# The provenance firewall
# --------------------------------------------------------------------------
def test_thesis_curated_provenance_is_refused(tmp_path):
    """The circularity proposal 5.2 exists to prevent."""
    path = _write(tmp_path, [_row()])
    with pytest.raises(ProvenanceViolation):
        load_dataset(path, provenance="thesis-curated AD guideline corpus")


def test_an_external_provenance_is_accepted(tmp_path):
    items = load_dataset(_write(tmp_path, [_row()]),
                         provenance="MedChangeQA (Vladika et al., EMNLP 2025)")
    assert len(items) == 1


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
def test_a_wrong_field_map_fails_loudly_rather_than_yielding_nothing(tmp_path):
    with pytest.raises(UnusableDataset) as caught:
        load_dataset(_write(tmp_path, [_row()]),
                     field_map={"older_text": "not_a_column"})
    assert "field-map" in str(caught.value)


def test_a_pair_whose_newer_side_is_not_newer_is_rejected(tmp_path):
    rows = [_row(0, old_year=2023, new_year=2015)]
    with pytest.raises(UnusableDataset):
        load_dataset(_write(tmp_path, rows))


def test_items_missing_a_date_are_skipped_not_guessed(tmp_path):
    bad = _row(1)
    bad["old_date"] = ""
    items = load_dataset(_write(tmp_path, [_row(0), bad]))
    assert len(items) == 1


def test_a_json_array_loads_as_well_as_jsonl(tmp_path):
    path = tmp_path / "medchangeqa.json"
    path.write_text(json.dumps([_row(0), _row(1)]), encoding="utf-8")
    assert len(load_dataset(str(path))) == 2


def test_a_year_only_date_is_accepted(tmp_path):
    row = _row()
    row["old_date"], row["new_date"] = "2015", "2023"
    assert len(load_dataset(_write(tmp_path, [row]))) == 1


# --------------------------------------------------------------------------
# Pair construction
# --------------------------------------------------------------------------
def test_length_mismatched_pairs_are_excluded_and_counted(tmp_path):
    items = load_dataset(_write(tmp_path, [
        _row(0, old_tokens=100, new_tokens=100),
        _row(1, old_tokens=100, new_tokens=400)]))
    built = build_pairs(items)
    assert built["constructed"] == 1
    assert built["excluded"]["token length outside the tolerance band"] == 1


def test_the_tolerance_band_is_reported_not_hidden(tmp_path):
    items = load_dataset(_write(tmp_path, [_row()]))
    assert build_pairs(items)["length_tolerance"] == DEFAULT_LENGTH_TOLERANCE


def test_falling_short_of_the_target_pair_count_is_flagged(tmp_path):
    items = load_dataset(_write(tmp_path, [_row(i) for i in range(10)]))
    built = build_pairs(items)
    assert built["below_target"] is True
    assert built["meets_target"] is False


def test_meeting_the_target_pair_count_is_recognised(tmp_path):
    items = load_dataset(_write(tmp_path, [_row(i) for i in range(TARGET_PAIRS[0])]))
    assert build_pairs(items)["meets_target"] is True


# --------------------------------------------------------------------------
# The permutation control
# --------------------------------------------------------------------------
def test_permutation_preserves_the_pair_count_and_the_texts():
    pairs = [_pair(f"p{i}") for i in range(50)]
    permuted = permute(pairs)
    assert len(permuted) == len(pairs)
    original = {(p["older"]["text"], p["newer"]["text"]) for p in pairs}
    swapped = {(p["newer"]["text"], p["older"]["text"]) for p in permuted}
    assert swapped <= original | {tuple(reversed(t)) for t in original}


def test_permutation_actually_flips_roughly_half():
    permuted = permute([_pair(f"p{i}") for i in range(200)])
    flipped = sum(1 for p in permuted if p["flipped"])
    assert 60 < flipped < 140


def test_permutation_is_reproducible():
    pairs = [_pair(f"p{i}") for i in range(30)]
    assert [p["flipped"] for p in permute(pairs, seed=5)] == \
        [p["flipped"] for p in permute(pairs, seed=5)]


# --------------------------------------------------------------------------
# The estimator
# --------------------------------------------------------------------------
def test_a_filter_indifferent_to_age_gives_delta_zero():
    pairs = [_pair(f"p{i}") for i in range(60)]
    result = probe(pairs, lambda q, t: 0.5, resamples=FAST)
    assert result["delta"]["point"] == pytest.approx(0.0)
    assert result["delta"]["excludes_zero"] is False


def test_a_filter_that_prefers_older_evidence_is_detected():
    """The hypothesised direction, injected on purpose."""
    pairs = [_pair(f"p{i}") for i in range(80)]
    result = probe(pairs, lambda q, t: 0.9 if "newer" not in t else 0.2,
                   resamples=FAST)
    assert result["delta"]["point"] > 0.5
    assert result["delta"]["excludes_zero"] is True
    assert "H1 direction" in result["direction"]


def test_a_filter_that_prefers_newer_evidence_reports_the_opposite_direction():
    pairs = [_pair(f"p{i}") for i in range(80)]
    result = probe(pairs, lambda q, t: 0.2 if "newer" not in t else 0.9,
                   resamples=FAST)
    assert result["delta"]["point"] < 0
    assert "opposite to H1" in result["direction"]


def test_an_inconclusive_result_is_named_as_inconclusive():
    """Proposal 8: the most likely adverse outcome, and it must not read as a null."""
    pairs = [_pair(f"p{i}") for i in range(12)]
    calls = {"n": 0}

    def noisy(question, text):
        calls["n"] += 1
        return 0.5 + 0.4 * ((calls["n"] % 5) - 2) / 2.0

    result = probe(pairs, noisy, resamples=FAST)
    if not result["delta"]["excludes_zero"]:
        assert "inconclusive" in result["direction"]


def test_the_resampling_unit_is_the_matched_pair():
    interval = paired_bootstrap([0.1] * 40, resamples=FAST)
    assert interval["unit_of_analysis"] == "matched pair"
    assert interval["pairs"] == 40


def test_the_interval_brackets_the_point_estimate():
    interval = paired_bootstrap([0.1 * i for i in range(-10, 11)], resamples=FAST)
    assert interval["ci_low"] <= interval["point"] <= interval["ci_high"]


# --------------------------------------------------------------------------
# Equivalence, and the blocking gate
# --------------------------------------------------------------------------
def test_equivalence_requires_containment_not_merely_overlapping_zero():
    """A wide interval containing zero is not a passing control."""
    wide = {"ci_low": -0.4, "ci_high": 0.4, "point": 0.0}
    assert equivalence(wide, band=0.05)["passed"] is False
    tight = {"ci_low": -0.01, "ci_high": 0.01, "point": 0.0}
    assert equivalence(tight, band=0.05)["passed"] is True


def test_the_band_is_declared_as_pre_specified():
    assert "before the probe is run" in equivalence(
        {"ci_low": -0.01, "ci_high": 0.01, "point": 0.0}, 0.05)["pre_specified"]


def test_h1_is_not_reportable_when_the_control_fails():
    """The blocking gate. A probe that fails V1 may not report a primary result."""
    pairs = [_pair(f"p{i}") for i in range(60)]
    # A filter keyed to the passage text, not to age: permuting cannot cancel it.
    result = run_probe(pairs, lambda q, t: 0.9 if "newer" not in t else 0.1,
                       equivalence_band=0.02, resamples=FAST)
    assert result["H4_equivalence"]["passed"] is False
    assert result["reportable"] is False
    assert "BLOCKING" in result["reportability_note"] or \
        "BLOCKING" in result["H4_equivalence"]["reason"]


def test_h1_is_reportable_when_the_control_passes():
    pairs = [_pair(f"p{i}") for i in range(60)]
    result = run_probe(pairs, lambda q, t: 0.5, equivalence_band=0.05,
                       resamples=FAST)
    assert result["H4_equivalence"]["passed"] is True
    assert result["reportable"] is True


def test_an_underpowered_run_says_so():
    pairs = [_pair(f"p{i}") for i in range(20)]
    result = run_probe(pairs, lambda q, t: 0.5, equivalence_band=0.05,
                       resamples=FAST)
    assert result["underpowered"] is True
    assert "inconclusive" in result["power_note"]


# --------------------------------------------------------------------------
# Batched scoring: the one bug that would silently invalidate the probe
# --------------------------------------------------------------------------
def _length_scorer(rendered):
    """A deterministic 'model': the score encodes the passage, so a mix-up shows."""
    return [len(text) / 1000.0 for text in rendered]


def test_each_side_gets_back_its_own_score():
    pairs = [_pair(f"p{i}", tokens=10 + i) for i in range(30)]
    admit_fn, distinct = batched_admit_fn(
        pairs, lambda q, t: t, _length_scorer, batch_size=7)
    for pair in pairs:
        assert admit_fn(pair["question"], pair["older"]["text"]) == \
            pytest.approx(len(pair["older"]["text"]) / 1000.0)
        assert admit_fn(pair["question"], pair["newer"]["text"]) == \
            pytest.approx(len(pair["newer"]["text"]) / 1000.0)
    assert distinct == 60


@pytest.mark.parametrize("batch_size", [1, 2, 7, 16, 1000])
def test_the_batch_size_never_changes_a_score(batch_size):
    pairs = [_pair(f"p{i}", tokens=10 + i) for i in range(20)]
    admit_fn, _ = batched_admit_fn(pairs, lambda q, t: t, _length_scorer,
                                   batch_size=batch_size)
    reference, _ = batched_admit_fn(pairs, lambda q, t: t, _length_scorer,
                                    batch_size=1)
    for pair in pairs:
        for side in ("older", "newer"):
            key = (pair["question"], pair[side]["text"])
            assert admit_fn(*key) == reference(*key)


def test_a_repeated_passage_is_scored_once():
    """Deduplication is what makes the batching worth doing."""
    shared = _pair("p1")
    pairs = [shared, {**shared, "pair_id": "p2"}]
    calls = []

    def counting(rendered):
        calls.extend(rendered)
        return _length_scorer(rendered)

    _, distinct = batched_admit_fn(pairs, lambda q, t: t, counting)
    assert distinct == 2 and len(calls) == 2


def test_a_scorer_returning_the_wrong_count_is_refused():
    """Silently truncating would misalign every passage after the gap."""
    pairs = [_pair(f"p{i}") for i in range(5)]
    with pytest.raises(UnusableDataset):
        batched_admit_fn(pairs, lambda q, t: t, lambda batch: [0.5], batch_size=4)


def test_the_probe_runs_end_to_end_through_the_batched_lookup():
    pairs = [_pair(f"p{i}", tokens=50) for i in range(40)]
    admit_fn, _ = batched_admit_fn(pairs, lambda q, t: t, _length_scorer)
    result = probe(pairs, admit_fn, resamples=FAST)
    # "newer" text is longer by one token, so the older side scores lower.
    assert result["delta"]["point"] < 0


# --------------------------------------------------------------------------
# Provenance of the result
# --------------------------------------------------------------------------
def test_the_pair_digest_identifies_the_pair_set():
    pairs = [_pair(f"p{i}") for i in range(10)]
    assert digest_pairs(pairs) == digest_pairs(list(reversed(pairs)))
    changed = [dict(p) for p in pairs]
    changed[0] = {**changed[0], "older": {**changed[0]["older"], "text": "different"}}
    assert digest_pairs(changed) != digest_pairs(pairs)


def test_the_result_carries_the_digest_of_the_pairs_it_used():
    pairs = [_pair(f"p{i}") for i in range(10)]
    result = run_probe(pairs, lambda q, t: 0.5, equivalence_band=0.5, resamples=FAST)
    assert result["pair_digest"] == digest_pairs(pairs)
