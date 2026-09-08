"""Tests for the SCAF admission policy.

    cd scaf && python3 -m pytest        (or: python3 -m pytest scaf/tests)

Offline: no models, no GPU, no index. SCAF admission is a deterministic function
of metadata and text, so all of it is testable here.
"""

import json
import math
import os
import sys

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _p in (_ROOT, os.path.join(_ROOT, "rag2")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from rag2.config import FilterConfig
from rag2.filtering.base import build_filter
from rag2.schema import Evidence, Question

import scaf
from scaf.policy import (
    AuthorityScorer,
    CurrencyScorer,
    SCAFFilter,
    SupportScorer,
    _year_fraction,
    tokenize,
)


def question(text="Which biomarker is recommended for early Alzheimer diagnosis?", **meta):
    return Question(qid="q1", question=text, options={}, answer=None, metadata=meta)


def evidence(chunk_id="c1", text="plasma p-tau217 biomarker for early alzheimer diagnosis",
             date="2024-01-01", tier="", category="pmc-fulltext", retracted="no", **meta):
    return Evidence(
        text=text, source=category, doc_id=chunk_id.split("#")[0], passage_id=chunk_id,
        metadata={"canonical_date": date, "date_precision": "day",
                  "authority_tier_label": tier, "retracted": retracted,
                  "in_currency_pack": "no", **meta},
    )


def scaf_filter(**options):
    return SCAFFilter(FilterConfig(kind="scaf", options=options))


# ---------------------------------------------------------------- support
class TestSupport:
    def test_more_overlap_scores_higher(self):
        scorer = SupportScorer()
        q = question("amyloid PET imaging in Alzheimer diagnosis")
        on_topic = evidence("c1", "amyloid PET imaging supports alzheimer diagnosis")
        off_topic = evidence("c2", "renal dialysis schedule for chronic kidney failure")
        idf = scorer.idf([on_topic, off_topic])
        assert scorer.score(q, on_topic, idf)[0] > scorer.score(q, off_topic, idf)[0]

    def test_score_is_bounded(self):
        scorer = SupportScorer()
        q = question("amyloid")
        for ev in (evidence("c1", "amyloid amyloid amyloid"), evidence("c2", "")):
            value, _ = scorer.score(q, ev, scorer.idf([ev]))
            assert 0.0 <= value <= 1.0

    def test_detail_names_the_matched_terms(self):
        scorer = SupportScorer()
        q = question("amyloid PET imaging")
        ev = evidence("c1", "amyloid imaging findings")
        _, detail = scorer.score(q, ev, scorer.idf([ev]))
        assert "amyloid" in detail["matched"] and "imaging" in detail["matched"]
        assert detail["num_matched"] == 2

    def test_method_is_declared_as_lexical_not_entailment(self):
        """The largest approximation in SCAF must be visible in every record."""
        assert "lexical" in SupportScorer.method
        assert scaf_filter().describe()["support_is_entailment"] is False

    def test_stopwords_do_not_create_support(self):
        scorer = SupportScorer()
        q = question("what is the most likely of these patients")
        ev = evidence("c1", "the study of these patients is what most results conclude")
        assert scorer.score(q, ev, scorer.idf([ev]))[0] == 0.0


# --------------------------------------------------------------- currency
class TestCurrency:
    def test_retracted_evidence_scores_zero(self):
        gamma, detail = CurrencyScorer(as_of="2026-01-01").score(
            question(), evidence(date="2025-01-01", retracted="yes"))
        assert gamma == 0.0 and detail["state"] == "retracted"

    def test_non_time_sensitive_claims_are_exempt_from_decay(self):
        """psi(q) = 0: an old paper is not stale if the claim does not age."""
        gamma, detail = CurrencyScorer(as_of="2026-01-01").score(
            question(time_sensitive=False), evidence(date="1995-01-01"))
        assert gamma == 1.0 and detail["state"] == "not_time_sensitive"

    def test_decay_is_the_specified_half_life_curve(self):
        scorer = CurrencyScorer(half_life_years=5.0, as_of="2026-01-01")
        recent, _ = scorer.score(question(), evidence(date="2026-01-01"))
        one_life, _ = scorer.score(question(), evidence(date="2021-01-01"))
        two_lives, _ = scorer.score(question(), evidence(date="2016-01-01"))
        assert recent == pytest.approx(1.0, abs=0.02)
        assert one_life == pytest.approx(0.5, abs=0.02)
        assert two_lives == pytest.approx(0.25, abs=0.02)

    def test_age_never_hard_rejects(self):
        """The mirror-image error the thesis must not make."""
        gamma, _ = CurrencyScorer(as_of="2026-01-01").score(
            question(), evidence(date="1960-01-01"))
        assert gamma > 0.0

    def test_superseded_evidence_is_discounted_not_rejected(self):
        scorer = CurrencyScorer(half_life_years=5.0, superseded_discount=0.5,
                                as_of="2026-01-01")
        plain, _ = scorer.score(question(), evidence(date="2021-01-01"))
        superseded, detail = scorer.score(
            question(), evidence(date="2021-01-01", superseded_by="PMC999"))
        assert superseded == pytest.approx(plain * 0.5, abs=1e-6)
        assert detail["state"] == "superseded" and detail["supersession"] == "PMC999"
        assert superseded > 0.0

    def test_undated_evidence_is_flagged_not_guessed(self):
        gamma, detail = CurrencyScorer(as_of="2026-01-01").score(
            question(), evidence(date=""))
        assert detail["state"] == "undated"
        assert 0.0 < gamma < 1.0

    def test_supersession_is_reported_unknown_when_the_corpus_is_silent(self):
        """Absence of evidence must not be reported as 'current'."""
        _, detail = CurrencyScorer(as_of="2026-01-01").score(
            question(), evidence(date="2024-01-01"))
        assert detail["supersession"] == "unknown"

    def test_future_dates_do_not_exceed_one(self):
        gamma, _ = CurrencyScorer(as_of="2020-01-01").score(
            question(), evidence(date="2026-01-01"))
        assert gamma <= 1.0


class TestYearFraction:
    def test_precision_is_respected(self):
        assert _year_fraction("2020") == pytest.approx(2020.5)
        assert _year_fraction("2020-01") == pytest.approx(2020.0, abs=0.05)
        assert _year_fraction("2020-07-15") == pytest.approx(2020.5, abs=0.06)

    def test_unparseable_dates_return_none(self):
        for value in ("", "n/a", "not-a-date", "20", "9999999"):
            assert _year_fraction(value) is None


# -------------------------------------------------------------- authority
class TestAuthority:
    def test_tier_ordering_comes_from_config_not_code(self):
        """Ablation A12: authority ordering is a tested variable."""
        inverted = {"clinical-practice-guideline": 0.1, "": 0.9}
        scorer = AuthorityScorer(tiers=inverted)
        guideline, _ = scorer.score(evidence(tier="clinical-practice-guideline"))
        plain, _ = scorer.score(evidence(tier=""))
        assert guideline < plain              # the ordering really did invert

    def test_default_ordering_prefers_guidelines(self):
        scorer = AuthorityScorer()
        guideline, detail = scorer.score(evidence(tier="clinical-practice-guideline"))
        plain, _ = scorer.score(evidence(tier=""))
        assert guideline > plain
        assert detail["basis"] == "authority_tier_label"

    def test_falls_back_to_source_category(self):
        scorer = AuthorityScorer()
        value, detail = scorer.score(evidence(tier="", category="currency-pack"))
        assert detail["basis"] == "source_category"
        assert value == scaf.DEFAULT_CATEGORY_AUTHORITY["currency-pack"]


# -------------------------------------------------------------- admission
class TestAdmission:
    def test_one_decision_per_candidate_in_order(self):
        f = scaf_filter()
        candidates = [evidence(f"c{i}") for i in range(5)]
        decisions = f.decide(question(), candidates)
        assert len(decisions) == 5
        assert [d.detail["chunk_id"] for d in decisions] == [c.passage_id for c in candidates]

    def test_retracted_evidence_is_hard_gated_regardless_of_score(self):
        f = scaf_filter(admit_threshold=0.0)          # everything would pass on score
        decisions = f.decide(question(), [
            evidence("good", retracted="no"),
            evidence("bad", retracted="yes"),
        ])
        assert decisions[0].keep is True
        assert decisions[1].keep is False
        assert decisions[1].detail["gate"] == "retracted"

    def test_threshold_governs_admission(self):
        candidates = [evidence("c1", "amyloid pet imaging alzheimer diagnosis biomarker")]
        q = question("amyloid pet imaging alzheimer diagnosis biomarker")
        assert scaf_filter(admit_threshold=0.0).decide(q, candidates)[0].keep is True
        assert scaf_filter(admit_threshold=1.01).decide(q, candidates)[0].keep is False

    def test_weights_are_configurable_and_recorded(self):
        f = scaf_filter(weights={"support": 1.0, "currency": 0.0,
                                 "corroboration": 0.0, "authority": 0.0})
        decision = f.decide(question(), [evidence()])[0]
        assert decision.detail["weights"]["currency"] == 0.0
        assert decision.detail["scaf_score"] == pytest.approx(
            decision.detail["sigma_support"], abs=1e-6)

    def test_currency_can_change_an_admission(self):
        """The thesis's mechanism: age moves the decision, it does not gate it."""
        q = question("amyloid pet imaging")
        text = "amyloid pet imaging"
        weights = {"support": 0.5, "currency": 0.5, "corroboration": 0.0, "authority": 0.0}
        f = scaf_filter(weights=weights, admit_threshold=0.7, as_of="2026-01-01")
        new = f.decide(q, [evidence("new", text, date="2026-01-01")])[0]
        old = f.decide(q, [evidence("old", text, date="1996-01-01")])[0]
        assert new.keep is True and old.keep is False
        assert old.detail["gate"] == ""            # rejected on score, not gated

    def test_max_admit_caps_deterministically_by_score(self):
        q = question("amyloid pet imaging biomarker")
        candidates = [
            evidence("weak", "unrelated renal text"),
            evidence("strong", "amyloid pet imaging biomarker"),
            evidence("medium", "amyloid imaging"),
        ]
        decisions = scaf_filter(admit_threshold=0.0, max_admit=1).decide(q, candidates)
        kept = [d.detail["chunk_id"] for d in decisions if d.keep]
        assert kept == ["strong"]
        assert any(d.detail["gate"] == "max_admit" for d in decisions if not d.keep)

    def test_abstains_when_nothing_clears(self):
        f = scaf_filter(admit_threshold=1.01)
        f.decide(question(), [evidence("c1"), evidence("c2")])
        assert f.abstained() is True

    def test_does_not_abstain_when_something_is_admitted(self):
        f = scaf_filter(admit_threshold=0.0)
        f.decide(question(), [evidence("c1")])
        assert f.abstained() is False

    def test_admission_is_deterministic(self):
        q = question()
        candidates = [evidence(f"c{i}", f"amyloid text {i}", date=f"20{10 + i}-01-01")
                      for i in range(8)]
        first = [d.keep for d in scaf_filter().decide(q, candidates)]
        second = [d.keep for d in scaf_filter().decide(q, candidates)]
        assert first == second

    def test_scores_are_finite(self):
        decisions = scaf_filter().decide(question(), [
            evidence("c1", ""), evidence("c2", "x", date=""),
            evidence("c3", "amyloid", date="1900-01-01"),
        ])
        for d in decisions:
            assert math.isfinite(d.score)
            for key in ("sigma_support", "gamma_currency", "tau_authority", "scaf_score"):
                assert math.isfinite(d.detail[key])

    def test_scaf_never_retrieves(self):
        """SCAF must consume the given candidates and nothing else."""
        candidates = [evidence("only")]
        decisions = scaf_filter(admit_threshold=0.0).decide(question(), candidates)
        assert len(decisions) == 1
        assert decisions[0].detail["chunk_id"] == "only"

    def test_empty_candidate_list_is_handled(self):
        assert scaf_filter().decide(question(), []) == []


# ----------------------------------------------------------- inspectability
class TestInspectability:
    def test_every_decision_explains_itself(self):
        decisions = scaf_filter().decide(question(), [evidence()])
        detail = decisions[0].detail
        for key in ("scaf_score", "sigma_support", "gamma_currency", "rho_corroboration",
                    "tau_authority", "weights", "threshold", "gate", "reason",
                    "support_method", "corroboration_status", "support_detail",
                    "currency_detail", "authority_detail"):
            assert key in detail, f"missing {key}"
        assert detail["reason"]

    def test_unimplemented_parts_say_so(self):
        detail = scaf_filter().decide(question(), [evidence()])[0].detail
        assert detail["corroboration_status"] == "not_implemented"
        assert detail["weights"]["corroboration"] == 0.0
        assert "lexical" in detail["support_method"]

    def test_decision_record_is_json_serialisable(self):
        decisions = scaf_filter().decide(question(), [evidence()])
        json.dumps([d.detail for d in decisions])       # must not raise

    def test_describe_records_the_full_configuration(self):
        described = scaf_filter(half_life_years=3.0, admit_threshold=0.6).describe()
        assert described["half_life_years"] == 3.0
        assert described["threshold"] == 0.6
        assert described["authority_tiers"]


class TestRegistry:
    def test_scaf_builds_through_the_baseline_registry(self):
        built = build_filter(FilterConfig(kind="scaf"))
        assert isinstance(built, SCAFFilter)

    def test_baseline_does_not_import_scaf(self):
        """The dependency must run one way: scaf -> rag2, never rag2 -> scaf."""
        base = os.path.join(_ROOT, "rag2", "rag2")
        offenders = []
        for dirpath, _, filenames in os.walk(base):
            for name in filenames:
                if not name.endswith(".py"):
                    continue
                path = os.path.join(dirpath, name)
                with open(path, "r", encoding="utf-8") as handle:
                    for lineno, line in enumerate(handle, 1):
                        stripped = line.strip()
                        if stripped.startswith(("import scaf", "from scaf")):
                            offenders.append(f"{path}:{lineno}")
        assert not offenders, "baseline imports the thesis extension:\n" + "\n".join(offenders)
