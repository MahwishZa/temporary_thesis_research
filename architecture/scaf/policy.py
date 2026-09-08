"""SCAF -- Support / Currency / Authority admission policy. **Minimum viable.**

The thesis contribution. It replaces RAG2's perplexity-derived filter at the one
seam the baseline already exposes (:class:`rag2.filtering.base.EvidenceFilter`),
so the two can be compared on an identical candidate set with nothing else
changed. No baseline file is modified by this module, and it lives outside
``rag2/`` so the baseline's metadata-isolation guard stays meaningful -- see
``architecture/scaf/__init__.py``.

    A(s) = w_sigma * sigma  +  w_gamma * gamma  +  w_rho * rho  +  w_tau * tau

    sigma  SUPPORT     does the passage bear on the question?
    gamma  CURRENCY    is it current enough for a time-sensitive claim?
    rho    CORROBORATION  is it corroborated / contested?   (NOT IMPLEMENTED, see below)
    tau    AUTHORITY   what is the source tier worth?

Scope of this version
---------------------
This is the *first* runnable SCAF, built for a preliminary comparison, not the
final thesis instrument. Three deliberate limits, each surfaced in the output
rather than hidden:

* **sigma is lexical, not entailment.** The thesis specifies an entailment model.
  That needs a trained NLI model this stage does not have, so sigma here is a
  prototype: question-term coverage combined with corpus-IDF-weighted overlap
  (:class:`SupportScorer`, v2). Every decision record carries ``support_method``
  and ``support_detail.idf_source`` so no reader can mistake one for the other.
  **This is the single largest approximation in SCAF and must be replaced before
  any thesis claim.**
* **rho (corroboration/contested) is not implemented.** Detecting that two
  passages disagree needs cross-passage inference. ``w_rho`` defaults to 0 and
  every record reports ``corroboration: "not_implemented"``. It is in the formula
  so the interface does not change when it arrives.
* **Supersession is only detected where the corpus states it.** The thesis's
  three-state currency has a "superseded" state discounted by delta. Nothing in
  the corpus marks one document as superseding another, so a passage is
  superseded here only when its own metadata says so. Absence of the state is
  reported as ``supersession: "unknown"``, never as "current".

Authority is a *tested variable*, never a constant
--------------------------------------------------
Thesis ablation A12 treats the authority ordering as something to be measured.
So the tier -> weight map lives in configuration (``options.authority_tiers``)
and this module ships only a documented default. Changing the ordering must be a
config edit that shows up in the run manifest, never a code edit.

Every decision is inspectable
-----------------------------
:class:`SCAFDecision` carries every input, every sub-score, the weights, the
threshold, the gate that fired, and a human-readable reason. A passage that was
rejected can always be explained.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Dict, List, Optional, Sequence, Tuple

from rag2.config import FilterConfig
from rag2.schema import Evidence, FilterDecision, Question
from rag2.filtering.base import EvidenceFilter, register_filter

# --------------------------------------------------------------------------
# Defaults. Every one of these is overridable from config; none is a constant
# of the method.
# --------------------------------------------------------------------------

#: Authority tier -> tau. A DEFAULT ORDERING, NOT A FINDING (ablation A12).
#: Guidelines above consensus above primary literature is the conventional
#: evidence hierarchy; the thesis is meant to test it, not assume it.
DEFAULT_AUTHORITY_TIERS: Dict[str, float] = {
    "clinical-practice-guideline": 1.00,
    "appropriate-use-criteria": 0.85,
    "diagnostic-criteria": 0.85,
    "consensus-recommendation": 0.70,
    "": 0.40,                      # ordinary primary literature: no tier assigned
}

#: Fallback when source_category is all we have.
DEFAULT_CATEGORY_AUTHORITY: Dict[str, float] = {
    "currency-pack": 0.60,
    "pmc-fulltext": 0.45,
    "pubmed-abstract": 0.40,
}

DEFAULT_WEIGHTS: Dict[str, float] = {
    "support": 0.50,
    "currency": 0.30,
    "corroboration": 0.0,          # rho is not implemented; see the module docstring
    "authority": 0.20,
}

#: Currency half-life in years: gamma = 2 ** (-age / H) for a time-sensitive claim.
DEFAULT_HALF_LIFE_YEARS = 5.0

#: Multiplier applied to gamma when the corpus marks a passage superseded.
DEFAULT_SUPERSEDED_DISCOUNT = 0.5

DEFAULT_ADMIT_THRESHOLD = 0.45

#: Tokens too common in this corpus to carry topical signal.
_STOPWORDS = frozenset("""
a an the and or but if then than that this these those of in on at to for from by with
without within is are was were be been being do does did doing have has had having
we our us it its as into about over under between during after before more most other
some such no nor not only own same so too very can will just should now which who whom
what when where why how all any both each few many patients patient study studies
results conclusion conclusions background methods method objective purpose
""".split())

_TOKEN = re.compile(r"[a-z0-9][a-z0-9\-']+")


def tokenize(text: str) -> List[str]:
    return [t for t in _TOKEN.findall((text or "").lower()) if t not in _STOPWORDS]


# --------------------------------------------------------------------------
# Sub-scorers. Each is a pure function of one candidate, so each is testable in
# isolation and each can be swapped without touching the admission policy.
# --------------------------------------------------------------------------
class SupportScorer:
    """sigma -- does this passage bear on the question?

    **PROTOTYPE SUPPORT SCORER. NOT semantic entailment.** The thesis specifies
    an entailment model; none is available in this environment, so this is a
    lexical stand-in and every decision record says so via ``support_method``.
    It is deliberately behind one small interface so the entailment model can
    replace it without touching the admission policy.

    Fix to the v1 collapse (see docs/experiments/preliminary_rag2_vs_scaf.md, example E)
    -------------------------------------------------------------------------
    v1 computed IDF **over the candidate set**. When retrieval does its job every
    candidate is on-topic, so the question's own terms appear in all of them, IDF
    drives their weight to nearly zero, and sigma collapses towards zero for
    every candidate at once -- the score stopped measuring topicality and became
    a within-set contrast. On alz-016 a passage matching *neuropathological*,
    *hallmark* and *disease* scored 0.033.

    v2 fixes it two ways:

    * **IDF comes from the corpus, not the candidate set.** A term is rare
      because it is rare in the corpus, which is what IDF is supposed to mean.
      Pass ``document_frequency`` + ``corpus_size`` (the freeze step computes
      them from the same chunk file the candidates came from).
    * **Coverage is scored alongside it.** ``coverage`` is the plain fraction of
      the question's content terms present in the passage. It cannot collapse,
      because it does not depend on any other candidate. sigma is the mean of
      coverage and the IDF-weighted overlap.

    Without a corpus ``document_frequency`` the scorer still runs, but falls back
    to coverage alone and reports ``idf_source: "none"`` -- never silently back to
    the v1 behaviour.
    """

    method = "lexical-coverage-corpus-idf-v2"

    def __init__(self, floor: float = 0.0,
                 document_frequency: Optional[Dict[str, int]] = None,
                 corpus_size: int = 0) -> None:
        self.floor = floor
        self.document_frequency = dict(document_frequency or {})
        self.corpus_size = int(corpus_size or 0)

    @property
    def idf_source(self) -> str:
        return "corpus" if (self.document_frequency and self.corpus_size) else "none"

    def corpus_idf(self, token: str) -> float:
        """Smoothed IDF over the corpus. 1.0 when no corpus statistics exist."""
        if not self.document_frequency or not self.corpus_size:
            return 1.0
        df = self.document_frequency.get(token, 0)
        return math.log((self.corpus_size + 1.0) / (df + 1.0)) + 1.0

    def idf(self, candidates: Sequence[Evidence]) -> Dict[str, float]:
        """Kept for interface compatibility; the corpus table is what is used.

        Returns an empty map: per-question IDF is exactly the thing v2 removed.
        """
        return {}

    def score(self, question: Question, candidate: Evidence,
              idf: Optional[Dict[str, float]] = None) -> Tuple[float, Dict[str, Any]]:
        q_tokens = set(tokenize(question.question))
        if not q_tokens:
            return self.floor, {"matched": [], "question_tokens": 0,
                                "coverage": 0.0, "idf_overlap": 0.0,
                                "idf_source": self.idf_source}

        text_tokens = set(tokenize(candidate.text))
        matched = sorted(q_tokens & text_tokens)

        # Coverage: cannot collapse, because it depends on this passage alone.
        coverage = len(matched) / len(q_tokens)

        if self.idf_source == "corpus":
            total = sum(self.corpus_idf(t) for t in q_tokens)
            hit = sum(self.corpus_idf(t) for t in matched)
            idf_overlap = (hit / total) if total > 0 else 0.0
            value = 0.5 * coverage + 0.5 * idf_overlap
        else:
            idf_overlap = 0.0
            value = coverage

        return max(self.floor, min(1.0, value)), {
            "matched": matched[:12],
            "num_matched": len(matched),
            "question_tokens": len(q_tokens),
            "coverage": round(coverage, 6),
            "idf_overlap": round(idf_overlap, 6),
            "idf_source": self.idf_source,
            "corpus_size": self.corpus_size,
        }


class CurrencyScorer:
    """gamma -- three-state currency, per the thesis specification.

        gamma = 0                              if retracted or withdrawn
        gamma = 1                              if the claim is not time-sensitive
        gamma = delta * 2^(-age / H)           if superseded
        gamma = 2^(-age / H)                   otherwise

    Hard rejection is confined to retraction/withdrawal. Age never rejects on its
    own -- it down-weights. That distinction is the point of the thesis: the
    complaint against a confidence filter is that it silently drops recent
    evidence, and a currency policy that hard-rejects old evidence would make the
    mirror-image error.
    """

    def __init__(self, half_life_years: float = DEFAULT_HALF_LIFE_YEARS,
                 superseded_discount: float = DEFAULT_SUPERSEDED_DISCOUNT,
                 as_of: Optional[str] = None) -> None:
        self.half_life_years = half_life_years or DEFAULT_HALF_LIFE_YEARS
        self.superseded_discount = superseded_discount
        self.as_of = as_of

    def _as_of_year(self) -> float:
        if self.as_of:
            return _year_fraction(self.as_of) or float(date.today().year)
        return float(date.today().year)

    def score(self, question: Question, candidate: Evidence) -> Tuple[float, Dict[str, Any]]:
        meta = candidate.metadata or {}
        detail: Dict[str, Any] = {
            "canonical_date": meta.get("canonical_date", ""),
            "date_precision": meta.get("date_precision", ""),
            "retracted": meta.get("retracted", ""),
            "half_life_years": self.half_life_years,
        }

        if str(meta.get("retracted", "")).strip().lower() in {"yes", "true", "1"}:
            detail["state"] = "retracted"
            detail["supersession"] = "n/a"
            return 0.0, detail

        # psi(q): is the claim time-sensitive? Carried on the question so it is
        # explicit and inspectable, never inferred silently.
        time_sensitive = bool(question.metadata.get("time_sensitive", True))
        detail["time_sensitive"] = time_sensitive
        if not time_sensitive:
            detail["state"] = "not_time_sensitive"
            detail["supersession"] = "n/a"
            return 1.0, detail

        published = _year_fraction(str(meta.get("canonical_date", "")))
        if published is None:
            # No date: cannot claim currency, must not invent it. Neutral, flagged.
            detail["state"] = "undated"
            detail["supersession"] = "unknown"
            return 0.5, detail

        age = max(0.0, self._as_of_year() - published)
        decay = 2.0 ** (-age / self.half_life_years)
        detail["age_years"] = round(age, 3)

        superseded = str(meta.get("superseded_by", "")).strip()
        if superseded:
            detail["state"] = "superseded"
            detail["supersession"] = superseded
            detail["superseded_discount"] = self.superseded_discount
            return max(0.0, min(1.0, self.superseded_discount * decay)), detail

        detail["state"] = "current"
        # Nothing in this corpus marks supersession, so "unknown" is the honest
        # label -- never "current" as a positive claim.
        detail["supersession"] = "unknown"
        return max(0.0, min(1.0, decay)), detail


class AuthorityScorer:
    """tau -- source tier. The ordering is configuration, never a constant."""

    def __init__(self, tiers: Optional[Dict[str, float]] = None,
                 category_fallback: Optional[Dict[str, float]] = None,
                 default: float = 0.35) -> None:
        self.tiers = dict(tiers if tiers is not None else DEFAULT_AUTHORITY_TIERS)
        self.category_fallback = dict(
            category_fallback if category_fallback is not None else DEFAULT_CATEGORY_AUTHORITY
        )
        self.default = default

    def score(self, candidate: Evidence) -> Tuple[float, Dict[str, Any]]:
        meta = candidate.metadata or {}
        tier = str(meta.get("authority_tier_label", "") or "")
        detail: Dict[str, Any] = {
            "authority_tier_label": tier,
            "source_category": candidate.source or "",
            "in_currency_pack": meta.get("in_currency_pack", ""),
        }
        if tier and tier in self.tiers:
            detail["basis"] = "authority_tier_label"
            return self.tiers[tier], detail
        category = candidate.source or ""
        if category in self.category_fallback:
            detail["basis"] = "source_category"
            return self.category_fallback[category], detail
        if "" in self.tiers:
            detail["basis"] = "untiered_default"
            return self.tiers[""], detail
        detail["basis"] = "default"
        return self.default, detail


def _year_fraction(value: str) -> Optional[float]:
    """A YYYY / YYYY-MM / YYYY-MM-DD date as a float year. None if unparseable.

    Precision is respected rather than invented: a year-only date becomes mid-year
    so it is not silently treated as 1 January.
    """
    text = (value or "").strip()
    match = re.match(r"^(\d{4})(?:-(\d{2}))?(?:-(\d{2}))?", text)
    if not match:
        return None
    year = int(match.group(1))
    if not 1500 <= year <= 2200:
        return None
    if match.group(2):
        month = min(12, max(1, int(match.group(2))))
        day = int(match.group(3)) if match.group(3) else 15
        return year + ((month - 1) + (min(28, max(1, day)) - 1) / 30.0) / 12.0
    return year + 0.5


# --------------------------------------------------------------------------
# Decision record
# --------------------------------------------------------------------------
@dataclass
class SCAFDecision:
    """Everything that produced one ADMIT/REJECT. Nothing is hidden."""

    chunk_id: str
    admit: bool
    score: float
    support: float
    currency: float
    corroboration: float
    authority: float
    weights: Dict[str, float]
    threshold: float
    gate: str                       # "" when no hard gate fired
    reason: str
    support_method: str
    support_detail: Dict[str, Any] = field(default_factory=dict)
    currency_detail: Dict[str, Any] = field(default_factory=dict)
    authority_detail: Dict[str, Any] = field(default_factory=dict)
    corroboration_status: str = "not_implemented"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "admit": self.admit,
            "scaf_score": round(self.score, 6),
            "sigma_support": round(self.support, 6),
            "gamma_currency": round(self.currency, 6),
            "rho_corroboration": round(self.corroboration, 6),
            "tau_authority": round(self.authority, 6),
            "weights": self.weights,
            "threshold": self.threshold,
            "gate": self.gate,
            "reason": self.reason,
            "support_method": self.support_method,
            "corroboration_status": self.corroboration_status,
            "support_detail": self.support_detail,
            "currency_detail": self.currency_detail,
            "authority_detail": self.authority_detail,
        }


# --------------------------------------------------------------------------
# The filter
# --------------------------------------------------------------------------
class SCAFFilter(EvidenceFilter):
    """Admission by weighted Support / Currency / Authority, with hard gates.

    Consumes the same :class:`~rag2.schema.Evidence` list the RAG2 filter does,
    and returns the same :class:`~rag2.schema.FilterDecision` list, so the
    pipeline cannot tell them apart. It never retrieves.
    """

    name = "scaf"

    def __init__(self, config: FilterConfig, prompts: Any = None) -> None:
        self.config = config
        options = dict(config.options or {})
        weights = dict(DEFAULT_WEIGHTS)
        weights.update(options.get("weights", {}) or {})
        self.weights = {k: float(v) for k, v in weights.items()}
        self.threshold = float(options.get("admit_threshold", DEFAULT_ADMIT_THRESHOLD))
        self.max_admit = int(options.get("max_admit", 0))         # 0 = no cap
        self.abstain_enabled = bool(options.get("abstain", True))

        self.support = SupportScorer(
            floor=float(options.get("support_floor", 0.0)),
            document_frequency=options.get("document_frequency"),
            corpus_size=int(options.get("corpus_size", 0) or 0),
        )
        self.currency = CurrencyScorer(
            half_life_years=float(options.get("half_life_years", DEFAULT_HALF_LIFE_YEARS)),
            superseded_discount=float(
                options.get("superseded_discount", DEFAULT_SUPERSEDED_DISCOUNT)),
            as_of=options.get("as_of") or None,
        )
        self.authority = AuthorityScorer(
            tiers=options.get("authority_tiers"),
            category_fallback=options.get("category_authority"),
        )
        self.reject_retracted = bool(options.get("reject_retracted", True))
        self.last_decisions: List[SCAFDecision] = []

    # -- scoring -----------------------------------------------------------
    def evaluate(self, question: Question,
                 candidates: Sequence[Evidence]) -> List[SCAFDecision]:
        idf = self.support.idf(candidates)
        out: List[SCAFDecision] = []
        for candidate in candidates:
            sigma, sigma_detail = self.support.score(question, candidate, idf)
            gamma, gamma_detail = self.currency.score(question, candidate)
            tau, tau_detail = self.authority.score(candidate)
            rho = 0.0                                  # not implemented; w_rho = 0

            score = (self.weights["support"] * sigma
                     + self.weights["currency"] * gamma
                     + self.weights["corroboration"] * rho
                     + self.weights["authority"] * tau)

            gate = ""
            admit = score >= self.threshold
            reason = (f"score {score:.3f} "
                      f"{'>=' if admit else '<'} threshold {self.threshold:.3f}")
            # Hard gate: retraction. The one place currency rejects outright.
            if self.reject_retracted and gamma_detail.get("state") == "retracted":
                gate, admit = "retracted", False
                reason = "hard gate: evidence is retracted or withdrawn"

            out.append(SCAFDecision(
                chunk_id=str(candidate.passage_id or candidate.doc_id or ""),
                admit=admit, score=score, support=sigma, currency=gamma,
                corroboration=rho, authority=tau, weights=dict(self.weights),
                threshold=self.threshold, gate=gate, reason=reason,
                support_method=self.support.method,
                support_detail=sigma_detail, currency_detail=gamma_detail,
                authority_detail=tau_detail,
            ))
        return out

    # -- EvidenceFilter ----------------------------------------------------
    def decide(self, question: Question,
               candidates: Sequence[Evidence]) -> List[FilterDecision]:
        decisions = self.evaluate(question, candidates)

        if self.max_admit > 0:
            # Cap admissions at the highest-scoring max_admit, ties by position so
            # the result is deterministic.
            ranked = sorted(
                (i for i, d in enumerate(decisions) if d.admit),
                key=lambda i: (-decisions[i].score, i),
            )
            for index in ranked[self.max_admit:]:
                decisions[index].admit = False
                decisions[index].gate = decisions[index].gate or "max_admit"
                decisions[index].reason = (
                    f"{decisions[index].reason}; dropped by max_admit={self.max_admit}")

        self.last_decisions = decisions
        return [
            FilterDecision(
                keep=d.admit,
                label="[ADMIT]" if d.admit else "[REJECT]",
                score=float(d.score),
                detail=d.to_dict(),
            )
            for d in decisions
        ]

    def abstained(self) -> bool:
        """True when nothing cleared admission -- the thesis's abstention case."""
        return self.abstain_enabled and bool(self.last_decisions) and not any(
            d.admit for d in self.last_decisions)

    def describe(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "class": type(self).__name__,
            "weights": self.weights,
            "threshold": self.threshold,
            "max_admit": self.max_admit,
            "abstain": self.abstain_enabled,
            "support_method": self.support.method,
            "support_is_entailment": False,
            "support_idf_source": self.support.idf_source,
            "support_corpus_size": self.support.corpus_size,
            "corroboration_status": "not_implemented",
            "half_life_years": self.currency.half_life_years,
            "superseded_discount": self.currency.superseded_discount,
            "as_of": self.currency.as_of,
            "authority_tiers": self.authority.tiers,
            "category_authority": self.authority.category_fallback,
            "reject_retracted": self.reject_retracted,
        }


@register_filter("scaf")
def _build_scaf(config: FilterConfig, prompts: Any = None) -> EvidenceFilter:
    return SCAFFilter(config, prompts)
