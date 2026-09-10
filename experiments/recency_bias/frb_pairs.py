"""FRB-PAIRS: the Filter Recency-Bias Probe. The thesis's primary contribution.

What this measures
------------------
For a filter ``f`` and a set of temporal-counterfactual passage pairs,

    Delta = E_pairs[ P(admit | older) - P(admit | newer) ]

A value above zero means the filter preferentially admits *older* evidence
(hypothesis H1). The pairs are matched on claim, source tier and token length,
so age is intended to be the only systematic difference between the two sides.

Why the pairs come from outside this repository
-----------------------------------------------
Proposal 5.2 is explicit, and it is a correction the proposal makes to an
earlier draft of itself:

    "[DES] Provenance separation is mandatory and is the single most important
     correction in this revision. An earlier formulation authored evaluation
     items, the supersession table and the currency corpus from the same
     twenty-odd documents, rendering a positive result unfalsifiable by
     construction. The probe's primary material is now drawn from MedChangeQA
     -- externally authored, peer-reviewed in provenance, publicly released,
     and independent of any supersession table this thesis builds."

There is a **provenance firewall**: the primary claim rests on the external
lane, and the thesis-curated Alzheimer material supports replication and case
study only. The lanes must not cross. So this module refuses to build pairs from
the thesis corpus, however convenient that would be -- a probe built on
thesis-authored pairs would not be a weaker version of this experiment, it would
be the exact circularity the proposal revised itself to remove.

This also corrects a conclusion recorded elsewhere in this repository: that
FRB-PAIRS cannot be constructed because the Alzheimer corpus has no pre-2021
stratum. That is true of the corpus and beside the point -- the corpus was never
the intended source.

Why the filter is injected
--------------------------
``probe()`` takes an ``admit_fn``. The trained Flan-T5 filter lives on the
machine that trained it, so the estimator is separated from the model: the
statistics are testable offline here, and the same code path runs against the
real filter there with nothing recompiled.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import re
from collections import Counter
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

#: Proposal 7.3: paired bootstrap, 10,000 resamples, unit of analysis = the pair.
BOOTSTRAP_RESAMPLES = 10000
BOOTSTRAP_SEED = 20260910

#: Proposal 4.2 leaves the length tolerance band [OPEN]. This is a DESIGN
#: DECISION recorded here rather than a finding: pairs whose two sides differ in
#: token count by more than this fraction are excluded, because a filter may
#: respond to length instead of age. Report it; do not tune it against results.
DEFAULT_LENGTH_TOLERANCE = 0.25

#: Target pair count from proposal 5.2.
TARGET_PAIRS = (150, 300)

_TOKEN = re.compile(r"\S+")


class ProvenanceViolation(Exception):
    """Pairs were sourced from the thesis-curated lane. The firewall holds."""


class UnusableDataset(Exception):
    """The supplied dataset cannot yield valid temporal-counterfactual pairs."""


def _tokens(text: str) -> int:
    return len(_TOKEN.findall(text or ""))


def _year(value: Any) -> Optional[float]:
    """A year from a date string, an int year, or None."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value) if 1500 <= float(value) <= 2200 else None
    match = re.match(r"^\s*(\d{4})(?:-(\d{2}))?", str(value))
    if not match:
        return None
    year = int(match.group(1))
    if not 1500 <= year <= 2200:
        return None
    month = int(match.group(2)) if match.group(2) else 7
    return year + (month - 1) / 12.0


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
#: MedChangeQA's released field names are not assumed. The loader is told which
#: columns carry what, so the same code reads the dataset whatever it calls
#: them, and a mis-mapping fails loudly instead of silently pairing the wrong
#: text with the wrong date.
DEFAULT_FIELD_MAP: Dict[str, str] = {
    "id": "id",
    "question": "question",
    "older_text": "old_abstract",
    "older_date": "old_date",
    "newer_text": "new_abstract",
    "newer_date": "new_date",
    "tier": "source_tier",
}


def load_dataset(path: str, field_map: Optional[Dict[str, str]] = None,
                 provenance: str = "") -> List[Dict[str, Any]]:
    """Read MedChangeQA (or any dataset in its shape) into probe items.

    ``provenance`` must name an external source. Passing a thesis-curated one
    raises rather than proceeding -- see the module docstring.
    """
    if provenance and "thesis" in provenance.lower():
        raise ProvenanceViolation(
            f"provenance {provenance!r} is thesis-curated. The primary claim "
            "must rest on the external lane (proposal 5.2, provenance firewall).")
    fields = {**DEFAULT_FIELD_MAP, **(field_map or {})}
    rows = _read_rows(path)
    items: List[Dict[str, Any]] = []
    skipped: Counter = Counter()
    for index, row in enumerate(rows):
        older_text = row.get(fields["older_text"])
        newer_text = row.get(fields["newer_text"])
        older_year = _year(row.get(fields["older_date"]))
        newer_year = _year(row.get(fields["newer_date"]))
        if not older_text or not newer_text:
            skipped["missing text"] += 1
            continue
        if older_year is None or newer_year is None:
            skipped["missing or unparseable date"] += 1
            continue
        if newer_year <= older_year:
            skipped["newer side is not newer"] += 1
            continue
        items.append({
            "pair_id": str(row.get(fields["id"], f"item-{index:04d}")),
            "question": row.get(fields["question"], ""),
            "older": {"text": older_text, "year": older_year,
                      "date": row.get(fields["older_date"]),
                      "tokens": _tokens(older_text)},
            "newer": {"text": newer_text, "year": newer_year,
                      "date": row.get(fields["newer_date"]),
                      "tokens": _tokens(newer_text)},
            "tier": row.get(fields["tier"], ""),
        })
    if not items:
        raise UnusableDataset(
            f"no usable pairs in {path}. Rejected: {dict(skipped)}. Check the "
            f"--field-map: the loader looked for {fields}.")
    return items


def _read_rows(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        first = handle.read(1)
        handle.seek(0)
        if first == "[":
            return json.load(handle)
        return [json.loads(line) for line in handle if line.strip()]


# --------------------------------------------------------------------------
# Pair construction
# --------------------------------------------------------------------------
def build_pairs(items: Sequence[Dict[str, Any]],
                length_tolerance: float = DEFAULT_LENGTH_TOLERANCE,
                require_same_tier: bool = True) -> Dict[str, Any]:
    """Matched temporal-counterfactual pairs, with every exclusion counted.

    Matching reduces but cannot eliminate the confound that older and newer
    statements of the same claim differ in hedging, citation density and prose
    style. Proposal 4.2 says so, and the permutation control (H4) is what
    detects it. Exclusions are reported rather than silently applied, because
    the exclusion rule is itself a design decision.
    """
    kept: List[Dict[str, Any]] = []
    excluded: Counter = Counter()
    for item in items:
        older, newer = item["older"], item["newer"]
        if require_same_tier and item.get("tier_older") and item.get("tier_newer") \
                and item["tier_older"] != item["tier_newer"]:
            excluded["source tier differs"] += 1
            continue
        longer = max(older["tokens"], newer["tokens"])
        shorter = min(older["tokens"], newer["tokens"])
        if longer == 0:
            excluded["empty passage"] += 1
            continue
        if (longer - shorter) / longer > length_tolerance:
            excluded["token length outside the tolerance band"] += 1
            continue
        kept.append({**item, "age_gap_years": round(newer["year"] - older["year"], 3),
                     "length_ratio": round(shorter / longer, 4)})
    return {
        "pairs": kept,
        "constructed": len(kept),
        "excluded": dict(excluded),
        "excluded_total": sum(excluded.values()),
        "length_tolerance": length_tolerance,
        "target_range": list(TARGET_PAIRS),
        "meets_target": TARGET_PAIRS[0] <= len(kept) <= TARGET_PAIRS[1],
        "below_target": len(kept) < TARGET_PAIRS[0],
        "matching_note": (
            "Matched on claim (both sides address the same changed verdict), on "
            "source tier where the dataset records one, and on token length "
            f"within {length_tolerance:.0%}. Matching cannot remove differences "
            "in hedging or prose era; the permutation control is what detects "
            "those (proposal 4.2, control V1)."),
    }


def permute(pairs: Sequence[Dict[str, Any]],
            seed: int = BOOTSTRAP_SEED) -> List[Dict[str, Any]]:
    """Validity control V1: randomly reassign which side is 'older'.

    On the permuted set Delta must collapse to zero (H4). If it does not, the
    probe is responding to prose-era features rather than to recency, the
    primary result is uninterpretable, and proposal 4.6 requires it be withheld
    rather than explained.
    """
    rng = random.Random(seed)
    out: List[Dict[str, Any]] = []
    for pair in pairs:
        flipped = rng.random() < 0.5
        older, newer = (pair["newer"], pair["older"]) if flipped else \
            (pair["older"], pair["newer"])
        out.append({**pair, "older": older, "newer": newer, "permuted": True,
                    "flipped": flipped})
    return out


# --------------------------------------------------------------------------
# The estimator
# --------------------------------------------------------------------------
def paired_bootstrap(differences: Sequence[float],
                     resamples: int = BOOTSTRAP_RESAMPLES,
                     seed: int = BOOTSTRAP_SEED,
                     confidence: float = 0.95) -> Dict[str, Any]:
    """Proposal 7.3: paired bootstrap, unit of analysis = the matched pair."""
    if not differences:
        return {"point": None, "ci_low": None, "ci_high": None, "pairs": 0}
    observed = sum(differences) / len(differences)
    rng = random.Random(seed)
    indices = range(len(differences))
    draws: List[float] = []
    for _ in range(resamples):
        picked = [differences[rng.choice(indices)] for _ in indices]
        draws.append(sum(picked) / len(picked))
    draws.sort()
    tail = (1.0 - confidence) / 2.0
    low = draws[max(0, int(math.floor(tail * len(draws))))]
    high = draws[min(len(draws) - 1, int(math.ceil((1 - tail) * len(draws))) - 1)]
    below = sum(1 for d in draws if d <= 0.0)
    above = sum(1 for d in draws if d >= 0.0)
    return {
        "point": round(observed, 6),
        "ci_low": round(low, 6),
        "ci_high": round(high, 6),
        "confidence": confidence,
        "excludes_zero": bool(low > 0.0 or high < 0.0),
        "p_value": round(min(1.0, 2.0 * (min(below, above) + 1) / (len(draws) + 1)), 6),
        "pairs": len(differences),
        "resamples": len(draws),
        "seed": seed,
        "unit_of_analysis": "matched pair",
    }


def equivalence(interval: Dict[str, Any], band: float) -> Dict[str, Any]:
    """H4: is the permuted Delta inside a pre-specified band around zero?

    An equivalence test, not a failure to reject. A wide interval that merely
    contains zero is not evidence the control passed; the interval must sit
    *inside* the band.
    """
    if interval.get("ci_low") is None:
        return {"band": band, "passed": None, "reason": "no interval"}
    inside = (interval["ci_low"] > -band) and (interval["ci_high"] < band)
    return {
        "band": band,
        "passed": bool(inside),
        "interval": [interval["ci_low"], interval["ci_high"]],
        "reason": ("the permuted interval lies inside the band" if inside else
                   "the permuted interval is not contained in the band: the "
                   "probe may be responding to prose-era features rather than "
                   "to recency. Proposal 4.6 (V1) makes this BLOCKING -- the "
                   "primary result is withheld and pair construction revisited."),
        "pre_specified": ("The band must be fixed before the probe is run "
                          "(proposal 7.3, pre-registration). It is an input to "
                          "this function, never chosen from the output."),
    }


def probe(pairs: Sequence[Dict[str, Any]],
          admit_fn: Callable[[str, str], float],
          resamples: int = BOOTSTRAP_RESAMPLES,
          seed: int = BOOTSTRAP_SEED) -> Dict[str, Any]:
    """Delta = mean over pairs of P(admit | older) - P(admit | newer).

    ``admit_fn(question, passage_text) -> probability in [0, 1]`` is the filter
    under test. Pass the RAG2 perplexity-labelled filter for H1; pass the
    entailment-labelled filter for H2; the two must share backbone, capacity and
    training data or the comparison measures capacity instead of the label
    function (proposal 4.2, Figure 2).
    """
    per_pair: List[Dict[str, Any]] = []
    for pair in pairs:
        older_p = float(admit_fn(pair["question"], pair["older"]["text"]))
        newer_p = float(admit_fn(pair["question"], pair["newer"]["text"]))
        per_pair.append({
            "pair_id": pair["pair_id"],
            "p_admit_older": round(older_p, 6),
            "p_admit_newer": round(newer_p, 6),
            "difference": round(older_p - newer_p, 6),
            "age_gap_years": pair.get("age_gap_years"),
        })
    differences = [row["difference"] for row in per_pair]
    interval = paired_bootstrap(differences, resamples, seed)
    return {
        "delta": interval,
        "per_pair": per_pair,
        "mean_p_admit_older": round(
            sum(r["p_admit_older"] for r in per_pair) / len(per_pair), 6)
        if per_pair else None,
        "mean_p_admit_newer": round(
            sum(r["p_admit_newer"] for r in per_pair) / len(per_pair), 6)
        if per_pair else None,
        "direction": _direction(interval),
        "hypothesis": ("H1: Delta > 0, i.e. the filter preferentially admits "
                       "OLDER evidence. Direction is stated in the proposal; "
                       "the magnitude is not, and no expected value appears "
                       "anywhere in it."),
    }


def _direction(interval: Dict[str, Any]) -> str:
    if interval.get("point") is None:
        return "undetermined"
    if not interval["excludes_zero"]:
        return ("no asymmetry distinguishable from zero at this pair count -- "
                "an inconclusive result, which proposal 8 names as the most "
                "likely adverse outcome and materially worse than a clean null")
    return ("prefers older evidence (H1 direction)" if interval["point"] > 0
            else "prefers newer evidence (opposite to H1)")


def batched_admit_fn(pairs: Sequence[Dict[str, Any]],
                     render: Callable[[str, str], str],
                     score_batch: Callable[[List[str]], List[float]],
                     batch_size: int = 16,
                     progress: Optional[Callable[[int, int], None]] = None
                     ) -> Tuple[Callable[[str, str], float], int]:
    """Score every distinct (question, passage) once, then look scores up.

    A model call per passage would be needlessly slow, but batching introduces
    the one bug that would silently invalidate the whole probe: a passage
    receiving another passage's score. So the mapping is built explicitly and
    ``test_frb_pairs`` checks that each side gets back its own score.

    Returns the lookup and the number of distinct passages scored.
    """
    order: Dict[Tuple[str, str], int] = {}
    rendered: List[str] = []
    for pair in pairs:
        for side in ("older", "newer"):
            key = (pair["question"], pair[side]["text"])
            if key not in order:
                order[key] = len(rendered)
                rendered.append(render(*key))
    scores: List[float] = []
    for start in range(0, len(rendered), batch_size):
        scores.extend(score_batch(rendered[start:start + batch_size]))
        if progress:
            progress(min(start + batch_size, len(rendered)), len(rendered))
    if len(scores) != len(rendered):
        raise UnusableDataset(
            f"the scorer returned {len(scores)} scores for {len(rendered)} "
            "passages; the mapping from passage to score would be wrong")

    def admit_fn(question: str, text: str) -> float:
        return float(scores[order[(question, text)]])

    return admit_fn, len(rendered)


def digest_pairs(pairs: Sequence[Dict[str, Any]]) -> str:
    """A stable fingerprint of the pair set, so a result names its own inputs."""
    hasher = hashlib.sha256()
    for pair in sorted(pairs, key=lambda p: p["pair_id"]):
        hasher.update(pair["pair_id"].encode("utf-8"))
        hasher.update(str(pair["older"]["year"]).encode("utf-8"))
        hasher.update(str(pair["newer"]["year"]).encode("utf-8"))
        hasher.update(hashlib.sha256(
            pair["older"]["text"].encode("utf-8")).hexdigest().encode("utf-8"))
        hasher.update(hashlib.sha256(
            pair["newer"]["text"].encode("utf-8")).hexdigest().encode("utf-8"))
    return hasher.hexdigest()


def run_probe(pairs: Sequence[Dict[str, Any]],
              admit_fn: Callable[[str, str], float],
              equivalence_band: float,
              resamples: int = BOOTSTRAP_RESAMPLES,
              seed: int = BOOTSTRAP_SEED) -> Dict[str, Any]:
    """H1 and H4 together -- the proposal's Minimum Viable Implementation.

    Section 5.4 names exactly this as "a complete and defensible thesis should
    all later phases fail": FRB-PAIRS and the permutation control constructed
    from MedChangeQA, the label-function comparison executed, and H1 and H4
    reported with confidence intervals.
    """
    primary = probe(pairs, admit_fn, resamples, seed)
    control = probe(permute(pairs, seed), admit_fn, resamples, seed)
    gate = equivalence(control["delta"], equivalence_band)
    return {
        "pairs": len(pairs),
        "pair_digest": digest_pairs(pairs),
        "H1_primary": primary,
        "H4_permutation_control": control,
        "H4_equivalence": gate,
        "reportable": bool(gate["passed"]),
        "reportability_note": (
            "H1 is reportable only if the permutation control passes. Proposal "
            "4.6 lists V1 as BLOCKING: 'If the control fails, the probe is "
            "responding to prose-era features rather than recency and the "
            "primary result is uninterpretable.'"),
        "underpowered": len(pairs) < TARGET_PAIRS[0],
        "power_note": (
            f"Proposal 5.2 targets {TARGET_PAIRS[0]}-{TARGET_PAIRS[1]} pairs and "
            "8 warns that a modest asymmetry may fail to clear a conventional "
            "threshold at that count. Report an inconclusive result as "
            "inconclusive; do not adjust the target effect afterwards."),
    }
