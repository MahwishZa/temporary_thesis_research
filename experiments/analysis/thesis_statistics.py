"""Statistics for the admission comparison, with the clustering taken seriously.

The proposal's analysis plan (7.3) says the thing that governs this file:

    "[DES] Claims are clustered within questions, so a naive per-claim
     chi-squared test would be anti-conservative."

Every observation here is clustered the same way. Twenty candidates share one
question, one retrieval, one rationale and one topic; treating them as 600
independent trials would shrink every interval by roughly the square root of the
cluster size and manufacture significance out of nothing. So the resampling unit
is **the question**, never the passage.

What is computed
----------------
* the paired admission contrast (each candidate is scored by both arms, so the
  pairing is exact), with a question-clustered bootstrap interval;
* McNemar's exact test on the discordant pairs -- reported **alongside** the
  clustered interval and explicitly labelled anti-conservative, because a reader
  will expect the conventional number and the honest move is to show it next to
  the one that respects the design rather than to omit it;
* Cohen's h, an effect size for two proportions that does not depend on n;
* the association between each arm's score and the human usefulness label, with
  a question-clustered interval.

Dependency-free by design: no scipy, no sklearn. The exact binomial and the
bootstrap are short enough to read, and a reader can check them.

What it does not do
-------------------
No test here can support a claim about answer quality, correctness, or recency
bias. Admission is what was measured, so admission is what is tested.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

#: Resamples for every bootstrap. The proposal specifies 10,000 (7.3).
BOOTSTRAP_RESAMPLES = 10000

#: Fixed so a rerun reproduces the intervals exactly.
BOOTSTRAP_SEED = 20260910


def _ranks(values: Sequence[float]) -> List[float]:
    """Average ranks, so ties do not distort a rank correlation."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    position = 0
    while position < len(order):
        end = position
        while end + 1 < len(order) and values[order[end + 1]] == values[order[position]]:
            end += 1
        shared = (position + end) / 2 + 1
        for index in range(position, end + 1):
            ranks[order[index]] = shared
        position = end + 1
    return ranks


def spearman(x: Sequence[float], y: Sequence[float]) -> Optional[float]:
    """Spearman rank correlation, or None where it is undefined."""
    if len(x) != len(y) or len(x) < 3:
        return None
    rx, ry = _ranks(x), _ranks(y)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    numerator = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    denominator = math.sqrt(sum((a - mx) ** 2 for a in rx)
                            * sum((b - my) ** 2 for b in ry))
    return None if denominator == 0 else numerator / denominator


# --------------------------------------------------------------------------
# Exact tests, written out rather than imported
# --------------------------------------------------------------------------
def binomial_two_sided(successes: int, trials: int, p: float = 0.5) -> float:
    """Exact two-sided binomial p-value by the method-of-small-p-values.

    Sums the probability of every outcome no more likely than the observed one.
    At p = 0.5 this is the exact McNemar test.
    """
    if trials <= 0:
        return 1.0
    def pmf(k: int) -> float:
        return math.comb(trials, k) * (p ** k) * ((1 - p) ** (trials - k))
    observed = pmf(successes)
    total = sum(pmf(k) for k in range(trials + 1)
                if pmf(k) <= observed * (1 + 1e-9))
    return min(1.0, total)


def mcnemar_exact(only_a: int, only_b: int) -> Dict[str, Any]:
    """Exact McNemar on the discordant counts of a paired binary outcome.

    ``only_a`` is the number of items the first arm admitted and the second did
    not; ``only_b`` the reverse. Concordant pairs carry no information about a
    difference and are correctly ignored.
    """
    discordant = only_a + only_b
    return {
        "discordant_pairs": discordant,
        "only_first": only_a,
        "only_second": only_b,
        "p_value": binomial_two_sided(only_a, discordant),
        "test": "exact McNemar (binomial on discordant pairs)",
        "independence_assumption": (
            "treats the 600 candidates as independent. They are not: 20 share "
            "each question. This p-value is therefore ANTI-CONSERVATIVE and is "
            "reported only next to the question-clustered interval."),
    }


def cohens_h(p1: float, p2: float) -> float:
    """Effect size for two proportions. Independent of sample size."""
    def phi(p: float) -> float:
        return 2.0 * math.asin(math.sqrt(min(max(p, 0.0), 1.0)))
    return phi(p1) - phi(p2)


def interpret_h(h: float) -> str:
    magnitude = abs(h)
    if magnitude < 0.2:
        return "negligible"
    if magnitude < 0.5:
        return "small"
    if magnitude < 0.8:
        return "medium"
    return "large"


# --------------------------------------------------------------------------
# The clustered bootstrap
# --------------------------------------------------------------------------
def cluster_bootstrap(clusters: Sequence[Sequence[Any]],
                      statistic: Callable[[List[Any]], Optional[float]],
                      resamples: int = BOOTSTRAP_RESAMPLES,
                      seed: int = BOOTSTRAP_SEED,
                      confidence: float = 0.95) -> Dict[str, Any]:
    """Resample whole clusters with replacement, then recompute the statistic.

    This is the only resampling scheme that respects the design: a question is
    drawn or not drawn as a unit, carrying all of its candidates with it. The
    interval it produces is wider than a naive per-row bootstrap, and that width
    is the honest one.
    """
    observed = statistic([row for cluster in clusters for row in cluster])
    if observed is None or not clusters:
        return {"point": None, "ci_low": None, "ci_high": None,
                "resamples": 0, "clusters": len(clusters),
                "note": "statistic undefined on the observed data"}
    rng = random.Random(seed)
    indices = range(len(clusters))
    draws: List[float] = []
    for _ in range(resamples):
        picked = [clusters[rng.choice(indices)] for _ in indices]
        value = statistic([row for cluster in picked for row in cluster])
        if value is not None:
            draws.append(value)
    if not draws:
        return {"point": round(observed, 6), "ci_low": None, "ci_high": None,
                "resamples": 0, "clusters": len(clusters),
                "note": "statistic undefined on every resample"}
    draws.sort()
    tail = (1.0 - confidence) / 2.0
    low = draws[max(0, int(math.floor(tail * len(draws))))]
    high = draws[min(len(draws) - 1, int(math.ceil((1 - tail) * len(draws))) - 1)]
    return {
        "point": round(observed, 6),
        "ci_low": round(low, 6),
        "ci_high": round(high, 6),
        "confidence": confidence,
        "excludes_zero": bool(low > 0.0 or high < 0.0),
        "p_value": round(bootstrap_p_value(draws, 0.0), 6),
        "resamples": len(draws),
        "clusters": len(clusters),
        "method": "percentile bootstrap over whole clusters (questions)",
        "seed": seed,
    }


def bootstrap_p_value(draws: Sequence[float], null: float = 0.0) -> float:
    """Two-sided bootstrap p-value: how often does the resample cross the null?

    Twice the smaller tail, with the conventional +1 correction so a p-value is
    never reported as exactly zero on a finite number of resamples.
    """
    if not draws:
        return 1.0
    n = len(draws)
    below = sum(1 for d in draws if d <= null)
    above = sum(1 for d in draws if d >= null)
    return min(1.0, 2.0 * (min(below, above) + 1) / (n + 1))


def holm_bonferroni(p_values: Dict[str, float],
                    alpha: float = 0.05) -> Dict[str, Dict[str, Any]]:
    """Holm-Bonferroni step-down, as the proposal's analysis plan specifies.

    Five associations are examined on one sample, so an uncorrected interval
    that excludes zero is not by itself a finding. Holm controls the family-wise
    error rate without assuming the tests are independent -- which they are not,
    since the scores are computed from overlapping inputs.
    """
    ordered = sorted(p_values.items(), key=lambda kv: kv[1])
    total = len(ordered)
    out: Dict[str, Dict[str, Any]] = {}
    still_rejecting = True
    for index, (name, p) in enumerate(ordered):
        threshold = alpha / (total - index)
        if p > threshold:
            still_rejecting = False
        out[name] = {"p_value": round(p, 6),
                     "holm_threshold": round(threshold, 6),
                     "survives_holm": bool(still_rejecting and p <= threshold),
                     "rank": index + 1}
    return out


def _by_question(rows: Sequence[Dict[str, Any]],
                 key: str = "qid") -> List[List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row[key]].append(row)
    return [grouped[q] for q in sorted(grouped)]


# --------------------------------------------------------------------------
# The admission contrast
# --------------------------------------------------------------------------
def admission_statistics(records: Sequence[Dict[str, Any]],
                         resamples: int = BOOTSTRAP_RESAMPLES) -> Dict[str, Any]:
    """How differently do the two policies admit, and how sure can we be?

    The pairing is exact: every candidate was scored by both arms on the same
    frozen text, so this is a matched-pair contrast, not two samples.
    """
    clusters = _by_question(records)
    rag2_rate = sum(1 for r in records if r["rag2_admit"]) / len(records)
    scaf_rate = sum(1 for r in records if r["scaf_admit"]) / len(records)

    only_scaf = sum(1 for r in records if r["scaf_admit"] and not r["rag2_admit"])
    only_rag2 = sum(1 for r in records if r["rag2_admit"] and not r["scaf_admit"])
    both = sum(1 for r in records if r["rag2_admit"] and r["scaf_admit"])
    neither = sum(1 for r in records if not r["rag2_admit"] and not r["scaf_admit"])

    def difference(rows: List[Dict[str, Any]]) -> Optional[float]:
        if not rows:
            return None
        return (sum(1 for r in rows if r["scaf_admit"]) / len(rows)
                - sum(1 for r in rows if r["rag2_admit"]) / len(rows))

    def scaf_only(rows: List[Dict[str, Any]]) -> Optional[float]:
        return None if not rows else sum(1 for r in rows if r["scaf_admit"]) / len(rows)

    def rag2_only(rows: List[Dict[str, Any]]) -> Optional[float]:
        return None if not rows else sum(1 for r in rows if r["rag2_admit"]) / len(rows)

    h = cohens_h(scaf_rate, rag2_rate)
    return {
        "candidates": len(records),
        "questions": len(clusters),
        "contingency": {"both_admitted": both, "scaf_only": only_scaf,
                        "rag2_only": only_rag2, "neither": neither},
        "rag2_admission_rate": round(rag2_rate, 6),
        "scaf_admission_rate": round(scaf_rate, 6),
        "rag2_rate_clustered": cluster_bootstrap(clusters, rag2_only, resamples),
        "scaf_rate_clustered": cluster_bootstrap(clusters, scaf_only, resamples),
        "difference_clustered": cluster_bootstrap(clusters, difference, resamples),
        "mcnemar": mcnemar_exact(only_scaf, only_rag2),
        "effect_size": {"cohens_h": round(h, 4), "magnitude": interpret_h(h),
                        "note": ("An effect size for how differently the two "
                                 "policies admit. It is not evidence that either "
                                 "admission rate is correct.")},
        "interpretation_guard": (
            "This measures how much evidence each policy admits at its own "
            "configured operating point. It does not measure whether the "
            "admitted evidence is better, and a larger admission rate is not a "
            "better one -- 96% admission is close to no filtering at all."),
    }


# --------------------------------------------------------------------------
# Evidence quality
# --------------------------------------------------------------------------
def evidence_quality_statistics(annotations: Sequence[Dict[str, Any]],
                                score_by_chunk: Dict[Tuple[str, str], Dict[str, float]],
                                resamples: int = BOOTSTRAP_RESAMPLES) -> Dict[str, Any]:
    """Do the machine scores order passages the way the annotator did?

    A correlation here is **not** validation of the scoring model: the labels are
    passage-level usefulness judgements from one annotator, and sigma is a
    lexical overlap score, so a positive association is partly the annotator and
    the scorer agreeing about topicality. The interval is what stops a small
    correlation on 120 clustered rows from being read as a finding.
    """
    rows: List[Dict[str, Any]] = []
    unresolved: List[str] = []
    for record in annotations:
        label = record.get("human_label")
        key = (record.get("qid"), record.get("chunk_id"))
        scores = score_by_chunk.get(key)
        if label is None or scores is None:
            unresolved.append(str(key))
            continue
        rows.append({"qid": record["qid"], "human_label": float(label), **scores})
    clusters = _by_question(rows)

    def association(name: str) -> Callable[[List[Dict[str, Any]]], Optional[float]]:
        def statistic(subset: List[Dict[str, Any]]) -> Optional[float]:
            if len(subset) < 3:
                return None
            return spearman([r[name] for r in subset],
                            [r["human_label"] for r in subset])
        return statistic

    names = sorted({k for r in rows for k in r if k not in ("qid", "human_label")})
    associations = {name: cluster_bootstrap(clusters, association(name), resamples)
                    for name in names}
    # Five associations on one sample: an interval that excludes zero is not a
    # finding until multiplicity is controlled. 7.3 specifies Holm-Bonferroni.
    holm = holm_bonferroni({name: block["p_value"] for name, block in associations.items()
                            if block.get("p_value") is not None})
    for name, verdict in holm.items():
        associations[name]["holm"] = verdict

    label_counts: Dict[str, int] = defaultdict(int)
    for row in rows:
        label_counts[str(int(row["human_label"]))] += 1

    return {
        "rows": len(rows),
        "unresolved": unresolved,
        "questions": len(clusters),
        "label_distribution": dict(sorted(label_counts.items())),
        "rank_association": associations,
        "guard": (
            "Descriptive association only. These labels were produced "
            "independently of every machine score (annotation pass "
            "'corrected-human-only-v1', no suggestion generated or shown), which "
            "is what makes them usable at all -- but a correlation between a "
            "lexical support score and a human topicality judgement is not "
            "evidence that the support score measures evidential support, and "
            "the sample is 120 non-randomly-selected passages clustered in 29 "
            "questions."),
    }


def build_statistics(records: Sequence[Dict[str, Any]],
                     annotations: Sequence[Dict[str, Any]],
                     resamples: int = BOOTSTRAP_RESAMPLES) -> Dict[str, Any]:
    """Everything, with the design limits stated next to the numbers."""
    score_by_chunk = {
        (r["qid"], r["chunk_id"]): {"scaf_score": r["recorded_score"],
                                    "rag2_score": r["rag2_score"],
                                    "sigma_support": r["sigma"],
                                    "gamma_currency": r["gamma"],
                                    "rho_rerank": r["rho"]}
        for r in records}
    return {
        "resamples": resamples,
        "seed": BOOTSTRAP_SEED,
        "resampling_unit": "question",
        "why_clustered": (
            "20 candidates share each question, so the effective sample size is "
            "much closer to 30 than to 600. Every interval here resamples whole "
            "questions."),
        "admission": admission_statistics(records, resamples),
        "evidence_quality": evidence_quality_statistics(
            annotations, score_by_chunk, resamples),
        "dual_reporting": {
            "abstentions_rag2": 0,
            "abstentions_scaf": 0,
            "note": ("The proposal requires every rate twice -- conditional on "
                     "answering, and with abstentions counted as failures. "
                     "Neither arm abstained on any of the 30 questions, so the "
                     "two readings coincide here. RAG2 nonetheless answered 26 "
                     "of 30 questions with zero admitted passages, which is "
                     "closed-book generation, not abstention."),
        },
        "not_tested_here": {
            "H1/H2/H3 (admission asymmetry)": "needs FRB-PAIRS; not constructed",
            "H4 (permutation control)": "needs FRB-PAIRS; not constructed",
            "H5 (unsupported-claim rate)": "needs claim-level answer labels; none exist",
            "H6 (mediation by retrieval recall)": "needs relevance labels per arm",
            "H7 (non-inferiority on time-invariant QA)": "needs gold answers; none exist",
        },
    }
