"""RAG2-vs-SCAF controlled comparison over one frozen candidate set.

Two arms, one variable:

    ARM A   frozen candidates -> RAG2 filter -> context -> generator -> answer
    ARM B   frozen candidates -> SCAF filter -> context -> generator -> answer

Everything except the admission policy is shared by construction: both arms are
handed the *same* :class:`~scaf.frozen.FrozenCandidateSet` objects, build their
Evidence through the same converter, and use the same generator instance and
decoding settings. :func:`fairness_report` then re-checks that programmatically
after the fact, because "by construction" is a claim and the manifest should
carry a measurement.

Nothing here retrieves. If a future edit made an arm retrieve, the candidate
digests recorded per arm would diverge and :func:`fairness_report` would fail.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping, Optional, Sequence

from .frozen import FrozenCandidateSet, candidate_digest, set_digest


@dataclass
class ArmResult:
    """One arm's outcome for one question."""

    qid: str
    arm: str
    admitted_chunk_ids: List[str] = field(default_factory=list)
    rejected_chunk_ids: List[str] = field(default_factory=list)
    decisions: List[Dict[str, Any]] = field(default_factory=list)
    context: str = ""
    context_chars: int = 0
    answer: str = ""
    abstained: bool = False
    error: str = ""
    candidate_digest: str = ""

    @property
    def num_candidates(self) -> int:
        return len(self.admitted_chunk_ids) + len(self.rejected_chunk_ids)

    @property
    def admission_rate(self) -> float:
        return (len(self.admitted_chunk_ids) / self.num_candidates) if self.num_candidates else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "qid": self.qid, "arm": self.arm,
            "num_candidates": self.num_candidates,
            "num_admitted": len(self.admitted_chunk_ids),
            "admission_rate": round(self.admission_rate, 6),
            "admitted_chunk_ids": self.admitted_chunk_ids,
            "rejected_chunk_ids": self.rejected_chunk_ids,
            "context_chars": self.context_chars,
            "answer": self.answer,
            "abstained": self.abstained,
            "error": self.error,
            "candidate_digest": self.candidate_digest,
            "decisions": self.decisions,
        }


def build_context(frozen: FrozenCandidateSet, admitted: Sequence[str]) -> str:
    """The evidence block, in frozen order. Identical rule for both arms."""
    keep = [c for c in frozen.candidates if c.chunk_id in set(admitted)]
    return "\n".join(f"[{i}] {c.text.strip()}" for i, c in enumerate(keep, start=1))


def run_arm(arm: str, evidence_filter: Any, frozen_sets: Sequence[FrozenCandidateSet],
            generator: Optional[Any] = None) -> List[ArmResult]:
    """Score every question with one admission policy.

    ``generator`` is optional: with none, the arm still records admission and
    context, which is the part the preliminary milestone needs. A failing
    question is *recorded* with its error, never dropped -- a silently shrinking
    denominator is how a comparison stops meaning anything.
    """
    results: List[ArmResult] = []
    for frozen in frozen_sets:
        result = ArmResult(qid=frozen.qid, arm=arm,
                           candidate_digest=candidate_digest(frozen.candidates))
        try:
            question = frozen.to_question()
            evidence = [c.to_evidence() for c in frozen.candidates]
            decisions = evidence_filter.decide(question, evidence)
            if len(decisions) != len(evidence):
                raise ValueError(
                    f"filter returned {len(decisions)} decisions for {len(evidence)} candidates")

            for candidate, decision in zip(frozen.candidates, decisions):
                record = {
                    "chunk_id": candidate.chunk_id,
                    "keep": bool(decision.keep),
                    "label": decision.label,
                    "score": decision.score,
                    "canonical_date": candidate.canonical_date,
                    "authority_tier_label": candidate.authority_tier_label,
                    "source_category": candidate.source_category,
                    "rerank_rank": candidate.rerank_rank,
                }
                if getattr(decision, "detail", None):
                    record["detail"] = decision.detail
                result.decisions.append(record)
                (result.admitted_chunk_ids if decision.keep
                 else result.rejected_chunk_ids).append(candidate.chunk_id)

            result.context = build_context(frozen, result.admitted_chunk_ids)
            result.context_chars = len(result.context)
            result.abstained = bool(getattr(evidence_filter, "abstained", lambda: False)()) \
                if not result.admitted_chunk_ids else False

            if generator is not None:
                result.answer = generator(question, result.context)
        except Exception as exc:                       # recorded, never swallowed
            result.error = f"{type(exc).__name__}: {exc}"
        results.append(result)
    return results


# --------------------------------------------------------------------------
# Fairness
# --------------------------------------------------------------------------
def fairness_report(frozen_sets: Sequence[FrozenCandidateSet],
                    arm_a: Sequence[ArmResult],
                    arm_b: Sequence[ArmResult],
                    config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Re-check, after the fact, that only the admission policy differed.

    Every check here is one way the comparison could be quietly invalid. A
    failure means the numbers must not be reported.
    """
    checks: List[Dict[str, Any]] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        checks.append({"check": name, "pass": bool(ok), "detail": detail})

    a_by_qid = {r.qid: r for r in arm_a}
    b_by_qid = {r.qid: r for r in arm_b}
    frozen_qids = [f.qid for f in frozen_sets]

    check("same questions in both arms",
          sorted(a_by_qid) == sorted(b_by_qid) == sorted(frozen_qids),
          f"frozen={len(frozen_qids)} armA={len(a_by_qid)} armB={len(b_by_qid)}")

    digest_mismatch = [q for q in frozen_qids
                       if a_by_qid.get(q) and b_by_qid.get(q)
                       and a_by_qid[q].candidate_digest != b_by_qid[q].candidate_digest]
    check("same candidate set and ordering in both arms", not digest_mismatch,
          "identical per-question digests" if not digest_mismatch
          else f"mismatched: {digest_mismatch[:5]}")

    frozen_digests = {f.qid: candidate_digest(f.candidates) for f in frozen_sets}
    drifted = [q for q in frozen_qids
               if a_by_qid.get(q) and a_by_qid[q].candidate_digest != frozen_digests[q]]
    check("neither arm retrieved its own candidates", not drifted,
          "arm digests match the frozen file" if not drifted
          else f"drifted from frozen: {drifted[:5]}")

    count_mismatch = [q for q in frozen_qids
                      if a_by_qid.get(q) and b_by_qid.get(q)
                      and a_by_qid[q].num_candidates != b_by_qid[q].num_candidates]
    check("same number of candidates scored per question", not count_mismatch,
          "" if not count_mismatch else f"mismatched: {count_mismatch[:5]}")

    cfg = dict(config or {})
    check("same generator in both arms",
          cfg.get("arm_a_generator") == cfg.get("arm_b_generator"),
          f"{cfg.get('arm_a_generator')!r} vs {cfg.get('arm_b_generator')!r}")
    check("same decoding settings in both arms",
          cfg.get("arm_a_generation") == cfg.get("arm_b_generation"),
          json.dumps(cfg.get("arm_a_generation"), sort_keys=True))
    check("same corpus and index in both arms",
          cfg.get("arm_a_index") == cfg.get("arm_b_index"),
          f"{cfg.get('arm_a_index')!r}")

    # An admission policy must not be able to see the gold answer.
    leaked = [q for q in frozen_qids
              if (frozen_sets[frozen_qids.index(q)].question_metadata.get("answer")
                  and any("answer" in str(d.get("detail", {})).lower()
                          for d in (a_by_qid.get(q).decisions if a_by_qid.get(q) else [])))]
    check("no gold answer reachable from admission records", not leaked,
          "" if not leaked else f"suspect: {leaked[:5]}")

    check("failures reported rather than dropped",
          len(arm_a) == len(arm_b) == len(frozen_sets),
          f"armA={len(arm_a)} armB={len(arm_b)} frozen={len(frozen_sets)}")

    failed = [c for c in checks if not c["pass"]]
    return {
        "checks": checks,
        "passed": len(checks) - len(failed),
        "total": len(checks),
        "all_passed": not failed,
        "failed": [c["check"] for c in failed],
    }


# --------------------------------------------------------------------------
# Aggregation and manifest
# --------------------------------------------------------------------------
def summarise(results: Sequence[ArmResult]) -> Dict[str, Any]:
    ok = [r for r in results if not r.error]
    admitted = [len(r.admitted_chunk_ids) for r in ok]
    return {
        "questions_attempted": len(results),
        "questions_succeeded": len(ok),
        "questions_failed": len(results) - len(ok),
        "errors": [{"qid": r.qid, "error": r.error} for r in results if r.error][:20],
        "mean_candidates": round(sum(r.num_candidates for r in ok) / len(ok), 3) if ok else 0.0,
        "mean_admitted": round(sum(admitted) / len(ok), 3) if ok else 0.0,
        "total_admitted": sum(admitted),
        "total_candidates": sum(r.num_candidates for r in ok),
        "admission_rate": round(
            sum(admitted) / sum(r.num_candidates for r in ok), 6
        ) if ok and sum(r.num_candidates for r in ok) else 0.0,
        "questions_with_no_evidence": sum(1 for r in ok if not r.admitted_chunk_ids),
        "abstentions": sum(1 for r in ok if r.abstained),
        "mean_context_chars": round(
            sum(r.context_chars for r in ok) / len(ok), 1) if ok else 0.0,
        "answers_generated": sum(1 for r in ok if r.answer),
    }


def recency_profile(results: Sequence[ArmResult], boundary_year: int = 2020) -> Dict[str, Any]:
    """Descriptive admission-by-age. **Not** the FRB-PAIRS test.

    Splits candidates at ``boundary_year`` and reports the admission rate each
    side. Undated candidates are counted separately rather than assigned a side.
    """
    buckets = {"older": [0, 0], "newer": [0, 0], "undated": [0, 0]}   # [admitted, total]
    for result in results:
        for decision in result.decisions:
            date = str(decision.get("canonical_date", ""))[:4]
            if date.isdigit():
                key = "older" if int(date) < boundary_year else "newer"
            else:
                key = "undated"
            buckets[key][1] += 1
            if decision.get("keep"):
                buckets[key][0] += 1
    out: Dict[str, Any] = {"boundary_year": boundary_year}
    for key, (admitted, total) in buckets.items():
        out[key] = {
            "candidates": total,
            "admitted": admitted,
            "admission_rate": round(admitted / total, 6) if total else None,
        }
    older, newer = out["older"]["admission_rate"], out["newer"]["admission_rate"]
    out["newer_minus_older"] = (
        round(newer - older, 6) if older is not None and newer is not None else None)
    out["note"] = ("Descriptive only. Formal FRB-PAIRS analysis remains pending: "
                   "matched older/newer evidence pairs have not been constructed.")
    return out


def environment() -> Dict[str, Any]:
    def _git(*args: str) -> Optional[str]:
        try:
            return subprocess.check_output(["git", *args], stderr=subprocess.DEVNULL,
                                           text=True).strip()
        except Exception:
            return None

    packages: Dict[str, Optional[str]] = {}
    for name in ("torch", "transformers", "numpy", "faiss"):
        try:
            packages[name] = str(getattr(__import__(name), "__version__", "unknown"))
        except Exception:
            packages[name] = None
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "packages": packages,
    }


def write_manifest(path: str, payload: Dict[str, Any]) -> str:
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)
        handle.write("\n")
    return path


def build_manifest(frozen_sets: Sequence[FrozenCandidateSet],
                   arm_a: Sequence[ArmResult], arm_b: Sequence[ArmResult],
                   config: Dict[str, Any]) -> Dict[str, Any]:
    """The machine-readable record of the run and its fairness controls."""
    return {
        "label": "PRELIMINARY / DEVELOPMENT RESULTS",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "questions": len(frozen_sets),
        "frozen_set_digest": set_digest(frozen_sets),
        "config": config,
        "environment": environment(),
        "fairness": fairness_report(frozen_sets, arm_a, arm_b, config),
        "arm_a_rag2": summarise(arm_a),
        "arm_b_scaf": summarise(arm_b),
        "recency_arm_a_rag2": recency_profile(arm_a),
        "recency_arm_b_scaf": recency_profile(arm_b),
    }


# --------------------------------------------------------------------------
# Scientific-run preconditions
# --------------------------------------------------------------------------
#: Retrieval sources that may NOT back a reported comparison.
DEVELOPMENT_RETRIEVAL_MARKERS = ("lexical", "tfidf", "bm25", "mock", "stub", "dev")


def scientific_preconditions(config: Mapping[str, Any],
                             frozen_sets: Sequence[FrozenCandidateSet],
                             arm_a_filter: Any, arm_b_filter: Any) -> List[Dict[str, Any]]:
    """The conditions a *reportable* RAG2-vs-SCAF run must satisfy.

    These are separate from :func:`fairness_report`, which asks "were the two
    arms treated identically?". These ask the prior question: "is either arm the
    thing it claims to be?". A run can be perfectly fair and still be worthless
    because Arm A had no filter or retrieval was a lexical stand-in.

    Returns one record per condition. The caller must abort if any fails --
    warning and continuing is how a prototype gets reported as a result.
    """
    checks: List[Dict[str, Any]] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        checks.append({"check": name, "pass": bool(ok), "detail": detail})

    # -- Arm A really is the trained RAG2 perplexity filter --------------
    arm_a_kind = str(config.get("arm_a_filter", ""))
    check("Arm A is the RAG2 perplexity filter", arm_a_kind == "rag2_perplexity",
          f"arm_a_filter={arm_a_kind!r}")
    check("Arm A is NOT passthrough", arm_a_kind != "passthrough",
          "passthrough is the paper's 'w/o filter' ablation, not the RAG2 filter")
    checkpoint = (config.get("arm_a_filter_config") or {}).get("checkpoint")
    check("Arm A has a trained checkpoint", bool(checkpoint), f"checkpoint={checkpoint!r}")
    if checkpoint:
        check("Arm A checkpoint exists on disk", os.path.isdir(str(checkpoint)),
              str(checkpoint))
    else:
        check("Arm A checkpoint exists on disk", False, "no checkpoint configured")
    # A loaded Flan-T5 filter exposes the two label token ids; a stand-in does not.
    check("Arm A filter loaded its label tokens",
          all(getattr(arm_a_filter, attr, None) is not None
              for attr in ("helpful_id", "not_helpful_id")),
          f"{type(arm_a_filter).__name__}")

    # -- retrieval really is production MedCPT ---------------------------
    is_medcpt = config.get("retrieval_is_medcpt")
    check("production MedCPT retrieval was used", is_medcpt is True,
          f"retrieval_is_medcpt={is_medcpt!r}")
    source = str(config.get("retrieval_source", "")).lower()
    offending = [m for m in DEVELOPMENT_RETRIEVAL_MARKERS if m in source]
    check("no development retrieval stand-in", not offending,
          f"retrieval_source={config.get('retrieval_source')!r}"
          + (f" matched {offending}" if offending else ""))

    # -- SCAF really computed its components -----------------------------
    described = arm_b_filter.describe() if hasattr(arm_b_filter, "describe") else {}
    check("Arm B is SCAF", described.get("name") == "scaf", str(described.get("name")))
    check("SCAF support has corpus statistics",
          described.get("support_idf_source") == "corpus",
          f"idf_source={described.get('support_idf_source')!r}")
    weights = described.get("weights") or {}
    check("SCAF currency is active", float(weights.get("currency", 0)) > 0,
          f"w_currency={weights.get('currency')}")
    check("SCAF authority is active", float(weights.get("authority", 0)) > 0,
          f"w_authority={weights.get('authority')}")
    check("SCAF retraction gate is active", bool(described.get("reject_retracted")),
          f"reject_retracted={described.get('reject_retracted')}")
    check("SCAF abstention is enabled", bool(described.get("abstain")),
          f"abstain={described.get('abstain')}")

    # -- answers really were generated -----------------------------------
    generator = config.get("arm_a_generator")
    check("a generator is configured", bool(generator) and generator != "none",
          f"generator={generator!r}")

    check("at least 20 questions", len(frozen_sets) >= 20, f"n={len(frozen_sets)}")
    return checks


def scientific_report(config: Mapping[str, Any], frozen_sets: Sequence[FrozenCandidateSet],
                      arm_a_filter: Any, arm_b_filter: Any) -> Dict[str, Any]:
    checks = scientific_preconditions(config, frozen_sets, arm_a_filter, arm_b_filter)
    failed = [c for c in checks if not c["pass"]]
    return {
        "checks": checks,
        "passed": len(checks) - len(failed),
        "total": len(checks),
        "all_passed": not failed,
        "failed": [c["check"] for c in failed],
    }


def components_actually_computed(results: Sequence[ArmResult]) -> Dict[str, Any]:
    """Evidence, from the recorded decisions, that SCAF really scored each term.

    A constant sub-score across every candidate means the component is inert --
    which is exactly how the v1 support collapse hid in plain sight.
    """
    values: Dict[str, List[float]] = {"sigma_support": [], "gamma_currency": [],
                                      "tau_authority": [], "scaf_score": []}
    for result in results:
        for decision in result.decisions:
            detail = decision.get("detail") or {}
            for key in values:
                if key in detail:
                    values[key].append(float(detail[key]))
    out: Dict[str, Any] = {}
    for key, series in values.items():
        distinct = len(set(round(v, 6) for v in series))
        out[key] = {
            "n": len(series),
            "distinct_values": distinct,
            "min": round(min(series), 6) if series else None,
            "max": round(max(series), 6) if series else None,
            "varies": distinct > 1,
        }
    return out
