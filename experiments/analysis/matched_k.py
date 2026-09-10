"""Matched-k selection: give both arms exactly the same evidence budget.

The completed admission comparison measured what each policy *naturally* admits,
and the answer is lopsided -- SCAF passed ~19 passages per question, RAG2 passed
0.13. Any answer difference between those two runs is confounded by roughly 688x
more context on one side, so it cannot be attributed to the admission policy.

This module answers a different question:

    when both policies must hand the generator exactly five passages,
    do they choose different evidence?

The rule, and why it is the established one
-------------------------------------------
Each arm ranks the **same 20 frozen candidates** by **its own score** and takes
the top five. That is the procedure already used by ``matched_budget`` in
``evidence_quality.py`` and already reported in the results analysis; nothing
new is invented here, no weight is changed and no threshold is tuned.

Selection deliberately ignores each arm's admission threshold. RAG2 admits only
4 passages across all 600 decisions, so a threshold-respecting rule could not
reach k=5 for 26 of the 30 questions, and the experiment would collapse back
into the confound it exists to remove. The threshold governs *how much* evidence
a policy admits -- which the original experiment measured. This one holds the
amount fixed and varies only *which* evidence, so the score's ranking is the
relevant output.

This module selects and checks. It does not generate: generation needs the
candidate text and a live model, neither of which belongs in a pure selection
step.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence

#: The scientific run this experiment must be built on. A different digest means
#: a different candidate set, and therefore a different experiment.
SCIENTIFIC_FROZEN_DIGEST = (
    "316260f04c1720fbc20c1b584ea9f0a093dedbae9fd46476d8a7f45380e86aad")

DEFAULT_K = 5
ARMS = ("rag2", "scaf")


class ProvenanceFailure(Exception):
    """Raised when the inputs are not the scientific run's inputs."""


def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(65536), b""):
            digest.update(block)
    return digest.hexdigest()


def select_top_k(decisions: Sequence[Dict[str, Any]], arm: str,
                 k: int = DEFAULT_K) -> List[str]:
    """The arm's own top-k chunk ids, deterministically.

    Sorted by descending score with ``chunk_id`` as the tie-break, so two runs
    over the same file always produce the same five passages even when scores
    are equal.
    """
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}; expected one of {ARMS}")
    ordered = sorted(decisions, key=lambda d: (-float(d["score"]), str(d["chunk_id"])))
    return [str(d["chunk_id"]) for d in ordered[:k]]


def verify_inputs(per_question_path: str, run_manifest_path: str,
                  frozen_meta_path: str, questions_path: str,
                  k: int = DEFAULT_K) -> Dict[str, Any]:
    """Everything that must hold before a single answer is generated.

    Returns a report. The caller decides whether to proceed; ``run_matched_k``
    refuses when any check fails, because a matched-k run built on the wrong
    candidate set is not a weaker version of this experiment -- it is a
    different one wearing its name.
    """
    checks: List[Dict[str, Any]] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        checks.append({"check": name, "pass": bool(ok), "detail": detail})

    rows = _read_jsonl(per_question_path) if os.path.isfile(per_question_path) else []
    check("scientific decisions file exists", bool(rows), per_question_path)
    check("exactly 30 questions", len(rows) == 30, f"{len(rows)} questions")

    counts = Counter(len(r[arm]["decisions"]) for r in rows for arm in ARMS)
    check("exactly 20 candidates per question per arm",
          set(counts) == {20} if counts else False, f"candidate counts {dict(counts)}")

    same_ids = all(
        [d["chunk_id"] for d in r["rag2"]["decisions"]]
        == [d["chunk_id"] for d in r["scaf"]["decisions"]] for r in rows)
    check("both arms hold identical candidate ids in identical order", same_ids)

    enough = all(len(r[arm]["decisions"]) >= k for r in rows for arm in ARMS)
    check(f"every arm-question has at least k={k} candidates", enough)

    run_manifest: Dict[str, Any] = {}
    if os.path.isfile(run_manifest_path):
        with open(run_manifest_path, "r", encoding="utf-8") as handle:
            run_manifest = json.load(handle)
    recorded = run_manifest.get("frozen_set_digest")
    check("scientific run digest is the expected one",
          recorded == SCIENTIFIC_FROZEN_DIGEST,
          f"run manifest records {recorded}")

    frozen_meta: Dict[str, Any] = {}
    if frozen_meta_path and os.path.isfile(frozen_meta_path):
        with open(frozen_meta_path, "r", encoding="utf-8") as handle:
            frozen_meta = json.load(handle)
    local_digest = frozen_meta.get("frozen_set_digest")
    provenance = frozen_meta.get("provenance") or {}
    check("frozen candidate file on disk IS the scientific one",
          local_digest == SCIENTIFIC_FROZEN_DIGEST,
          f"local file records {local_digest} (source: {provenance.get('source')})"
          if local_digest else "no frozen candidate metadata found on disk")
    check("frozen candidate retrieval is production MedCPT",
          provenance.get("retrieval_is_medcpt") is True,
          f"retrieval_is_medcpt={provenance.get('retrieval_is_medcpt')}")

    if os.path.isfile(questions_path):
        questions = _read_jsonl(questions_path)
        by_qid = {q["qid"]: q["question"] for q in questions}
        matching = sum(1 for r in rows if by_qid.get(r["qid"]) == r["question"])
        check("question ids and text match the committed question set",
              matching == len(rows) and len(questions) == 30,
              f"{matching}/{len(rows)} match; question set holds {len(questions)}")
    else:
        check("question ids and text match the committed question set", False,
              f"not found: {questions_path}")

    generation = (run_manifest.get("config") or {}).get("arm_a_generation") or {}
    same_generation = json.dumps(
        (run_manifest.get("config") or {}).get("arm_a_generation"), sort_keys=True) == \
        json.dumps((run_manifest.get("config") or {}).get("arm_b_generation"),
                   sort_keys=True)
    check("generation configuration identical across arms in the source run",
          same_generation)

    return {
        "all_passed": all(c["pass"] for c in checks),
        "checks": checks,
        "questions": len(rows),
        "k": k,
        "expected_frozen_digest": SCIENTIFIC_FROZEN_DIGEST,
        "run_manifest_digest": recorded,
        "local_frozen_digest": local_digest,
        "local_frozen_source": provenance.get("source"),
        "generation_config": generation,
    }


def build_selection(per_question_path: str, k: int = DEFAULT_K) -> List[Dict[str, Any]]:
    """The exact five passages each arm would receive, per question.

    Needs only the saved scores: no retrieval, no reranking, no index, no
    candidate text. The result is a specification that can be checked against
    the frozen set before anything is generated.
    """
    selection = []
    for row in _read_jsonl(per_question_path):
        entry: Dict[str, Any] = {
            "qid": row["qid"], "question": row["question"], "k": k,
            "candidate_ids": [d["chunk_id"] for d in row["rag2"]["decisions"]],
        }
        for arm in ARMS:
            entry[f"{arm}_selected_chunk_ids"] = select_top_k(
                row[arm]["decisions"], arm, k)
        a, b = set(entry["rag2_selected_chunk_ids"]), set(entry["scaf_selected_chunk_ids"])
        entry["overlap_count"] = len(a & b)
        entry["identical_sets"] = a == b
        entry["jaccard"] = round(len(a & b) / len(a | b), 6) if a | b else None
        selection.append(entry)
    return selection


def selection_statistics(selection: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Descriptive only. How different are the two evidence sets at matched k?

    None of this is an answer-quality result. It describes what the generator
    would be handed, not what it would produce.
    """
    overlaps = [e["overlap_count"] for e in selection]
    return {
        "questions": len(selection),
        "k": selection[0]["k"] if selection else None,
        "questions_with_identical_sets": sum(1 for e in selection if e["identical_sets"]),
        "questions_with_any_overlap": sum(1 for o in overlaps if o > 0),
        "questions_with_no_overlap": sum(1 for o in overlaps if o == 0),
        "questions_where_evidence_differs": sum(
            1 for e in selection if not e["identical_sets"]),
        "overlap_distribution": dict(sorted(Counter(overlaps).items())),
        "mean_overlap": round(sum(overlaps) / len(overlaps), 4) if overlaps else None,
        "mean_jaccard": round(
            sum(e["jaccard"] for e in selection) / len(selection), 6) if selection else None,
        "guard": ("descriptive statistics about which passages each policy would "
                  "hand the generator. They are not answer-quality measurements "
                  "and must not be reported as one."),
    }


def check_selection_invariants(selection: Sequence[Dict[str, Any]],
                               k: int = DEFAULT_K) -> Dict[str, Any]:
    """The post-selection fairness invariants, before generation."""
    checks: List[Dict[str, Any]] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        checks.append({"check": name, "pass": bool(ok), "detail": detail})

    check("30 questions selected", len(selection) == 30, f"{len(selection)}")
    for arm in ARMS:
        field = f"{arm}_selected_chunk_ids"
        sizes = Counter(len(e[field]) for e in selection)
        check(f"{arm}: exactly {k} passages for every question",
              set(sizes) == {k} if sizes else False, f"sizes {dict(sizes)}")
        dupes = [e["qid"] for e in selection if len(set(e[field])) != len(e[field])]
        check(f"{arm}: no duplicate passage within a question", not dupes, f"{dupes[:5]}")
        outside = [e["qid"] for e in selection
                   if not set(e[field]) <= set(e["candidate_ids"])]
        check(f"{arm}: every selected passage is in that question's frozen candidates",
              not outside, f"{outside[:5]}")

    pools = [tuple(e["candidate_ids"]) for e in selection]
    check("no cross-question candidate leakage",
          len(set(pools)) == len(pools),
          "each question draws from its own distinct candidate pool")
    return {"all_passed": all(c["pass"] for c in checks), "checks": checks}
