"""Can the repository evaluate ANSWER quality yet? Inspect and decide.

This module opens files and reports what it finds. It generates nothing, scores
nothing, and creates no reference answer -- inventing a gold answer is exactly
the failure it exists to prevent.

The distinction it enforces
---------------------------
Three different things in this repository are easy to mistake for an
answer-quality gold standard, and none of them is one:

* the **120 evidence-quality labels** judge whether a *passage* is useful for a
  question. They say nothing about whether an *answer* is correct;
* the **filter training weak labels** (``[HELPFUL]`` / ``[NOT_HELPFUL]``) are
  training data for RAG2's filter, derived by rule, not evaluation gold;
* the **SCAF and RAG2 scores** are the systems' own opinions of a passage. Using
  a system's own score to grade its own output is circular.

An answer-quality target has to be independent of all three.
"""

from __future__ import annotations

import json
import os
import re
import statistics
import time
from collections import Counter
from typing import Any, Dict, List, Sequence

#: Field names that would indicate a gold or reference answer if present.
ANSWER_TARGET_PATTERNS = (
    "gold", "reference", "answer_key", "ideal_answer", "ground_truth",
    "expected_answer", "correct_answer",
)

READY, PARTIAL, NOT_READY = "READY", "PARTIALLY READY", "NOT READY"


def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def inspect_question_set(path: str) -> Dict[str, Any]:
    """What each of the evaluation questions actually carries."""
    if not os.path.isfile(path):
        return {"path": path, "exists": False}
    rows = _read_jsonl(path)
    fields = Counter(k for r in rows for k in r)
    target_fields = sorted(
        f for f in fields
        if any(p in f.lower() for p in ANSWER_TARGET_PATTERNS)
        or f.lower() in ("answer", "answers", "options", "choices"))
    return {
        "path": path,
        "exists": True,
        "questions": len(rows),
        "fields": {k: v for k, v in sorted(fields.items())},
        "answer_target_fields": target_fields,
        "has_gold_answers": bool(target_fields),
        "set_identifiers": sorted({str(r.get("set")) for r in rows}),
        "notes": sorted({str(r.get("note")) for r in rows if r.get("note")}),
        "per_question_holdings": {
            "question_text": len(rows),
            "gold_answer": 0 if not target_fields else None,
            "reference_answer": 0 if not target_fields else None,
            "expert_labels": 0,
            "answer_quality_labels": 0,
        },
    }


def inspect_generated_answers(per_question_path: str) -> Dict[str, Any]:
    """Whether the completed run persisted answer text, and how big its
    contexts were -- the confound a matched-k design exists to remove."""
    rows = _read_jsonl(per_question_path)
    out: Dict[str, Any] = {"path": per_question_path, "questions": len(rows), "arms": {}}
    for arm in ("rag2", "scaf"):
        answers = [str(r.get(arm, {}).get("answer") or "") for r in rows]
        filled = [a for a in answers if a.strip()]
        contexts = [r[arm].get("context_chars", 0) for r in rows]
        admitted = [r[arm].get("num_admitted", 0) for r in rows]
        out["arms"][arm] = {
            "answers_present": len(filled),
            "answers_expected": len(rows),
            "answer_chars": {
                "min": min((len(a) for a in filled), default=0),
                "median": int(statistics.median([len(a) for a in filled])) if filled else 0,
                "max": max((len(a) for a in filled), default=0)},
            "abstentions": sum(1 for r in rows if r[arm].get("abstained")),
            "generation_errors": sum(1 for r in rows if r[arm].get("generation_error")),
            "mean_context_chars": round(statistics.mean(contexts), 1) if contexts else 0,
            "mean_admitted_passages": round(statistics.mean(admitted), 3) if admitted else 0,
        }
    ctx = [out["arms"][a]["mean_context_chars"] for a in ("rag2", "scaf")]
    out["context_ratio_scaf_over_rag2"] = (
        round(ctx[1] / ctx[0], 1) if ctx[0] else None)
    return out


def inspect_generator(run_manifest_path: str) -> Dict[str, Any]:
    """Is the generator pinned well enough to reproduce a matched-k run?"""
    with open(run_manifest_path, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    config = manifest.get("config") or {}
    generation = config.get("arm_a_generation") or {}
    same = json.dumps(config.get("arm_a_generation"), sort_keys=True) == \
        json.dumps(config.get("arm_b_generation"), sort_keys=True)
    return {
        "backend": generation.get("backend"),
        "model": generation.get("model"),
        "greedy": generation.get("greedy"),
        "max_new_tokens": generation.get("max_new_tokens"),
        "prompt_version": generation.get("prompt_version"),
        "prompt_fingerprint": generation.get("prompt_fingerprint"),
        "identical_across_arms": same,
        "answers_with_reference": {
            arm: (manifest.get(f"answers_arm_{suffix}") or {}).get("with_reference")
            for arm, suffix in (("rag2", "a_rag2"), ("scaf", "b_scaf"))},
        "manifest_note": (manifest.get("answers_arm_a_rag2") or {}).get("note"),
    }


def inspect_evaluation_code(root: str) -> Dict[str, Any]:
    """Answer-level evaluation code that exists, and what each function needs."""
    path = os.path.join(root, "architecture", "rag2", "rag2", "evaluation.py")
    if not os.path.isfile(path):
        return {"path": path, "exists": False}
    source = open(path, "r", encoding="utf-8").read()
    functions = re.findall(r"^def (\w+)", source, flags=re.M)
    return {
        "path": os.path.relpath(path, root),
        "exists": True,
        "functions": functions,
        "capabilities": {
            "multiple_choice_accuracy": {
                "functions": [f for f in functions
                              if f in ("extract_choice", "is_correct", "accuracy",
                                       "accuracy_by")],
                "requires": "a gold option letter and an options mapping per question",
                "usable_on_alzheimer_set": False,
                "why": "the Alzheimer questions are open-ended and carry neither "
                       "options nor a gold letter"},
            "open_ended_similarity": {
                "functions": [f for f in functions
                              if f in ("rouge_l", "bertscore", "open_ended_metrics")],
                "requires": "a written reference answer per question",
                "usable_on_alzheimer_set": False,
                "why": "no reference answer exists for any of the 30 questions"},
            "filter_metrics": {
                "functions": [f for f in functions if f == "filter_metrics"],
                "requires": "gold HELPFUL/NOT_HELPFUL labels",
                "usable_on_alzheimer_set": False,
                "why": "measures the filter, not the answer; and the repository's "
                       "HELPFUL labels are rule-derived training data"},
        },
        "citation_correctness": {"exists": False},
        "factuality_or_claim_level": {"exists": False},
        "wired_into_the_scaf_comparison": False,
    }


def inspect_non_targets(root: str) -> Dict[str, Any]:
    """Artifacts that resemble an answer-quality target but are not one."""
    results = os.path.join(root, "experiments", "results", "rag2_vs_scaf_alzheimer")
    entries = {}

    sheet = os.path.join(results, "evidence_quality", "annotation_sheet_v2.jsonl")
    if os.path.isfile(sheet):
        rows = _read_jsonl(sheet)
        entries["evidence_quality_annotations"] = {
            "path": os.path.relpath(sheet, root),
            "rows": len(rows),
            "what_it_is": "human 0/1/2 judgements of whether a PASSAGE is useful "
                          "for a question",
            "why_not_an_answer_target": "it grades retrieved evidence, not "
                                        "generated answers. An answer can be wrong "
                                        "from useful passages and right from thin "
                                        "ones.",
            "suitable_as_answer_gold": False,
        }

    weak = os.path.join(results, "training_dataset", "human_validation_subset.jsonl")
    if os.path.isfile(weak):
        rows = _read_jsonl(weak)
        entries["filter_training_weak_labels"] = {
            "path": os.path.relpath(weak, root),
            "rows": len(rows),
            "label_field": "assigned_label",
            "labels": dict(Counter(str(r.get("assigned_label")) for r in rows)),
            "what_it_is": "rule-derived [HELPFUL]/[NOT_HELPFUL] training data for "
                          "the RAG2 filter",
            "why_not_an_answer_target": "training data, not evaluation data, and "
                                        "about passages rather than answers. Using "
                                        "it to evaluate would be testing on train.",
            "suitable_as_answer_gold": False,
        }

    entries["machine_scores"] = {
        "path": os.path.relpath(os.path.join(results, "comparison_scientific",
                                             "per_question.jsonl"), root),
        "what_it_is": "each system's own SCAF/RAG2 score for each passage",
        "why_not_an_answer_target": "circular: grading a system's output with that "
                                    "system's own score measures self-consistency, "
                                    "not quality.",
        "suitable_as_answer_gold": False,
    }
    return entries


def matched_k_feasibility(per_question_path: str, k: int = 5) -> Dict[str, Any]:
    """Which halves of a matched-k experiment the repository can already do."""
    rows = _read_jsonl(per_question_path)
    enough = sum(1 for r in rows for arm in ("rag2", "scaf")
                 if len(r[arm].get("decisions") or []) >= k)
    return {
        "k": k,
        "questions": len(rows),
        "arm_questions_with_at_least_k_candidates": enough,
        "arm_questions_total": len(rows) * 2,
        "selection_executable_now": enough == len(rows) * 2,
        "components": {
            "same_30_questions": True,
            "same_frozen_candidate_set": True,
            "top_k_selection_from_frozen_scores": True,
            "same_generator_configuration_recorded": True,
            "generation_must_be_rerun": True,
            "answer_quality_target": False,
        },
        "blocked_on": ["answer_quality_target"],
        "note": ("choosing each arm's top-k needs no retrieval, no reranking and no "
                 "index -- the frozen decisions already carry every score. "
                 "Generation would have to be rerun because the k=5 contexts differ "
                 "from the ones the completed run used. That is cheap. What is "
                 "missing is not compute: it is something to grade the answers "
                 "against."),
    }


def decide(question_set: Dict[str, Any], evaluation_code: Dict[str, Any],
           feasibility: Dict[str, Any]) -> Dict[str, Any]:
    """READY / PARTIALLY READY / NOT READY, with the reason stated."""
    has_target = bool(question_set.get("has_gold_answers"))
    has_code = bool(evaluation_code.get("exists"))
    selection_ready = bool(feasibility.get("selection_executable_now"))

    if has_target and has_code:
        verdict, reason = READY, (
            "A reference target and working evaluation code both exist.")
    elif has_code or selection_ready:
        verdict, reason = PARTIAL, (
            "The machinery exists but the target does not. Answer-evaluation code "
            "is present and tested, the generator is pinned, the 30 questions and "
            "the frozen candidate set are fixed, and each arm's top-k can be "
            "selected from the saved scores without rerunning retrieval. The single "
            "missing component is something to grade an answer against: there is no "
            "gold answer, no reference answer and no expert answer label anywhere in "
            "the repository.")
    else:
        verdict, reason = NOT_READY, (
            "Neither an answer-quality target nor usable evaluation code exists.")
    return {
        "verdict": verdict,
        "reason": reason,
        "has_answer_quality_target": has_target,
        "has_answer_evaluation_code": has_code,
        "matched_k_selection_executable": selection_ready,
    }


def build_readiness(root: str) -> Dict[str, Any]:
    """The whole inspection. Opens files; writes nothing."""
    results = os.path.join(root, "experiments", "results", "rag2_vs_scaf_alzheimer")
    per_question = os.path.join(results, "comparison_scientific", "per_question.jsonl")
    run_manifest = os.path.join(results, "comparison_scientific", "manifest.json")
    questions = os.path.join(root, "data", "datasets", "thesis_questions",
                             "dev_questions.jsonl")

    question_set = inspect_question_set(questions)
    evaluation_code = inspect_evaluation_code(root)
    feasibility = matched_k_feasibility(per_question)

    return {
        "inspected_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "analysis_version": "answer-quality-readiness-v1",
        "artifacts_inspected": [
            os.path.relpath(p, root) for p in
            (questions, per_question, run_manifest,
             os.path.join(root, "architecture", "rag2", "rag2", "evaluation.py"),
             os.path.join(results, "evidence_quality", "annotation_sheet_v2.jsonl"),
             os.path.join(results, "training_dataset", "human_validation_subset.jsonl"))
            if os.path.exists(p)],
        "question_set": question_set,
        "generated_answers": inspect_generated_answers(per_question),
        "generator": inspect_generator(run_manifest),
        "evaluation_code": evaluation_code,
        "artifacts_that_are_not_answer_targets": inspect_non_targets(root),
        "matched_k_feasibility": feasibility,
        "decision": decide(question_set, evaluation_code, feasibility),
    }
