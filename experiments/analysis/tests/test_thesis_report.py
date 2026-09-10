"""Tests for the reportability gates and the final package.

The failure this file exists to prevent is a non-reportable result quietly
appearing as a finding -- either because a gate passed that should not have, or
because the renderer showed a number without the condition attached to it.
"""

import json
import os
import subprocess
import sys

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from experiments.analysis.render_thesis_final import render_markdown  # noqa: E402
from experiments.analysis.thesis_report import (  # noqa: E402
    ANNOTATION_SHEET_SHA256,
    CORRECTED_ANNOTATION_PASS,
    SCIENTIFIC_FROZEN_DIGEST,
    answer_the_thesis_questions,
    gate,
    reportability,
    sha256,
)

RESULTS = os.path.join(_ROOT, "experiments", "results", "rag2_vs_scaf_alzheimer")
FINAL = os.path.join(RESULTS, "final")
SCRIPT = os.path.join(_ROOT, "experiments", "scripts", "thesis_final_analysis.py")
SHEET = os.path.join(RESULTS, "evidence_quality", "annotation_sheet_v2.jsonl")

_ALL_GOOD = dict(executed=True, correct_corpus=True, correct_candidates=True,
                 fairness_passed=True, intended_generator=True,
                 provenance_complete=True, labels_support_the_claim=True)


# --------------------------------------------------------------------------
# The gate
# --------------------------------------------------------------------------
def test_a_result_meeting_every_condition_is_reportable():
    assert gate("ok", **_ALL_GOOD)["reportable"] is True


@pytest.mark.parametrize("condition", sorted(_ALL_GOOD))
def test_every_single_condition_can_block_on_its_own(condition):
    """No condition may be decorative."""
    result = gate("x", **{**_ALL_GOOD, condition: False})
    assert result["reportable"] is False
    assert result["blocking"], condition


def test_a_known_invalidating_issue_blocks_even_when_all_else_passes():
    result = gate("x", **_ALL_GOOD, known_invalidating_issue="the arms differ upstream")
    assert result["reportable"] is False


def test_the_blocking_reason_is_recorded_not_just_the_verdict():
    result = gate("x", **{**_ALL_GOOD, "executed": False})
    assert "the experiment was actually executed, not inferred" in result["blocking"]


# --------------------------------------------------------------------------
# The suite of gates
# --------------------------------------------------------------------------
def _gates(matched_k_ready=True, matched_k_generated=False):
    ablations = {"reproduction_check": {"all_passed": True},
                 "variance_decomposition": {"terms": {}}}
    return reportability(ablations, {}, matched_k_ready, matched_k_generated)


def test_the_recency_probe_is_never_reportable_while_unexecuted():
    """The proposal's primary claim must not be softened into a finding."""
    gates = {g["result"]: g for g in _gates()["gates"]}
    probe = gates["Filter Recency-Bias Probe (H1-H4), the proposal's primary claim"]
    assert probe["reportable"] is False
    assert probe["conditions"][
        "the experiment was actually executed, not inferred"] is False
    # Implemented is not executed: the gate must not soften as tooling lands.
    assert "Not executed" in probe["known_invalidating_issue"]


def test_the_anchored_pilot_is_never_reportable():
    gates = {g["result"]: g for g in _gates()["gates"]}
    pilot = gates["Anchored evidence-quality pilot"]
    assert pilot["reportable"] is False
    assert "measured the interface" in pilot["known_invalidating_issue"]


def test_the_natural_admission_answer_comparison_is_blocked_by_the_context_confound():
    gates = {g["result"]: g for g in _gates()["gates"]}
    answers = gates["Answer-quality comparison from the natural-admission run"]
    assert answers["reportable"] is False
    assert "688x more context" in answers["known_invalidating_issue"]


def test_matched_k_stays_non_reportable_until_it_is_generated():
    before = {g["result"]: g for g in _gates(matched_k_generated=False)["gates"]}
    after = {g["result"]: g for g in _gates(matched_k_generated=True)["gates"]}
    key = "Matched-k = 5 answer comparison"
    assert before[key]["reportable"] is False
    # Even generated, it cannot support an answer-QUALITY claim.
    assert after[key]["reportable"] is False
    assert after[key]["conditions"][
        "the evaluation labels support the claim being made"] is False


def test_a_broken_reproduction_check_blocks_the_ablations():
    ablations = {"reproduction_check": {"all_passed": False},
                 "variance_decomposition": {"terms": {}}}
    gates = {g["result"]: g for g in reportability(ablations, {}, True, False)["gates"]}
    assert gates["Offline ablations (A4, A5, A7, A11, A12)"]["reportable"] is False


def test_non_reportable_results_are_kept_rather_than_deleted():
    assert "kept, not deleted" in _gates()["policy"]


# --------------------------------------------------------------------------
# The answers
# --------------------------------------------------------------------------
def _payload():
    with open(os.path.join(FINAL, "thesis_final_results.json"), encoding="utf-8") as fh:
        return json.load(fh)


def test_all_thirteen_questions_are_answered():
    payload = _payload()
    assert len(payload["thesis_questions"]) == 13
    for entry in payload["thesis_questions"]:
        assert entry["answer"].strip()
        assert entry["status"].strip()


def test_the_evidence_quality_answer_is_negative_not_hedged():
    answers = {e["question"][:5]: e for e in _payload()["thesis_questions"]}
    entry = answers["5. Di"]
    assert entry["answer"].startswith("No.")
    assert "NOT supported" in entry["status"] or "NOT supported" in entry["answer"]


def test_the_recency_answer_says_unresolved_rather_than_claiming_a_direction():
    answers = {e["question"][:5]: e for e in _payload()["thesis_questions"]}
    entry = answers["7. Wa"]
    assert "Left unresolved" in entry["answer"]
    assert "not evidence of anything" in entry["answer"]


# --------------------------------------------------------------------------
# The renderer
# --------------------------------------------------------------------------
def test_the_rendered_report_shows_non_reportable_results_in_the_same_table():
    text = open(os.path.join(FINAL, "THESIS_FINAL_RESULTS.md"), encoding="utf-8").read()
    for name in _payload()["reportability"]["not_reportable"]:
        assert name in text, name


def test_every_forbidden_claim_is_printed():
    text = open(os.path.join(FINAL, "THESIS_FINAL_RESULTS.md"), encoding="utf-8").read()
    for claim in _payload()["forbidden_claims"]:
        assert claim in text


def test_each_forbidden_claim_carries_its_own_negation():
    """So that no bullet can be lifted out of the list and read as a finding."""
    for claim in _payload()["forbidden_claims"]:
        assert claim.startswith("Do NOT claim"), claim


#: Words that turn a superiority phrase into a prohibition rather than a claim.
_NEGATORS = ("not claimable", "not supported", "does not", "do not", "must never",
             "never", "cannot", "no ", "not ", "forbidden", "is not")


def test_every_superiority_phrase_appears_only_inside_a_prohibition():
    """Banning the substring outright is wrong: the report must be *able* to
    name the claim it is forbidding. What it must never do is assert it."""
    text = open(os.path.join(FINAL, "THESIS_FINAL_RESULTS.md"),
                encoding="utf-8").read().replace("**", "").lower()
    sentences = [s.strip() for s in text.replace("\n", " ").split(". ")]
    for phrase in ("scaf outperforms", "scaf is better", "scaf wins",
                   "scaf improves evidence quality", "proves superiority",
                   "scaf improves answer"):
        for sentence in sentences:
            if phrase in sentence:
                assert any(n in sentence for n in _NEGATORS), \
                    f"{phrase!r} asserted rather than forbidden in: {sentence}"


def test_the_prohibitions_are_actually_present():
    """The negation test above is only meaningful if the phrases appear at all."""
    text = open(os.path.join(FINAL, "THESIS_FINAL_RESULTS.md"),
                encoding="utf-8").read().lower()
    assert "scaf is better" in text.replace("**", "")
    assert "not claimable" in text.replace("**", "")


def test_the_renderer_reads_every_number_from_the_payload():
    """A hand-typed number in the prose could drift from the JSON."""
    payload = _payload()
    text = render_markdown(payload)
    admission = payload["statistics"]["admission"]
    assert f"{admission['rag2_admission_rate']:.4f}" in text
    assert f"{admission['scaf_admission_rate']:.4f}" in text


# --------------------------------------------------------------------------
# Provenance refusals
# --------------------------------------------------------------------------
def _run(args):
    return subprocess.run([sys.executable, SCRIPT, *args], capture_output=True,
                          text=True, cwd=_ROOT, timeout=900)


def test_the_script_refuses_an_annotation_sheet_from_the_anchored_pilot(tmp_path):
    rows = [json.loads(l) for l in open(SHEET, encoding="utf-8") if l.strip()]
    for row in rows:
        row["annotation_pass"] = "pilot-anchored-v1"
        row["ai_suggestion_shown"] = True
    anchored = tmp_path / "anchored.jsonl"
    with open(anchored, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    result = _run(["--sheet", str(anchored), "--out", str(tmp_path / "out"),
                   "--resamples", "50"])
    assert result.returncode == 1
    assert "REFUSING" in result.stdout
    assert not (tmp_path / "out").exists()


def test_the_script_refuses_a_run_that_is_not_the_scientific_one(tmp_path):
    manifest = json.load(open(os.path.join(
        RESULTS, "comparison_scientific", "manifest.json"), encoding="utf-8"))
    manifest["frozen_set_digest"] = "0" * 64
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    result = _run(["--run-manifest", str(path), "--out", str(tmp_path / "out"),
                   "--resamples", "50"])
    assert result.returncode == 1
    assert "does not record the scientific frozen candidate digest" in result.stdout


def test_the_committed_annotation_sheet_still_has_its_recorded_digest():
    """If this fails the labels changed, and every association above is stale."""
    assert sha256(SHEET) == ANNOTATION_SHEET_SHA256


def test_the_committed_package_records_the_scientific_candidate_set():
    identity = _payload()["identity"]
    assert identity["run manifest records"] == SCIENTIFIC_FROZEN_DIGEST
    assert identity["annotation pass"] == CORRECTED_ANNOTATION_PASS
