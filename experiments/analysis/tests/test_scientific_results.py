"""The results analysis: read-only, honest about small cells, and pilot-proof.

The failure that matters most here is not a wrong number -- it is analysing the
wrong annotation pass. The retired pilot reproduced a displayed lexical
suggestion on all 120 rows; running this analysis over it would produce a
strong, entirely artefactual SCAF correlation and put it in a thesis report. So
the refusal is tested first and hardest.
"""

import json
import os
import sys

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from experiments.analysis.evidence_quality import (  # noqa: E402
    load_decisions,
    split_blind_and_key,
    stratified_sample,
)
from experiments.analysis.scientific_results import (  # noqa: E402
    CORRECTED_PASS_LABEL,
    NotTheCorrectedPass,
    admission_behaviour,
    build_results,
    human_evidence_quality,
    retraction_census,
    temporal_feasibility,
)
from experiments.analysis.render_results import render_markdown  # noqa: E402

DIGEST = "316260f04c1720fbc20c1b584ea9f0a093dedbae9fd46476d8a7f45380e86aad"


def _decision(qid, i, rag2_score, scaf_score, rag2_keep, scaf_keep, year=2024):
    common = {"chunk_id": f"{qid}-c{i:02d}", "rerank_rank": i + 1,
              "source_category": "pmc-fulltext",
              "canonical_date": f"{year}-01-01", "authority_tier_label": ""}
    rag2 = {**common, "score": rag2_score, "keep": rag2_keep, "label": ""}
    scaf = {**common, "score": scaf_score, "keep": scaf_keep, "label": "admit",
            "detail": {"sigma_support": scaf_score * 0.8, "gamma_currency": 0.9,
                       "tau_authority": 0.45, "rho_corroboration": 0.0,
                       "scaf_score": scaf_score, "threshold": 0.45, "gate": "",
                       "weights": {"support": 0.5, "currency": 0.3,
                                   "authority": 0.2, "corroboration": 0.0},
                       "support_method": "lexical-coverage-corpus-idf-v2",
                       "corroboration_status": "not_implemented",
                       "currency_detail": {"retracted": "no", "supersession": "unknown",
                                           "state": "current", "date_precision": "day"},
                       "authority_detail": {"basis": "source_category"},
                       "support_detail": {"coverage": 0.5}}}
    return rag2, scaf


@pytest.fixture()
def study(tmp_path):
    run_dir = tmp_path / "comparison_scientific"
    run_dir.mkdir()
    per_question = run_dir / "per_question.jsonl"
    with open(per_question, "w", encoding="utf-8") as fh:
        for q in range(10):
            qid = f"alz-{q:03d}"
            rag2s, scafs = [], []
            for i in range(20):
                r, s = _decision(qid, i, 0.20 + 0.01 * i, 0.30 + 0.03 * i,
                                 rag2_keep=(q == 0 and i == 19),
                                 scaf_keep=(0.30 + 0.03 * i) >= 0.45)
                rag2s.append(r); scafs.append(s)
            fh.write(json.dumps({
                "qid": qid, "question": f"Question {q}?", "time_sensitive": True,
                "num_candidates": 20,
                "rag2": {"decisions": rag2s, "admitted_chunk_ids":
                         [d["chunk_id"] for d in rag2s if d["keep"]]},
                "scaf": {"decisions": scafs, "admitted_chunk_ids":
                         [d["chunk_id"] for d in scafs if d["keep"]]},
            }) + "\n")
    run_manifest = run_dir / "manifest.json"
    run_manifest.write_text(json.dumps({
        "frozen_set_digest": DIGEST, "reportable": True,
        "fairness": {"all_passed": True}, "scientific": {"all_passed": True}}),
        encoding="utf-8")

    decisions = load_decisions(str(per_question))
    enriched = [{**d, "candidate_text": f"passage {d['chunk_id']}", "title": "T",
                 "pmid": "", "pmcid": ""} for d in decisions]
    sample, provenance = stratified_sample(enriched, size=40, seed=42)
    blind, _ = split_blind_and_key(sample)
    for i, row in enumerate(blind):
        row["human_label"] = [2, 1, 2, 0, 1][i % 5]
        row["annotation_pass"] = CORRECTED_PASS_LABEL
        row["ai_suggestion_shown"] = False
        row["ai_suggestion_generated"] = False
        row["annotated_utc"] = "2026-09-10T12:00:00Z"

    sheet = tmp_path / "annotation_sheet_v2.jsonl"
    with open(sheet, "w", encoding="utf-8") as fh:
        for row in blind:
            fh.write(json.dumps(row, sort_keys=True) + "\n")
    manifest = tmp_path / "sample_manifest.json"
    manifest.write_text(json.dumps({**provenance, "frozen_set_digest": DIGEST,
                                    "retrieval_is_medcpt": True}), encoding="utf-8")
    return {"dir": tmp_path, "per_question": str(per_question), "sheet": str(sheet),
            "manifest": str(manifest), "run_manifest": str(run_manifest),
            "decisions": decisions, "blind": blind}


def _build(study):
    return build_results(study["per_question"], study["sheet"], study["manifest"],
                         study["run_manifest"])


# --------------------------------------------------------------------------
# The refusal that protects the thesis
# --------------------------------------------------------------------------
def test_refuses_the_anchored_pilot(study, tmp_path):
    """Analysing the pilot would produce a strong, artefactual correlation."""
    rows = [dict(r) for r in study["blind"]]
    for row in rows:
        row.pop("annotation_pass")
        row["ai_suggestion_shown"] = True
    pilot = tmp_path / "annotation_sheet.jsonl"
    with open(pilot, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True) + "\n")
    with pytest.raises(NotTheCorrectedPass):
        build_results(study["per_question"], str(pilot), study["manifest"],
                      study["run_manifest"])


def test_refuses_a_mixed_sheet(study, tmp_path):
    rows = [dict(r) for r in study["blind"]]
    rows[0]["annotation_pass"] = "something-else"
    mixed = tmp_path / "mixed.jsonl"
    with open(mixed, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True) + "\n")
    with pytest.raises(NotTheCorrectedPass):
        build_results(study["per_question"], str(mixed), study["manifest"],
                      study["run_manifest"])


def test_accepts_the_corrected_pass(study):
    assert _build(study)["provenance"]["annotation_pass"] == [CORRECTED_PASS_LABEL]


# --------------------------------------------------------------------------
# Read-only
# --------------------------------------------------------------------------
def test_building_results_writes_nothing(study):
    before = {p: os.path.getmtime(os.path.join(study["dir"], p))
              for p in os.listdir(study["dir"])}
    _build(study)
    after = {p: os.path.getmtime(os.path.join(study["dir"], p))
             for p in os.listdir(study["dir"])}
    assert before == after


def test_results_are_deterministic(study):
    a, b = _build(study), _build(study)
    for block in ("admission_behaviour", "human_evidence_quality", "scaf_components",
                  "matched_context", "temporal_feasibility", "retraction_census",
                  "examples"):
        assert a[block] == b[block], block


# --------------------------------------------------------------------------
# The numbers
# --------------------------------------------------------------------------
def test_admission_counts_add_up(study):
    c = admission_behaviour(study["decisions"])["candidate_level"]
    assert c["both_admit"] + c["rag2_only"] + c["scaf_only"] + c["neither_admits"] \
        == c["decisions"]
    assert c["both_admit"] + c["rag2_only"] == c["rag2_admitted"]
    assert c["both_admit"] + c["scaf_only"] == c["scaf_admitted"]


def test_question_level_is_reported_separately_from_candidate_level(study):
    a = admission_behaviour(study["decisions"])
    assert a["question_level"]["questions"] == 10
    assert a["question_level"]["questions_with_zero_rag2_evidence"] == 9


def test_every_annotation_joins_to_a_decision(study):
    r = _build(study)
    assert r["join"]["joined"] == r["provenance"]["counts"]["human_annotations"]
    assert r["join"]["unmatched"] == []


def test_group_means_cover_every_decision_cell(study):
    g = _build(study)["human_evidence_quality"]["by_group"]
    assert g["scaf_admitted"]["n"] + g["scaf_rejected"]["n"] == g["all_annotated"]["n"]
    assert g["rag2_admitted"]["n"] + g["rag2_rejected"]["n"] == g["all_annotated"]["n"]
    assert (g["both_admit"]["n"] + g["scaf_only"]["n"] + g["rag2_only"]["n"]
            + g["neither_admits"]["n"]) == g["all_annotated"]["n"]


def test_empty_cells_report_n_zero_rather_than_a_fake_mean(study):
    """rag2_only is empty in the real study; it must not invent a mean."""
    g = _build(study)["human_evidence_quality"]["by_group"]
    for cell in g.values():
        if cell["n"] == 0:
            assert "mean" not in cell


def test_irrelevant_passage_census(study):
    hq = _build(study)["human_evidence_quality"]
    zeros = sum(1 for r in study["blind"] if r["human_label"] == 0)
    assert hq["irrelevant_passages"]["n"] == zeros


def test_temporal_feasibility_reports_the_year_span(study):
    t = temporal_feasibility(study["decisions"])
    assert t["earliest_year"] == "2024"
    assert t["candidates_before_2020"] == 0


def test_retraction_census_counts_every_decision(study):
    raw = [json.loads(l) for l in open(study["per_question"]) if l.strip()]
    details = [d["detail"] for r in raw for d in r["scaf"]["decisions"]]
    census = retraction_census(details)
    assert sum(census["retracted"].values()) == len(details)
    assert census["retracted"] == {"no": 200}


def test_examples_are_chosen_by_a_stated_rule(study):
    ex = _build(study)["examples"]
    assert "deterministic" in ex["selection_rule"]
    for rows in (ex["scaf_admitted_but_human_says_irrelevant"], ex["rag2_admitted"]):
        for row in rows:
            assert row["passage_excerpt"]
            assert row["human_label"] in (0, 1, 2)


def test_guards_are_present_on_every_interpretable_block(study):
    r = _build(study)
    for block in ("admission_behaviour", "human_evidence_quality", "scaf_components",
                  "matched_context"):
        assert r[block]["guard"].strip()


# --------------------------------------------------------------------------
# The report
# --------------------------------------------------------------------------
def test_markdown_renders_without_placeholders(study):
    md = render_markdown(_build(study))
    assert "None" not in md
    assert "{}" not in md
    for heading in ("Executive summary", "Data and provenance", "Admission behaviour",
                    "Human evidence-quality results", "SCAF component behaviour",
                    "Matched-k", "Retracted and superseded", "Temporal",
                    "Claims supported", "Claims that are NOT established",
                    "Limitations", "Recommended next step"):
        assert heading in md, heading


def test_markdown_states_it_is_derived_not_evidence(study):
    md = render_markdown(_build(study))
    assert "generated" in md.lower()
    assert "derived" in md.lower()


def test_markdown_carries_the_provenance_identifiers(study):
    r = _build(study)
    md = render_markdown(r)
    assert r["provenance"]["annotation_sheet_sha256"] in md
    assert r["provenance"]["label_fingerprint"] in md
    assert DIGEST in md


def test_markdown_lists_the_overclaims_as_not_established(study):
    """The banned claims must appear -- but only in the table that refutes them.

    A substring ban would be the wrong test: "SCAF is superior to RAG2" belongs
    in the report, as a row saying it is not established, and "superior" also
    turns up inside quoted passage text. What matters is that every overclaim is
    named and marked.
    """
    md = render_markdown(_build(study))
    section = md.split("## 15. Claims that are NOT established")[1].split("## 16.")[0]
    for claim in ("SCAF improves evidence quality", "SCAF is superior to RAG",
                  "SCAF improves answer accuracy", "SCAF reduces outdated evidence",
                  "RAG² has recency bias", "SCAF improves citation quality",
                  "SCAF reduces unsupported claims"):
        row = next((l for l in section.splitlines() if claim in l), None)
        assert row is not None, f"{claim} is not listed"
        assert ("Not established" in row or "Requires additional experiment" in row), row


def test_markdown_never_asserts_superiority_outside_the_refutation(study):
    """Outside the refuted-claims table, no sentence may assert superiority."""
    md = render_markdown(_build(study))
    body = md.split("## 15. Claims that are NOT established")[0]
    for forbidden in ("SCAF outperforms", "proves that SCAF", "validates SCAF",
                      "SCAF is better than"):
        assert forbidden not in body, forbidden
