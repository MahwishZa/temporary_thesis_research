"""The annotation interface: what it must show, what it must never show.

Two kinds of guarantee are pinned here, and they fail in different ways.

*Correctness* failures are loud -- a label is lost, a resume starts in the wrong
place. Tests for those are ordinary.

*Blinding* failures are silent. If a SCAF score reaches the page, every label
collected afterwards is contaminated and nothing downstream can detect it: the
numbers still compute, the analysis still runs, and the conclusion is wrong. So
the blinding tests here render the real HTML and search it for the forbidden
values, rather than checking that some helper was called.
"""

import importlib.util
import json
import os
import re
import sys
import threading
import urllib.error
import urllib.parse
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from experiments.analysis.evidence_quality import (  # noqa: E402
    BLIND_FIELDS,
    check_annotations,
    split_blind_and_key,
)
from experiments.analysis.annotation_ui import (  # noqa: E402
    FORBIDDEN_IN_UI,
    LABEL_TEXT,
    KeyFileAccess,
    content_terms,
    is_labelled,
    load_sheet,
    next_unlabelled,
    plain_summary,
    progress,
    record_label,
    save_sheet,
    suggest_label,
    suggestion_for,
    view_model,
)


def _load_script(name):
    path = os.path.join(_ROOT, "experiments", "scripts", f"{name}.py")
    spec = importlib.util.spec_from_file_location(f"_script_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


annotate = _load_script("annotate")


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------
def _row(i, question="What is known about amyloid clearance in Alzheimer disease?",
         text="Amyloid clearance in Alzheimer disease declines with age. "
              "Impaired clearance raises plaque burden."):
    return {
        "annotation_id": f"a{i:03d}",
        "qid": f"alz-{i:03d}",
        "question": question,
        "chunk_id": f"chunk-{i}",
        "title": f"Paper {i}",
        "candidate_text": text,
        "canonical_date": "2023-05-01",
        "source_category": "pmc-fulltext",
        "human_label": "",
        "human_notes": "",
    }


@pytest.fixture()
def sheet(tmp_path):
    path = tmp_path / "annotation_sheet.jsonl"
    rows = [_row(i) for i in range(5)]
    save_sheet(str(path), rows)
    return str(path)


class _Server:
    """The real handler over a real socket, on an ephemeral port."""

    def __init__(self, app):
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), annotate.Handler)
        self.httpd.app = app
        self.app = app
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()

    def get(self, path):
        try:
            with urllib.request.urlopen(self.base + path) as response:
                return response.status, response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            return exc.code, exc.read().decode("utf-8")

    def post(self, path, fields, follow=True):
        data = urllib.parse.urlencode(fields).encode("utf-8")
        opener = urllib.request.build_opener() if follow else urllib.request.build_opener(
            _NoRedirect)
        try:
            with opener.open(urllib.request.Request(self.base + path, data=data)) as r:
                return r.status, r.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            return exc.code, exc.read().decode("utf-8")

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()
        self.thread.join(timeout=5)


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


def _visible(page: str) -> str:
    """The document minus its stylesheet.

    The CSS is a fixed template, not data, and searching it for leaked values
    produces false alarms: ``overflow`` contains "low", ``#3b6fd4`` contains
    "d4". Only what came from a row can leak, so only that is searched.
    """
    return re.sub(r"<style>.*?</style>", "", page, flags=re.S)


@pytest.fixture()
def server(sheet):
    srv = _Server(annotate.AnnotationApp(sheet))
    yield srv
    srv.close()


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
def test_loads_the_annotation_sheet(sheet):
    rows = load_sheet(sheet)
    assert len(rows) == 5
    assert rows[0]["annotation_id"] == "a000"


def test_blank_lines_do_not_become_rows(tmp_path):
    path = tmp_path / "annotation_sheet.jsonl"
    path.write_text(json.dumps(_row(0)) + "\n\n" + json.dumps(_row(1)) + "\n",
                    encoding="utf-8")
    assert len(load_sheet(str(path))) == 2


def test_refuses_to_open_the_machine_score_key(tmp_path):
    """The de-blinding file must be unreachable from this process, by name."""
    key = tmp_path / "annotation_key.jsonl"
    key.write_text(json.dumps({"annotation_id": "a000", "scaf_score": 0.91}) + "\n",
                   encoding="utf-8")
    with pytest.raises(KeyFileAccess):
        load_sheet(str(key))


def test_key_refusal_is_not_case_sensitive(tmp_path):
    key = tmp_path / "ANNOTATION_KEY.JSONL"
    key.write_text("{}\n", encoding="utf-8")
    with pytest.raises(KeyFileAccess):
        load_sheet(str(key))


def test_cli_refuses_a_key_path_and_does_not_start(tmp_path, capsys):
    key = tmp_path / "annotation_key.jsonl"
    key.write_text(json.dumps({"annotation_id": "a000", "scaf_score": 0.91}) + "\n",
                   encoding="utf-8")
    assert annotate.main(["--sheet", str(key), "--no-browser"]) == 2
    assert "refusing to open" in capsys.readouterr().out


def test_cli_reports_a_missing_sheet_instead_of_starting(tmp_path, capsys):
    assert annotate.main(["--sheet", str(tmp_path / "nope.jsonl")]) == 2
    assert "not found" in capsys.readouterr().out


# --------------------------------------------------------------------------
# The suggestion
# --------------------------------------------------------------------------
def test_suggestion_is_generated_with_a_label_and_a_reason():
    s = suggest_label("What is known about amyloid clearance?",
                      "Amyloid clearance is impaired in Alzheimer disease.")
    assert s["ai_suggested_label"] in LABEL_TEXT
    assert s["ai_explanation"].strip()
    assert s["ai_rule_version"] == "lexical-overlap-v1"


def test_suggestion_is_high_when_the_words_match_and_low_when_they_do_not():
    question = "What is known about amyloid clearance in Alzheimer disease?"
    on_topic = suggest_label(question, "Amyloid clearance in Alzheimer disease is slow.")
    off_topic = suggest_label(question, "Tomato cultivation requires warm soil.")
    assert on_topic["ai_suggested_label"] == 2
    assert off_topic["ai_suggested_label"] == 0
    assert on_topic["ai_coverage"] > off_topic["ai_coverage"]


def test_suggestion_is_deterministic():
    args = ("What is known about tau?", "Tau tangles accumulate in the cortex.")
    assert suggest_label(*args) == suggest_label(*args)


def test_suggestion_explains_which_words_were_missing():
    s = suggest_label("What is known about amyloid clearance in Alzheimer disease?",
                      "Amyloid deposits were measured in the cortex.")
    assert s["ai_missing_terms"]
    assert any(term in s["ai_explanation"] for term in s["ai_missing_terms"][:4])


def test_suggestion_survives_an_empty_question_without_raising():
    s = suggest_label("", "Some passage text.")
    assert s["ai_suggested_label"] == 0
    assert s["ai_coverage"] == 0.0


def test_content_terms_drop_question_scaffolding():
    assert content_terms("What is known about tau?") == {"known", "tau"}


def test_plain_summary_quotes_the_passage_rather_than_inventing_text():
    passage = "First sentence here. Second sentence here. Third sentence here."
    summary = plain_summary(passage)
    assert summary == "First sentence here. Second sentence here."
    assert summary in passage


# --------------------------------------------------------------------------
# Recording a label
# --------------------------------------------------------------------------
@pytest.mark.parametrize("label", [0, 1, 2, "0", "1", "2", " 2 "])
def test_accepts_every_valid_label(sheet, label):
    rows = load_sheet(sheet)
    updated = record_label(rows, "a000", label)
    assert updated["human_label"] == int(str(label).strip())


@pytest.mark.parametrize("label", [3, -1, 99, "", "yes", "two", None, "1.5"])
def test_rejects_invalid_labels_and_leaves_the_row_alone(sheet, label):
    rows = load_sheet(sheet)
    with pytest.raises(ValueError):
        record_label(rows, "a000", label)
    assert rows[0]["human_label"] == ""


def test_unknown_annotation_id_raises(sheet):
    with pytest.raises(KeyError):
        record_label(load_sheet(sheet), "nope", 1)


def test_human_label_is_stored_separately_from_the_ai_suggestion(sheet):
    """The distinction the whole experiment rests on."""
    rows = load_sheet(sheet)
    suggestion = suggestion_for(rows[0], shown=True)
    assert suggestion["ai_suggested_label"] == 2          # the rule says 2 here
    record_label(rows, "a000", 0, suggestion=suggestion)  # the human says 0

    assert rows[0]["human_label"] == 0
    assert rows[0]["ai_suggested_label"] == 2
    assert rows[0]["ai_suggestion_shown"] is True
    assert rows[0]["ai_explanation"]


def test_a_suggestion_alone_never_counts_as_a_label(sheet):
    """Storing the suggestion must not make a row look answered."""
    rows = load_sheet(sheet)
    rows[0].update(suggestion_for(rows[0], shown=True))
    assert not is_labelled(rows[0])
    assert progress(rows)["completed"] == 0


def test_hidden_suggestions_are_recorded_as_not_shown(sheet):
    """The anchoring baseline needs to be distinguishable afterwards."""
    rows = load_sheet(sheet)
    record_label(rows, "a000", 1, suggestion=suggestion_for(rows[0], shown=False))
    assert rows[0]["ai_suggestion_shown"] is False
    assert "ai_suggested_label" in rows[0]


def test_notes_are_saved_with_the_label(sheet):
    rows = load_sheet(sheet)
    record_label(rows, "a000", 1, notes="unsure, mentions tau not amyloid")
    assert rows[0]["human_notes"] == "unsure, mentions tau not amyloid"


def test_a_label_can_be_changed(sheet):
    rows = load_sheet(sheet)
    record_label(rows, "a000", 0)
    record_label(rows, "a000", 2)
    assert rows[0]["human_label"] == 2


# --------------------------------------------------------------------------
# Progress and resuming
# --------------------------------------------------------------------------
def test_progress_counts_only_human_answers(sheet):
    rows = load_sheet(sheet)
    record_label(rows, "a000", 1)
    record_label(rows, "a001", 2)
    assert progress(rows) == {"total": 5, "completed": 2, "remaining": 3}


def test_next_unlabelled_skips_answered_rows(sheet):
    rows = load_sheet(sheet)
    record_label(rows, "a000", 1)
    assert next_unlabelled(rows)["annotation_id"] == "a001"


def test_next_unlabelled_continues_after_the_row_just_answered(sheet):
    rows = load_sheet(sheet)
    record_label(rows, "a002", 1)
    assert next_unlabelled(rows, after="a002")["annotation_id"] == "a003"


def test_next_unlabelled_wraps_back_to_an_earlier_gap(sheet):
    rows = load_sheet(sheet)
    for aid in ("a002", "a003", "a004"):
        record_label(rows, aid, 1)
    assert next_unlabelled(rows, after="a004")["annotation_id"] == "a000"


def test_next_unlabelled_is_none_when_everything_is_answered(sheet):
    rows = load_sheet(sheet)
    for row in rows:
        record_label(rows, row["annotation_id"], 1)
    assert next_unlabelled(rows) is None


def test_reopening_the_file_preserves_existing_answers(sheet):
    """Resuming: quit halfway, come back, nothing is lost and nothing repeats."""
    first = annotate.AnnotationApp(sheet)
    first.submit("a000", "2", "")
    first.submit("a001", "0", "keeps this note")

    resumed = annotate.AnnotationApp(sheet)
    assert resumed.progress() == {"total": 5, "completed": 2, "remaining": 3}
    assert resumed.next_row()["annotation_id"] == "a002"

    stored = {r["annotation_id"]: r for r in load_sheet(sheet)}
    assert stored["a000"]["human_label"] == 2
    assert stored["a001"]["human_notes"] == "keeps this note"


def test_saving_preserves_every_field_the_export_wrote(sheet):
    before = load_sheet(sheet)[0]
    app = annotate.AnnotationApp(sheet)
    app.submit("a000", "1", "")
    after = load_sheet(sheet)[0]
    for field in BLIND_FIELDS:
        if field not in ("human_label", "human_notes"):
            assert after.get(field) == before.get(field), field


def test_save_is_atomic_and_leaves_no_partial_file(tmp_path):
    path = tmp_path / "annotation_sheet.jsonl"
    rows = [_row(i) for i in range(3)]
    save_sheet(str(path), rows)
    save_sheet(str(path), rows)
    assert len(load_sheet(str(path))) == 3
    assert not [p for p in os.listdir(tmp_path) if p.endswith(".tmp")]


# --------------------------------------------------------------------------
# Blinding
# --------------------------------------------------------------------------
def _contaminated_row():
    """A row carrying every machine field, as if the export had leaked them."""
    row = _row(0)
    row.update({"scaf_score": 0.9137, "rag2_score": 0.2481, "scaf_admitted": True,
                "rag2_admitted": False, "scaf_bin": "high", "rag2_bin": "low",
                "sigma_support": 0.71, "gamma_currency": 0.88, "tau_authority": 0.45,
                "rho_corroboration": 0.0, "rerank_rank": 3})
    return row


def test_view_model_drops_every_machine_field():
    view = view_model(_contaminated_row())
    for field in FORBIDDEN_IN_UI:
        assert field not in view, field


def test_view_model_omits_the_scaf_currency_and_authority_inputs():
    """Date and source are what SCAF's gamma and tau read; the annotator is told
    to judge relevance only, so the page must not put them in front of them."""
    view = view_model(_contaminated_row())
    assert "canonical_date" not in view
    assert "source_category" not in view


def test_rendered_page_contains_no_machine_value():
    row = _contaminated_row()
    page = _visible(annotate.render_annotate(
        view_model(row), {"total": 5, "completed": 0, "remaining": 5}, "tok").decode())
    for field in FORBIDDEN_IN_UI:
        assert field not in page, field
    for value in ("0.9137", "0.2481", "0.71", "0.88", "0.45", "high", "low"):
        assert value not in page, value
    assert "2023-05-01" not in page
    assert "pmc-fulltext" not in page


def test_served_page_contains_no_machine_value(tmp_path):
    path = tmp_path / "annotation_sheet.jsonl"
    save_sheet(str(path), [_contaminated_row()])
    srv = _Server(annotate.AnnotationApp(str(path)))
    try:
        status, page = srv.get("/annotate")
    finally:
        srv.close()
    assert status == 200
    page = _visible(page)
    for value in ("0.9137", "0.2481", "scaf_score", "rag2_score", "scaf_admitted",
                  "2023-05-01", "pmc-fulltext"):
        assert value not in page, value


def test_the_blind_export_itself_carries_no_machine_field():
    """Belt and braces: the sheet the interface reads is already blind."""
    sample = [{**_contaminated_row(), "scaf_score": 0.9137}]
    blind, key = split_blind_and_key(sample)
    for field in FORBIDDEN_IN_UI:
        assert field not in blind[0], field
    assert "scaf_score" in key[0]


def test_page_escapes_html_in_the_data():
    row = _row(0, question="<script>alert('x')</script>",
               text="passage with <b>markup</b> & an ampersand")
    page = annotate.render_annotate(view_model(row), {"total": 1, "completed": 0,
                                                      "remaining": 1}, "tok").decode()
    assert "<script>alert" not in page
    assert "&lt;script&gt;" in page
    assert "<b>markup</b>" not in page


# --------------------------------------------------------------------------
# The page the annotator actually sees
# --------------------------------------------------------------------------
def test_page_shows_question_passage_explanation_suggestion_and_buttons(server):
    _, page = server.get("/annotate")
    assert "amyloid clearance" in page
    assert "What does this mean?" in page
    assert "Computer suggestion" in page
    assert "Suggested label:" in page
    assert "Why:" in page
    for value, text in LABEL_TEXT.items():
        assert f"{value} &mdash; {text}" in page
    assert "0 / 5 completed" in page


def test_no_suggestions_mode_hides_the_suggestion(sheet):
    srv = _Server(annotate.AnnotationApp(sheet, show_suggestion=False))
    try:
        _, page = srv.get("/annotate")
    finally:
        srv.close()
    assert "Computer suggestion" not in page
    assert "Suggested label:" not in page
    assert "amyloid clearance" in page          # the passage is still there
    for value, text in LABEL_TEXT.items():      # and the buttons still are
        assert f"{value} &mdash; {text}" in page


def test_pressing_a_button_records_that_label_and_moves_on(server):
    server.post("/label", {"token": server.app.token, "annotation_id": "a000",
                           "label": "1", "notes": ""})
    stored = {r["annotation_id"]: r for r in load_sheet(server.app.sheet)}
    assert stored["a000"]["human_label"] == 1
    assert server.app.next_row()["annotation_id"] == "a001"


def test_the_page_offers_no_way_to_submit_a_label_other_than_0_1_2(server):
    _, page = server.get("/annotate")
    assert page.count('name="label"') == 3
    for value in (0, 1, 2):
        assert f'name="label" value="{value}"' in page


def test_a_forged_label_value_is_rejected_and_nothing_is_written(server):
    status, page = server.post("/label", {"token": server.app.token,
                                          "annotation_id": "a000", "label": "7"})
    assert status == 400
    assert "not a valid answer" in page
    assert load_sheet(server.app.sheet)[0]["human_label"] == ""


def test_a_stale_form_token_is_rejected(server):
    status, _ = server.post("/label", {"token": "wrong", "annotation_id": "a000",
                                       "label": "1"})
    assert status == 403
    assert load_sheet(server.app.sheet)[0]["human_label"] == ""


def test_finishing_every_row_reaches_the_done_page(server):
    for i in range(5):
        server.post("/label", {"token": server.app.token,
                               "annotation_id": f"a{i:03d}", "label": "2"})
    _, page = server.get("/annotate")
    assert "All done" in page
    assert "5 / 5 completed" in page


def test_an_answered_row_can_be_revisited_and_overridden(server):
    server.post("/label", {"token": server.app.token, "annotation_id": "a000",
                           "label": "0"})
    _, page = server.get("/annotate?id=a000")
    assert "You answered" in page
    assert "will replace that answer" in page

    server.post("/label", {"token": server.app.token, "annotation_id": "a000",
                           "label": "2"})
    assert load_sheet(server.app.sheet)[0]["human_label"] == 2


def test_root_redirects_to_the_interface(server):
    status, page = server.get("/")
    assert status == 200
    assert "Does this passage help answer this question?" in page


def test_unknown_page_is_a_404_not_a_traceback(server):
    status, page = server.get("/robots.txt")
    assert status == 404
    assert "no such page" in page


# --------------------------------------------------------------------------
# Compatibility with the rest of the workflow
# --------------------------------------------------------------------------
def test_a_sheet_finished_here_passes_the_check_command(sheet):
    app = annotate.AnnotationApp(sheet)
    for i, label in enumerate([0, 1, 2, 2, 1]):
        app.submit(f"a{i:03d}", str(label), "")

    report = check_annotations(load_sheet(sheet))
    assert report["total_rows"] == 5
    assert report["completed"] == 5
    assert report["missing"] == 0
    assert report["invalid"] == 0
    assert report["ready_for_analysis"] is True
    assert report["label_distribution"] == {0: 1, 1: 2, 2: 2}


def test_a_half_finished_sheet_is_reported_as_not_ready(sheet):
    app = annotate.AnnotationApp(sheet)
    app.submit("a000", "1", "")
    report = check_annotations(load_sheet(sheet))
    assert report["completed"] == 1
    assert report["missing"] == 4
    assert report["ready_for_analysis"] is False


def test_the_extra_ai_fields_do_not_confuse_the_check_command(sheet):
    app = annotate.AnnotationApp(sheet)
    for i in range(5):
        app.submit(f"a{i:03d}", "1", "")
    rows = load_sheet(sheet)
    assert all("ai_suggested_label" in r for r in rows)
    assert check_annotations(rows)["ready_for_analysis"] is True
