#!/usr/bin/env python3
"""Label the evidence sample, one passage at a time, in a browser.

    python experiments/scripts/annotate.py

Opens a page on your own computer. Nothing is uploaded; nothing leaves the
machine. Read one passage, read the computer's suggestion, then press the button
*you* think is right. You can stop whenever you like and run the same command
again -- it picks up at the first passage you have not answered.

    --no-suggestions   hide the computer's suggestion (see below)
    --sheet PATH       a different annotation file
    --port N           a different port (default 8765)
    --no-browser       do not open a browser window automatically

Why the suggestion is not the answer
------------------------------------
The suggestion counts how many words from the question appear in the passage.
That is all it does. It cannot tell that "memory decline" and "cognitive
impairment" mean nearly the same thing, so it will be wrong whenever a passage
answers the question in different words. Those are the rows where your judgement
is the only thing of any value.

There is a second reason, and it matters for the thesis rather than for the
annotator. People agree with a displayed recommendation more often than they
would unprompted. The suggestion is lexical, and the SCAF support term this
annotation exists to test is *also* lexical -- so simply pressing whatever the
computer suggests would partly manufacture the agreement being measured instead
of measuring it. Every row therefore records the suggestion, and whether it was
on screen, next to the human answer, so that the agreement rate stays checkable
afterwards. ``--no-suggestions`` annotates without them, which is what an
anchoring baseline needs.

What this program never sees
----------------------------
The SCAF score, the RAG2 score, which system admitted the passage, and the
machine's ranking are all in a separate key file. This program refuses to open
any file whose name contains "key", and it builds each page from a fixed list of
allowed fields rather than by hiding forbidden ones, so a field added upstream
later cannot leak into the page by accident.
"""

from __future__ import annotations

import argparse
import html
import os
import secrets
import sys
import threading
import urllib.parse
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Optional

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from experiments.analysis.annotation_ui import (  # noqa: E402
    LABEL_HELP,
    LABEL_TEXT,
    KeyFileAccess,
    is_labelled,
    load_sheet,
    next_unlabelled,
    progress,
    record_label,
    save_sheet,
    suggestion_for,
    view_model,
)

DEFAULT_SHEET = os.path.join(_ROOT, "experiments", "results", "rag2_vs_scaf_alzheimer",
                             "evidence_quality", "annotation_sheet.jsonl")

# --------------------------------------------------------------------------
# Rendering. No CDN, no external font, no JavaScript: the page must work with
# the network cable unplugged, because that is the point of a local tool.
# --------------------------------------------------------------------------
CSS = """
*{box-sizing:border-box}
body{margin:0;background:#f4f5f7;color:#1a1c1f;
     font:17px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Arial,sans-serif}
.wrap{max-width:820px;margin:0 auto;padding:24px 16px 64px}
h1{font-size:20px;margin:0 0 4px}
.sub{color:#5b6270;font-size:14px;margin:0 0 24px}
.card{background:#fff;border:1px solid #dfe2e8;border-radius:10px;
      padding:20px;margin:0 0 18px}
.card h2{font-size:13px;letter-spacing:.08em;text-transform:uppercase;
         color:#5b6270;margin:0 0 10px;font-weight:600}
.question{font-size:21px;font-weight:600;line-height:1.45}
.passage{font-size:17px;white-space:pre-wrap}
.title{color:#5b6270;font-size:14px;margin:0 0 10px;font-style:italic}
.plain{background:#f0f6ff;border-left:4px solid #7aa7e8}
.sugg{background:#fffaf0;border-left:4px solid #e8b44a}
.sugg .big{font-size:19px;font-weight:600;margin:0 0 8px}
.note{color:#5b6270;font-size:14px;margin:10px 0 0}
.choices{display:flex;flex-direction:column;gap:12px;margin:14px 0 0}
button.choice{display:block;width:100%;text-align:left;cursor:pointer;
       padding:16px 18px;font:inherit;border-radius:10px;
       border:2px solid #c9cedb;background:#fff}
button.choice:hover{border-color:#3b6fd4;background:#f5f8ff}
button.choice b{display:block;font-size:19px;margin:0 0 3px}
button.choice span{color:#4b515e;font-size:15px}
textarea{width:100%;font:inherit;padding:10px;border:1px solid #c9cedb;
         border-radius:8px;resize:vertical}
.bar{height:10px;background:#e2e5ec;border-radius:6px;overflow:hidden;margin:8px 0}
.bar i{display:block;height:100%;background:#3b6fd4}
.msg{background:#eaf6ec;border:1px solid #b6ddbe;border-radius:8px;
     padding:10px 14px;margin:0 0 18px;font-size:15px}
.msg.warn{background:#fdecea;border-color:#f0b8b2}
.links{font-size:14px;color:#5b6270}
.links a{color:#3b6fd4}
.done{font-size:19px}
"""


def _page(title: str, body: str) -> bytes:
    return (f"<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
            f"<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
            f"<title>{html.escape(title)}</title><style>{CSS}</style></head>"
            f"<body><div class=\"wrap\">{body}</div></body></html>").encode("utf-8")


def _progress_card(prog: Dict[str, int]) -> str:
    total = max(prog["total"], 1)
    pct = 100.0 * prog["completed"] / total
    return (f"<div class=\"card\"><h2>Progress</h2>"
            f"<div class=\"bar\"><i style=\"width:{pct:.1f}%\"></i></div>"
            f"<div>{prog['completed']} / {prog['total']} completed"
            f" &nbsp;&middot;&nbsp; {prog['remaining']} left</div>"
            f"<p class=\"note\">You can close this window at any time. Run the same "
            f"command again to carry on from here.</p></div>")


def _choice_buttons() -> str:
    out = []
    for value in (0, 1, 2):
        out.append(
            f"<button class=\"choice\" type=\"submit\" name=\"label\" value=\"{value}\">"
            f"<b>{value} &mdash; {html.escape(LABEL_TEXT[value])}</b>"
            f"<span>{html.escape(LABEL_HELP[value])}</span></button>")
    return f"<div class=\"choices\">{''.join(out)}</div>"


def render_annotate(view: Dict[str, Any], prog: Dict[str, int], token: str,
                    message: str = "", warn: bool = False) -> bytes:
    """One passage. Every value that reaches the page is escaped."""
    banner = (f"<div class=\"msg{' warn' if warn else ''}\">{html.escape(message)}</div>"
              if message else "")

    title = view.get("title") or ""
    title_html = f"<p class=\"title\">From: {html.escape(title)}</p>" if title else ""

    plain = view.get("plain_summary") or ""
    plain_html = ""
    if plain:
        plain_html = (f"<div class=\"card plain\"><h2>What does this mean?</h2>"
                      f"<div>This passage is about: {html.escape(plain)}</div></div>")

    sugg = view.get("suggestion")
    sugg_html = ""
    if sugg:
        value = sugg["ai_suggested_label"]
        sugg_html = (
            f"<div class=\"card sugg\"><h2>Computer suggestion</h2>"
            f"<p class=\"big\">Suggested label: {value} &mdash; "
            f"{html.escape(LABEL_TEXT[value])}</p>"
            f"<div>Why: {html.escape(sugg['ai_explanation'])}</div>"
            f"<p class=\"note\">This is only a word-matching guess. It cannot tell that "
            f"two different words mean the same thing, so it is often wrong. "
            f"<b>Please decide for yourself.</b></p></div>")

    existing = view.get("human_label")
    prior = ""
    if is_labelled(view):
        value = int(str(existing).strip())
        prior = (f"<p class=\"note\">You answered <b>{value} &mdash; "
                 f"{html.escape(LABEL_TEXT[value])}</b> for this one. "
                 f"Pressing a button will replace that answer.</p>")

    notes = html.escape(str(view.get("human_notes") or ""))

    body = (
        f"<h1>Does this passage help answer this question?</h1>"
        f"<p class=\"sub\">Judge usefulness only &mdash; not whether it is true, "
        f"recent, or from a good journal.</p>"
        f"{banner}"
        f"<div class=\"card\"><h2>Question</h2>"
        f"<div class=\"question\">{html.escape(view.get('question') or '')}</div></div>"
        f"<div class=\"card\"><h2>Medical passage</h2>{title_html}"
        f"<div class=\"passage\">{html.escape(view.get('candidate_text') or '')}</div></div>"
        f"{plain_html}{sugg_html}"
        f"<form method=\"post\" action=\"/label\">"
        f"<input type=\"hidden\" name=\"token\" value=\"{html.escape(token)}\">"
        f"<input type=\"hidden\" name=\"annotation_id\" "
        f"value=\"{html.escape(str(view.get('annotation_id') or ''))}\">"
        f"<div class=\"card\"><h2>Please make your own final decision</h2>"
        f"{prior}{_choice_buttons()}"
        f"<p class=\"note\">Optional &mdash; anything you want to record about this "
        f"passage:</p><textarea name=\"notes\" rows=\"2\">{notes}</textarea></div>"
        f"</form>"
        f"{_progress_card(prog)}")
    return _page("Passage review", body)


def render_done(prog: Dict[str, int], last_id: Optional[str]) -> bytes:
    back = (f" &nbsp;&middot;&nbsp; <a href=\"/annotate?id={urllib.parse.quote(last_id)}\">"
            f"change my last answer</a>") if last_id else ""
    body = (f"<h1>All done</h1>"
            f"<p class=\"done\">You have answered all {prog['total']} passages. "
            f"Your answers are saved.</p>"
            f"{_progress_card(prog)}"
            f"<div class=\"card\"><h2>Next</h2>"
            f"<p>Close this window, then close the black command window "
            f"(press Ctrl+C in it).</p>"
            f"<p>Tell your supervisor the annotation is finished. They will run the "
            f"analysis.</p>"
            f"<p class=\"links\"><a href=\"/annotate\">review the passages again</a>"
            f"{back}</p></div>")
    return _page("All done", body)


def render_error(message: str) -> bytes:
    return _page("Problem", f"<h1>Something went wrong</h1>"
                            f"<div class=\"msg warn\">{html.escape(message)}</div>"
                            f"<p class=\"links\"><a href=\"/annotate\">go back</a></p>")


# --------------------------------------------------------------------------
# State
# --------------------------------------------------------------------------
class AnnotationApp:
    """The rows, the file they came from, and a lock around every write.

    Saved after every single button press. A crash, a closed laptop or a power
    cut costs at most the row in progress, and the save is atomic, so the file
    is never left half-written.
    """

    def __init__(self, sheet: str, show_suggestion: bool = True) -> None:
        self.sheet = sheet
        self.show_suggestion = show_suggestion
        self.rows: List[Dict[str, Any]] = load_sheet(sheet)
        self.lock = threading.Lock()
        self.token = secrets.token_urlsafe(16)
        self.last_id: Optional[str] = None

    def progress(self) -> Dict[str, int]:
        with self.lock:
            return progress(self.rows)

    def row_by_id(self, annotation_id: str) -> Optional[Dict[str, Any]]:
        with self.lock:
            for row in self.rows:
                if str(row.get("annotation_id")) == str(annotation_id):
                    return dict(row)
        return None

    def next_row(self, after: Optional[str] = None) -> Optional[Dict[str, Any]]:
        with self.lock:
            row = next_unlabelled(self.rows, after=after)
            return dict(row) if row else None

    def submit(self, annotation_id: str, label: str, notes: str) -> Dict[str, Any]:
        """Record one human decision and persist it before returning."""
        with self.lock:
            target = next((r for r in self.rows
                           if str(r.get("annotation_id")) == str(annotation_id)), None)
            if target is None:
                raise KeyError(f"unknown annotation_id {annotation_id!r}")
            # With suggestions off, none is computed at all -- not computed and
            # withheld. A suggestion that is never generated cannot leak into the
            # page, cannot be stored beside the label, and cannot be mistaken for
            # one later. Only the fact of its absence is recorded.
            suggestion = (suggestion_for(target, shown=True) if self.show_suggestion
                          else {"ai_suggestion_shown": False,
                                "ai_suggestion_generated": False})
            updated = record_label(self.rows, annotation_id, label,
                                   notes=notes, suggestion=suggestion)
            save_sheet(self.sheet, self.rows)
            self.last_id = str(annotation_id)
            return dict(updated)


# --------------------------------------------------------------------------
# HTTP
# --------------------------------------------------------------------------
class Handler(BaseHTTPRequestHandler):
    server_version = "annotate/1.0"
    protocol_version = "HTTP/1.1"

    @property
    def app(self) -> AnnotationApp:
        return self.server.app  # type: ignore[attr-defined]

    def log_message(self, fmt, *args):  # quiet: the console shows instructions only
        pass

    def _send(self, payload: bytes, status: int = HTTPStatus.OK) -> None:
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(payload)

    def _redirect(self, location: str) -> None:
        self.send_response(HTTPStatus.SEE_OTHER)
        self.send_header("Location", location)
        self.send_header("Content-Length", "0")
        self.end_headers()

    # -- GET ---------------------------------------------------------------
    def do_GET(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        query = urllib.parse.parse_qs(parsed.query)

        if parsed.path in ("/", ""):
            self._redirect("/annotate")
            return
        if parsed.path == "/done":
            self._send(render_done(self.app.progress(), self.app.last_id))
            return
        if parsed.path != "/annotate":
            self._send(render_error(f"no such page: {parsed.path}"),
                       HTTPStatus.NOT_FOUND)
            return

        wanted = (query.get("id") or [""])[0]
        row = self.app.row_by_id(wanted) if wanted else self.app.next_row()
        if row is None:
            if wanted:
                self._send(render_error(f"no passage with id {wanted}"),
                           HTTPStatus.NOT_FOUND)
            else:
                self._redirect("/done")
            return

        message = (query.get("saved") or [""])[0]
        banner = f"Saved: {message}." if message else ""
        self._send(render_annotate(view_model(row, self.app.show_suggestion),
                                   self.app.progress(), self.app.token, banner))

    # -- POST --------------------------------------------------------------
    def do_POST(self) -> None:
        if urllib.parse.urlparse(self.path).path != "/label":
            self._send(render_error("unexpected form target"), HTTPStatus.NOT_FOUND)
            return

        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length).decode("utf-8") if length else ""
        form = urllib.parse.parse_qs(raw, keep_blank_values=True)

        def field(name: str) -> str:
            return (form.get(name) or [""])[0]

        # A per-process token. The server listens on the loopback interface only,
        # but any page in the annotator's browser can POST to localhost blind;
        # without a token such a post would silently write a label.
        if not secrets.compare_digest(field("token"), self.app.token):
            self._send(render_error(
                "This page is out of date -- the program was restarted. "
                "Nothing was saved. Go back and try again."), HTTPStatus.FORBIDDEN)
            return

        annotation_id = field("annotation_id")
        try:
            updated = self.app.submit(annotation_id, field("label"), field("notes").strip())
        except ValueError as exc:
            self._send(render_error(f"That is not a valid answer ({exc}). "
                                    "Nothing was saved."), HTTPStatus.BAD_REQUEST)
            return
        except KeyError as exc:
            self._send(render_error(str(exc)), HTTPStatus.NOT_FOUND)
            return
        except OSError as exc:
            self._send(render_error(
                f"Your answer could NOT be saved: {exc}. Do not carry on -- "
                "the previous answers are still in the file."),
                HTTPStatus.INTERNAL_SERVER_ERROR)
            return

        chosen = int(updated["human_label"])
        note = urllib.parse.quote(f"{chosen} - {LABEL_TEXT[chosen]}")
        nxt = self.app.next_row(after=annotation_id)
        if nxt is None:
            self._redirect("/done")
        else:
            self._redirect(f"/annotate?saved={note}")


def serve(app: AnnotationApp, port: int, open_browser: bool = True) -> None:
    httpd = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    httpd.app = app  # type: ignore[attr-defined]
    url = f"http://127.0.0.1:{httpd.server_address[1]}/annotate"
    prog = app.progress()

    print(f"\n  Annotation file : {app.sheet}")
    print(f"  Passages        : {prog['total']}  "
          f"({prog['completed']} already answered, {prog['remaining']} to go)")
    if not app.show_suggestion:
        print("  Suggestions     : HIDDEN (--no-suggestions)")
    print(f"\n  Open this page in your browser:\n\n      {url}\n")
    print("  Your answers are saved as you go. Press Ctrl+C here when you are done.\n")

    if open_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n  Stopped. Your answers are saved in the file above.")
    finally:
        httpd.server_close()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sheet", default=DEFAULT_SHEET)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--no-suggestions", action="store_true",
                        help="hide the computer's suggestion (anchoring baseline)")
    args = parser.parse_args(argv)

    if not os.path.isfile(args.sheet):
        print(f"ERROR: annotation file not found:\n  {args.sheet}\n\n"
              "Build it first, on the machine that ran the comparison:\n"
              "  python experiments/scripts/evidence_quality.py export")
        return 2
    try:
        app = AnnotationApp(args.sheet, show_suggestion=not args.no_suggestions)
    except KeyFileAccess as exc:
        print(f"ERROR: {exc}")
        return 2
    if not app.rows:
        print(f"ERROR: {args.sheet} has no rows.")
        return 2

    serve(app, args.port, open_browser=not args.no_browser)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
