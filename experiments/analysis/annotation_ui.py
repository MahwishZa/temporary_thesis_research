"""A local, beginner-facing interface for the evidence-quality annotation.

One passage at a time: the question, the passage, a plain-language reading of
what the passage is about, a computer suggestion with its reason, and three
buttons. Only the button press is stored as ``human_label``.

Runs entirely on the annotator's machine. Nothing is sent anywhere, and the file
holding the machine scores (``annotation_key.jsonl``) is never opened by this
module -- ``load_sheet`` refuses any path whose name contains ``key``.

A warning that belongs in the thesis, not only in this docstring
-----------------------------------------------------------------
**Showing a suggestion anchors the annotator.** People agree with a displayed
recommendation more often than they would have unprompted; that is a robust,
well-documented effect, not a hypothetical. The labels collected with
suggestions visible are therefore not fully independent of the suggestion.

Two consequences specific to this experiment:

1. The suggestion here is **lexical overlap** between question and passage.
   SCAF's support term sigma is *also* lexical overlap (corpus-IDF weighted).
   So to whatever extent the annotator defers to the suggestion, a
   SCAF-versus-human correlation is partly manufactured rather than measured.
   That is the exact claim the annotation exists to test.
2. It is measurable rather than merely arguable. Every row records
   ``ai_suggested_label`` beside ``human_label``, so the agreement rate is
   recoverable afterwards. A high agreement rate is a warning about the labels,
   not a validation of the rule.

``--no-suggestions`` runs the same interface with the suggestion hidden.
Annotating a subset that way first gives an anchoring baseline to compare
against, and costs only the time of those rows.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
import time
from typing import Any, Dict, List, Optional, Sequence, Set

#: Bump when the suggestion rule changes; stored on every suggested row.
SUGGESTION_RULE_VERSION = "lexical-overlap-v1"

#: Coverage thresholds, fixed here rather than tuned against any label.
STRONG_COVERAGE = 0.60
WEAK_COVERAGE = 0.30

LABEL_TEXT = {
    0: "Not relevant",
    1: "Partially relevant",
    2: "Clearly relevant",
}
LABEL_HELP = {
    0: "This passage does not help answer the question.",
    1: "This passage is related to the question, but it only gives weak, "
       "indirect, or incomplete information.",
    2: "This passage directly provides useful information for answering the question.",
}

_WORD = re.compile(r"[A-Za-z0-9][A-Za-z0-9\-]*")

#: Ordinary English words plus question scaffolding. Kept small and explicit so
#: the rule stays inspectable; this is deliberately NOT scaf.policy's stopword
#: list, so the suggestion cannot inherit SCAF's own tokenisation.
_STOPWORDS = frozenset("""
a about above after again against all also am an and any are as at be because been
before being below between both but by can cannot could did do does doing down
during each few for from further had has have having he her here hers him his how
i if in into is it its itself just me more most my no nor not now of off on once
only or other our out over own same she should so some such than that the their
them then there these they this those through to too under until up very was we
were what when where which while who whom why will with you your
what's whats does do is are which how why when who
study studies patient patients evidence show shows shown report reports
""".split())


def content_terms(text: str) -> Set[str]:
    """Lowercased content words: the units the suggestion rule compares."""
    return {w.lower() for w in _WORD.findall(text or "")
            if len(w) > 2 and w.lower() not in _STOPWORDS}


def suggest_label(question: str, passage: str) -> Dict[str, Any]:
    """A transparent computer suggestion from the question and passage alone.

    No model, no machine score, no SCAF or RAG2 value: how many of the
    question's content words appear in the passage, and nothing else. Chosen
    over a language model because it is deterministic, reproducible, costs
    nothing, sends no thesis data anywhere, and -- most importantly -- its bias
    is legible. A reader can see exactly why it said what it said.

    Its weakness is equally legible: it cannot recognise a paraphrase, so a
    passage that answers the question in different words scores low. Say so in
    the interface, because that is precisely where the annotator must override
    it.
    """
    q_terms = content_terms(question)
    p_terms = content_terms(passage)
    matched = sorted(q_terms & p_terms)
    missing = sorted(q_terms - p_terms)
    coverage = (len(matched) / len(q_terms)) if q_terms else 0.0

    if coverage >= STRONG_COVERAGE:
        label = 2
    elif coverage >= WEAK_COVERAGE:
        label = 1
    else:
        label = 0

    if label == 2:
        why = (f"The passage mentions most of what the question asks about "
               f"({', '.join(matched[:6])}). It looks like it is on topic.")
    elif label == 1:
        why = (f"The passage mentions some of what the question asks about "
               f"({', '.join(matched[:5]) or 'very little'}), but not "
               f"{', '.join(missing[:4])}. It may only be partly useful.")
    else:
        why = (f"The passage does not mention most of what the question asks "
               f"about ({', '.join(missing[:5])}). It may be off topic.")

    return {
        "ai_suggested_label": label,
        "ai_explanation": why,
        "ai_rule_version": SUGGESTION_RULE_VERSION,
        "ai_matched_terms": matched[:12],
        "ai_missing_terms": missing[:12],
        "ai_coverage": round(coverage, 4),
    }


def plain_summary(passage: str, limit: int = 2) -> str:
    """A plain-language lead-in: the passage's own first sentences.

    Quoting the passage rather than paraphrasing it is deliberate. A generated
    paraphrase would be a second opinion the annotator has to evaluate, and a
    wrong one would mislead. The passage's own opening says what it is about.
    """
    text = " ".join((passage or "").split())
    if not text:
        return ""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    lead = " ".join(sentences[:limit])
    return lead if len(lead) <= 400 else lead[:397] + "..."


# --------------------------------------------------------------------------
# Sheet I/O -- the key file is never opened here
# --------------------------------------------------------------------------
class KeyFileAccess(Exception):
    """Raised when something tries to load the de-blinding key."""


def load_sheet(path: str) -> List[Dict[str, Any]]:
    """Read the annotation sheet. Refuses the machine-score key.

    The guard is on the filename rather than the content because it must fire
    *before* the file is read: the point is that this process never holds the
    scores in memory while the annotator is judging.
    """
    name = os.path.basename(path).lower()
    if "key" in name:
        raise KeyFileAccess(
            f"refusing to open {name}: this interface must stay blind to the "
            "machine scores. Point it at annotation_sheet.jsonl.")
    with open(path, "r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


#: Fields the interface may add. Everything else in a row is left untouched.
WRITABLE = ("human_label", "human_notes", "ai_suggested_label", "ai_explanation",
            "ai_rule_version", "ai_coverage", "ai_suggestion_shown",
            "ai_suggestion_generated", "annotated_utc")

#: Copied from the suggestion onto the row, in this order. ``human_label`` is
#: deliberately absent: no path through this module writes a suggestion there.
SUGGESTION_FIELDS = ("ai_suggested_label", "ai_explanation", "ai_rule_version",
                     "ai_coverage", "ai_suggestion_shown", "ai_suggestion_generated")

#: Anything matching these must never reach the browser.
FORBIDDEN_IN_UI = ("scaf_score", "rag2_score", "scaf_admitted", "rag2_admitted",
                   "scaf_bin", "rag2_bin", "sigma_support", "gamma_currency",
                   "tau_authority", "rho_corroboration", "rerank_rank")


def save_sheet(path: str, rows: Sequence[Dict[str, Any]]) -> str:
    """Rewrite the sheet atomically, so an interrupted save cannot truncate it."""
    directory = os.path.dirname(os.path.abspath(path)) or "."
    handle = tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="\n",
                                         dir=directory, delete=False, suffix=".tmp")
    try:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.close()
        os.replace(handle.name, path)
    except BaseException:
        handle.close()
        if os.path.exists(handle.name):
            os.unlink(handle.name)
        raise
    return path


def is_labelled(row: Dict[str, Any]) -> bool:
    """Has a human actually chosen a label for this row?

    A suggestion present with no human choice is *not* labelled. That is the
    whole distinction the experiment rests on.
    """
    raw = str(row.get("human_label", "")).strip()
    return raw.isdigit() and int(raw) in LABEL_TEXT


def progress(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    done = sum(1 for r in rows if is_labelled(r))
    return {"total": len(rows), "completed": done, "remaining": len(rows) - done}


def next_unlabelled(rows: Sequence[Dict[str, Any]], after: Optional[str] = None
                    ) -> Optional[Dict[str, Any]]:
    """The next row still needing a human decision, resuming where work stopped."""
    started = after is None
    for row in rows:
        if not started:
            started = str(row.get("annotation_id")) == after
            continue
        if not is_labelled(row):
            return row
    for row in rows:                       # wrap around
        if not is_labelled(row):
            return row
    return None


def record_label(rows: List[Dict[str, Any]], annotation_id: str, label: Any,
                 notes: str = "", suggestion: Optional[Dict[str, Any]] = None
                 ) -> Dict[str, Any]:
    """Store one human decision. Rejects anything that is not 0, 1 or 2.

    ``suggestion`` is written to its own fields and never to ``human_label`` --
    the two are kept apart in storage as well as in the interface, so a later
    analysis cannot mistake one for the other.
    """
    try:
        value = int(str(label).strip())
    except (TypeError, ValueError):
        raise ValueError(f"invalid label {label!r}; expected 0, 1 or 2")
    if value not in LABEL_TEXT:
        raise ValueError(f"invalid label {value}; expected 0, 1 or 2")

    for row in rows:
        if str(row.get("annotation_id")) == str(annotation_id):
            row["human_label"] = value
            if notes:
                row["human_notes"] = notes
            if suggestion:
                for field in SUGGESTION_FIELDS:
                    if field in suggestion:
                        row[field] = suggestion[field]
            row["annotated_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            return row
    raise KeyError(f"unknown annotation_id {annotation_id!r}")


def suggestion_for(row: Dict[str, Any], shown: bool) -> Dict[str, Any]:
    """The suggestion to *store* on a row, tagged with whether it was displayed.

    ``shown`` is written on every labelled row and never inferred later: it is
    the field that separates an anchored label from an unanchored one, and the
    first pilot pass showed why that matters -- 120 of 120 rows were shown a
    suggestion and 120 of 120 labels matched it exactly, which is unusable as
    independent validation.

    Callers running a human-only pass should not call this at all. Storing
    ``{"ai_suggestion_shown": False, "ai_suggestion_generated": False}`` records
    that no suggestion existed, which is a stronger guarantee than one that was
    computed and withheld.
    """
    payload = suggest_label(row.get("question", ""), row.get("candidate_text", ""))
    payload["ai_suggestion_shown"] = bool(shown)
    payload["ai_suggestion_generated"] = True
    return payload


def view_model(row: Dict[str, Any], show_suggestion: bool = True) -> Dict[str, Any]:
    """What the browser is allowed to see for one row.

    Built by naming the permitted fields rather than by removing forbidden ones,
    so a new machine field added upstream cannot leak by default.

    ``canonical_date`` and ``source_category`` are in the blind sheet but are
    deliberately **not** here. They are the inputs to SCAF's currency and
    authority terms, and the written instructions tell the annotator to judge
    relevance only -- not recency, not journal quality. Displaying them would
    invite exactly the judgement the instructions forbid, and any resulting
    SCAF-versus-human agreement on those two terms would be an artefact of the
    interface. The annotator cannot be influenced by a number they never see.
    """
    payload = {
        "annotation_id": row.get("annotation_id", ""),
        "qid": row.get("qid", ""),
        "question": row.get("question", ""),
        "chunk_id": row.get("chunk_id", ""),
        "title": row.get("title", ""),
        "candidate_text": row.get("candidate_text", ""),
        "human_label": row.get("human_label", ""),
        "human_notes": row.get("human_notes", ""),
        "plain_summary": plain_summary(row.get("candidate_text", "")),
    }
    if show_suggestion:
        payload["suggestion"] = suggest_label(row.get("question", ""),
                                              row.get("candidate_text", ""))
    return payload
