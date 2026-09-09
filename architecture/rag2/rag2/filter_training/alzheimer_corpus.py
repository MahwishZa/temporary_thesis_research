"""Corpus-grounded construction of an Alzheimer-domain filter training set.

Why this module exists
----------------------
The paper's own labelling procedure (``build_labels.py``, Figure 2) needs a base
LLM to answer every question twice and to score a rationale's perplexity twice
per snippet. That path is reproduced in this repository and stays the canonical
one for MedQA/MedMCQA. It is unusable for this thesis' immediate milestone for
two reasons:

1. it requires gold answers, and the thesis' Alzheimer questions have none;
2. it requires Llama-3-8B, which does not fit the 4 GB RTX 2050 the thesis is
   being run on.

This module therefore builds an Alzheimer-specific training set **from the
frozen corpus itself**, deterministically and without any model in the loop. No
LLM is asked "is this relevant?"; no label is invented by hand.

The construction, stated plainly so it can be judged
----------------------------------------------------
For every source document D that (a) mentions Alzheimer and (b) states its own
research objective in its abstract:

*   **question** -- the authors' own words. Either the title, when the title is
    already a question, or the objective's noun phrase wrapped in one of four
    content-free interrogative frames. Nothing clinical is authored here; the
    substance is always the source document's.
*   **positive evidence** -- an abstract chunk of D itself. An abstract answers
    the objective its own abstract states. Where D has more than one abstract
    window, a window other than the one the objective was read from is
    preferred, so the positive is not a verbatim restatement of the question.
*   **hard negative** -- a chunk from a *different* document, taken from a
    mid-rank band of a BM25 ranking over the same Alzheimer corpus. These are
    the near misses a retriever actually surfaces: same domain, same vocabulary,
    different subject. The top of the ranking is deliberately skipped, because
    a document that ranks first for D's objective may genuinely answer it and
    labelling it ``[NOT_HELPFUL]`` would be a false negative.
*   **easy negative** -- a chunk from a document sharing no distinctive term
    with the question, drawn with a fixed seed.

What this is, and is not
------------------------
These are **weakly supervised** labels. They are derived by a documented,
deterministic rule from real corpus structure, not from human judgement and not
from a model's opinion. Two known ways they can be wrong:

*   a hard negative may in fact be helpful (mitigated, not eliminated, by
    skipping the top of the ranking);
*   a positive is helpful *by construction* rather than by inspection.

``build_human_validation_subset`` exists so that a human can measure both rates
on a stratified sample. Nothing in this module fills that file in.

The corpus is opened read-only. Nothing here writes to ``data/``.

No date field is read anywhere below, and that is deliberate. RAG2 is
date-blind; treating evidence age as a signal is SCAF's contribution and the
whole subject of the comparison. A training set for the RAG2 filter that
selected or labelled examples by age would put a piece of SCAF into the baseline
and make the two arms less distinguishable. ``tests/test_metadata_isolation.py``
enforces this across the whole ``rag2`` package. Provenance records the PMID and
PMCID of every passage, so the publication date of any training example remains
recoverable from the corpus without it ever entering this code.
"""

from __future__ import annotations

import json
import math
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

from ..prompts import DEFAULT_PROMPTS, LABEL_HELPFUL, LABEL_NOT_HELPFUL, PromptSet
from ..schema import Evidence, Question
from .labeling import LabeledPair

# --------------------------------------------------------------------------
# Domain scope.
#
# One literal term, matched case-insensitively against title + text. The corpus
# is broader than Alzheimer alone (it covers cognition and dementia generally),
# so a scope test is needed; using a single author-independent keyword rather
# than a hand-written ontology keeps the inclusion rule inspectable and keeps
# this module from smuggling in a subject-matter judgement of its own.
# --------------------------------------------------------------------------
DOMAIN_TERM = "alzheimer"

# --------------------------------------------------------------------------
# Question frames. Content-free by design: each one adds an interrogative
# wrapper and nothing else, so the clinical substance of every question is the
# source document's own objective, verbatim. Four rather than one because a
# single frame would put an identical 40-character prefix on every training
# question, which the filter could learn instead of the evidence relation.
# The frame is chosen deterministically from the question index.
# --------------------------------------------------------------------------
QUESTION_FRAMES: Tuple[str, ...] = (
    "What does the research evidence show regarding {need}?",
    "What is known about {need}?",
    "What do studies report about {need}?",
    "What is the evidence on {need}?",
)

DATASET_VERSION = "alzheimer-filter-train-v1"

# Sentence boundaries, kept away from the abbreviations that actually occur in
# biomedical abstracts. Splitting "vs." or "et al." as a sentence end truncates
# the objective mid-clause and produces a malformed question.
_ABBREVIATIONS = (
    "vs", "cf", "eg", "ie", "al", "fig", "figs", "no", "ca", "approx", "etc",
    "dr", "prof", "st", "mg", "kg", "ml", "yr", "yrs", "min", "max", "ref",
)
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(])")

_OBJECTIVE_HEADER = re.compile(
    r"\b(?:OBJECTIVES?|AIMS?|PURPOSE|GOAL)\s*[:.]\s*", re.IGNORECASE)
_AIM_SENTENCE = re.compile(
    r"(?:\bwe\s+aim(?:ed)?\s+to\b"
    r"|\bthis\s+(?:study|review|analysis|work|paper)\s+aim(?:ed|s)?\s+to\b"
    r"|\bthe\s+(?:aim|objective|purpose|goal)\s+of\s+(?:this|the)\s+"
    r"(?:study|review|analysis|work|paper)\s+was\s+to\b)\s*",
    re.IGNORECASE)
#: ``To <verb> <noun phrase>`` -- the objective's own object becomes the need.
_TO_NOUN_PHRASE = re.compile(r"^to\s+([a-z]+)\s+(.{15,220})$", re.IGNORECASE)

#: A need ending on one of these is a clause that was cut short; drop it rather
#: than emit an ungrammatical question.
_DANGLING = frozenset("""
and or but with without to for in on of at by from as than then that which who
whom whose the a an is are was were be been being it its their his her our
between among during within versus vs compared while whether if not no nor
""".split())

_TOKEN = re.compile(r"[a-z0-9]+")

#: Function words carry no retrieval signal and would dominate the BM25 overlap.
STOPWORDS = frozenset("""
a about above after again against all also am an and any are as at be because
been before being below between both but by can cannot could did do does doing
down during each few for from further had has have having he her here hers him
his how i if in into is it its itself just me more most my no nor not now of
off on once only or other our out over own same she should so some such than
that the their them then there these they this those through to too under
until up very was we were what when where which while who whom why will with
you your study studies aim aims aimed objective objectives purpose goal
patients patient results result conclusion conclusions method methods
background introduction using used use showed shown show found find analysis
associated association may might data group groups
""".split())


def tokenize(text: str) -> List[str]:
    """Lowercased alphanumeric tokens, stopwords and 1-character tokens removed.

    Deliberately independent of ``scaf.policy.tokenize``: the training set must
    not be constructed with the tokenizer of one of the two arms being compared.
    """
    return [t for t in _TOKEN.findall(text.lower())
            if len(t) > 1 and t not in STOPWORDS]


def _ends_with_abbreviation(text: str) -> bool:
    words = text.rstrip().rstrip(".").split()
    return bool(words) and words[-1].lower().strip("(),;:") in _ABBREVIATIONS


def split_sentences(text: str) -> List[str]:
    """Sentence split that does not break on biomedical abbreviations."""
    merged: List[str] = []
    for part in _SENTENCE_END.split(text.strip()):
        if merged and _ends_with_abbreviation(merged[-1]):
            merged[-1] = f"{merged[-1]} {part}"
        else:
            merged.append(part)
    return [m.strip() for m in merged if m.strip()]


def is_domain_text(text: str) -> bool:
    """Is this Alzheimer-domain material?"""
    return DOMAIN_TERM in text.lower()


# --------------------------------------------------------------------------
# Corpus reading
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class CorpusChunk:
    """One frozen-corpus chunk, reduced to the fields this module reads."""

    chunk_id: str
    document_id: str
    chunk_index: int
    text: str
    title: str
    location: str
    source_category: str
    pmid: str
    pmcid: str
    journal: str

    @classmethod
    def from_record(cls, record: Dict[str, Any]) -> "CorpusChunk":
        """Read the fields this module uses. Date fields are not among them."""
        return cls(
            chunk_id=str(record.get("chunk_id", "")),
            document_id=str(record.get("document_id", "")),
            chunk_index=int(record.get("chunk_index") or 0),
            text=str(record.get("text") or ""),
            title=str(record.get("title") or ""),
            location=str(record.get("location") or ""),
            source_category=str(record.get("source_category") or ""),
            pmid=str(record.get("pmid") or ""),
            pmcid=str(record.get("pmcid") or ""),
            journal=str(record.get("journal") or ""),
        )

    def to_evidence(self) -> Evidence:
        """The ``Evidence`` the filter prompt is rendered from.

        Only ``text`` reaches the model. The rest is provenance, and carries no
        date -- see this module's header.
        """
        return Evidence(
            text=self.text,
            source=self.source_category,
            doc_id=self.document_id,
            passage_id=self.chunk_id,
            metadata={
                "journal": self.journal,
                "pmid": self.pmid,
                "pmcid": self.pmcid,
                "source_category": self.source_category,
            },
        )


def iter_corpus(path: str) -> Iterator[Dict[str, Any]]:
    """Stream the frozen chunk file. Read-only; the corpus is never rewritten."""
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_domain_chunks(path: str, location: str = "abstract") -> List[CorpusChunk]:
    """Alzheimer-domain chunks of one corpus layer, in file order.

    ``location='abstract'`` is the layer available in full here; it is also the
    layer whose text is guaranteed to state a document's objective.
    """
    chunks: List[CorpusChunk] = []
    for record in iter_corpus(path):
        if location and str(record.get("location") or "") != location:
            continue
        if not is_domain_text(f"{record.get('title') or ''} {record.get('text') or ''}"):
            continue
        chunks.append(CorpusChunk.from_record(record))
    return chunks


def group_by_document(chunks: Sequence[CorpusChunk]) -> Dict[str, List[CorpusChunk]]:
    """Document id -> its chunks, each list ordered by ``chunk_index``."""
    grouped: Dict[str, List[CorpusChunk]] = defaultdict(list)
    for chunk in chunks:
        grouped[chunk.document_id].append(chunk)
    return {doc: sorted(items, key=lambda c: (c.chunk_index, c.chunk_id))
            for doc, items in sorted(grouped.items())}


# --------------------------------------------------------------------------
# Information needs
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class InformationNeed:
    """A question and exactly where in the corpus its wording came from."""

    document_id: str
    need: str
    method: str
    source_chunk_id: str
    verb: str = ""

    def render(self, frame_index: int) -> str:
        if self.method == "title_question":
            return self.need
        return QUESTION_FRAMES[frame_index % len(QUESTION_FRAMES)].format(need=self.need)


def _clean_need(text: str) -> Optional[str]:
    """Normalise an extracted noun phrase, or reject it."""
    need = " ".join(text.split()).rstrip(".;:,").strip()
    if not (15 <= len(need) <= 220):
        return None
    if "?" in need:
        return None
    words = need.split()
    if len(words) < 4:
        return None
    if words[-1].lower().strip("(),;:") in _DANGLING:
        return None
    if need.count("(") != need.count(")"):
        return None
    return need


def extract_information_need(
    document_id: str, chunks: Sequence[CorpusChunk]
) -> Optional[InformationNeed]:
    """The document's own stated information need, or ``None``.

    Three sources, in priority order, all of them the authors' words:
    a title that is already a question; an ``OBJECTIVE:``-style header; an
    "we aimed to ..." sentence. Documents stating no objective are skipped --
    being selective costs nothing when the corpus offers far more candidates
    than the target size.
    """
    if not chunks:
        return None
    first = chunks[0]

    title = " ".join((first.title or "").split()).strip()
    if title.endswith("?") and 15 <= len(title) <= 220:
        return InformationNeed(document_id, title, "title_question", first.chunk_id)

    for chunk in chunks:
        text = chunk.text or ""
        segment: Optional[str] = None

        header = _OBJECTIVE_HEADER.search(text)
        if header:
            following = split_sentences(text[header.end():])
            segment = following[0] if following else None
        else:
            aim = _AIM_SENTENCE.search(text)
            if aim:
                following = split_sentences(text[aim.end():])
                if following:
                    segment = following[0]
                    if not segment.lower().startswith("to "):
                        segment = "to " + segment
        if not segment:
            continue

        match = _TO_NOUN_PHRASE.match(segment.strip().rstrip("."))
        if not match:
            continue
        need = _clean_need(match.group(2))
        if need:
            return InformationNeed(
                document_id, need, "objective_np", chunk.chunk_id,
                verb=match.group(1).lower())
    return None


# --------------------------------------------------------------------------
# BM25 over the frozen corpus -- used only to mine hard negatives
# --------------------------------------------------------------------------
class BM25Index:
    """Okapi BM25 over the domain chunks.

    Not a retrieval system and not a stand-in for MedCPT: it never touches the
    evaluation questions and never produces a candidate that either arm sees.
    Its one job is to find, for a training question, the passages that *look*
    retrievable but come from another document -- i.e. realistic hard negatives.

    Terms appearing in more than ``max_df_ratio`` of the corpus are dropped from
    scoring. In an all-Alzheimer corpus "alzheimer" and "dementia" match almost
    everything, so keeping them would rank by domain rather than by subject and
    the resulting negatives would be uninformative.
    """

    def __init__(self, chunks: Sequence[CorpusChunk], k1: float = 1.5, b: float = 0.75,
                 max_df_ratio: float = 0.10) -> None:
        self.chunks = list(chunks)
        self.k1 = k1
        self.b = b
        self.doc_len: List[int] = []
        postings: Dict[str, List[Tuple[int, int]]] = defaultdict(list)

        for position, chunk in enumerate(self.chunks):
            tokens = tokenize(chunk.text)
            self.doc_len.append(len(tokens))
            for term, count in sorted(Counter(tokens).items()):
                postings[term].append((position, count))

        self.n = len(self.chunks)
        self.avgdl = (sum(self.doc_len) / self.n) if self.n else 0.0
        max_df = max(1, int(self.n * max_df_ratio))
        self.postings = {term: plist for term, plist in postings.items()
                         if len(plist) <= max_df}
        self.dropped_terms = sorted(set(postings) - set(self.postings))
        self.idf = {
            term: math.log(1.0 + (self.n - len(plist) + 0.5) / (len(plist) + 0.5))
            for term, plist in self.postings.items()
        }

    def score(self, query_tokens: Sequence[str]) -> Dict[int, float]:
        """Sparse BM25 scores: position -> score, only for positions that match."""
        scores: Dict[int, float] = defaultdict(float)
        for term in sorted(set(query_tokens)):
            plist = self.postings.get(term)
            if not plist:
                continue
            idf = self.idf[term]
            for position, freq in plist:
                length = self.doc_len[position] or 1
                denominator = freq + self.k1 * (1 - self.b + self.b * length / self.avgdl)
                scores[position] += idf * freq * (self.k1 + 1) / denominator
        return dict(scores)

    def ranked(self, query_tokens: Sequence[str], exclude_documents: Iterable[str] = ()) -> List[Tuple[int, float]]:
        """Matching positions, best first. Ties broken by chunk_id for determinism."""
        blocked = set(exclude_documents)
        scored = [(position, value) for position, value in self.score(query_tokens).items()
                  if self.chunks[position].document_id not in blocked]
        scored.sort(key=lambda pair: (-pair[1], self.chunks[pair[0]].chunk_id))
        return scored


# --------------------------------------------------------------------------
# Example construction
# --------------------------------------------------------------------------
@dataclass
class TrainingExample:
    """One (question, evidence, label) triple with full provenance."""

    qid: str
    question_text: str
    label: str
    role: str  # positive | hard_negative | easy_negative
    chunk: CorpusChunk
    need: InformationNeed
    bm25_rank: Optional[int] = None
    bm25_score: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_question(self) -> Question:
        return Question(
            qid=self.qid,
            question=self.question_text,
            options={},          # no options: matches the evaluation questions exactly
            answer=None,
            dataset=DATASET_VERSION,
            metadata={"construction": self.need.method},
        )

    def to_labeled_pair(self, index: int, dataset_name: str,
                        prompts: Optional[PromptSet] = None) -> LabeledPair:
        prompts = prompts or DEFAULT_PROMPTS
        return LabeledPair(
            id=f"{dataset_name}_{index}",
            question=prompts.render_filter_prompt(self.to_question(), self.chunk.to_evidence()),
            answer=self.label,
            dataset_name=dataset_name,
            provenance={
                "qid": self.qid,
                "role": self.role,
                "construction_method": self.need.method,
                "label_source": "weak_supervision_corpus_structure",
                "question_text": self.question_text,
                "question_source_document": self.need.document_id,
                "question_source_chunk": self.need.source_chunk_id,
                "chunk_id": self.chunk.chunk_id,
                "document_id": self.chunk.document_id,
                "pmid": self.chunk.pmid,
                "pmcid": self.chunk.pmcid,
                "source_category": self.chunk.source_category,
                "bm25_rank": self.bm25_rank,
                "bm25_score": None if self.bm25_score is None else round(self.bm25_score, 6),
                **self.metadata,
            },
        )


@dataclass
class BuildStats:
    """Counters the quality report is written from."""

    documents_scanned: int = 0
    documents_with_need: int = 0
    questions_built: int = 0
    skipped_no_need: int = 0
    skipped_leakage: int = 0
    skipped_duplicate_question: int = 0
    skipped_no_hard_negative: int = 0
    skipped_no_answer_window: int = 0
    method_counts: Dict[str, int] = field(default_factory=dict)
    role_counts: Dict[str, int] = field(default_factory=dict)


def normalise_question(text: str) -> str:
    """Comparison form for duplicate and leakage checks."""
    return " ".join(re.sub(r"[^a-z0-9 ]+", " ", text.lower()).split())


def build_examples(
    documents: Dict[str, List[CorpusChunk]],
    index: BM25Index,
    evaluation_questions: Sequence[str],
    target_questions: int = 800,
    positives_per_question: int = 2,
    hard_negatives_per_positive: int = 1,
    easy_negatives_per_positive: int = 1,
    hard_negative_skip: int = 5,
    hard_negative_band: int = 25,
    require_answer_window: bool = True,
    seed: int = 42,
) -> Tuple[List[TrainingExample], BuildStats]:
    """Build the training examples deterministically.

    ``hard_negative_skip`` is the false-negative guard: the top of the BM25
    ranking is the region most likely to contain a passage that genuinely
    answers the question, so negatives are drawn below it.

    ``require_answer_window`` keeps only documents whose abstract has a window
    *other* than the one the objective was read from, and uses only those
    windows as positives. It costs candidate documents and is worth it: the
    objective window restates the question almost verbatim, so a filter trained
    on it can score word overlap instead of helpfulness. The later windows
    carry the results and conclusions -- what actually answers the question.
    """
    rng = random.Random(seed)
    stats = BuildStats()
    blocked = {normalise_question(q) for q in evaluation_questions}
    seen_questions: set = set()
    examples: List[TrainingExample] = []

    for document_id in sorted(documents):
        if stats.questions_built >= target_questions:
            break
        stats.documents_scanned += 1
        chunks = documents[document_id]
        need = extract_information_need(document_id, chunks)
        if need is None:
            stats.skipped_no_need += 1
            continue
        stats.documents_with_need += 1

        question_index = stats.questions_built
        question_text = need.render(question_index)
        normalised = normalise_question(question_text)
        if normalised in blocked:
            stats.skipped_leakage += 1
            continue
        if normalised in seen_questions:
            stats.skipped_duplicate_question += 1
            continue

        # -- positives: the document's own abstract, excluding (or at worst
        #    deprioritising) the window the objective was read from.
        answer_windows = [c for c in chunks if c.chunk_id != need.source_chunk_id]
        if require_answer_window:
            if not answer_windows:
                stats.skipped_no_answer_window += 1
                continue
            positives = answer_windows[:positives_per_question]
        else:
            ordered = answer_windows + [c for c in chunks if c.chunk_id == need.source_chunk_id]
            positives = ordered[:positives_per_question]
        if not positives:
            continue

        query_tokens = tokenize(question_text)
        ranking = index.ranked(query_tokens, exclude_documents={document_id})
        band = ranking[hard_negative_skip:hard_negative_skip + hard_negative_band]
        if len(band) < hard_negatives_per_positive * len(positives):
            stats.skipped_no_hard_negative += 1
            continue

        qid = f"alz-train-{stats.questions_built:05d}"
        pending: List[TrainingExample] = []

        for positive in positives:
            pending.append(TrainingExample(
                qid=qid, question_text=question_text, label=LABEL_HELPFUL,
                role="positive", chunk=positive, need=need,
                metadata={"same_document_as_question": True,
                          "is_objective_window": positive.chunk_id == need.source_chunk_id},
            ))

        chosen_hard = rng.sample(range(len(band)),
                                 k=min(hard_negatives_per_positive * len(positives), len(band)))
        for offset in sorted(chosen_hard):
            position, score = band[offset]
            pending.append(TrainingExample(
                qid=qid, question_text=question_text, label=LABEL_NOT_HELPFUL,
                role="hard_negative", chunk=index.chunks[position], need=need,
                bm25_rank=hard_negative_skip + offset + 1, bm25_score=score,
                metadata={"same_document_as_question": False},
            ))

        # Easy negatives must share no distinctive term at all, so anything the
        # BM25 query matched is off limits.
        matched = {position for position, _ in ranking}
        wanted = easy_negatives_per_positive * len(positives)
        attempts = 0
        added = 0
        while added < wanted and attempts < 200:
            attempts += 1
            position = rng.randrange(index.n)
            chunk = index.chunks[position]
            if position in matched or chunk.document_id == document_id:
                continue
            pending.append(TrainingExample(
                qid=qid, question_text=question_text, label=LABEL_NOT_HELPFUL,
                role="easy_negative", chunk=chunk, need=need,
                bm25_rank=None, bm25_score=0.0,
                metadata={"same_document_as_question": False},
            ))
            added += 1

        examples.extend(pending)
        seen_questions.add(normalised)
        stats.questions_built += 1
        stats.method_counts[need.method] = stats.method_counts.get(need.method, 0) + 1
        for example in pending:
            stats.role_counts[example.role] = stats.role_counts.get(example.role, 0) + 1

    return examples, stats


# --------------------------------------------------------------------------
# Split and validation subset
# --------------------------------------------------------------------------
def split_by_question(
    examples: Sequence[TrainingExample], validation_fraction: float = 0.2, seed: int = 42
) -> Tuple[List[TrainingExample], List[TrainingExample]]:
    """Deterministic split **by question**, never by example.

    Splitting by example would put a question's positives in train and its
    negatives in validation, and the validation score would then be measuring
    memorisation of that question's wording.
    """
    qids = sorted({example.qid for example in examples})
    rng = random.Random(seed)
    shuffled = list(qids)
    rng.shuffle(shuffled)
    cut = int(round(len(shuffled) * (1.0 - validation_fraction)))
    train_qids = set(shuffled[:cut])
    train = [e for e in examples if e.qid in train_qids]
    validation = [e for e in examples if e.qid not in train_qids]
    return train, validation


def build_human_validation_subset(
    examples: Sequence[TrainingExample], size: int = 120, seed: int = 42
) -> List[Dict[str, Any]]:
    """A stratified sample for a human to judge. Labels are left blank.

    The ``human_label`` field is empty on purpose: filling it in is the one step
    of this pipeline a model must not perform. ``score_human_validation``
    computes agreement once a person has completed it.
    """
    rng = random.Random(seed)
    by_role: Dict[str, List[TrainingExample]] = defaultdict(list)
    for example in examples:
        by_role[example.role].append(example)

    rows: List[Dict[str, Any]] = []
    roles = sorted(by_role)
    per_role = max(1, size // max(1, len(roles)))
    for role in roles:
        pool = sorted(by_role[role], key=lambda e: (e.qid, e.chunk.chunk_id))
        picks = pool if len(pool) <= per_role else rng.sample(pool, per_role)
        for example in sorted(picks, key=lambda e: (e.qid, e.chunk.chunk_id)):
            rows.append({
                "qid": example.qid,
                "question": example.question_text,
                "evidence": example.chunk.text,
                "chunk_id": example.chunk.chunk_id,
                "document_id": example.chunk.document_id,
                "assigned_label": example.label,
                "role": example.role,
                # To be completed by a human: [HELPFUL] or [NOT_HELPFUL].
                "human_label": "",
                "human_notes": "",
            })
    return sorted(rows, key=lambda r: (r["role"], r["qid"], r["chunk_id"]))


def score_human_validation(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Agreement between the weak labels and a completed human subset."""
    judged = [r for r in rows if str(r.get("human_label") or "").strip()]
    if not judged:
        return {"judged": 0, "note": "human_label is empty; no agreement can be reported"}
    agree = sum(1 for r in judged
                if str(r["human_label"]).strip().upper() == str(r["assigned_label"]).strip().upper())
    by_role: Dict[str, List[int]] = defaultdict(list)
    for row in judged:
        by_role[str(row.get("role"))].append(
            1 if str(row["human_label"]).strip().upper() == str(row["assigned_label"]).strip().upper() else 0)
    return {
        "judged": len(judged),
        "agreements": agree,
        "agreement_rate": round(agree / len(judged), 4),
        "by_role": {role: {"judged": len(v), "agreement_rate": round(sum(v) / len(v), 4)}
                    for role, v in sorted(by_role.items())},
    }


def leakage_report(
    examples: Sequence[TrainingExample], evaluation_questions: Sequence[str]
) -> Dict[str, Any]:
    """Overlap between training questions and the evaluation set.

    Exact (normalised) duplicates are a hard failure. Token-Jaccard is reported
    for the closest pairs so a near-duplicate cannot hide behind punctuation.
    """
    training = sorted({e.question_text for e in examples})
    evaluation = list(evaluation_questions)
    normalised_eval = {normalise_question(q) for q in evaluation}
    exact = sorted(q for q in training if normalise_question(q) in normalised_eval)

    eval_tokens = [(q, set(tokenize(q))) for q in evaluation]
    closest: List[Dict[str, Any]] = []
    for question in training:
        tokens = set(tokenize(question))
        if not tokens:
            continue
        best = 0.0
        best_q = ""
        for evaluation_question, evaluation_token_set in eval_tokens:
            union = tokens | evaluation_token_set
            if not union:
                continue
            score = len(tokens & evaluation_token_set) / len(union)
            if score > best:
                best, best_q = score, evaluation_question
        closest.append({"training_question": question, "evaluation_question": best_q,
                        "jaccard": round(best, 4)})
    closest.sort(key=lambda row: -row["jaccard"])
    return {
        "training_questions": len(training),
        "evaluation_questions": len(evaluation),
        "exact_duplicates": exact,
        "exact_duplicate_count": len(exact),
        "max_jaccard": closest[0]["jaccard"] if closest else 0.0,
        "closest_pairs": closest[:10],
    }
