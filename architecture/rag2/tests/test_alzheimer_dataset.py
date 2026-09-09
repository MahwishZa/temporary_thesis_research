"""Construction of the Alzheimer-specific filter training set.

The dataset is the one artifact of this route that nothing downstream can
check: a wrong label is invisible to the training script, to the checkpoint
verifier and to the comparison. So the rules that produce labels are pinned
here -- what becomes a question, what becomes a positive, what may become a
negative, and what must never enter the set at all.
"""

import json

import pytest

from rag2.filter_training.alzheimer_corpus import (
    DOMAIN_TERM,
    QUESTION_FRAMES,
    BM25Index,
    CorpusChunk,
    InformationNeed,
    build_examples,
    build_human_validation_subset,
    extract_information_need,
    group_by_document,
    is_domain_text,
    leakage_report,
    normalise_question,
    score_human_validation,
    split_by_question,
    split_sentences,
    tokenize,
)
from rag2.prompts import DEFAULT_PROMPTS, LABEL_HELPFUL, LABEL_NOT_HELPFUL


def _chunk(chunk_id, document_id, text, title="", index=1):
    return CorpusChunk(
        chunk_id=chunk_id, document_id=document_id, chunk_index=index, text=text,
        title=title, location="abstract", source_category="pmc-fulltext",
        pmid=f"pmid-{document_id}", pmcid=document_id, journal="J Test",
    )


# --------------------------------------------------------------------------
# Text handling
# --------------------------------------------------------------------------
def test_tokenize_drops_stopwords_and_single_characters():
    tokens = tokenize("The plasma p-tau217 of a patient")
    assert "the" not in tokens and "of" not in tokens and "a" not in tokens
    assert "plasma" in tokens and "tau217" in tokens


def test_sentence_split_does_not_break_on_biomedical_abbreviations():
    text = "We compared donepezil vs. placebo. Results were positive."
    assert split_sentences(text) == [
        "We compared donepezil vs. placebo.", "Results were positive."]


def test_domain_filter_is_the_single_documented_term():
    assert is_domain_text("Alzheimer's disease progression")
    assert is_domain_text("ALZHEIMER DISEASE")
    assert not is_domain_text("Parkinson disease and gait")
    assert DOMAIN_TERM == "alzheimer"


# --------------------------------------------------------------------------
# Information needs -- the authors' words, never invented ones
# --------------------------------------------------------------------------
def test_an_interrogative_title_is_used_verbatim():
    chunks = [_chunk("c1", "D1", "Some abstract text about Alzheimer.",
                     title="Does lecanemab slow decline in early Alzheimer disease?")]
    need = extract_information_need("D1", chunks)
    assert need.method == "title_question"
    assert need.need == "Does lecanemab slow decline in early Alzheimer disease?"
    # A title question is asked as written: no frame is wrapped around it.
    assert need.render(0) == need.need


def test_objective_header_supplies_the_noun_phrase():
    chunks = [_chunk("c1", "D1",
                     "BACKGROUND: Alzheimer disease is common. OBJECTIVE: To evaluate the "
                     "diagnostic accuracy of plasma p-tau217 for amyloid pathology. "
                     "METHODS: We enrolled 80 participants.")]
    need = extract_information_need("D1", chunks)
    assert need.method == "objective_np"
    assert need.verb == "evaluate"
    assert need.need == "the diagnostic accuracy of plasma p-tau217 for amyloid pathology"
    assert need.render(0) == QUESTION_FRAMES[0].format(need=need.need)


def test_an_aim_sentence_is_used_when_there_is_no_header():
    chunks = [_chunk("c1", "D1",
                     "Alzheimer disease is common. We aimed to compare hippocampal volumetry "
                     "with amyloid PET in memory clinic patients. Methods follow.")]
    need = extract_information_need("D1", chunks)
    assert need.method == "objective_np"
    assert need.need == "hippocampal volumetry with amyloid PET in memory clinic patients"


def test_a_truncated_objective_is_rejected_rather_than_asked():
    """'...the relationship between X and' is a cut clause, not a question."""
    chunks = [_chunk("c1", "D1",
                     "OBJECTIVE: To examine the relationship between amyloid burden and. "
                     "METHODS: none.")]
    assert extract_information_need("D1", chunks) is None


def test_a_document_stating_no_objective_yields_nothing():
    chunks = [_chunk("c1", "D1", "Alzheimer disease affects many people worldwide.")]
    assert extract_information_need("D1", chunks) is None


def test_frames_rotate_so_the_set_is_not_one_constant_prefix():
    need = InformationNeed("D1", "the role of amyloid PET", "objective_np", "c1")
    rendered = {need.render(i) for i in range(len(QUESTION_FRAMES))}
    assert len(rendered) == len(QUESTION_FRAMES)


# --------------------------------------------------------------------------
# BM25 -- negative mining only
# --------------------------------------------------------------------------
def _corpus():
    return [
        _chunk("a1", "A", "Plasma p-tau217 predicts amyloid pathology in Alzheimer disease."),
        _chunk("b1", "B", "Amyloid PET imaging quantifies plaque burden in Alzheimer disease."),
        _chunk("c1", "C", "Hospital catering budgets and staff rotas in Alzheimer wards."),
        _chunk("d1", "D", "Sleep architecture and circadian rhythm in Alzheimer disease."),
    ]


def test_bm25_ranks_a_term_match_above_a_non_match():
    index = BM25Index(_corpus(), max_df_ratio=1.0)
    ranked = index.ranked(tokenize("plasma p-tau217 amyloid"))
    assert index.chunks[ranked[0][0]].document_id == "A"
    assert "C" not in [index.chunks[p].document_id for p, _ in ranked]


def test_bm25_excludes_the_source_document():
    index = BM25Index(_corpus(), max_df_ratio=1.0)
    ranked = index.ranked(tokenize("plasma p-tau217 amyloid"), exclude_documents={"A"})
    assert "A" not in [index.chunks[p].document_id for p, _ in ranked]


def test_bm25_drops_terms_that_match_almost_everything():
    """'alzheimer' is in every document here, so it must not drive the ranking."""
    index = BM25Index(_corpus(), max_df_ratio=0.5)
    assert "alzheimer" in index.dropped_terms
    assert not index.ranked(tokenize("alzheimer"))


# --------------------------------------------------------------------------
# Example construction
# --------------------------------------------------------------------------
def _documents(n=6):
    chunks = []
    for i in range(n):
        document = f"D{i}"
        chunks.append(_chunk(f"{document}#1", document,
                             f"OBJECTIVE: To evaluate biomarker{i} for Alzheimer disease staging "
                             f"in cohort{i}. METHODS: enrolled.", index=1))
        chunks.append(_chunk(f"{document}#2", document,
                             f"RESULTS: biomarker{i} separated groups in cohort{i}. "
                             f"CONCLUSION: biomarker{i} is informative for Alzheimer disease.",
                             index=2))
    return chunks


def _build(**kwargs):
    chunks = _documents()
    index = BM25Index(chunks, max_df_ratio=1.0)
    options = dict(target_questions=10, positives_per_question=1,
                   hard_negatives_per_positive=1, easy_negatives_per_positive=0,
                   hard_negative_skip=0, hard_negative_band=5, seed=42)
    options.update(kwargs)
    return build_examples(group_by_document(chunks), index, [], **options)


def test_positives_come_from_the_questions_own_document_and_negatives_do_not():
    examples, _ = _build()
    assert examples
    for example in examples:
        if example.role == "positive":
            assert example.label == LABEL_HELPFUL
            assert example.chunk.document_id == example.need.document_id
        else:
            assert example.label == LABEL_NOT_HELPFUL
            assert example.chunk.document_id != example.need.document_id


def test_the_positive_is_never_the_window_the_question_was_read_from():
    """Otherwise the evidence restates the question and overlap does the work."""
    examples, _ = _build()
    positives = [e for e in examples if e.role == "positive"]
    assert positives
    for example in positives:
        assert example.chunk.chunk_id != example.need.source_chunk_id


def test_a_document_with_no_answer_window_is_skipped_by_default():
    single = [_chunk("D9#1", "D9", "OBJECTIVE: To evaluate biomarker9 for Alzheimer disease "
                                   "staging in cohort9. METHODS: enrolled.")]
    index = BM25Index(single, max_df_ratio=1.0)
    examples, stats = build_examples(group_by_document(single), index, [],
                                     target_questions=5, hard_negative_skip=0)
    assert examples == []
    assert stats.skipped_no_answer_window == 1


def test_the_objective_window_may_be_used_when_explicitly_allowed():
    single = [_chunk("D9#1", "D9", "OBJECTIVE: To evaluate biomarker9 for Alzheimer disease "
                                   "staging in cohort9. METHODS: enrolled.")]
    other = [_chunk("D8#1", "D8", "OBJECTIVE: To evaluate biomarker8 for Alzheimer disease "
                                  "staging in cohort8. METHODS: enrolled.")]
    chunks = single + other
    index = BM25Index(chunks, max_df_ratio=1.0)
    examples, _ = build_examples(group_by_document(chunks), index, [], target_questions=5,
                                 positives_per_question=1, hard_negatives_per_positive=1,
                                 easy_negatives_per_positive=0, hard_negative_skip=0,
                                 require_answer_window=False)
    assert any(e.role == "positive" for e in examples)


def test_construction_is_deterministic_for_a_fixed_seed():
    first, _ = _build()
    second, _ = _build()
    assert [(e.qid, e.chunk.chunk_id, e.label) for e in first] == \
           [(e.qid, e.chunk.chunk_id, e.label) for e in second]


def test_hard_negative_skip_keeps_the_top_of_the_ranking_out_of_the_negatives():
    """The false-negative guard: rank 1 may genuinely answer the question."""
    examples, _ = _build(hard_negative_skip=2, hard_negative_band=3)
    ranks = [e.bm25_rank for e in examples if e.role == "hard_negative"]
    assert ranks and all(rank > 2 for rank in ranks)


def test_a_question_matching_an_evaluation_question_is_never_built():
    chunks = _documents()
    index = BM25Index(chunks, max_df_ratio=1.0)
    every, _ = _build()
    blocked = every[0].question_text
    examples, stats = build_examples(group_by_document(chunks), index, [blocked],
                                     target_questions=10, positives_per_question=1,
                                     hard_negatives_per_positive=1,
                                     easy_negatives_per_positive=0, hard_negative_skip=0,
                                     hard_negative_band=5, seed=42)
    assert stats.skipped_leakage >= 1
    assert blocked not in {e.question_text for e in examples}


# --------------------------------------------------------------------------
# Training-record shape
# --------------------------------------------------------------------------
def test_the_training_record_matches_the_release_schema():
    examples, _ = _build()
    record = examples[0].to_labeled_pair(0, "unit").to_training_record()
    assert set(record) == {"id", "answer", "dataset_name", "question"}
    assert all(isinstance(v, str) for v in record.values())
    assert record["answer"] in (LABEL_HELPFUL, LABEL_NOT_HELPFUL)


def test_the_filter_input_carries_no_options_so_training_matches_inference():
    """The 30 evaluation questions have no options; training must not either.

    A filter trained on 'question A) .. B) .. C) .. D) ..' and used on a bare
    question is being asked something it never saw. Keeping ``options`` empty
    makes the rendered training input character-identical in form to what
    ``rag2.filtering`` builds at evaluation time.
    """
    examples, _ = _build()
    example = examples[0]
    rendered = example.to_labeled_pair(0, "unit").question
    assert rendered.startswith("Given the following evidence")
    assert "\n\nEvidence: " in rendered and "\n\nQuestion: " in rendered
    assert rendered.endswith(example.question_text)
    assert " A) " not in rendered and " B) " not in rendered
    # The same text the inference path would build for this pair.
    assert rendered == DEFAULT_PROMPTS.render_filter_prompt(
        example.to_question(), example.chunk.to_evidence())


def test_provenance_records_where_every_label_came_from():
    examples, _ = _build()
    provenance = examples[0].to_labeled_pair(0, "unit").provenance
    assert provenance["label_source"] == "weak_supervision_corpus_structure"
    for key in ("qid", "role", "chunk_id", "document_id", "pmid", "pmcid",
                "question_source_document", "question_source_chunk", "construction_method"):
        assert key in provenance
    # Provenance stays out of the model's input.
    assert "chunk_id" not in json.dumps(examples[0].to_labeled_pair(0, "u").to_training_record())


# --------------------------------------------------------------------------
# Split, leakage, human validation
# --------------------------------------------------------------------------
def test_the_split_never_puts_one_question_on_both_sides():
    examples, _ = _build()
    train, validation = split_by_question(examples, validation_fraction=0.4, seed=42)
    assert train and validation
    assert not ({e.qid for e in train} & {e.qid for e in validation})
    assert len(train) + len(validation) == len(examples)


def test_the_split_is_deterministic():
    examples, _ = _build()
    a, _ = split_by_question(examples, seed=42)
    b, _ = split_by_question(examples, seed=42)
    assert [e.qid for e in a] == [e.qid for e in b]


def test_leakage_report_catches_an_exact_duplicate_through_punctuation():
    examples, _ = _build()
    question = examples[0].question_text
    disguised = question.upper().replace("?", " ?")
    report = leakage_report(examples, [disguised])
    assert report["exact_duplicate_count"] == 1
    assert normalise_question(disguised) == normalise_question(question)


def test_leakage_report_is_clean_for_unrelated_questions():
    examples, _ = _build()
    report = leakage_report(examples, ["What is the capital of France?"])
    assert report["exact_duplicate_count"] == 0
    assert report["max_jaccard"] < 0.5


def test_the_human_validation_subset_is_left_for_a_human_to_fill_in():
    examples, _ = _build()
    rows = build_human_validation_subset(examples, size=6, seed=42)
    assert rows
    assert all(row["human_label"] == "" for row in rows), \
        "labels in this file must be written by a person, never generated"
    for row in rows:
        assert row["assigned_label"] in (LABEL_HELPFUL, LABEL_NOT_HELPFUL)
        assert row["question"] and row["evidence"]


def test_agreement_is_absent_rather_than_zero_until_a_human_has_judged():
    examples, _ = _build()
    rows = build_human_validation_subset(examples, size=4, seed=42)
    empty = score_human_validation(rows)
    assert empty["judged"] == 0
    assert "agreement_rate" not in empty

    for row in rows:
        row["human_label"] = row["assigned_label"]
    scored = score_human_validation(rows)
    assert scored["judged"] == len(rows)
    assert scored["agreement_rate"] == 1.0

    rows[0]["human_label"] = (LABEL_NOT_HELPFUL if rows[0]["assigned_label"] == LABEL_HELPFUL
                              else LABEL_HELPFUL)
    assert score_human_validation(rows)["agreement_rate"] < 1.0


@pytest.mark.parametrize("fraction", [0.1, 0.2, 0.5])
def test_validation_fraction_is_honoured_within_rounding(fraction):
    examples, _ = _build(target_questions=10)
    _, validation = split_by_question(examples, validation_fraction=fraction, seed=42)
    qids = {e.qid for e in examples}
    expected = round(len(qids) * fraction)
    assert abs(len({e.qid for e in validation}) - expected) <= 1


def test_the_dataset_carries_no_date_field_anywhere():
    """RAG2 is date-blind; evidence age is SCAF's contribution, not the baseline's.

    A training set that selected or labelled by publication age would put part of
    SCAF into the arm SCAF is being compared against. tests/test_metadata_isolation.py
    enforces this for the whole package; this pins the produced artifact too, since
    a date could reach the model through provenance rather than through code.
    """
    examples, _ = _build()
    pair = examples[0].to_labeled_pair(0, "unit")
    serialised = json.dumps({"record": pair.to_training_record(),
                             "provenance": pair.provenance,
                             "evidence_metadata": examples[0].chunk.to_evidence().metadata})
    for banned in ("publication_date", "canonical_date", "pub_date", "recency", "published_at"):
        assert banned not in serialised, f"{banned} reached the training artifact"
    assert "pmid" in pair.provenance and "pmcid" in pair.provenance, \
        "the date must stay recoverable from the corpus via these identifiers"
