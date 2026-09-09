#!/usr/bin/env python
"""Stage 3a (Alzheimer route): build the filter training set from the frozen corpus.

This is the sibling of ``03_build_filter_labels.py``, not a replacement for it.
That script implements the paper's own perplexity procedure (Figure 2) and stays
the canonical path for MedQA/MedMCQA. It cannot be used for this thesis'
milestone: it needs gold answers, which the Alzheimer questions do not have, and
Llama-3-8B, which does not fit a 4 GB RTX 2050.

This script instead derives an Alzheimer-specific training set deterministically
from the frozen corpus -- no LLM, no hand-written labels. The construction and
its known failure modes are documented in
``rag2/filter_training/alzheimer_corpus.py``; the labels it produces are **weakly
supervised** and every artifact written here says so.

The corpus is opened read-only. Nothing under ``data/`` is written.

    python scripts/03b_build_alzheimer_filter_labels.py \\
        --chunks ../../data/corpora/pmc/chunks/chunks.jsonl \\
        --eval-questions ../../data/datasets/thesis_questions/dev_questions.jsonl \\
        --out ../../experiments/results/rag2_vs_scaf_alzheimer/training_dataset \\
        --target-questions 800
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from collections import Counter
from typing import Any, Dict, List, Sequence

_HERE = os.path.dirname(os.path.abspath(__file__))
_RAG2 = os.path.dirname(_HERE)
if _RAG2 not in sys.path:
    sys.path.insert(0, _RAG2)

from rag2.filter_training.alzheimer_corpus import (  # noqa: E402
    DATASET_VERSION,
    QUESTION_FRAMES,
    BM25Index,
    build_examples,
    build_human_validation_subset,
    group_by_document,
    leakage_report,
    load_domain_chunks,
    split_by_question,
    tokenize,
)
from rag2.filter_training.train import write_training_file  # noqa: E402
from rag2.prompts import LABEL_HELPFUL, LABEL_NOT_HELPFUL  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(_RAG2))
DEFAULT_CHUNKS = os.path.join(REPO_ROOT, "data", "corpora", "pmc", "chunks", "chunks.jsonl")
DEFAULT_EVAL = os.path.join(REPO_ROOT, "data", "datasets", "thesis_questions", "dev_questions.jsonl")
DEFAULT_OUT = os.path.join(REPO_ROOT, "experiments", "results", "rag2_vs_scaf_alzheimer",
                           "training_dataset")


def write_json(path: str, payload: Any) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")
    return path


def write_jsonl(path: str, rows: Sequence[Dict[str, Any]]) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return path


def digest_of(path: str) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            hasher.update(block)
    return hasher.hexdigest()


def read_eval_questions(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def overlap_statistics(examples) -> Dict[str, Any]:
    """Lexical overlap between question and evidence, by role.

    Reported because the construction has a known artifact: a positive drawn
    from the question's own document shares vocabulary with it more than a real
    retrieved passage would. Making that visible is the point -- a filter that
    learns word overlap rather than helpfulness would show up here first.
    """
    buckets: Dict[str, List[float]] = {}
    for example in examples:
        question_tokens = set(tokenize(example.question_text))
        evidence_tokens = set(tokenize(example.chunk.text))
        union = question_tokens | evidence_tokens
        jaccard = (len(question_tokens & evidence_tokens) / len(union)) if union else 0.0
        buckets.setdefault(example.role, []).append(jaccard)
    return {
        role: {
            "n": len(values),
            "mean_jaccard": round(sum(values) / len(values), 4),
            "min": round(min(values), 4),
            "max": round(max(values), 4),
        }
        for role, values in sorted(buckets.items())
    }


def label_counts(examples) -> Dict[str, int]:
    counter = Counter(example.label for example in examples)
    return {LABEL_HELPFUL: counter.get(LABEL_HELPFUL, 0),
            LABEL_NOT_HELPFUL: counter.get(LABEL_NOT_HELPFUL, 0)}


def render_report(manifest: Dict[str, Any]) -> str:
    """The human-readable quality report, rendered from the manifest.

    Generated rather than written by hand so the prose and the numbers cannot
    drift apart: every figure below is read out of the same dict that is
    serialised to manifest.json.
    """
    counts = manifest["counts"]
    balance = manifest["class_balance"]
    selection = manifest["selection"]
    leakage = manifest["evaluation_leakage"]
    overlap = manifest["statistics"]["question_evidence_overlap"]
    corpus = manifest["source_corpus"]
    construction = manifest["construction"]
    provenance = manifest["label_provenance"]

    def row(label: str, value: Any) -> str:
        return f"| {label} | {value} |"

    lines: List[str] = [
        f"# Alzheimer filter training dataset -- quality report",
        "",
        f"`{manifest['dataset_name']}` (`{manifest['dataset_version']}`), built "
        f"{manifest['created_utc']} by `{manifest['created_by']}`.",
        "",
        "## 1. Label provenance -- read this first",
        "",
        f"**Label class: {provenance['class'].replace('_', ' ').upper()}.**",
        "",
        provenance["statement"],
        "",
        f"- human-validated: **{provenance['human_validated']}** "
        f"(subset prepared at `{provenance['human_validation_subset']}`, "
        f"completed: {provenance['human_validation_completed']})",
        f"- generated by a language model: **{provenance['llm_generated']}**",
        f"- gold: **{provenance['gold']}**",
        "",
        "Known failure modes of this construction:",
        "",
    ]
    lines += [f"{i}. {mode}" for i, mode in enumerate(provenance["known_failure_modes"], 1)]
    lines += [
        "",
        "## 2. Size and class balance",
        "",
        "| quantity | value |",
        "| --- | --- |",
        row("questions", counts["questions"]),
        row("examples (question, evidence, label)", counts["examples_total"]),
        row("training examples", counts["examples_train"]),
        row("validation examples", counts["examples_validation"]),
        row("training questions", counts["questions_train"]),
        row("validation questions", counts["questions_validation"]),
        row("positives `[HELPFUL]`", counts["label_counts_total"]["[HELPFUL]"]),
        row("negatives `[NOT_HELPFUL]`", counts["label_counts_total"]["[NOT_HELPFUL]"]),
        row("&nbsp;&nbsp;of which hard negatives", counts["hard_negatives"]),
        row("&nbsp;&nbsp;of which easy negatives", counts["easy_negatives"]),
        row("positive fraction (all / train / validation)",
            f"{balance['positive_fraction_total']} / {balance['positive_fraction_train']} / "
            f"{balance['positive_fraction_validation']}"),
        row("examples per question", manifest["statistics"]["examples_per_question"]),
        "",
        "## 3. Sources",
        "",
        "| quantity | value |",
        "| --- | --- |",
        row("corpus file (read-only)", f"`{corpus['chunks_file']}`"),
        row("corpus sha256", f"`{corpus['chunks_file_sha256'][:32]}...`"),
        row("corpus layer", corpus["layer"]),
        row("domain filter", f"`{corpus['domain_filter']}`"),
        row("Alzheimer chunks available", corpus["domain_chunks"]),
        row("Alzheimer documents available", corpus["domain_documents"]),
        row("source documents used (one per question)", counts["unique_source_documents"]),
        row("documents contributing evidence", counts["unique_evidence_documents"]),
        row("distinct chunks used", counts["unique_chunks"]),
        row("distinct PMIDs", counts["unique_pmids"]),
        row("distinct PMCIDs", counts["unique_pmcids"]),
        "",
        f"The corpus was **not modified**: `corpus_modified: {construction['corpus_modified']}`, "
        f"`index_used: {construction['index_used']}`. "
        + construction["bm25_note"],
        "",
        "## 4. How each part was constructed",
        "",
        f"- **questions** -- {construction['question_source']}. Frames: "
        + ", ".join(f"`{frame}`" for frame in construction["question_frames"]) + ".",
        f"- **positives** -- {construction['positive_rule']}.",
        f"- **hard negatives** -- {construction['hard_negative_rule']}.",
        f"- **easy negatives** -- {construction['easy_negative_rule']}.",
        f"- **no LLM in the loop**: {construction['no_llm_in_the_loop']}.",
        "",
        "Question construction methods: "
        + ", ".join(f"`{k}` {v}" for k, v in selection["question_construction_methods"].items())
        + ".",
        "",
        "Documents examined and why they were dropped:",
        "",
        "| outcome | documents |",
        "| --- | --- |",
        row("scanned", selection["documents_scanned"]),
        row("stated an objective", selection["documents_with_stated_objective"]),
        row("skipped: no stated objective", selection["skipped_no_stated_objective"]),
        row("skipped: no answer window besides the objective",
            selection["skipped_no_answer_window"]),
        row("skipped: no hard negative available", selection["skipped_no_hard_negative"]),
        row("skipped: duplicate question", selection["skipped_duplicate_question"]),
        row("skipped: matched an evaluation question", selection["skipped_evaluation_leakage"]),
        row("**questions built**", selection["questions_built"]),
        "",
        "## 5. Evaluation leakage",
        "",
        "The 30 questions in `data/datasets/thesis_questions/dev_questions.jsonl` are "
        "**evaluation only**. They were read here for one purpose: to check that none of "
        "them appears in training. No training label was derived from them.",
        "",
        "| check | result |",
        "| --- | --- |",
        row("distinct training questions", leakage["training_questions"]),
        row("evaluation questions checked", leakage["evaluation_questions"]),
        row("exact duplicates (case/punctuation normalised)",
            f"**{leakage['exact_duplicate_count']}**"),
        row("highest token Jaccard against any evaluation question", leakage["max_jaccard"]),
        "",
        "Closest pair found:",
        "",
    ]
    if leakage["closest_pairs"]:
        closest = leakage["closest_pairs"][0]
        lines += [
            f"- training: *{closest['training_question']}*",
            f"- evaluation: *{closest['evaluation_question']}*",
            f"- Jaccard {closest['jaccard']} -- different questions, not a duplicate.",
            "",
        ]
    lines += [
        "## 6. Question/evidence lexical overlap",
        "",
        "Reported because the construction has a known artifact: evidence taken from the "
        "question's own document shares vocabulary with it more than an independently "
        "retrieved passage would. If a filter learns word overlap instead of helpfulness, "
        "this table is where that shows up.",
        "",
        "| role | n | mean Jaccard | min | max |",
        "| --- | --- | --- | --- | --- |",
    ]
    for role, values in overlap.items():
        lines.append(f"| {role} | {values['n']} | {values['mean_jaccard']} | "
                     f"{values['min']} | {values['max']} |")
    lines += [
        "",
        "## 7. Split",
        "",
        f"- rule: **{manifest['split']['rule']}**",
        f"- validation fraction: {manifest['split']['validation_fraction']}",
        f"- random seed: **{manifest['split']['seed']}**",
        f"- train / validation questions: {manifest['split']['questions_train']} / "
        f"{manifest['split']['questions_validation']}, disjoint: "
        f"{manifest['split']['disjoint']}",
        "",
        "Splitting by question rather than by example matters: a per-example split would "
        "put one question's positives in train and its negatives in validation, and the "
        "validation score would then measure memorisation of that question's wording.",
        "",
        "## 8. Files",
        "",
        "| file | sha256 | bytes |",
        "| --- | --- | --- |",
    ]
    for name, entry in sorted(manifest["files"].items()):
        lines.append(f"| `{entry['path']}` | `{entry['sha256'][:16]}...` | {entry['bytes']} |")
    lines += [
        "",
        f"Build took {manifest['build_seconds']}s on Python {manifest['python']}. "
        "`manifest.json` is the machine-readable source of truth for every number above; "
        "this file is generated from it.",
        "",
        "## 9. Reproducing it",
        "",
        "```",
        "python architecture/rag2/scripts/03b_build_alzheimer_filter_labels.py \\",
        f"    --target-questions {manifest['parameters']['target_questions']} \\",
        f"    --seed {manifest['parameters']['seed']}",
        "```",
        "",
        "Deterministic given the same corpus file and seed. If the corpus sha256 above "
        "differs from yours, you are building from a different corpus and the datasets "
        "are not comparable.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--chunks", default=DEFAULT_CHUNKS, help="frozen corpus chunk file (read-only)")
    parser.add_argument("--eval-questions", default=DEFAULT_EVAL,
                        help="the 30 evaluation questions; used ONLY for the leakage check")
    parser.add_argument("--out", default=DEFAULT_OUT, help="output directory")
    parser.add_argument("--target-questions", type=int, default=800)
    parser.add_argument("--positives-per-question", type=int, default=2)
    parser.add_argument("--hard-negatives-per-positive", type=int, default=1)
    parser.add_argument("--easy-negatives-per-positive", type=int, default=1)
    parser.add_argument("--hard-negative-skip", type=int, default=5,
                        help="BM25 ranks skipped before negatives are drawn (false-negative guard)")
    parser.add_argument("--hard-negative-band", type=int, default=25)
    parser.add_argument("--allow-objective-window-positive", action="store_true",
                        help="permit a positive drawn from the abstract window the question "
                             "was read from (off by default: it restates the question)")
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--human-validation-size", type=int, default=120)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset-name", default="alzheimer_corpus_weak_v1")
    args = parser.parse_args()

    started = time.time()
    print(f"corpus      {args.chunks}")
    if not os.path.isfile(args.chunks):
        print(f"ERROR: corpus chunk file not found: {args.chunks}")
        return 2

    chunks = load_domain_chunks(args.chunks, location="abstract")
    documents = group_by_document(chunks)
    print(f"  {len(chunks)} Alzheimer abstract chunks over {len(documents)} documents")

    print("building BM25 index over the frozen corpus (read-only, hard-negative mining only)")
    index = BM25Index(chunks)
    print(f"  {index.n} chunks, {len(index.postings)} terms kept, "
          f"{len(index.dropped_terms)} high-frequency terms dropped")

    evaluation = read_eval_questions(args.eval_questions)
    evaluation_texts = [row["question"] for row in evaluation]
    print(f"evaluation  {len(evaluation_texts)} questions (leakage check only -- never trained on)")

    examples, stats = build_examples(
        documents,
        index,
        evaluation_texts,
        target_questions=args.target_questions,
        positives_per_question=args.positives_per_question,
        hard_negatives_per_positive=args.hard_negatives_per_positive,
        easy_negatives_per_positive=args.easy_negatives_per_positive,
        hard_negative_skip=args.hard_negative_skip,
        hard_negative_band=args.hard_negative_band,
        require_answer_window=not args.allow_objective_window_positive,
        seed=args.seed,
    )
    if not examples:
        print("ERROR: no examples were built")
        return 1

    train, validation = split_by_question(
        examples, validation_fraction=args.validation_fraction, seed=args.seed)

    os.makedirs(args.out, exist_ok=True)
    train_path = os.path.join(args.out, "train.json")
    validation_path = os.path.join(args.out, "validation.json")

    train_pairs = [e.to_labeled_pair(i, args.dataset_name) for i, e in enumerate(train)]
    validation_pairs = [e.to_labeled_pair(i, args.dataset_name) for i, e in enumerate(validation)]
    write_training_file(train_path, train_pairs)
    write_training_file(validation_path, validation_pairs)

    write_jsonl(os.path.join(args.out, "train.provenance.jsonl"),
                [{"id": p.id, **p.provenance} for p in train_pairs])
    write_jsonl(os.path.join(args.out, "validation.provenance.jsonl"),
                [{"id": p.id, **p.provenance} for p in validation_pairs])

    human_rows = build_human_validation_subset(
        examples, size=args.human_validation_size, seed=args.seed)
    human_path = os.path.join(args.out, "human_validation_subset.jsonl")
    write_jsonl(human_path, human_rows)

    leakage = leakage_report(examples, evaluation_texts)
    if leakage["exact_duplicate_count"]:
        print(f"ERROR: {leakage['exact_duplicate_count']} training question(s) duplicate an "
              "evaluation question. Refusing to write a leaking manifest.")
        return 1

    question_ids = sorted({e.qid for e in examples})
    source_documents = sorted({e.need.document_id for e in examples})
    evidence_documents = sorted({e.chunk.document_id for e in examples})
    pmids = sorted({e.chunk.pmid for e in examples if e.chunk.pmid})
    pmcids = sorted({e.chunk.pmcid for e in examples if e.chunk.pmcid})
    chunk_ids = sorted({e.chunk.chunk_id for e in examples})

    train_qids = sorted({e.qid for e in train})
    validation_qids = sorted({e.qid for e in validation})
    assert not (set(train_qids) & set(validation_qids)), "split leaked a question across sides"

    manifest: Dict[str, Any] = {
        "dataset_version": DATASET_VERSION,
        "dataset_name": args.dataset_name,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "created_by": os.path.relpath(os.path.abspath(__file__), REPO_ROOT).replace(os.sep, "/"),
        "python": platform.python_version(),

        "label_provenance": {
            "class": "weak_supervision",
            "human_validated": False,
            "llm_generated": False,
            "gold": False,
            "statement": (
                "Labels are WEAKLY SUPERVISED. They were derived deterministically from "
                "corpus structure: a passage from the document whose stated objective the "
                "question was taken from is labelled [HELPFUL]; passages from other "
                "documents, drawn below the top of a BM25 ranking, are labelled "
                "[NOT_HELPFUL]. No human judged these pairs and no language model was "
                "asked whether a passage is relevant. They are NOT gold labels and must "
                "not be reported as such."
            ),
            "known_failure_modes": [
                "a hard negative may genuinely answer the question (false negative); the "
                "top {} BM25 ranks are skipped to reduce this, not to eliminate it".format(
                    args.hard_negative_skip),
                "a positive is helpful by construction rather than by inspection",
                "positives share a document with the question, so their lexical overlap "
                "with it exceeds that of a genuinely retrieved passage; see "
                "statistics.question_evidence_overlap",
            ],
            "human_validation_subset": os.path.relpath(human_path, REPO_ROOT).replace(os.sep, "/"),
            "human_validation_completed": False,
        },

        "construction": {
            "method": "corpus_grounded_deterministic",
            "question_source": "source document's own title (when interrogative) or "
                               "OBJECTIVE/AIM statement, authors' wording preserved",
            "question_frames": list(QUESTION_FRAMES),
            "positive_rule": (
                "abstract window of the question's own source document, excluding the "
                "window the objective was read from (the later windows carry results and "
                "conclusions, so the positive answers the question rather than restating it)"
                if not args.allow_objective_window_positive else
                "abstract chunk of the question's own source document, preferring a window "
                "other than the one the objective was read from"),
            "hard_negative_rule": f"BM25 ranks {args.hard_negative_skip + 1}"
                                  f"-{args.hard_negative_skip + args.hard_negative_band} over the "
                                  "same Alzheimer corpus, excluding the source document",
            "easy_negative_rule": "seeded random chunk sharing no distinctive term with the question",
            "no_llm_in_the_loop": True,
            "corpus_modified": False,
            "index_used": False,
            "retrieval_model_used": None,
            "bm25_note": "BM25 is used only to mine negatives. It never scores an evaluation "
                         "question and never produces a candidate either arm sees; MedCPT "
                         "remains the sole retrieval model of the experiment.",
        },

        "source_corpus": {
            "chunks_file": os.path.relpath(os.path.abspath(args.chunks), REPO_ROOT).replace(os.sep, "/"),
            "chunks_file_sha256": digest_of(args.chunks),
            "layer": "abstract",
            "domain_filter": "'alzheimer' in title+text (case-insensitive)",
            "domain_chunks": len(chunks),
            "domain_documents": len(documents),
            "read_only": True,
        },

        "counts": {
            "questions": len(question_ids),
            "examples_total": len(examples),
            "examples_train": len(train),
            "examples_validation": len(validation),
            "questions_train": len(train_qids),
            "questions_validation": len(validation_qids),
            "positives": stats.role_counts.get("positive", 0),
            "hard_negatives": stats.role_counts.get("hard_negative", 0),
            "easy_negatives": stats.role_counts.get("easy_negative", 0),
            "label_counts_total": label_counts(examples),
            "label_counts_train": label_counts(train),
            "label_counts_validation": label_counts(validation),
            "unique_source_documents": len(source_documents),
            "unique_evidence_documents": len(evidence_documents),
            "unique_chunks": len(chunk_ids),
            "unique_pmids": len(pmids),
            "unique_pmcids": len(pmcids),
        },

        "class_balance": {
            "positive_fraction_total": round(
                label_counts(examples)[LABEL_HELPFUL] / len(examples), 4),
            "positive_fraction_train": round(
                label_counts(train)[LABEL_HELPFUL] / len(train), 4) if train else 0.0,
            "positive_fraction_validation": round(
                label_counts(validation)[LABEL_HELPFUL] / len(validation), 4) if validation else 0.0,
        },

        "split": {
            "rule": "by question, never by example",
            "validation_fraction": args.validation_fraction,
            "seed": args.seed,
            "questions_train": len(train_qids),
            "questions_validation": len(validation_qids),
            "disjoint": True,
        },

        "selection": {
            "documents_scanned": stats.documents_scanned,
            "documents_with_stated_objective": stats.documents_with_need,
            "questions_built": stats.questions_built,
            "skipped_no_stated_objective": stats.skipped_no_need,
            "skipped_duplicate_question": stats.skipped_duplicate_question,
            "skipped_evaluation_leakage": stats.skipped_leakage,
            "skipped_no_hard_negative": stats.skipped_no_hard_negative,
            "skipped_no_answer_window": stats.skipped_no_answer_window,
            "question_construction_methods": dict(sorted(stats.method_counts.items())),
        },

        "evaluation_leakage": leakage,

        "statistics": {
            "question_evidence_overlap": overlap_statistics(examples),
            "examples_per_question": round(len(examples) / len(question_ids), 3),
        },

        "parameters": vars(args),

        "files": {},
        "build_seconds": None,
    }

    by_qid: Dict[str, List[Any]] = {}
    for example in examples:
        by_qid.setdefault(example.qid, []).append(example)
    train_qid_set = set(train_qids)
    train_ids_path = write_jsonl(
        os.path.join(args.out, "question_index.jsonl"),
        [{"qid": qid,
          "split": "train" if qid in train_qid_set else "validation",
          "question": by_qid[qid][0].question_text,
          "construction_method": by_qid[qid][0].need.method,
          "source_document_id": by_qid[qid][0].need.document_id,
          "source_chunk_id": by_qid[qid][0].need.source_chunk_id,
          "source_pmid": next((e.chunk.pmid for e in by_qid[qid] if e.role == "positive"), ""),
          "source_pmcid": next((e.chunk.pmcid for e in by_qid[qid] if e.role == "positive"), ""),
          "evidence_chunk_ids": sorted(e.chunk.chunk_id for e in by_qid[qid])}
         for qid in question_ids])

    manifest["build_seconds"] = round(time.time() - started, 1)
    for name, path in (("train", train_path), ("validation", validation_path),
                       ("question_index", train_ids_path),
                       ("human_validation_subset", human_path)):
        manifest["files"][name] = {
            "path": os.path.relpath(path, REPO_ROOT).replace(os.sep, "/"),
            "sha256": digest_of(path),
            "bytes": os.path.getsize(path),
        }

    manifest_path = write_json(os.path.join(args.out, "manifest.json"), manifest)
    manifest["dataset_digest"] = digest_of(manifest_path)
    write_json(manifest_path, manifest)

    report_path = os.path.join(args.out, "TRAINING_DATA_REPORT.md")
    with open(report_path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(render_report(manifest))

    print()
    print(f"questions            {len(question_ids)}")
    print(f"examples             {len(examples)}  "
          f"(train {len(train)} / validation {len(validation)})")
    print(f"labels               {label_counts(examples)}")
    print(f"positive fraction    {manifest['class_balance']['positive_fraction_total']}")
    print(f"leakage              {leakage['exact_duplicate_count']} exact, "
          f"max jaccard {leakage['max_jaccard']}")
    print(f"labels are           WEAK SUPERVISION (not human-validated)")
    print(f"written to           {args.out}")
    print(f"quality report       {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
