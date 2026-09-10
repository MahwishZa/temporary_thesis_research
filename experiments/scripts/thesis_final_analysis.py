#!/usr/bin/env python3
"""Build the final thesis results package from the frozen artifacts.

    python experiments/scripts/thesis_final_analysis.py

Writes into experiments/results/rag2_vs_scaf_alzheimer/analysis/:

    THESIS_FINAL_RESULTS.md    for a person to read
    thesis_final_results.json  the same content, structured
    thesis_ablations.json      counterfactual re-scoring, on its own
    thesis_statistics.json     clustered intervals, on their own

Read-only with respect to every scientific artifact. It performs no retrieval,
no reranking, no generation, no annotation and no sampling: it re-scores the
admission arithmetic that the completed run already recorded, computes intervals
that respect the question clustering, applies the reportability gates and
renders. Rerun it whenever the analysis is intentionally repeated -- the
provenance block records which inputs produced the package, so a stale copy can
always be told from a current one.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from experiments.analysis.scaf_ablations import (  # noqa: E402
    NotTheScientificRun,
    build_ablations,
    load_records,
)
from experiments.analysis.thesis_statistics import (  # noqa: E402
    BOOTSTRAP_RESAMPLES,
    build_statistics,
)
from experiments.analysis.thesis_report import (  # noqa: E402
    ANNOTATION_SHEET_SHA256,
    CORRECTED_ANNOTATION_PASS,
    PRODUCTION_INDEX_DIGEST,
    SCIENTIFIC_FROZEN_DIGEST,
    answer_the_thesis_questions,
    reportability,
    sha256,
)
from experiments.analysis.render_thesis_final import render_markdown  # noqa: E402

RESULTS = os.path.join(_ROOT, "experiments", "results", "rag2_vs_scaf_alzheimer")
DEFAULT_PER_QUESTION = os.path.join(RESULTS, "comparison_scientific", "per_question.jsonl")
DEFAULT_RUN_MANIFEST = os.path.join(RESULTS, "comparison_scientific", "manifest.json")
DEFAULT_SHEET = os.path.join(RESULTS, "evidence_quality", "annotation_sheet_v2.jsonl")
DEFAULT_READINESS = os.path.join(RESULTS, "analysis", "matched_k5_readiness.json")
DEFAULT_MATCHED_K = os.path.join(RESULTS, "matched_k5", "manifest.json")
DEFAULT_OUT = os.path.join(RESULTS, "analysis")

#: Each entry begins with its own negation, so that no bullet can be quoted out
#: of this list and read as an assertion. That is the point of the list.
FORBIDDEN_CLAIMS = [
    "Do NOT claim that SCAF is superior to RAG² because it admits more "
    "passages — admission volume is not evidence quality, and 96% admission is "
    "close to no filtering at all.",
    "Do NOT claim that SCAF improves evidence quality — its association with "
    "independent human labels has a 95% interval that includes zero.",
    "Do NOT claim that SCAF improves answer accuracy — no answer-level "
    "correctness label exists for any of the 30 questions.",
    "Do NOT claim that RAG² exhibits recency bias — the probe that would test "
    "this was never built, and the four admissions it made are all recent, if "
    "anything pointing the other way.",
    "Do NOT claim that the human labels validate the SCAF scoring formula — "
    "they were collected independently of it, which is what makes them usable, "
    "and they do not corroborate it.",
    "Do NOT claim that a passage is factually correct because the annotator "
    "marked it relevant — the labels are passage-level usefulness judgements, "
    "not correctness judgements.",
    "Do NOT claim that the matched-k results prove superiority — different "
    "evidence producing different answers is not better answers.",
    "Do NOT claim that this reproduction matches the RAG² paper's reported "
    "accuracy — no accuracy was measured, and the reproduction is structural.",
]


def remaining_work(matched_k_generated: bool) -> list:
    return [
        {"work": "Run the Filter Recency-Bias Probe for H1 + H4 — the "
                 "proposal's primary contribution C1 and its Minimum Viable "
                 "Implementation (§5.4). Pipeline implemented and tested; "
                 "needs data.",
         "blocker": "MedChangeQA is not in the repository and this container "
                    "cannot reach huggingface.co (the proxy returns 403). "
                    "Everything else it needs — pair construction, the "
                    "permutation control, the estimator, and the "
                    "already-trained filter — is in place.",
         "where": "Student's Windows machine: download MedChangeQA, then "
                  "`python experiments/scripts/run_frb_probe.py --dataset <path> "
                  "--checkpoint architecture\\rag2\\runs\\filter-alz "
                  "--equivalence-band 0.05`. Validate pair construction "
                  "anywhere first with --dry-run."},
        {"work": "Train the second filter on the entailment label (ablation A1 "
                 "and H2, the primary causal claim)",
         "blocker": "needs a GPU and the entailment teacher; only the "
                    "perplexity-labelled filter exists. H1 and H4 do not "
                    "depend on this.",
         "where": "Student's Windows machine (RTX 2050)"},
        {"work": "Matched-k = 5 answer generation (60 answers)",
         "blocker": ("already generated" if matched_k_generated else
                     "needs the scientific frozen candidate TEXT and the local "
                     "Ollama generator; this container has neither"),
         "where": "Student's Windows machine"},
        {"work": "Blind pairwise answer evaluation",
         "blocker": "requires the matched-k answers to exist first",
         "where": "Anywhere, once the answers exist"},
        {"work": "Replace σ with an entailment model (the largest "
                 "proposal-to-implementation gap)",
         "blocker": "needs a trained NLI model; scoped as future work by the "
                    "implementation's own docstring",
         "where": "Student's Windows machine"},
    ]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--per-question", default=DEFAULT_PER_QUESTION)
    parser.add_argument("--run-manifest", default=DEFAULT_RUN_MANIFEST)
    parser.add_argument("--sheet", default=DEFAULT_SHEET)
    parser.add_argument("--readiness", default=DEFAULT_READINESS)
    parser.add_argument("--matched-k", default=DEFAULT_MATCHED_K)
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--resamples", type=int, default=BOOTSTRAP_RESAMPLES)
    args = parser.parse_args(argv)

    for path in (args.per_question, args.run_manifest, args.sheet):
        if not os.path.isfile(path):
            print(f"ERROR: not found: {path}")
            return 2

    annotations = [json.loads(line) for line in open(args.sheet, encoding="utf-8")
                   if line.strip()]
    passes = sorted({r.get("annotation_pass") for r in annotations})
    if passes != [CORRECTED_ANNOTATION_PASS]:
        print(f"REFUSING: the annotation sheet is not the corrected independent "
              f"pass. Found {passes}, expected ['{CORRECTED_ANNOTATION_PASS}'].")
        return 1
    if any(r.get("ai_suggestion_shown") for r in annotations):
        print("REFUSING: a machine suggestion was shown to the annotator in this "
              "sheet. That is the anchored pilot, not the independent pass.")
        return 1

    try:
        ablations = build_ablations(args.per_question)
    except NotTheScientificRun as exc:
        print(f"REFUSING: {exc}")
        return 1

    records = load_records(args.per_question)
    statistics = build_statistics(records, annotations, resamples=args.resamples)

    matched_k_generated = os.path.isfile(args.matched_k)
    readiness = {}
    if os.path.isfile(args.readiness):
        with open(args.readiness, encoding="utf-8") as handle:
            readiness = json.load(handle)
    matched_k_ready = bool(readiness.get("selection_invariants", {}).get("all_passed"))

    run_manifest = json.load(open(args.run_manifest, encoding="utf-8"))
    payload = {
        "built_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_run": os.path.relpath(args.per_question, _ROOT).replace("\\", "/"),
        "identity": {
            "scientific frozen candidate digest": SCIENTIFIC_FROZEN_DIGEST,
            "run manifest records": run_manifest.get("frozen_set_digest"),
            "production MedCPT index digest": PRODUCTION_INDEX_DIGEST,
            "annotation pass": CORRECTED_ANNOTATION_PASS,
            "annotation sheet sha256": ANNOTATION_SHEET_SHA256,
            "generator": (run_manifest.get("config", {})
                          .get("arm_a_generation", {}).get("model")),
            "prompt fingerprint": (run_manifest.get("config", {})
                                   .get("arm_a_generation", {})
                                   .get("prompt_fingerprint")),
            "source run git commit": (run_manifest.get("environment", {})
                                      .get("git_commit")),
        },
        "inputs": {os.path.relpath(p, _ROOT).replace("\\", "/"): sha256(p)
                   for p in (args.per_question, args.run_manifest, args.sheet)},
        "ablations": ablations,
        "statistics": statistics,
        "reportability": reportability(ablations, statistics,
                                       matched_k_ready, matched_k_generated),
        "thesis_questions": answer_the_thesis_questions(ablations, statistics),
        "remaining_work": remaining_work(matched_k_generated),
        "forbidden_claims": FORBIDDEN_CLAIMS,
    }

    if payload["identity"]["run manifest records"] != SCIENTIFIC_FROZEN_DIGEST:
        print("REFUSING: the run manifest does not record the scientific frozen "
              f"candidate digest (found {payload['identity']['run manifest records']}).")
        return 1

    os.makedirs(args.out, exist_ok=True)
    for name, content in (("thesis_ablations.json", ablations),
                          ("thesis_statistics.json", statistics),
                          ("thesis_final_results.json", payload)):
        with open(os.path.join(args.out, name), "w", encoding="utf-8",
                  newline="\n") as handle:
            json.dump(content, handle, indent=2, ensure_ascii=False, sort_keys=True)
            handle.write("\n")
    md_path = os.path.join(args.out, "THESIS_FINAL_RESULTS.md")
    with open(md_path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(render_markdown(payload))

    admission = statistics["admission"]
    quality = statistics["evidence_quality"]["rank_association"]
    print(f"candidates            {admission['candidates']} in "
          f"{admission['questions']} questions")
    print(f"admission  RAG2 {admission['rag2_admission_rate']:.4f}   "
          f"SCAF {admission['scaf_admission_rate']:.4f}   "
          f"difference {admission['difference_clustered']['point']:+.4f} "
          f"[{admission['difference_clustered']['ci_low']:+.4f}, "
          f"{admission['difference_clustered']['ci_high']:+.4f}]")
    print(f"\nassociation with the human usefulness label "
          f"({statistics['resamples']:,} clustered resamples, Holm-corrected):")
    for name, block in sorted(quality.items(), key=lambda kv: -abs(kv[1]["point"] or 0)):
        if block.get("point") is None:
            continue
        print(f"  {name:16s} {block['point']:+.4f} "
              f"[{block['ci_low']:+.4f}, {block['ci_high']:+.4f}]  "
              f"{'SURVIVES HOLM' if block['holm']['survives_holm'] else ''}")
    print(f"\nreportable      {len(payload['reportability']['reportable'])}")
    print(f"not reportable  {len(payload['reportability']['not_reportable'])}")
    for name in payload["reportability"]["not_reportable"]:
        print(f"    - {name}")
    print(f"\nwritten:\n  {md_path}\n  {os.path.join(args.out, 'thesis_final_results.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
