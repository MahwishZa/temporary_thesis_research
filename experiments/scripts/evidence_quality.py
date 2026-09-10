#!/usr/bin/env python3
"""Evidence-quality validation -- one command per step.

The completed comparison established how much evidence each policy admits. It
could not establish whether either admits *better* evidence, because the
repository holds no evidence-quality signal. This tool collects that signal from
one human pass and then tests the question that decides whether SCAF is a viable
extension rather than a differently-calibrated one:

    does a higher SCAF score correspond to better human-judged evidence?

Four steps, in order:

    1  export       build the blind annotation file (do this once)
    2  check        how many labels are filled, and are they valid?
    3  analyse      run the analysis (refuses while labels are missing)
    4  diagnostics  offline matched-k / component / time splits (no labels needed)

Nothing here writes a human label. Step 3 refuses to treat a missing label as a
zero, because a missing judgement and a judgement of "not relevant" are
different facts.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from experiments.analysis.evidence_quality import (  # noqa: E402
    LABELS,
    SCHEMA_VERSION,
    analyse,
    attach_candidate_text,
    check_annotations,
    component_diagnostics,
    load_decisions,
    matched_budget,
    qualitative_cases,
    split_blind_and_key,
    stratified_sample,
    time_sensitivity_split,
)

RESULTS = os.path.join(_ROOT, "experiments", "results", "rag2_vs_scaf_alzheimer")
DEFAULT_PER_QUESTION = os.path.join(RESULTS, "comparison_scientific", "per_question.jsonl")
DEFAULT_FROZEN = os.path.join(_ROOT, "experiments", "runs", "frozen_candidates.jsonl")
DEFAULT_OUT = os.path.join(RESULTS, "evidence_quality")

INSTRUCTIONS = """\
# Evidence-quality annotation -- instructions

You are judging ONE thing: does this passage help answer this question?

You are NOT judging whether the passage is true, whether it is recent, whether
it is from a good journal, or whether the system should have admitted it. Judge
only usefulness as evidence for THIS question.

## The scale

  0  Not relevant -- does not help answer the question.
  1  Partially relevant -- weak, indirect, or background support. It touches the
     topic, or supports part of the question, but does not answer it.
  2  Clearly relevant -- useful supporting evidence. A reader answering the
     question would want to see this passage.

Put the number in the `human_label` field. Leave `human_notes` empty unless
something is worth recording.

## Hard cases -- decide them this way

* **Medically related but off-question.** A passage about Alzheimer's that does
  not bear on what was asked is **0**, not 1. "About the same disease" is not
  relevance.
* **Useful background that does not answer the question.** Definitions,
  epidemiology, general mechanism -- **1**.
* **Contradicts other evidence, or contradicts what you believe.** Judge
  relevance only. A passage that directly addresses the question but reports a
  contrary finding is still **2**. Disagreement is not irrelevance.
* **Partly relevant, partly off-topic.** Judge the part that bears on the
  question. If that part is useful, **2**; if it is thin, **1**.
* **Truncated mid-sentence.** Judge what is there. Do not guess the rest.
* **You genuinely cannot tell.** Choose **1** and say why in `human_notes`.
  Do not leave it blank -- a blank means "not yet done" and blocks the analysis.

## Two rules that protect the result

1. **Do not look up the machine scores.** They are deliberately kept in a
   separate key file. If you judge with the score in view, the labels partly
   measure your agreement with the score, and the whole exercise collapses.
2. **Do not skip rows you find hard.** Skipped hard rows bias the sample toward
   easy ones. Label them 1 with a note.

Roughly 1-2 minutes per row. You can stop and resume; run `check` to see
progress.
"""


def _write_jsonl(path: str, rows: List[Dict[str, Any]]) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return path


def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_json(path: str, payload: Any) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")
    return path


def _provenance(args: argparse.Namespace, extra: Dict[str, Any]) -> Dict[str, Any]:
    frozen_meta_path = f"{os.path.splitext(args.frozen)[0]}.meta.json" \
        if getattr(args, "frozen", None) else ""
    frozen_meta: Dict[str, Any] = {}
    if frozen_meta_path and os.path.exists(frozen_meta_path):
        with open(frozen_meta_path, "r", encoding="utf-8") as handle:
            frozen_meta = json.load(handle)
    return {
        "schema_version": SCHEMA_VERSION,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_per_question": os.path.abspath(getattr(args, "per_question", "")),
        "source_frozen": os.path.abspath(getattr(args, "frozen", "")) if getattr(args, "frozen", "") else None,
        "frozen_set_digest": frozen_meta.get("frozen_set_digest"),
        "retrieval_is_medcpt": (frozen_meta.get("provenance") or {}).get("retrieval_is_medcpt"),
        **extra,
    }


# --------------------------------------------------------------------------
def cmd_export(args: argparse.Namespace) -> int:
    if not os.path.isfile(args.per_question):
        print(f"ERROR: not found: {args.per_question}")
        return 2
    if not os.path.isfile(args.frozen):
        print(f"ERROR: frozen candidate set not found: {args.frozen}\n"
              "  It holds the candidate TEXT, which per_question.jsonl does not.\n"
              "  It is gitignored, so it exists only on the machine that ran the\n"
              "  comparison. Run this command there, or pass --frozen <path>.")
        return 2

    records = load_decisions(args.per_question)
    print(f"decisions loaded      {len(records)}")

    enriched, missing = attach_candidate_text(records, args.frozen)
    if missing:
        print(f"ERROR: {len(missing)} chunk id(s) are absent from {args.frozen}.\n"
              "  That frozen file is not the one the comparison ran against.\n"
              f"  First few: {missing[:5]}\n"
              "  Refusing to export: annotating a different candidate set would\n"
              "  silently answer a different question.")
        return 1
    print(f"candidate text joined {len(enriched)}")

    sample, provenance = stratified_sample(enriched, size=args.size, seed=args.seed)
    blind, key = split_blind_and_key(sample)

    os.makedirs(args.out, exist_ok=True)
    blind_path = _write_jsonl(os.path.join(args.out, "annotation_sheet.jsonl"), blind)
    key_path = _write_jsonl(os.path.join(args.out, "annotation_key.jsonl"), key)
    manifest_path = _write_json(os.path.join(args.out, "sample_manifest.json"),
                                _provenance(args, provenance))
    with open(os.path.join(args.out, "ANNOTATION_INSTRUCTIONS.md"), "w",
              encoding="utf-8", newline="\n") as handle:
        handle.write(INSTRUCTIONS)

    print(f"\nsampled               {provenance['sampled']} of {provenance['population']}")
    print(f"  forced RAG2-admitted  {provenance['forced_rag2_admitted']}")
    print(f"  scaf tertile edges    {provenance['scaf_tertile_edges']}")
    print(f"  rag2 tertile edges    {provenance['rag2_tertile_edges']}")
    print(f"  cells                 {provenance['cell_counts']}")
    print(f"\nTO ANNOTATE ->  {blind_path}")
    print(f"instructions    {os.path.join(args.out, 'ANNOTATION_INSTRUCTIONS.md')}")
    print(f"machine key     {key_path}   (do NOT open while annotating)")
    print(f"manifest        {manifest_path}")
    return 0


def cmd_check(args: argparse.Namespace) -> int:
    path = args.sheet or os.path.join(DEFAULT_OUT, "annotation_sheet.jsonl")
    if not os.path.isfile(path):
        print(f"ERROR: not found: {path}  (run `export` first)")
        return 2
    report = check_annotations(_read_jsonl(path))
    print(f"file        {path}")
    print(f"total rows  {report['total_rows']}")
    print(f"completed   {report['completed']}")
    print(f"missing     {report['missing']}")
    print(f"invalid     {report['invalid']}")
    if report["duplicate_annotation_ids"]:
        print(f"DUPLICATE annotation_ids: {report['duplicate_annotation_ids'][:10]}")
    if report["invalid_entries"]:
        print(f"invalid entries (id, value): {report['invalid_entries']}")
        print(f"  valid labels are {sorted(LABELS)}")
    if report["label_distribution"]:
        print(f"labels so far {report['label_distribution']}")
    print("\nREADY FOR ANALYSIS" if report["ready_for_analysis"]
          else f"\nNOT READY -- {report['missing']} label(s) still to fill")
    return 0


def cmd_analyse(args: argparse.Namespace) -> int:
    sheet = args.sheet or os.path.join(DEFAULT_OUT, "annotation_sheet.jsonl")
    key = args.key or os.path.join(DEFAULT_OUT, "annotation_key.jsonl")
    for path in (sheet, key):
        if not os.path.isfile(path):
            print(f"ERROR: not found: {path}  (run `export` first)")
            return 2

    rows = _read_jsonl(sheet)
    report = check_annotations(rows)
    if not report["ready_for_analysis"] and not args.allow_partial:
        print(f"REFUSING: {report['missing']} missing, {report['invalid']} invalid, "
              f"{len(report['duplicate_annotation_ids'])} duplicate id(s).")
        print("  A missing label is not a label of 0. Finish the sheet, or pass")
        print("  --allow-partial to analyse only the completed rows (and say so).")
        return 1

    key_by_id = {r["annotation_id"]: r for r in _read_jsonl(key)}
    merged = [{**key_by_id.get(r["annotation_id"], {}), **r} for r in rows]

    result = analyse(merged)
    result["completeness"] = report
    result["partial"] = bool(args.allow_partial and not report["ready_for_analysis"])
    out = _write_json(os.path.join(args.out, "evidence_quality_analysis.json"),
                      {"provenance": _provenance(args, {"sheet": sheet, "key": key}),
                       "result": result})

    print(f"labelled rows  {result.get('labelled_rows')}")
    print(f"label spread   {result.get('human_label_distribution')}")
    print("\n-- rank association with the human label (PRIMARY) --")
    for name, value in (result.get("rank_association_with_human_label") or {}).items():
        print(f"   {name:22s} spearman {value['spearman']}  (n={value['n']})")
    print("\n-- mean human label by predeclared score bin --")
    for arm, bins in (result.get("by_score_bin") or {}).items():
        for level in ("low", "medium", "high"):
            if level in bins:
                print(f"   {arm:5s} {level:7s} mean {bins[level]['mean']}  (n={bins[level]['n']})")
    print("\n-- admitted vs rejected (threshold-dependent, SECONDARY) --")
    for arm, sides in (result.get("by_admission") or {}).items():
        print(f"   {arm:5s} admitted {sides['admitted'].get('mean')} (n={sides['admitted']['n']})"
              f"   rejected {sides['rejected'].get('mean')} (n={sides['rejected']['n']})")
    print("\n-- disagreement --")
    for name, stats in (result.get("disagreement_cases") or {}).items():
        print(f"   {name:28s} mean {stats.get('mean')}  (n={stats['n']})")
    print(f"\nwritten to {out}")
    return 0


def cmd_diagnostics(args: argparse.Namespace) -> int:
    if not os.path.isfile(args.per_question):
        print(f"ERROR: not found: {args.per_question}")
        return 2
    records = load_decisions(args.per_question)
    payload = {
        "provenance": _provenance(args, {"decisions": len(records)}),
        "matched_context_budget": matched_budget(records),
        "scaf_component_diagnostics": component_diagnostics(records),
        "time_sensitivity_split": time_sensitivity_split(records),
        "qualitative_cases": qualitative_cases(records),
    }
    out = _write_json(os.path.join(args.out, "offline_diagnostics.json"), payload)

    print("-- matched context budget (fairness guarantee 2: <= 5 passages) --")
    for k, stats in payload["matched_context_budget"]["k"].items():
        print(f"   k={k:>2s}  mean jaccard {stats['mean_jaccard']}   "
              f"shared {stats['shared_passages_total']}/{stats['possible_total']}")
    print("\n-- SCAF component diagnostics (NOT validated ablations) --")
    for name, stats in payload["scaf_component_diagnostics"]["variants"].items():
        print(f"   {name:20s} admitted {stats['admitted']:3d}/{stats['of']}  rate {stats['rate']}")
    print("\n-- time-sensitive split (flag from the committed question set) --")
    for key, stats in payload["time_sensitivity_split"]["groups"].items():
        print(f"   time_sensitive={key:5s} questions {stats['questions']:2d}  "
              f"scaf {stats['scaf_admitted']:3d}  rag2 {stats['rag2_admitted']:2d}  "
              f"mean gamma {stats['mean_gamma_currency']}")
    print(f"\nwritten to {out}")
    print("\nThese are diagnostics of admission behaviour. None of them measures "
          "evidence quality;\nthat requires the human labels.")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p, frozen=False):
        p.add_argument("--per-question", default=DEFAULT_PER_QUESTION)
        p.add_argument("--out", default=DEFAULT_OUT)
        if frozen:
            p.add_argument("--frozen", default=DEFAULT_FROZEN)
        return p

    export = common(sub.add_parser("export", help="build the blind annotation sheet"), frozen=True)
    export.add_argument("--size", type=int, default=120)
    export.add_argument("--seed", type=int, default=42)
    export.set_defaults(func=cmd_export)

    check = sub.add_parser("check", help="annotation completeness")
    check.add_argument("--sheet", default="")
    check.set_defaults(func=cmd_check, per_question=DEFAULT_PER_QUESTION, out=DEFAULT_OUT, frozen="")

    ana = common(sub.add_parser("analyse", help="run the analysis (needs labels)"))
    ana.add_argument("--sheet", default="")
    ana.add_argument("--key", default="")
    ana.add_argument("--allow-partial", action="store_true",
                     help="analyse only completed rows; the output records that it is partial")
    ana.set_defaults(func=cmd_analyse, frozen="")

    diag = common(sub.add_parser("diagnostics", help="offline matched-k / component / time splits"))
    diag.set_defaults(func=cmd_diagnostics, frozen="")

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
