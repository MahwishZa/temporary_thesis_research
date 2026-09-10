#!/usr/bin/env python3
"""Write the thesis-readable results report from the frozen artifacts.

    python experiments/scripts/analyse_results.py

Reads the completed comparison and the completed independent annotation, and
writes two files under experiments/results/rag2_vs_scaf_alzheimer/analysis/:

    scientific_results_analysis.md    for a person to read
    scientific_results_summary.json   the same numbers, for a machine

Both are DERIVED artifacts. Rerun this whenever the analysis is intentionally
repeated; the provenance block records which inputs produced them, so a stale
report can always be told apart from a current one. Nothing here retrieves,
scores, generates, samples or annotates anything, and it writes only inside
that analysis directory.

The prose is rendered from the same dictionary the JSON is dumped from, so a
number in the report and a number in the JSON cannot disagree.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from experiments.analysis.scientific_results import (  # noqa: E402
    NotTheCorrectedPass,
    build_results,
)
from experiments.analysis.render_results import render_markdown  # noqa: E402

RESULTS = os.path.join(_ROOT, "experiments", "results", "rag2_vs_scaf_alzheimer")
DEFAULT_PER_QUESTION = os.path.join(RESULTS, "comparison_scientific", "per_question.jsonl")
DEFAULT_RUN_MANIFEST = os.path.join(RESULTS, "comparison_scientific", "manifest.json")
DEFAULT_SHEET = os.path.join(RESULTS, "evidence_quality", "annotation_sheet_v2.jsonl")
DEFAULT_MANIFEST = os.path.join(RESULTS, "evidence_quality", "sample_manifest.json")
DEFAULT_OUT = os.path.join(RESULTS, "analysis")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--per-question", default=DEFAULT_PER_QUESTION)
    parser.add_argument("--run-manifest", default=DEFAULT_RUN_MANIFEST)
    parser.add_argument("--sheet", default=DEFAULT_SHEET)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--out", default=DEFAULT_OUT)
    args = parser.parse_args(argv)

    for path in (args.per_question, args.run_manifest, args.sheet, args.manifest):
        if not os.path.isfile(path):
            print(f"ERROR: not found: {path}")
            return 2

    try:
        results = build_results(args.per_question, args.sheet, args.manifest,
                                args.run_manifest)
    except NotTheCorrectedPass as exc:
        print(f"REFUSING: {exc}")
        return 1

    os.makedirs(args.out, exist_ok=True)
    json_path = os.path.join(args.out, "scientific_results_summary.json")
    with open(json_path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(results, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")

    md_path = os.path.join(args.out, "scientific_results_analysis.md")
    with open(md_path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(render_markdown(results))

    prov = results["provenance"]
    quality = results["human_evidence_quality"]
    admission = results["admission_behaviour"]["candidate_level"]
    print(f"decisions            {prov['counts']['decisions']}")
    print(f"human annotations    {prov['counts']['human_annotations']} "
          f"({results['join']['joined']} joined)")
    print(f"annotation pass      {', '.join(prov['annotation_pass'])}")
    print(f"sheet sha256         {prov['annotation_sheet_sha256']}")
    print(f"label fingerprint    {prov['label_fingerprint']}")
    print(f"frozen digests agree {prov['digests_agree']}")
    print(f"\nadmission  RAG2 {admission['rag2_admitted']}/{admission['decisions']}   "
          f"SCAF {admission['scaf_admitted']}/{admission['decisions']}")
    print("\nrank association with the human label (descriptive):")
    for name, stats in quality["rank_association"].items():
        print(f"  {name:22s} {stats['spearman']}")
    print(f"\nwritten:\n  {md_path}\n  {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
