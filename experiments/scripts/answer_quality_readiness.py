#!/usr/bin/env python3
"""Inspect whether an answer-quality experiment can be run yet, and record it.

    python experiments/scripts/answer_quality_readiness.py

Writes two derived artifacts into
experiments/results/rag2_vs_scaf_alzheimer/analysis/:

    answer_quality_readiness.md      for a person to read
    answer_quality_readiness.json    the same findings, structured

Read-only with respect to every scientific artifact. It runs no retrieval, no
generation, no annotation and no experiment; it opens files and reports.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from experiments.analysis.answer_quality_readiness import build_readiness  # noqa: E402
from experiments.analysis.render_readiness import render_markdown  # noqa: E402

DEFAULT_OUT = os.path.join(_ROOT, "experiments", "results", "rag2_vs_scaf_alzheimer",
                           "analysis")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default=_ROOT)
    parser.add_argument("--out", default=DEFAULT_OUT)
    args = parser.parse_args(argv)

    readiness = build_readiness(args.root)

    os.makedirs(args.out, exist_ok=True)
    json_path = os.path.join(args.out, "answer_quality_readiness.json")
    with open(json_path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(readiness, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")
    md_path = os.path.join(args.out, "answer_quality_readiness.md")
    with open(md_path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(render_markdown(readiness))

    q = readiness["question_set"]
    d = readiness["decision"]
    f = readiness["matched_k_feasibility"]
    print(f"questions                 {q['questions']}")
    print(f"gold/reference answers    {q['answer_target_fields'] or 'NONE'}")
    print(f"answer-evaluation code    {readiness['evaluation_code']['exists']} "
          f"({readiness['evaluation_code']['path']})")
    print(f"answers already generated "
          f"{readiness['generated_answers']['arms']['scaf']['answers_present']} SCAF / "
          f"{readiness['generated_answers']['arms']['rag2']['answers_present']} RAG2")
    print(f"matched-k selection ready {f['selection_executable_now']}")
    print(f"\nDECISION: {d['verdict']}")
    print(f"\nwritten:\n  {md_path}\n  {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
