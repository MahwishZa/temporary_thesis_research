#!/usr/bin/env python3
"""Stage 2: run both admission policies over one frozen candidate set.

    ARM A   frozen candidates -> RAG2 admission -> context -> (generator) -> answer
    ARM B   frozen candidates -> SCAF admission -> context -> (generator) -> answer

Which RAG2 arm, and what it licenses
------------------------------------
``--rag2-filter rag2_perplexity --rag2-checkpoint <dir>`` is the real baseline:
the paper's Flan-T5 filter trained on perplexity-derived labels. The checkpoint
is not distributed by the RAG2 authors and must be produced first with
``rag2/scripts/03_build_filter_labels.py`` then ``04_train_filter.py``.

``--rag2-filter passthrough`` runs **"RAG2 w/o filter"** -- the paper's own
ablation (Table 4), where balanced retrieval feeds the generator with no
admission step. It is a real, defined configuration, not a stand-in for the
filter, and the manifest records which arm actually ran. Use it to exercise the
comparison before the filter checkpoint exists; do not describe its output as
"RAG2 vs SCAF" without saying which RAG2.

Answer generation is optional. With no generator both arms still produce their
admission decisions and contexts, which is what the first milestone needs.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

_ROOT = Path(__file__).resolve().parent.parent.parent
for _p in (str(_ROOT), str(_ROOT / "rag2")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import scaf  # noqa: E402,F401  (registers the scaf filter)
from rag2.config import FilterConfig  # noqa: E402
from rag2.filtering.base import build_filter  # noqa: E402
from scaf.compare import build_manifest, run_arm  # noqa: E402
from scaf.frozen import load, read_meta  # noqa: E402

DEFAULT_FROZEN = _ROOT / "scaf" / "runs" / "frozen_candidates.jsonl"
DEFAULT_OUT = _ROOT / "scaf" / "runs"


def load_scaf_options(path: Path | None) -> Dict[str, Any]:
    if not path:
        return {}
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Run the RAG2-vs-SCAF comparison.")
    ap.add_argument("--frozen", type=Path, default=DEFAULT_FROZEN)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--rag2-filter", choices=["rag2_perplexity", "passthrough"],
                    default="passthrough",
                    help="'rag2_perplexity' is the real baseline and needs a trained "
                         "checkpoint; 'passthrough' is the paper's 'RAG2 w/o filter' ablation")
    ap.add_argument("--rag2-checkpoint", default="",
                    help="trained Flan-T5 filter, required for --rag2-filter rag2_perplexity")
    ap.add_argument("--scaf-config", type=Path, help="JSON of SCAF options")
    ap.add_argument("--generator", default="none",
                    help="'none' (admission only) or an llm backend name")
    args = ap.parse_args(argv)

    if args.rag2_filter == "rag2_perplexity" and not args.rag2_checkpoint:
        raise SystemExit(
            "--rag2-filter rag2_perplexity needs --rag2-checkpoint.\n"
            "The RAG2 authors do not distribute their trained filter; produce one with\n"
            "  python rag2/scripts/03_build_filter_labels.py -c <config> --candidates <cache>\n"
            "  python rag2/scripts/04_train_filter.py -c <config> --init-tokens\n"
            "  python rag2/scripts/04_train_filter.py -c <config> --train-file <labels> --select\n"
            "Or run the paper's own no-filter ablation with --rag2-filter passthrough."
        )

    frozen_sets = load(str(args.frozen))
    frozen_meta = read_meta(str(args.frozen)) or {}
    provenance = frozen_meta.get("provenance", {})
    print(f"frozen candidates: {len(frozen_sets)} questions from {args.frozen}")
    print(f"  digest           {frozen_meta.get('frozen_set_digest')}")
    print(f"  retrieval        {provenance.get('source')} "
          f"(MedCPT={provenance.get('retrieval_is_medcpt')})")

    rag2_config = FilterConfig(kind=args.rag2_filter, checkpoint=args.rag2_checkpoint)
    rag2_filter = build_filter(rag2_config)
    scaf_options = load_scaf_options(args.scaf_config)
    scaf_filter = build_filter(FilterConfig(kind="scaf", options=scaf_options))

    generator = None                     # answer generation needs model weights
    if args.generator != "none":
        raise SystemExit(
            f"generator backend {args.generator!r} is not wired into this script yet; "
            "this milestone compares admission. Use --generator none.")

    print(f"\nARM A  rag2   filter={args.rag2_filter}")
    arm_a = run_arm("rag2", rag2_filter, frozen_sets, generator)
    print(f"ARM B  scaf   {scaf_filter.describe()['support_method']}")
    arm_b = run_arm("scaf", scaf_filter, frozen_sets, generator)

    config: Dict[str, Any] = {
        "questions_file": provenance.get("questions_file"),
        "frozen_candidates": str(args.frozen),
        "frozen_set_digest": frozen_meta.get("frozen_set_digest"),
        "retrieval_source": provenance.get("source"),
        "retrieval_is_medcpt": provenance.get("retrieval_is_medcpt"),
        "candidate_depth": provenance.get("depth"),
        "arm_a_filter": args.rag2_filter,
        "arm_a_filter_config": {"checkpoint": args.rag2_checkpoint or None},
        "arm_b_filter": "scaf",
        "arm_b_filter_config": scaf_filter.describe(),
        # Identical by construction; fairness_report re-checks them.
        "arm_a_generator": args.generator, "arm_b_generator": args.generator,
        "arm_a_generation": {}, "arm_b_generation": {},
        "arm_a_index": provenance.get("chunks") or provenance.get("cache"),
        "arm_b_index": provenance.get("chunks") or provenance.get("cache"),
    }

    manifest = build_manifest(frozen_sets, arm_a, arm_b, config)
    args.out.mkdir(parents=True, exist_ok=True)

    with (args.out / "per_question.jsonl").open("w", encoding="utf-8", newline="\n") as fh:
        for a, b in zip(arm_a, arm_b):
            frozen = next(f for f in frozen_sets if f.qid == a.qid)
            fh.write(json.dumps({
                "qid": a.qid,
                "question": frozen.question,
                "time_sensitive": frozen.question_metadata.get("time_sensitive"),
                "num_candidates": a.num_candidates,
                "rag2": a.to_dict(),
                "scaf": b.to_dict(),
            }, ensure_ascii=False, sort_keys=True) + "\n")

    with (args.out / "manifest.json").open("w", encoding="utf-8", newline="\n") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True, default=str)
        fh.write("\n")

    fairness = manifest["fairness"]
    print(f"\nFairness: {fairness['passed']}/{fairness['total']} checks passed")
    for check in fairness["checks"]:
        print(f"  [{'PASS' if check['pass'] else 'FAIL'}] {check['check']}")
    if not fairness["all_passed"]:
        print("\nFAILED: the comparison is not valid. Do not report these numbers.")
        return 1

    for label, key in (("ARM A (rag2)", "arm_a_rag2"), ("ARM B (scaf)", "arm_b_scaf")):
        summary = manifest[key]
        print(f"\n{label}")
        print(f"  questions ok/failed  {summary['questions_succeeded']}/{summary['questions_failed']}")
        print(f"  mean admitted        {summary['mean_admitted']} of {summary['mean_candidates']}")
        print(f"  admission rate       {summary['admission_rate']:.3f}")
        print(f"  no-evidence / abstain {summary['questions_with_no_evidence']} / {summary['abstentions']}")

    print(f"\nWritten to {args.out}/  (per_question.jsonl, manifest.json)")
    print("PRELIMINARY / DEVELOPMENT RESULTS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
