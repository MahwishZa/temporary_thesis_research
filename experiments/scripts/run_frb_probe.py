#!/usr/bin/env python3
"""Execute the Filter Recency-Bias Probe -- the thesis's primary contribution.

    python experiments/scripts/run_frb_probe.py \
        --dataset data/datasets/medchangeqa/medchangeqa.jsonl \
        --checkpoint architecture\\rag2\\runs\\filter-alz \
        --equivalence-band 0.05

This reports **H1** (does the filter admit older evidence preferentially?) and
**H4** (does that asymmetry vanish when the dates are permuted?), which proposal
5.4 names as the Minimum Viable Implementation -- "a complete and defensible
thesis should all later phases fail".

Two things it will refuse to do
-------------------------------
* build pairs from thesis-curated material. Proposal 5.2 puts a provenance
  firewall around the primary claim: it must rest on MedChangeQA, externally
  authored, so a positive result is not unfalsifiable by construction;
* report H1 when the permutation control fails. Proposal 4.6 lists V1 as
  BLOCKING, not advisory.

``--dry-run`` builds and validates the pairs, loads nothing and scores nothing,
so pair construction can be checked on a machine with no GPU and no checkpoint.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _path in (_ROOT, os.path.join(_ROOT, "architecture"),
              os.path.join(_ROOT, "architecture", "rag2")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from experiments.recency_bias.frb_pairs import (  # noqa: E402
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    DEFAULT_LENGTH_TOLERANCE,
    ProvenanceViolation,
    UnusableDataset,
    batched_admit_fn,
    build_pairs,
    digest_pairs,
    load_dataset,
    run_probe,
)

DEFAULT_OUT = os.path.join(_ROOT, "experiments", "results", "recency_bias")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True,
                        help="MedChangeQA in JSONL or JSON-array form")
    parser.add_argument("--provenance", default="MedChangeQA (Vladika et al., "
                                                "Findings of EMNLP 2025)")
    parser.add_argument("--field-map", default="",
                        help='JSON, e.g. \'{"older_text": "abstract_old"}\' -- '
                             "only the columns whose names differ")
    parser.add_argument("--checkpoint", default="",
                        help="trained RAG2 filter checkpoint (Flan-T5-large)")
    parser.add_argument("--length-tolerance", type=float,
                        default=DEFAULT_LENGTH_TOLERANCE)
    parser.add_argument("--equivalence-band", type=float, default=0.05,
                        help="H4 band, PRE-SPECIFIED before the run (7.3)")
    parser.add_argument("--resamples", type=int, default=BOOTSTRAP_RESAMPLES)
    parser.add_argument("--seed", type=int, default=BOOTSTRAP_SEED)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--dry-run", action="store_true",
                        help="construct and validate pairs only; load no model")
    args = parser.parse_args(argv)

    field_map = json.loads(args.field_map) if args.field_map else None
    try:
        items = load_dataset(args.dataset, field_map, provenance=args.provenance)
    except ProvenanceViolation as exc:
        print(f"REFUSING: {exc}")
        return 2
    except UnusableDataset as exc:
        print(f"REFUSING: {exc}")
        return 2

    built = build_pairs(items, length_tolerance=args.length_tolerance)
    pairs = built["pairs"]
    print(f"-- pair construction {'-' * 50}")
    print(f"  dataset items            {len(items)}")
    print(f"  pairs constructed        {built['constructed']}")
    for reason, count in sorted(built["excluded"].items()):
        print(f"    excluded: {reason}: {count}")
    print(f"  target {built['target_range'][0]}-{built['target_range'][1]}: "
          f"{'MET' if built['meets_target'] else 'NOT MET'}")
    print(f"  pair digest              {digest_pairs(pairs)}")
    if built["below_target"]:
        print("  WARNING: below the proposal's target pair count. A modest "
              "asymmetry may fail to clear a conventional threshold; report an "
              "inconclusive result as inconclusive (proposal 8).")

    if args.dry_run:
        print("\n--dry-run: pairs only. No model loaded, nothing scored, "
              "nothing written.")
        return 0

    if not args.checkpoint:
        print("\nREFUSING: --checkpoint is required to score. The probe needs "
              "the trained RAG2 filter; it is the object under test, and no "
              "untrained substitute measures the same thing.")
        return 2

    # Imported here so --dry-run needs neither torch nor a checkpoint.
    from rag2.config import FilterConfig  # noqa: E402
    from rag2.filtering.rag2_filter import RAG2PerplexityFilter  # noqa: E402
    from rag2.prompts import DEFAULT_PROMPTS  # noqa: E402
    from rag2.schema import Question  # noqa: E402

    scorer = RAG2PerplexityFilter(
        FilterConfig(kind="rag2_perplexity", checkpoint=args.checkpoint))
    print(f"\n-- scoring {len(pairs) * 2} passages {'-' * 42}")

    # Rendered through the baseline's own filter prompt, so the probe measures
    # the filter as the pipeline uses it rather than a paraphrase of it.
    def _render(question_text: str, passage: str) -> str:
        return DEFAULT_PROMPTS.render_filter_prompt(
            Question(qid="frb", question=question_text, options={}, answer=None,
                     metadata={}),
            passage)

    admit_fn, distinct = batched_admit_fn(
        pairs, _render, scorer.score_pairs, batch_size=args.batch_size,
        progress=lambda done, total: print(f"  scored {done}/{total}", end="\r"))
    print(f"  scored {distinct} distinct passages")

    result = run_probe(pairs, admit_fn, args.equivalence_band,
                       resamples=args.resamples, seed=args.seed)

    payload = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dataset": os.path.basename(args.dataset),
        "provenance": args.provenance,
        "checkpoint": args.checkpoint,
        "pair_construction": {k: v for k, v in built.items() if k != "pairs"},
        "equivalence_band": args.equivalence_band,
        "seed": args.seed,
        "resamples": args.resamples,
        **result,
    }
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, "frb_probe.json")
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")

    delta = result["H1_primary"]["delta"]
    control = result["H4_permutation_control"]["delta"]
    print(f"\n-- results {'-' * 58}")
    print(f"  H1  Delta = {delta['point']:+.4f}  "
          f"95% CI [{delta['ci_low']:+.4f}, {delta['ci_high']:+.4f}]  "
          f"p = {delta['p_value']:.4f}")
    print(f"      {result['H1_primary']['direction']}")
    print(f"  H4  permuted Delta = {control['point']:+.4f}  "
          f"95% CI [{control['ci_low']:+.4f}, {control['ci_high']:+.4f}]")
    print(f"      equivalence band +/-{args.equivalence_band}: "
          f"{'PASS' if result['H4_equivalence']['passed'] else 'FAIL'}")
    if not result["reportable"]:
        print(f"\n  H1 IS NOT REPORTABLE: {result['H4_equivalence']['reason']}")
    print(f"\nwritten: {path}")
    return 0 if result["reportable"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
