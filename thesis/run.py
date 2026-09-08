#!/usr/bin/env python3
"""Entry point for the thesis architecture.

    python -m thesis.run -c configs/thesis/conditions/baseline.yaml
    python -m thesis.run -c configs/thesis/conditions/rag2.yaml \
        -o retrieval.final_top_k=16
    python -m thesis.run --list                     # conditions and temporal policies
    python -m thesis.run --smoke                    # offline wiring check, no weights

Overrides use the same ``section.key=value`` form as RAG2's stage scripts,
parsed by RAG2's own parser, so the two layers behave identically.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from thesis.conditions.base import available_conditions  # noqa: E402
from thesis.config import load_config, merge_overrides  # noqa: E402
from thesis.pipeline import run_pipeline  # noqa: E402
from thesis.recency import PLANNED_POLICIES, available_policies  # noqa: E402


def _list() -> int:
    print("experimental conditions:")
    for name in available_conditions():
        print(f"  {name}")
    print("\ntemporal policies:")
    for name in available_policies():
        planned = PLANNED_POLICIES.get(name)
        suffix = f"   NOT IMPLEMENTED -- {planned}" if planned else ""
        print(f"  {name}{suffix}")
    print(
        "\nA policy marked NOT IMPLEMENTED is a declared interface. Configuring it "
        "validates, and applying it raises: a recency arm must not silently produce "
        "baseline numbers."
    )
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-c", "--config", default="", help="thesis config YAML")
    parser.add_argument("-o", "--override", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--list", action="store_true", help="list conditions and policies")
    parser.add_argument("--smoke", action="store_true",
                        help="offline end-to-end wiring check (no models, no index)")
    parser.add_argument("--dry-run", action="store_true",
                        help="resolve config and open the corpus, then stop")
    args = parser.parse_args(argv)

    if args.list:
        return _list()
    if args.smoke:
        from thesis.smoke import run_smoke

        return run_smoke()
    if not args.config:
        parser.error("-c/--config is required (or use --smoke / --list)")

    config = load_config(args.config, merge_overrides(args.override))
    print(f"condition : {config.condition.name}")
    print(f"policy    : {config.recency.policy}")
    print(f"rag2 config: {config.rag2_config}")

    if args.dry_run:
        from thesis.corpus import CorpusHandle

        handle = CorpusHandle.open(config)
        print(json.dumps(handle.describe(), indent=2, default=str))
        return 0

    outcome = run_pipeline(
        config,
        progress=lambda done, total: print(f"[query] {done}/{total}", flush=True),
    )
    reportable, reasons = outcome.record.is_reportable()
    print(f"\nqueries   : {len(outcome.query_set)}")
    print(f"admitted  : {outcome.report['retrieval']['evidence_admitted']}"
          f" / {outcome.report['retrieval']['evidence_considered']} considered")
    print(f"output    : {outcome.output_dir}")
    print(f"reportable: {reportable}")
    for reason in reasons:
        print(f"    - {reason}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
