#!/usr/bin/env python
"""Load-test a trained filter checkpoint and score it on the validation split.

Training writing files is not evidence that a usable filter exists. A directory
can be present, be the right size, and still be unloadable -- which is exactly
what ``run_classifier.py``'s per-epoch ``accelerator.save_state`` directories
are. This script answers the only question that matters before the comparison
is run: **does the checkpoint load through the same code path the experiment
uses, and how good is it?**

Five checks, all of which must pass:

1. the path is a directory containing ``config.json`` (the discriminator that
   ``is_loadable_checkpoint`` applies);
2. a tokenizer is present;
3. ``RAG2PerplexityFilter`` -- the class the comparison instantiates, not a
   convenience loader -- builds from it;
4. ``[HELPFUL]`` and ``[NOT_HELPFUL]`` are single tokens with distinct ids
   (a filter whose label tokens were lost scores noise);
5. the model produces a probability for every validation pair.

Then it reports validation accuracy through ``rag2.evaluation.filter_metrics``,
plus the admitted fraction at the 0.5 decision boundary -- a filter that keeps
everything, or nothing, is degenerate even at high accuracy, and that has to be
visible before its admission behaviour is compared against SCAF.

    python scripts/verify_filter_checkpoint.py \\
        --checkpoint runs/filter-alzheimer \\
        --validation-file ../../experiments/results/rag2_vs_scaf_alzheimer/training_dataset/validation.json \\
        --out ../../experiments/results/rag2_vs_scaf_alzheimer/filter
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List

_HERE = os.path.dirname(os.path.abspath(__file__))
_RAG2 = os.path.dirname(_HERE)
if _RAG2 not in sys.path:
    sys.path.insert(0, _RAG2)

from rag2.config import FilterConfig, load_config  # noqa: E402
from rag2.evaluation import filter_metrics  # noqa: E402
from rag2.filter_training.train import is_loadable_checkpoint  # noqa: E402
from rag2.prompts import LABEL_HELPFUL, LABEL_NOT_HELPFUL  # noqa: E402

TOKENIZER_FILES = ("tokenizer_config.json", "tokenizer.json", "spiece.model",
                   "sentencepiece.bpe.model", "vocab.json")


def fail(check: str, detail: str) -> Dict[str, Any]:
    return {"check": check, "pass": False, "detail": detail}


def ok(check: str, detail: str = "") -> Dict[str, Any]:
    return {"check": check, "pass": True, "detail": detail}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True, help="trained filter directory")
    parser.add_argument("--validation-file", default="", help="labelled validation JSON")
    parser.add_argument("--config", default="", help="config supplying filter inference settings")
    parser.add_argument("--out", default="", help="directory for the report")
    parser.add_argument("--limit", type=int, default=0, help="score only the first N pairs")
    args = parser.parse_args()

    checks: List[Dict[str, Any]] = []
    report: Dict[str, Any] = {
        "checkpoint": os.path.abspath(args.checkpoint),
        "checked_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    # -- 1. structure ------------------------------------------------------
    if is_loadable_checkpoint(args.checkpoint):
        listing = sorted(os.listdir(args.checkpoint))
        checks.append(ok("checkpoint has config.json", f"{len(listing)} files"))
        report["files"] = [{"name": name,
                            "bytes": os.path.getsize(os.path.join(args.checkpoint, name))}
                           for name in listing
                           if os.path.isfile(os.path.join(args.checkpoint, name))]
    else:
        reason = ("not a directory" if not os.path.isdir(args.checkpoint)
                  else "no config.json -- this looks like an accelerator.save_state "
                       "directory (training state), not a model checkpoint")
        checks.append(fail("checkpoint has config.json", reason))
        report["checks"] = checks
        report["all_passed"] = False
        print(f"FAIL: {reason}")
        _write(args, report)
        return 1

    listing = set(os.listdir(args.checkpoint))
    if listing & set(TOKENIZER_FILES):
        checks.append(ok("tokenizer present", ", ".join(sorted(listing & set(TOKENIZER_FILES)))))
    else:
        checks.append(fail("tokenizer present",
                           "no tokenizer file; the filter cannot encode its input"))

    # -- 2. load through the experiment's own path -------------------------
    filter_config = FilterConfig(checkpoint=args.checkpoint)
    if args.config:
        configured = load_config(args.config, {}).filter
        filter_config = FilterConfig(**{**configured.__dict__, "checkpoint": args.checkpoint})
    report["filter_config"] = {k: v for k, v in filter_config.__dict__.items() if k != "options"}

    try:
        from rag2.filtering.rag2_filter import RAG2PerplexityFilter
        model = RAG2PerplexityFilter(filter_config)
        checks.append(ok("RAG2PerplexityFilter loaded the checkpoint"))
    except Exception as error:  # the whole point of this script
        checks.append(fail("RAG2PerplexityFilter loaded the checkpoint", f"{type(error).__name__}: {error}"))
        report["checks"] = checks
        report["all_passed"] = False
        print(f"FAIL: the checkpoint did not load: {error}")
        _write(args, report)
        return 1

    # -- 3. label tokens ---------------------------------------------------
    tokenizer = getattr(model, "tokenizer", None)
    if tokenizer is None:
        checks.append(fail("label tokens are distinct single ids", "filter exposes no tokenizer"))
    else:
        ids = {label: tokenizer.convert_tokens_to_ids(label)
               for label in (LABEL_HELPFUL, LABEL_NOT_HELPFUL)}
        unknown = getattr(tokenizer, "unk_token_id", None)
        distinct = len(set(ids.values())) == 2 and unknown not in set(ids.values())
        report["label_token_ids"] = ids
        checks.append((ok if distinct else fail)(
            "label tokens are distinct single ids",
            json.dumps(ids) + ("" if distinct else "  -- label tokens were not added to this "
                                                   "checkpoint's vocabulary")))

    # -- 4. inference and validation metrics -------------------------------
    if args.validation_file:
        with open(args.validation_file, "r", encoding="utf-8") as handle:
            records = json.load(handle)
        if args.limit:
            records = records[:args.limit]
        report["validation_file"] = os.path.abspath(args.validation_file)
        report["validation_examples"] = len(records)

        started = time.time()
        probabilities = model.score_pairs([r["question"] for r in records])
        report["scoring_seconds"] = round(time.time() - started, 2)

        if len(probabilities) != len(records):
            checks.append(fail("one probability per pair",
                               f"{len(probabilities)} for {len(records)} pairs"))
        else:
            checks.append(ok("one probability per pair", str(len(probabilities))))

        predictions = [LABEL_HELPFUL if p >= 0.5 else LABEL_NOT_HELPFUL for p in probabilities]
        metrics = filter_metrics([r["answer"] for r in records], predictions)
        kept = sum(1 for p in probabilities if p >= 0.5)
        metrics["kept_fraction"] = round(kept / len(records), 4) if records else 0.0
        metrics["mean_helpful_probability"] = round(sum(probabilities) / len(probabilities), 4) \
            if probabilities else 0.0
        gold_positive = sum(1 for r in records if r["answer"] == LABEL_HELPFUL)
        metrics["gold_positive_fraction"] = round(gold_positive / len(records), 4) if records else 0.0
        metrics["majority_class_baseline"] = round(
            100.0 * max(gold_positive, len(records) - gold_positive) / len(records), 2) if records else 0.0
        report["validation_metrics"] = metrics

        # A filter that keeps everything is passthrough wearing a checkpoint;
        # one that keeps nothing abstains on every question. Either makes the
        # RAG2 arm uninformative, so both are called out here rather than
        # discovered in the comparison.
        degenerate = metrics["kept_fraction"] in (0.0, 1.0)
        checks.append((fail if degenerate else ok)(
            "filter is not degenerate",
            f"keeps {metrics['kept_fraction']:.3f} of validation pairs"
            + (" -- this filter is equivalent to passthrough or to no_evidence"
               if degenerate else "")))
        checks.append((ok if metrics["final_acc_score"] > metrics["majority_class_baseline"] else fail)(
            "beats the majority-class baseline",
            f"{metrics['final_acc_score']:.2f}% vs {metrics['majority_class_baseline']:.2f}%"))

    report["checks"] = checks
    report["all_passed"] = all(c["pass"] for c in checks)

    print(f"\ncheckpoint {args.checkpoint}")
    for check in checks:
        print(f"  [{'PASS' if check['pass'] else 'FAIL'}] {check['check']}"
              + (f"  -- {check['detail']}" if check["detail"] else ""))
    if "validation_metrics" in report:
        metrics = report["validation_metrics"]
        print(f"\n  validation accuracy   {metrics['final_acc_score']:.2f}%  "
              f"(majority baseline {metrics['majority_class_baseline']:.2f}%)")
        print(f"  kept fraction         {metrics['kept_fraction']:.3f}")
        print(f"  per class             {json.dumps(metrics['per_class'])}")
    print(f"\n{'ALL CHECKS PASSED' if report['all_passed'] else 'CHECKS FAILED'}")
    _write(args, report)
    return 0 if report["all_passed"] else 1


def _write(args: argparse.Namespace, report: Dict[str, Any]) -> None:
    if not args.out:
        return
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, "checkpoint_verification.json")
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(report, handle, indent=2, sort_keys=True, default=str)
        handle.write("\n")
    print(f"written to {path}")


if __name__ == "__main__":
    raise SystemExit(main())
