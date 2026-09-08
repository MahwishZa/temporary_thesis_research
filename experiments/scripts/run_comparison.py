#!/usr/bin/env python3
"""Stage 2: run both admission policies over one frozen candidate set.

    ARM A   frozen candidates -> RAG2 admission -> context -> (generator) -> answer
    ARM B   frozen candidates -> SCAF admission -> context -> (generator) -> answer

Which RAG2 arm, and what it licenses
------------------------------------
``--rag2-filter rag2_perplexity --rag2-checkpoint <dir>`` is the real baseline:
the paper's Flan-T5 filter trained on perplexity-derived labels. The checkpoint
is not distributed by the RAG2 authors and must be produced first with
``architecture/rag2/scripts/03_build_filter_labels.py`` then ``04_train_filter.py``.

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
for _p in (str(_ROOT / "architecture"), str(_ROOT / "architecture" / "rag2")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import scaf  # noqa: E402,F401  (registers the scaf filter)
from rag2.config import FilterConfig  # noqa: E402
from rag2.filtering.base import build_filter  # noqa: E402
from scaf.compare import (  # noqa: E402
    build_manifest, checkpoint_identity, components_actually_computed, run_arm,
    scientific_report,
)
from scaf.frozen import load, read_meta  # noqa: E402
from scaf.generation import (  # noqa: E402
    SCIENTIFIC_BACKENDS, GeneratorSpec, GeneratorUnavailable, build_generator,
)

DEFAULT_FROZEN = _ROOT / "experiments" / "runs" / "frozen_candidates.jsonl"
DEFAULT_OUT = _ROOT / "experiments" / "runs"


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
                    help="'none' (admission only, no answers) or a backend: "
                         "huggingface | vllm | openai. Uses rag2's own LLM stack "
                         "and answer prompt; one generator object drives both arms")
    ap.add_argument("--generator-model", default="",
                    help="checkpoint override; defaults to the experiment config")
    ap.add_argument("--generator-revision", default="",
                    help="pin the model revision; recorded in the manifest")
    ap.add_argument("--experiment-config", type=Path,
                    default=_ROOT / "experiments" / "configs" / "preliminary_experiment.yaml",
                    help="canonical YAML; supplies generation settings and the seed")
    ap.add_argument("--scientific", action="store_true",
                    help="enforce every precondition for a REPORTABLE comparison and "
                         "abort if any fails: trained RAG2 checkpoint (never passthrough), "
                         "production MedCPT retrieval, active SCAF components, a generator")
    args = ap.parse_args(argv)

    if args.scientific and args.rag2_filter != "rag2_perplexity":
        raise SystemExit(
            "--scientific requires --rag2-filter rag2_perplexity.\n"
            "passthrough is the paper's 'RAG2 w/o filter' ablation, not the RAG2 filter, "
            "and must never back a reported RAG2-vs-SCAF result."
        )

    if args.scientific and args.generator == "none":
        raise SystemExit(
            "--scientific requires a real generator: pass --generator huggingface.\n"
            "A reported RAG2-vs-SCAF result must contain generated answers; "
            "--generator none produces admission decisions only."
        )

    # build_generator() also refuses a non-scientific backend, but only after the
    # filter is built -- which on the GPU machine means loading torch and a
    # multi-GB checkpoint first. Refuse here instead, beside the other cheap
    # checks, so 'stub' fails in the same second that 'none' does.
    if args.scientific and args.generator not in SCIENTIFIC_BACKENDS:
        raise SystemExit(
            f"--scientific will not run with --generator {args.generator!r}.\n"
            f"Allowed: {', '.join(SCIENTIFIC_BACKENDS)}.\n"
            "'stub' is the offline wiring backend and must never produce a "
            "reported answer."
        )

    if args.rag2_filter == "rag2_perplexity" and not args.rag2_checkpoint:
        raise SystemExit(
            "--rag2-filter rag2_perplexity needs --rag2-checkpoint.\n"
            "The RAG2 authors do not distribute their trained filter; produce one with\n"
            "  python architecture/rag2/scripts/03_build_filter_labels.py -c <config> --candidates <cache>\n"
            "  python architecture/rag2/scripts/04_train_filter.py -c <config> --init-tokens\n"
            "  python architecture/rag2/scripts/04_train_filter.py -c <config> --train-file <labels> --select\n"
            "Or run the paper's own no-filter ablation with --rag2-filter passthrough."
        )

    if args.rag2_checkpoint and not Path(args.rag2_checkpoint).is_dir():
        raise SystemExit(
            f"--rag2-checkpoint {args.rag2_checkpoint!r} is not a directory.\n"
            "It must be the trained Flan-T5 filter produced by "
            "architecture/rag2/scripts/04_train_filter.py --select (a HuggingFace model folder "
            "containing config.json and the weights)."
        )

    experiment_config: Dict[str, Any] = {}
    if args.experiment_config and args.experiment_config.exists():
        try:
            import yaml
            experiment_config = yaml.safe_load(
                args.experiment_config.read_text(encoding="utf-8")) or {}
        except ImportError:
            print("  WARNING: PyYAML absent; generation settings fall back to defaults")
    elif args.generator != "none":
        print(f"  WARNING: {args.experiment_config} not found; "
              "generation settings fall back to defaults")

    frozen_sets = load(str(args.frozen))
    frozen_meta = read_meta(str(args.frozen)) or {}
    provenance = frozen_meta.get("provenance", {})
    print(f"frozen candidates: {len(frozen_sets)} questions from {args.frozen}")
    print(f"  digest           {frozen_meta.get('frozen_set_digest')}")
    print(f"  retrieval        {provenance.get('source')} "
          f"(MedCPT={provenance.get('retrieval_is_medcpt')})")

    rag2_config = FilterConfig(kind=args.rag2_filter, checkpoint=args.rag2_checkpoint)
    try:
        rag2_filter = build_filter(rag2_config)
    except ImportError as exc:
        raise SystemExit(
            f"the RAG2 perplexity filter needs torch and transformers "
            f"({type(exc).__name__}: {exc}).\n"
            "  On the GPU machine:\n"
            "    pip install torch==2.4.1+cu121 --index-url "
            "https://download.pytorch.org/whl/cu121\n"
            "    pip install transformers accelerate sentencepiece\n"
            "  The comparison will not substitute a different filter."
        ) from exc
    except OSError as exc:
        raise SystemExit(
            f"could not load the RAG2 filter checkpoint {args.rag2_checkpoint!r}.\n"
            f"  ({type(exc).__name__}: {exc})\n"
            "  It must be a trained Flan-T5 filter directory from "
            "architecture/rag2/scripts/04_train_filter.py --select."
        ) from exc
    except ValueError as exc:
        raise SystemExit(f"the RAG2 filter refused to load: {exc}") from exc
    scaf_options = load_scaf_options(args.scaf_config)
    # Corpus statistics travel with the frozen set, so SCAF's support scorer uses
    # IDF from the same corpus the candidates came from (policy.SupportScorer v2).
    scaf_options.setdefault("document_frequency", provenance.get("document_frequency", {}))
    scaf_options.setdefault("corpus_size", provenance.get("corpus_size", 0))
    scaf_filter = build_filter(FilterConfig(kind="scaf", options=scaf_options))
    if scaf_filter.describe()["support_idf_source"] != "corpus":
        print("  WARNING: SCAF support has no corpus statistics; falling back to "
              "coverage only (re-run the freeze step to emit them)")

    # -- generator -------------------------------------------------------
    # Built ONCE and handed to both arms, so "same generator, same decoding" is a
    # property of the object rather than a claim. Failures here are fatal: the
    # comparison never downgrades to a mock or to a different model.
    generator = None
    generator_described: Dict[str, Any] = {}
    if args.generator != "none":
        spec = GeneratorSpec.from_mapping(experiment_config.get("generation"))
        spec.backend = args.generator
        if args.generator_model:
            spec.model = args.generator_model
        if args.generator_revision:
            spec.revision = args.generator_revision
        spec.seed = int(experiment_config.get("experiment", {}).get("seed", spec.seed))
        try:
            generator = build_generator(spec)
        except GeneratorUnavailable as exc:
            raise SystemExit(f"cannot build the generator:\n{exc}") from exc
        generator_described = generator.describe()
        print(f"  generator        {generator_described['model']} "
              f"({generator_described['backend']}, "
              f"greedy={generator_described['greedy']}, "
              f"max_new_tokens={generator_described['max_new_tokens']})")


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
        "arm_a_checkpoint_identity": checkpoint_identity(args.rag2_checkpoint),
        "arm_b_filter": "scaf",
        "arm_b_filter_config": scaf_filter.describe(),
        # Identical by construction; fairness_report re-checks them.
        # One object drives both arms, so these are the same dict by construction;
        # fairness_report re-checks them anyway.
        "arm_a_generator": args.generator, "arm_b_generator": args.generator,
        "arm_a_generation": generator_described, "arm_b_generation": generator_described,
        "arm_a_index": provenance.get("chunks") or provenance.get("cache"),
        "arm_b_index": provenance.get("chunks") or provenance.get("cache"),
    }

    manifest = build_manifest(frozen_sets, arm_a, arm_b, config)
    manifest["scientific"] = scientific_report(config, frozen_sets, rag2_filter, scaf_filter)
    manifest["scaf_components"] = components_actually_computed(arm_b)
    manifest["reportable"] = bool(
        manifest["fairness"]["all_passed"] and manifest["scientific"]["all_passed"])
    if not manifest["reportable"]:
        manifest["label"] = ("DEVELOPMENT RUN -- NOT A SCIENTIFIC RESULT "
                             "(see 'scientific' for the unmet preconditions)")
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

    science = manifest["scientific"]
    print(f"\nScientific preconditions: {science['passed']}/{science['total']} passed")
    for check in science["checks"]:
        print(f"  [{'PASS' if check['pass'] else 'FAIL'}] {check['check']}"
              + (f"  -- {check['detail']}" if not check['pass'] and check['detail'] else ""))
    if not science["all_passed"]:
        message = ("\nMILESTONE INCOMPLETE: this is a development run, not the "
                   "scientific RAG2-vs-SCAF result.\nUnmet: "
                   + ", ".join(science["failed"]))
        if args.scientific:
            print(message + "\nAborting because --scientific was requested.")
            return 1
        print(message)

    components = manifest["scaf_components"]
    inert = [k for k, v in components.items() if not v["varies"]]
    if inert:
        print(f"\n  WARNING: SCAF component(s) constant across all candidates: {inert}")

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
