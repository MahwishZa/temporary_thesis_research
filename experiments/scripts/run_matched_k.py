#!/usr/bin/env python3
"""Matched-k=5 answer generation: equal evidence budget, different policy.

    python experiments/scripts/run_matched_k.py --generator openai \
        --generator-model thesis-llama3-8b-q4

The completed admission comparison gave SCAF roughly 688x more context than
RAG2, so its answer difference cannot be attributed to the admission policy.
This run hands **both arms exactly five passages** from the same frozen
candidate set and generates one answer each, so the only thing that varies is
*which* five.

It does not replace the admission comparison. That experiment measured how much
evidence each policy admits, and remains the answer to that question. This one
holds the amount fixed to ask a different one.

What it will refuse to do
-------------------------
* run against any frozen candidate set other than the scientific one
  (digest 316260f0...). A matched-k run built on the lexical-development set is
  not a weaker version of this experiment, it is a different experiment wearing
  its name;
* run with a non-scientific generator backend;
* write a reportable manifest when any fairness invariant fails.

It performs no retrieval, no reranking and no index access: every candidate and
every score is read from the completed run.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.parse
from typing import Any, Dict, List

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for path in (_ROOT, os.path.join(_ROOT, "architecture"),
             os.path.join(_ROOT, "architecture", "rag2")):
    if path not in sys.path:
        sys.path.insert(0, path)

from experiments.analysis.matched_k import (  # noqa: E402
    DEFAULT_K,
    SCIENTIFIC_FROZEN_DIGEST,
    build_selection,
    check_selection_invariants,
    selection_statistics,
    sha256,
    verify_inputs,
)

#: Backends that reach a model over an OpenAI-compatible HTTP API. For these the
#: backend name alone does not say WHICH server answered, so the endpoint has to
#: be checked and recorded.
HTTP_OPENAI_BACKENDS = ("openai",)

#: Hosts that are the commercial OpenAI service (or Azure's hosted OpenAI).
#: Reaching one of these would mean a cloud model wrote thesis answers, which is
#: not the thesis generation setup.
COMMERCIAL_OPENAI_SUFFIXES = (".openai.com", ".openai.azure.com")
COMMERCIAL_OPENAI_HOSTS = ("openai.com",)

RESULTS = os.path.join(_ROOT, "experiments", "results", "rag2_vs_scaf_alzheimer")
DEFAULT_PER_QUESTION = os.path.join(RESULTS, "comparison_scientific", "per_question.jsonl")
DEFAULT_RUN_MANIFEST = os.path.join(RESULTS, "comparison_scientific", "manifest.json")
DEFAULT_FROZEN = os.path.join(_ROOT, "experiments", "runs", "frozen_candidates.jsonl")
DEFAULT_QUESTIONS = os.path.join(_ROOT, "data", "datasets", "thesis_questions",
                                 "dev_questions.jsonl")
DEFAULT_OUT = os.path.join(RESULTS, "matched_k5")

README = """\
# Matched-k = {k} answer generation

Both arms received **exactly {k} passages** from the same frozen candidate set,
so the only difference between them is *which* passages their own score ranked
highest. Generated {answers} answers ({questions} questions x 2 arms).

**This does not replace `comparison_scientific/`.** That run measured how much
evidence each policy naturally admits, and is still the answer to that question.
This run holds the evidence budget fixed at {k} to ask a different one: given an
equal budget, does the choice of evidence differ, and does it change the answer?

## What was NOT done here

No answer-quality judgement of any kind. No gold answers, no reference answers,
no correctness labels, no ROUGE or BERTScore, no human evaluation. The arms are
labelled in this directory, so these files must **not** be shown to a human
evaluator as they stand -- the blind pairwise evaluation is a separate stage
that anonymises them.

## Provenance

    frozen candidate digest   {digest}
    scientific run            comparison_scientific/
    generator                 {model} ({backend})
    prompt fingerprint        {prompt_fingerprint}
    generated                 {generated}
    reportable                {reportable}

`manifest.json` carries the full fairness report. `per_question.jsonl` carries
one record per question with both arms' selected passage ids, the passage text
they were given, and both answers.
"""


def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def check_generation_endpoint(backend: str, base_url: str,
                              api_key: str) -> Dict[str, Any]:
    """Where will `backend: openai` actually send the prompts?

    The scientific run recorded ``backend: openai`` and model
    ``thesis-llama3-8b-q4``. OpenAI serves no such model: the backend was
    pointed at a local OpenAI-compatible server (Ollama) through
    ``OPENAI_BASE_URL``, and the OpenAI SDK honours that variable without any
    code change. Nothing in the repository recorded it, so the manifest said
    "openai" and a reader could not tell a local run from a cloud one.

    Worse, the SDK's default when ``OPENAI_BASE_URL`` is unset is
    ``https://api.openai.com/v1`` -- so a run with the variable missing would
    quietly send thesis questions to a commercial API and produce a manifest
    indistinguishable from the local one. This makes that impossible: the
    endpoint must be present, must not be commercial, and is recorded.

    The API key only has to be non-empty. The local server ignores its value
    (``ollama`` is the conventional placeholder); the SDK merely refuses to
    construct a client without one.
    """
    checks: List[Dict[str, Any]] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        checks.append({"check": name, "pass": bool(ok), "detail": detail})

    if backend not in HTTP_OPENAI_BACKENDS:
        check(f"endpoint check applies to backend {backend!r}", True,
              "not an OpenAI-compatible HTTP backend; no endpoint to verify")
        return {"all_passed": True, "applicable": False, "backend": backend,
                "base_url": None, "checks": checks}

    check("OPENAI_BASE_URL is set", bool(base_url),
          base_url or "unset -- the SDK would default to https://api.openai.com/v1 "
                      "and send thesis questions to a commercial API")

    host = ""
    if base_url:
        host = (urllib.parse.urlparse(base_url).hostname or "").lower()
    commercial = bool(host) and (
        host in COMMERCIAL_OPENAI_HOSTS
        or any(host.endswith(suffix) for suffix in COMMERCIAL_OPENAI_SUFFIXES))
    check("endpoint is NOT the commercial OpenAI service", not commercial,
          f"host {host!r}" + (" -- this is a commercial API, not the thesis "
                              "generation setup" if commercial else ""))

    check("an API key value is present (any placeholder will do)", bool(api_key),
          "set" if api_key else "unset -- the local server ignores the value, but "
                                "the OpenAI SDK will not build a client without "
                                "one. Use a placeholder such as 'ollama'.")

    return {
        "all_passed": all(c["pass"] for c in checks),
        "applicable": True,
        "backend": backend,
        "base_url": base_url or None,
        "endpoint_host": host or None,
        "is_commercial_openai": commercial,
        "api_key_present": bool(api_key),
        "checks": checks,
        "note": ("the API key value is never recorded; only whether one was "
                 "present. The local server ignores it."),
    }


def _preflight(args) -> Dict[str, Any]:
    report = verify_inputs(args.per_question, args.run_manifest,
                           f"{os.path.splitext(args.frozen)[0]}.meta.json",
                           args.questions, k=args.k)
    print("-- pre-generation checks " + "-" * 46)
    for check in report["checks"]:
        print(f"  {'PASS' if check['pass'] else 'FAIL'}  {check['check']}")
        if check["detail"] and not check["pass"]:
            print(f"        {check['detail']}")
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--per-question", default=DEFAULT_PER_QUESTION)
    parser.add_argument("--run-manifest", default=DEFAULT_RUN_MANIFEST)
    parser.add_argument("--frozen", default=DEFAULT_FROZEN)
    parser.add_argument("--questions", default=DEFAULT_QUESTIONS)
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    parser.add_argument("--generator", default="openai",
                        help="generation backend; must be a scientific one")
    parser.add_argument("--generator-model", default="thesis-llama3-8b-q4")
    parser.add_argument("--dry-run", action="store_true",
                        help="run the checks and the selection, generate nothing, "
                             "write nothing")
    parser.add_argument("--document", action="store_true",
                        help="write the status report under analysis/ and exit. "
                             "Records whether this machine can run the experiment, "
                             "with the selection statistics that need no generator")
    parser.add_argument("--analysis-out",
                        default=os.path.join(RESULTS, "analysis"))
    args = parser.parse_args(argv)

    report = _preflight(args)

    endpoint = check_generation_endpoint(
        args.generator,
        os.environ.get("OPENAI_BASE_URL", ""),
        os.environ.get("OPENAI_API_KEY", ""))
    if endpoint["applicable"]:
        print("\n-- generation endpoint " + "-" * 48)
        for check in endpoint["checks"]:
            print(f"  {'PASS' if check['pass'] else 'FAIL'}  {check['check']}")
            if check["detail"]:
                print(f"        {check['detail']}")

    if args.document:
        selection = build_selection(args.per_question, k=args.k)
        payload = {
            "documented_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "k": args.k,
            "can_run_here": report["all_passed"],
            "pre_generation_checks": report,
            "selection_invariants": check_selection_invariants(selection, k=args.k),
            "selection_statistics": selection_statistics(selection),
            "generator_requested": {"backend": args.generator,
                                    "model": args.generator_model},
            "generation_endpoint": endpoint,
            "source_run_generation": (
                (json.load(open(args.run_manifest, encoding="utf-8")).get("config")
                 or {}).get("arm_a_generation") or {}),
            "inputs": {os.path.relpath(p, _ROOT): sha256(p)
                       for p in (args.per_question, args.run_manifest, args.questions)
                       if os.path.isfile(p)},
        }
        os.makedirs(args.analysis_out, exist_ok=True)
        # "readiness", not "generation": this file records whether the experiment
        # CAN run, and is written whether or not it has. Naming it after
        # generation invited the reading that answers already exist.
        json_path = os.path.join(args.analysis_out, "matched_k5_readiness.json")
        with open(json_path, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False, sort_keys=True)
            handle.write("\n")
        from experiments.analysis.render_matched_k import render_markdown
        md_path = os.path.join(args.analysis_out, "matched_k5_readiness.md")
        with open(md_path, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(render_markdown(payload))
        print(f"\ncan run here: {report['all_passed']}")
        print(f"written:\n  {md_path}\n  {json_path}")
        return 0 if report["all_passed"] else 3

    if not report["all_passed"]:
        print("\nREFUSING TO RUN. The inputs are not the scientific run's inputs.")
        if report["local_frozen_digest"] != SCIENTIFIC_FROZEN_DIGEST:
            print(f"  The frozen candidate file on this machine is\n"
                  f"    {report['local_frozen_digest']}  ({report['local_frozen_source']})\n"
                  f"  but this experiment requires\n    {SCIENTIFIC_FROZEN_DIGEST}\n"
                  "  Substituting a different candidate set would silently answer a\n"
                  "  different question. Run this on the machine holding the\n"
                  "  scientific frozen set, or pass --frozen <path> to it.")
        return 2

    selection = build_selection(args.per_question, k=args.k)
    invariants = check_selection_invariants(selection, k=args.k)
    stats = selection_statistics(selection)
    print("\n-- selection invariants " + "-" * 47)
    for check in invariants["checks"]:
        print(f"  {'PASS' if check['pass'] else 'FAIL'}  {check['check']}")
    print(f"\n  identical 5-passage sets : {stats['questions_with_identical_sets']}/30")
    print(f"  any overlap              : {stats['questions_with_any_overlap']}/30")
    print(f"  evidence differs         : {stats['questions_where_evidence_differs']}/30")
    print(f"  mean overlap             : {stats['mean_overlap']} of {args.k}")
    if not invariants["all_passed"]:
        print("\nREFUSING TO RUN: a selection invariant failed.")
        return 1
    if args.dry_run:
        print("\n--dry-run: checks and selection only. Nothing generated, nothing written.")
        return 0

    # Generation is the only mode that actually sends a prompt anywhere, so the
    # endpoint is enforced here rather than in the read-only modes above.
    if not endpoint["all_passed"]:
        print("\nREFUSING TO GENERATE. The endpoint is not the thesis generation "
              "setup.")
        for check in endpoint["checks"]:
            if not check["pass"]:
                print(f"  - {check['check']}: {check['detail']}")
        print("\n  The scientific run served thesis-llama3-8b-q4 from a LOCAL\n"
              "  OpenAI-compatible server. Restore that environment:\n\n"
              "    $env:OPENAI_BASE_URL = \"http://localhost:11434/v1\"\n"
              "    $env:OPENAI_API_KEY  = \"ollama\"   # any non-empty placeholder\n\n"
              "  Do NOT supply a commercial OpenAI key: a cloud model is not the\n"
              "  thesis generator, and the arms must share the frozen one.")
        return 2

    # -- generation ------------------------------------------------------
    from scaf.frozen import load as load_frozen  # noqa: E402
    from scaf.generation import (  # noqa: E402
        GeneratorSpec, GeneratorUnavailable, build_generator)
    from rag2.schema import Evidence, Question  # noqa: E402

    frozen = load_frozen(args.frozen, expected_digest=SCIENTIFIC_FROZEN_DIGEST)
    by_chunk = {c.chunk_id: c for fs in frozen for c in fs.candidates}

    source_generation = ((json.load(open(args.run_manifest, encoding="utf-8"))
                          .get("config") or {}).get("arm_a_generation") or {})
    spec = GeneratorSpec.from_mapping({**source_generation,
                                       "backend": args.generator,
                                       "model": args.generator_model})
    try:
        spec.validate()
        generator = build_generator(spec)
    except GeneratorUnavailable as exc:
        print(f"\nREFUSING TO RUN: {exc}")
        return 2

    print(f"\n-- generating {len(selection) * 2} answers " + "-" * 40)
    records, errors = [], []
    for entry in selection:
        record: Dict[str, Any] = {
            "qid": entry["qid"], "question": entry["question"], "k": args.k,
            "candidate_ids": entry["candidate_ids"],
            "overlap_count": entry["overlap_count"],
            "identical_sets": entry["identical_sets"],
        }
        for arm in ("rag2", "scaf"):
            ids = entry[f"{arm}_selected_chunk_ids"]
            passages = [by_chunk[cid] for cid in ids]
            evidences = [Evidence(chunk_id=p.chunk_id, text=p.text,
                                  title=getattr(p, "title", "")) for p in passages]
            question = Question(qid=entry["qid"], question=entry["question"])
            try:
                answer = generator(question, evidences)
                error = None
            except Exception as exc:                      # noqa: BLE001
                answer, error = "", f"{type(exc).__name__}: {exc}"
                errors.append({"qid": entry["qid"], "arm": arm, "error": error})
            record[arm] = {
                "selected_chunk_ids": ids,
                "selected_passages": [{"chunk_id": p.chunk_id,
                                       "text": p.text,
                                       "title": getattr(p, "title", "")}
                                      for p in passages],
                "context_chars": sum(len(p.text) for p in passages),
                "answer": answer,
                "answer_chars": len(answer),
                "generation_error": error,
            }
        records.append(record)
        print(f"  {entry['qid']}  rag2 {record['rag2']['answer_chars']:5d} chars   "
              f"scaf {record['scaf']['answer_chars']:5d} chars")

    # -- post-generation invariants --------------------------------------
    post: List[Dict[str, Any]] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        post.append({"check": name, "pass": bool(ok), "detail": detail})

    for arm in ("rag2", "scaf"):
        filled = sum(1 for r in records if str(r[arm]["answer"]).strip())
        check(f"{arm}: 30 answers", filled == 30, f"{filled}/30")
        sizes = {len(r[arm]["selected_chunk_ids"]) for r in records}
        check(f"{arm}: exactly {args.k} passages per question", sizes == {args.k},
              f"sizes {sorted(sizes)}")
    check("60 answers in total",
          sum(1 for r in records for a in ("rag2", "scaf")
              if str(r[a]["answer"]).strip()) == 60)
    check("zero generation errors", not errors, f"{len(errors)} error(s)")

    reportable = invariants["all_passed"] and all(c["pass"] for c in post)
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "per_question.jsonl"), "w",
              encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

    manifest = {
        "experiment": "matched-k answer generation",
        "k": args.k,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "reportable": reportable,
        "label": ("MATCHED-CONTEXT ANSWER GENERATION. Not an answer-quality result: "
                  "no reference answer exists and no judgement was made."),
        "distinct_from_admission_experiment": (
            "comparison_scientific/ measures how much evidence each policy admits. "
            "This run fixes the budget at k and measures only which evidence is "
            "chosen. Neither replaces the other."),
        "frozen_set_digest": SCIENTIFIC_FROZEN_DIGEST,
        "source_run": os.path.relpath(args.per_question, _ROOT),
        "input_digests": {
            os.path.relpath(p, _ROOT): sha256(p)
            for p in (args.per_question, args.run_manifest, args.questions)
            if os.path.isfile(p)},
        "generator": generator.describe(),
        "generation_endpoint": endpoint,
        "generation_config": spec.to_dict(),
        "source_run_generation": source_generation,
        "pre_generation_checks": report,
        "selection_invariants": invariants,
        "post_generation_checks": {"all_passed": all(c["pass"] for c in post),
                                   "checks": post},
        "selection_statistics": stats,
        "generation_errors": errors,
    }
    with open(os.path.join(args.out, "manifest.json"), "w",
              encoding="utf-8", newline="\n") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")
    with open(os.path.join(args.out, "README.md"), "w",
              encoding="utf-8", newline="\n") as handle:
        handle.write(README.format(
            k=args.k, answers=len(records) * 2, questions=len(records),
            digest=SCIENTIFIC_FROZEN_DIGEST, model=spec.model, backend=spec.backend,
            prompt_fingerprint=source_generation.get("prompt_fingerprint"),
            generated=manifest["created_at"], reportable=reportable))

    print("\n-- post-generation checks " + "-" * 45)
    for c in post:
        print(f"  {'PASS' if c['pass'] else 'FAIL'}  {c['check']}  {c['detail']}")
    print(f"\nREPORTABLE: {reportable}")
    print(f"written to {args.out}")
    return 0 if reportable else 1


if __name__ == "__main__":
    raise SystemExit(main())
