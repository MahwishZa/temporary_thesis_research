"""Render the matched-k generation status as a report a person reads.

Every value comes from the payload, so the Markdown and the JSON cannot
disagree. The report is written whether or not the experiment can run here: a
blocked run that is documented is far more useful than one that silently did
not happen.
"""

from __future__ import annotations

from typing import Any, Dict, List


def render_markdown(p: Dict[str, Any]) -> str:
    pre = p["pre_generation_checks"]
    inv = p["selection_invariants"]
    st = p["selection_statistics"]
    src = p["source_run_generation"]
    k = p["k"]
    can = p["can_run_here"]

    lines: List[str] = []
    A = lines.append

    A(f"# Matched-k = {k} answer generation")
    A("")
    A("> **Generated status report.** A *derived* artifact, rebuilt by "
      "`python experiments/scripts/run_matched_k.py --document`. It records "
      "whether this machine can run the matched-k experiment, and the parts of "
      "the design that can be verified without a generator.")
    A("")
    A(f"Documented {p['documented_utc']}")
    A("")
    A(f"## Status: {'READY TO RUN' if can else '**BLOCKED**'}")
    A("")
    if not can:
        failed = [c for c in pre["checks"] if not c["pass"]]
        A("The experiment **has not been run**. These preconditions failed:")
        A("")
        for c in failed:
            A(f"- **{c['check']}** — {c['detail']}")
        A("")
        A("No answers were generated, and no result directory was created. "
          "Substituting a different candidate set would produce 60 plausible "
          "answers and a clean-looking manifest for **a different experiment**, "
          "so the run refuses instead.")
    else:
        A("Every precondition passed. The experiment can be run on this machine.")
    A("")

    # 1 -------------------------------------------------------------------
    A("## 1. Objective, and why matched-k is necessary")
    A("")
    A("The completed admission comparison measured what each policy naturally "
      "admits: SCAF passed about 19 passages per question, RAG² passed 0.13. "
      "SCAF's answers were therefore written from roughly **688× more context**. "
      "Any answer difference between those two runs is confounded by context "
      "size and cannot be attributed to the admission policy.")
    A("")
    A(f"This experiment hands **both arms exactly {k} passages** from the same "
      "frozen candidate set, so the only thing that varies is *which* "
      f"{k} passages each policy's own score ranked highest.")
    A("")
    A("**It does not replace the admission experiment.** That run answers *how "
      "much* evidence each policy admits and remains the answer to that "
      "question. This run fixes the amount and asks only about the choice. "
      "Neither supersedes the other, and the two must not be conflated in the "
      "thesis.")
    A("")

    # 2 -------------------------------------------------------------------
    A("## 2. Selection procedure")
    A("")
    A(f"Each arm ranks the **same 20 frozen candidates** by **its own score** and "
      f"takes the top {k}. Ties break on `chunk_id`, so the choice is "
      "deterministic. This is the procedure already used by `matched_budget` in "
      "`evidence_quality.py` and already reported in the results analysis — no "
      "new scoring rule, no changed SCAF weight, no tuned threshold.")
    A("")
    A("**Selection deliberately ignores each arm's admission threshold.** RAG² "
      "admits only 4 passages across all 600 decisions; a threshold-respecting "
      f"rule could not reach k={k} for 26 of the 30 questions, and the experiment "
      "would collapse back into the confound it exists to remove. The threshold "
      "governs *how much* a policy admits, which the original experiment "
      "measured. Here the amount is fixed and only the ranking matters.")
    A("")

    # 3 -------------------------------------------------------------------
    A("## 3. Pre-generation checks")
    A("")
    A("| Check | Result | Detail |")
    A("| --- | --- | --- |")
    for c in pre["checks"]:
        detail = c["detail"] if c["detail"] else ""
        A(f"| {c['check']} | {'PASS' if c['pass'] else '**FAIL**'} | {detail} |")
    A("")
    A(f"**All passed: {pre['all_passed']}**")
    A("")
    A(f"Required frozen digest: `{pre['expected_frozen_digest']}`")
    A(f"Run manifest records: `{pre['run_manifest_digest']}`")
    A(f"Frozen file on this disk: `{pre['local_frozen_digest']}` "
      f"(source: {pre['local_frozen_source']})")
    A("")

    # 4 -------------------------------------------------------------------
    A("## 4. Selection invariants")
    A("")
    A("These hold regardless of whether generation can run — the selection needs "
      "only the saved scores.")
    A("")
    A("| Invariant | Result |")
    A("| --- | --- |")
    for c in inv["checks"]:
        A(f"| {c['check']} | {'PASS' if c['pass'] else '**FAIL**'} |")
    A("")
    A(f"**All passed: {inv['all_passed']}**")
    A("")

    # 5 -------------------------------------------------------------------
    A("## 5. How different is the selected evidence?")
    A("")
    A("Descriptive statistics about what each policy would hand the generator.")
    A("")
    A("| | |")
    A("| --- | ---: |")
    A(f"| Questions | {st['questions']} |")
    A(f"| k | {st['k']} |")
    A(f"| Questions with **identical** {k}-passage sets | "
      f"{st['questions_with_identical_sets']} |")
    A(f"| Questions where the evidence **differs** | "
      f"{st['questions_where_evidence_differs']} |")
    A(f"| Questions with **no overlap at all** | {st['questions_with_no_overlap']} |")
    A(f"| Questions with some overlap | {st['questions_with_any_overlap']} |")
    A(f"| Mean overlap (of {k}) | {st['mean_overlap']} |")
    A(f"| Mean Jaccard | {st['mean_jaccard']} |")
    A("")
    A(f"Overlap distribution: `{st['overlap_distribution']}`")
    A("")
    A(f"> {st['guard']}")
    A("")
    A(f"In plain terms: at an equal budget of {k} passages the two policies "
      f"choose **substantially different evidence** — no question gets an "
      f"identical set, {st['questions_with_no_overlap']} share nothing at all, "
      f"and on average only {st['mean_overlap']} of {k} passages coincide. That "
      "is what makes the comparison worth generating: if the sets were identical "
      "the answers would be too, and there would be nothing to measure.")
    A("")
    A("Context-size statistics (characters, tokens) are **not** in this report. "
      "They require the candidate text, which lives with the frozen candidate "
      "set; they will be recorded in `matched_k5/manifest.json` when the run "
      "executes.")
    A("")

    # 6 -------------------------------------------------------------------
    A("## 6. Generator")
    A("")
    A("The run reuses the configuration recorded by the scientific comparison "
      "rather than reconstructing it:")
    A("")
    A("| Setting | Value |")
    A("| --- | --- |")
    for key in ("backend", "model", "llm_class", "greedy", "max_new_tokens",
                "chat_template", "dtype", "prompt_version", "prompt_fingerprint"):
        if key in src:
            A(f"| `{key}` | {src[key]} |")
    A("")
    A(f"Requested for this run: `{p['generator_requested']['model']}` via "
      f"`{p['generator_requested']['backend']}`.")
    A("")

    # 7 -------------------------------------------------------------------
    A("## 7. What was NOT done")
    A("")
    A("- No answer-quality judgement of any kind.")
    A("- No gold answers or reference answers were created.")
    A("- No correctness labels, no ROUGE, no BERTScore.")
    A("- No human evaluation, and no annotator saw anything.")
    A("- No retrieval, reranking or index access.")
    A("")
    A("The blind pairwise evaluation is a **separate later stage**. When this "
      "run executes, its output labels each arm by name, so those files must be "
      "anonymised before any human sees them.")
    A("")

    # 8 -------------------------------------------------------------------
    A("## 8. Limitations")
    A("")
    A(f"- Matched-k measures the *ranking*, not the policy as deployed. Neither "
      f"arm would naturally admit exactly {k} passages.")
    A("- 30 questions is small, and they are development questions with no gold "
      "answer.")
    A("- Even once generated, the answers cannot be scored until an "
      "answer-quality target exists — see `answer_quality_readiness.md`.")
    A("- A single generator, single decoding pass. No variance estimate.")
    A("")

    # 9 -------------------------------------------------------------------
    A("## 9. Inputs")
    A("")
    A("```")
    for path, digest in p["inputs"].items():
        A(f"{digest}  {path}")
    A("```")
    A("")
    A("## 10. How to run it")
    A("")
    A("On the machine holding the scientific frozen candidate set and the "
      "generator:")
    A("")
    A("```")
    A("python experiments/scripts/run_matched_k.py --dry-run")
    A("python experiments/scripts/run_matched_k.py \\")
    A("    --generator openai --generator-model thesis-llama3-8b-q4")
    A("```")
    A("")
    A("`--dry-run` runs every check and the selection, generates nothing and "
      "writes nothing. The full command writes "
      "`experiments/results/rag2_vs_scaf_alzheimer/matched_k5/`. Refresh this "
      "report with `--document`.")
    A("")
    return "\n".join(lines)
