"""Render the answer-quality readiness inspection as a report a person reads.

Every value comes from the readiness dictionary, so the Markdown and the JSON
cannot disagree. Nothing is computed here.
"""

from __future__ import annotations

from typing import Any, Dict, List


def render_markdown(r: Dict[str, Any]) -> str:
    q = r["question_set"]
    gen = r["generated_answers"]
    generator = r["generator"]
    code = r["evaluation_code"]
    nots = r["artifacts_that_are_not_answer_targets"]
    feas = r["matched_k_feasibility"]
    dec = r["decision"]

    lines: List[str] = []
    A = lines.append

    A("# Can we evaluate ANSWER quality yet?")
    A("")
    A("> **Generated inspection report.** A *derived* artifact, rebuilt by "
      "`experiments/scripts/answer_quality_readiness.py`. It opens files and "
      "reports; it runs no retrieval, no generation, no annotation and no "
      "experiment, and it creates no reference answer.")
    A("")
    A(f"Inspected {r['inspected_utc']} · version `{r['analysis_version']}`")
    A("")

    # Decision first ------------------------------------------------------
    A(f"## Decision: **{dec['verdict']}**")
    A("")
    A(dec["reason"])
    A("")
    A("| | |")
    A("| --- | --- |")
    A(f"| Answer-quality target exists | **{dec['has_answer_quality_target']}** |")
    A(f"| Answer-evaluation code exists | **{dec['has_answer_evaluation_code']}** |")
    A(f"| Matched-k selection executable now | **{dec['matched_k_selection_executable']}** |")
    A("")
    A("In plain terms: **you can build the experiment, but you cannot yet mark "
      "it.** Everything needed to give both systems exactly five passages and "
      "produce two answers per question is already in the repository. What is "
      "missing is any independent statement of what a good answer to these "
      "questions would look like.")
    A("")

    # 1 -------------------------------------------------------------------
    A("## 1. The 30 evaluation questions")
    A("")
    A(f"`{q['path']}` — {q['questions']} questions, set "
      f"`{', '.join(q['set_identifiers'])}`.")
    A("")
    A("Fields present on each question record:")
    A("")
    A("| Field | Present on |")
    A("| --- | ---: |")
    for k, v in q["fields"].items():
        A(f"| `{k}` | {v}/{q['questions']} |")
    A("")
    A(f"**Answer/gold/reference fields: {q['answer_target_fields'] or 'NONE'}.**")
    A("")
    A("The question set says so itself. Every one of the 30 records carries:")
    A("")
    for note in q["notes"]:
        A(f"> {note}")
    A("")
    A("So for each of the 30 questions the repository holds **the question text "
      "and nothing else**: no gold answer, no reference answer, no expert label, "
      "no answer-quality label.")
    A("")

    # 2 -------------------------------------------------------------------
    A("## 2. Answers that already exist")
    A("")
    A("The completed comparison **did** save its generated answers, which is worth "
      "knowing — they are real text, not placeholders:")
    A("")
    A("| | RAG² | SCAF |")
    A("| --- | ---: | ---: |")
    for label, key in (("Answers present", "answers_present"),
                       ("Abstentions", "abstentions"),
                       ("Generation errors", "generation_errors"),
                       ("Mean context (chars)", "mean_context_chars"),
                       ("Mean admitted passages", "mean_admitted_passages")):
        A(f"| {label} | {gen['arms']['rag2'][key]} | {gen['arms']['scaf'][key]} |")
    A(f"| Answer length (median chars) | "
      f"{gen['arms']['rag2']['answer_chars']['median']} | "
      f"{gen['arms']['scaf']['answer_chars']['median']} |")
    A("")
    A(f"**This is the confound.** SCAF's answers were written from roughly "
      f"{gen['context_ratio_scaf_over_rag2']}× more context than RAG²'s "
      f"({gen['arms']['scaf']['mean_context_chars']} characters against "
      f"{gen['arms']['rag2']['mean_context_chars']}). Comparing these 60 answers "
      "as they stand would measure context size, not admission policy. That is "
      "precisely why a matched-k design is the right next experiment — and why "
      "the earlier preliminary answer figures must not be reported.")
    A("")

    # 3 -------------------------------------------------------------------
    A("## 3. Evaluation code that exists")
    A("")
    A(f"`{code['path']}` is real, tested evaluation machinery — but every one of "
      "its metrics needs a target this repository does not have.")
    A("")
    A("| Capability | Functions | Needs | Usable here? |")
    A("| --- | --- | --- | --- |")
    for name, cap in code["capabilities"].items():
        A(f"| {name.replace('_', ' ')} | `{', '.join(cap['functions'])}` | "
          f"{cap['requires']} | **{cap['usable_on_alzheimer_set']}** |")
    A("")
    for name, cap in code["capabilities"].items():
        A(f"- **{name.replace('_', ' ')}** — {cap['why']}.")
    A("")
    A(f"- **Citation correctness**: {code['citation_correctness']['exists']}. None "
      "exists.")
    A(f"- **Factuality / claim-level evaluation**: "
      f"{code['factuality_or_claim_level']['exists']}. None exists.")
    A(f"- **Wired into the SCAF comparison**: "
      f"{code['wired_into_the_scaf_comparison']}. The comparison records answer "
      "*counts* only.")
    A("")
    A("The run manifest states the same thing in its own words:")
    A("")
    A(f"> {generator['manifest_note']}")
    A("")
    A(f"`with_reference` is **{generator['answers_with_reference']['rag2']}** for "
      f"RAG² and **{generator['answers_with_reference']['scaf']}** for SCAF.")
    A("")

    # 4 -------------------------------------------------------------------
    A("## 4. Things that look like a gold standard but are not")
    A("")
    A("This section exists because these three are easy to reach for, and each "
      "would quietly invalidate the experiment.")
    A("")
    for name, entry in nots.items():
        A(f"### `{name}`")
        A("")
        A(f"- Path: `{entry['path']}`")
        if "rows" in entry:
            A(f"- Rows: {entry['rows']}")
        if "labels" in entry:
            A(f"- Labels: `{entry['labels']}`")
        A(f"- What it is: {entry['what_it_is']}")
        A(f"- **Why it is not an answer-quality target:** "
          f"{entry['why_not_an_answer_target']}")
        A(f"- Suitable as answer gold: **{entry['suitable_as_answer_gold']}**")
        A("")

    # 5 -------------------------------------------------------------------
    A("## 5. Matched-k feasibility")
    A("")
    A(f"The proposed design: same {feas['questions']} questions, same frozen "
      f"candidate set, same generator, exactly **k = {feas['k']}** admitted "
      "passages per arm, with only the admission policy differing.")
    A("")
    A("| Component | Available now |")
    A("| --- | --- |")
    for name, ok in feas["components"].items():
        A(f"| {name.replace('_', ' ')} | **{ok}** |")
    A("")
    A(f"Every arm-question pair has enough candidates to take a top-{feas['k']}: "
      f"**{feas['arm_questions_with_at_least_k_candidates']} of "
      f"{feas['arm_questions_total']}**.")
    A("")
    A(f"> {feas['note']}")
    A("")
    A(f"**Blocked on: {', '.join(feas['blocked_on'])}.**")
    A("")

    # 6 -------------------------------------------------------------------
    A("## 6. Limitations of this inspection")
    A("")
    A("- This is a search for artifacts, not a judgement of their scientific "
      "quality. A gold answer found would still have needed validating.")
    A("- It reports on this repository only. A reference standard may exist "
      "outside it — in a supervisor's notes, or a clinical guideline — and would "
      "not be visible here.")
    A("- It does not verify that the generator is currently reachable, only that "
      "its configuration is recorded.")
    A("- Absence of a field name is strong but not absolute evidence: a reference "
      "stored under an unexpected name would not be matched.")
    A("")

    # 7 -------------------------------------------------------------------
    A("## 7. Recommended next step")
    A("")
    A("**Decide and record the answer-quality standard before generating "
      "anything.** Two defensible routes exist, and the cheaper one does not "
      "require gold answers at all.")
    A("")
    A("### Route A — blind pairwise preference (recommended)")
    A("")
    A(f"Regenerate both arms at k = {feas['k']} from the frozen candidates, then "
      "show a human the question and the two answers **anonymised and in random "
      "order**, and ask which is better supported. No gold answer is needed, "
      "because the comparison is relative.")
    A("")
    A("Why this is the smallest defensible step:")
    A("")
    A("- it removes the context confound, which is the one thing that makes the "
      "current answer data uninterpretable;")
    A("- it needs no clinical expertise to *author* a reference, only to *compare* "
      "two answers;")
    A("- 30 comparisons is roughly an hour of work, against 30 written reference "
      "answers;")
    A("- the repository already has the machinery: a blind annotation interface, "
      "an audit that checks blinding and refuses anchored passes, and a stratified "
      "sampling design. The lesson already paid for — **never show the annotator "
      "a machine opinion** — applies directly.")
    A("")
    A("Its limitation, which must be stated in the thesis: preference is not "
      "correctness. It can show one arm's answers are *preferred*, not that they "
      "are *right*.")
    A("")
    A("### Route B — written reference answers")
    A("")
    A("Author a reference answer for each of the 30 questions, grounded in the "
      "corpus, and validate it with someone qualified. This unlocks the existing "
      "`rouge_l` / `bertscore` / `open_ended_metrics` code directly.")
    A("")
    A("It is more expensive and carries a real risk: a reference written by "
      "someone without clinical training, or written after seeing the systems' "
      "answers, would be worse than no reference at all. **If Route B is chosen, "
      "the references must be written before any matched-k answer is generated, "
      "and signed off by a supervisor.**")
    A("")
    A("### What not to do")
    A("")
    A("- Do not reuse the 120 evidence-quality labels as answer labels.")
    A("- Do not reuse the filter training weak labels as evaluation gold.")
    A("- Do not grade answers with SCAF or RAG² scores.")
    A("- Do not generate reference answers with the same model being evaluated.")
    A("- Do not report the earlier preliminary answer figures; they came from "
      "unmatched contexts.")
    A("")

    # 8 -------------------------------------------------------------------
    A("## 8. Artifacts inspected")
    A("")
    A("```")
    for path in r["artifacts_inspected"]:
        A(path)
    A("```")
    A("")
    A("Regenerate this report with:")
    A("")
    A("```")
    A("python experiments/scripts/answer_quality_readiness.py")
    A("```")
    A("")
    A("Structured form: `answer_quality_readiness.json` in this directory.")
    A("")
    return "\n".join(lines)
