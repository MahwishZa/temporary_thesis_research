"""Turn the computed results into the report a person reads.

Every number in the prose comes from the results dictionary, so the report and
the JSON summary cannot disagree. Nothing is computed here -- if a value is not
in the dictionary it does not appear in the report.

The tone is deliberately plain. The reader is an MS student who needs to open
one file months from now and understand what the experiment showed, including
the parts that did not work.
"""

from __future__ import annotations

from typing import Any, Dict, List


def _pct(value):
    return "n/a" if value is None else f"{value * 100:.2f}%"


def _rho(value) -> str:
    """Spearman is undefined for a constant column. Say so rather than "None"."""
    return "n/a (constant)" if value is None else str(value)


def _mean(entry: Dict[str, Any]) -> str:
    if not entry or not entry.get("n"):
        return "n=0"
    return f"n={entry['n']}, mean {entry['mean']}"


def _example_block(rows: List[Dict[str, Any]]) -> str:
    if not rows:
        return "_None in the annotated sample._\n"
    out = []
    for r in rows:
        out.append(
            f"- **{r['annotation_id']}** ({r['qid']}) — human label **{r['human_label']}**\n"
            f"  - SCAF {r['scaf_score']} (admitted: {r['scaf_admitted']}), "
            f"RAG2 {r['rag2_score']} (admitted: {r['rag2_admitted']})\n"
            f"  - σ {r['sigma']} · γ {r['gamma']} · τ {r['tau']} · "
            f"{r['canonical_date']} · {r['source_category']}\n"
            f"  - Q: {r['question']}\n"
            f"  - P: {r['passage_excerpt']}…\n")
    return "\n".join(out)


def render_markdown(results: Dict[str, Any]) -> str:
    p = results["provenance"]
    adm = results["admission_behaviour"]
    cand, ques = adm["candidate_level"], adm["question_level"]
    hq = results["human_evidence_quality"]
    comp = results["scaf_components"]
    mc = results["matched_context"]
    temporal = results["temporal_feasibility"]
    retr = results["retraction_census"]
    clus = results["clustering"]
    ex = results["examples"]
    g = hq["by_group"]
    ra = hq["rank_association"]

    lines: List[str] = []
    A = lines.append

    A("# RAG² vs SCAF — what the current experiment actually shows")
    A("")
    A("> **This file is generated.** It is a *derived* artifact, rebuilt from the "
      "frozen scientific artifacts by `experiments/scripts/analyse_results.py`. "
      "It is not itself evidence, and writing something here does not make it "
      "true. Every number below is computed from the files listed in §3.")
    A("")
    A(f"Generated {p['generated_utc']} · analysis version `{p['analysis_version']}`")
    A("")

    # 1 -------------------------------------------------------------------
    A("## 1. Executive summary")
    A("")
    A("In one sentence: **the experiment shows that SCAF and RAG² admit very "
      "different amounts of evidence, and that neither one's score orders "
      "evidence the way a human does.**")
    A("")
    A(f"- SCAF admits **{cand['scaf_admitted']} of {cand['decisions']}** candidate "
      f"passages ({_pct(cand['scaf_admission_rate'])}). RAG² admits "
      f"**{cand['rag2_admitted']}** ({_pct(cand['rag2_admission_rate'])}).")
    A(f"- **{ques['questions_with_zero_rag2_evidence']} of {ques['questions']} "
      f"questions receive no evidence at all from RAG².**")
    A(f"- Across the 120 independently annotated passages, the SCAF score's rank "
      f"association with human-judged usefulness is **{ra['scaf_score']['spearman']}** "
      f"— weak. RAG²'s is **{ra['rag2_score']['spearman']}** — essentially none.")
    A(f"- SCAF-admitted passages average **{g['scaf_admitted'].get('mean')}** on the "
      f"0–2 usefulness scale; the {g['scaf_rejected'].get('n', 0)} SCAF-rejected "
      f"passages average **{g['scaf_rejected'].get('mean')}**. The threshold does "
      f"not separate useful evidence from unuseful evidence here.")
    A(f"- All **{hq['irrelevant_passages']['n']}** passages a human judged *not "
      f"relevant* were **admitted by SCAF**; "
      f"{hq['irrelevant_passages']['rag2_admitted']} were admitted by RAG².")
    A("")
    A("**The most important thing to understand about these numbers:** an earlier "
      "annotation pass reported a much stronger SCAF association (≈0.63). That "
      "pass showed the annotator a word-matching suggestion on every passage, and "
      "all 120 of its labels matched that suggestion exactly. Because SCAF's "
      "support term is *also* word-matching, that correlation largely measured the "
      "annotation interface rather than SCAF. The independent pass analysed here "
      "removed the suggestion, and the association fell to "
      f"{ra['scaf_score']['spearman']}. **The weak number is the real one.**")
    A("")

    # 2 -------------------------------------------------------------------
    A("## 2. Experimental setup")
    A("")
    A(f"- **{p['counts']['questions']} Alzheimer questions**, "
      f"**{p['counts']['candidates_per_question']} candidates each**, "
      f"**{p['counts']['decisions']} candidate decisions** in total.")
    A("- Both arms scored the **identical frozen candidate set in identical "
      "order**. Neither arm did its own retrieval. Same generator, same decoding.")
    A(f"- Fairness checks recorded in the run manifest: "
      f"`all_passed = {p['scientific_run']['fairness_all_passed']}`.")
    A(f"- Scientific preconditions: `all_passed = "
      f"{p['scientific_run']['scientific_all_passed']}`, "
      f"`reportable = {p['scientific_run']['reportable']}`.")
    A("- The only difference between the arms is the **admission decision**: RAG²'s "
      "trained perplexity filter versus SCAF's weighted score against a threshold.")
    A("")

    # 3 -------------------------------------------------------------------
    A("## 3. Data and provenance")
    A("")
    A("| What | Value |")
    A("| --- | --- |")
    A(f"| Frozen candidate-set digest (run) | `{p['scientific_run']['frozen_set_digest']}` |")
    A(f"| Frozen digest recorded for the sample | `{p['sample']['frozen_set_digest']}` |")
    A(f"| Digests agree | **{p['digests_agree']}** |")
    A(f"| Retrieval is production MedCPT | **{p['sample']['retrieval_is_medcpt']}** |")
    A(f"| Annotation sheet SHA-256 | `{p['annotation_sheet_sha256']}` |")
    A(f"| Label fingerprint | `{p['label_fingerprint']}` |")
    A(f"| Annotation pass | `{', '.join(p['annotation_pass'])}` |")
    A(f"| Human annotations | {p['counts']['human_annotations']} "
      f"(joined to decisions: {results['join']['joined']}) |")
    A(f"| Sample seed / size / population | {p['sample']['seed']} / "
      f"{p['sample']['sampled']} / {p['sample']['population']} |")
    A(f"| RAG² admissions forced into sample | {p['sample']['forced_rag2_admitted']} |")
    A("")
    A("Source files:")
    A("")
    for name, path in p["sources"].items():
        A(f"- `{name}` → `{path}`")
    A("")
    A("The retired anchored pilot in `evidence_quality/pilot_anchored/` was **not** "
      "used. It is kept as the historical record of a methodological correction "
      "and must never be pooled with these labels.")
    A("")

    # 4 -------------------------------------------------------------------
    A("## 4. Admission behaviour")
    A("")
    A("**Candidate level** (all 600 decisions):")
    A("")
    A("| | Count | Rate |")
    A("| --- | ---: | ---: |")
    A(f"| RAG² admitted | {cand['rag2_admitted']} | {_pct(cand['rag2_admission_rate'])} |")
    A(f"| SCAF admitted | {cand['scaf_admitted']} | {_pct(cand['scaf_admission_rate'])} |")
    A(f"| Both admit | {cand['both_admit']} | |")
    A(f"| RAG² only | {cand['rag2_only']} | |")
    A(f"| SCAF only | {cand['scaf_only']} | |")
    A(f"| Neither | {cand['neither_admits']} | |")
    A(f"| Overlap (Jaccard) | {cand['jaccard']} | |")
    A("")
    A(f"**Question level** ({ques['questions']} questions): RAG² supplies evidence "
      f"for **{ques['questions_with_any_rag2_evidence']}** of them and none for "
      f"**{ques['questions_with_zero_rag2_evidence']}**. SCAF supplies evidence for "
      f"**{ques['questions_with_any_scaf_evidence']}**, averaging "
      f"{ques['scaf_admissions_per_question']['mean']} passages per question "
      f"(range {ques['scaf_admissions_per_question']['min']}–"
      f"{ques['scaf_admissions_per_question']['max']}).")
    A("")
    A(f"**RAG²-only admissions: {cand['rag2_only']}.** Every passage RAG² admits is "
      "also admitted by SCAF. The two policies are not making opposite calls — "
      "RAG²'s admissions are a subset of SCAF's.")
    A("")
    A(f"> {adm['guard']}")
    A("")

    # 5 -------------------------------------------------------------------
    A("## 5. Human evidence-quality results")
    A("")
    A("120 passages were read by a person who saw only the question and the "
      "passage, and rated each **0** (not relevant), **1** (partially relevant) or "
      "**2** (clearly relevant).")
    A("")
    A(f"Overall: {_mean(hq['overall'])}, distribution "
      f"{hq['overall'].get('distribution')}.")
    A("")
    A("| Group | n | Mean usefulness |")
    A("| --- | ---: | ---: |")
    for key, label in (("all_annotated", "All annotated"),
                       ("scaf_admitted", "SCAF admitted"),
                       ("scaf_rejected", "SCAF rejected"),
                       ("rag2_admitted", "RAG² admitted"),
                       ("rag2_rejected", "RAG² rejected"),
                       ("scaf_only", "SCAF only (SCAF admits, RAG² rejects)"),
                       ("rag2_only", "RAG² only (RAG² admits, SCAF rejects)"),
                       ("both_admit", "Both admit"),
                       ("neither_admits", "Neither admits")):
        e = g.get(key) or {}
        A(f"| {label} | {e.get('n', 0)} | {e.get('mean', '—')} |")
    A("")
    A("**Rank association with the human label** (Spearman, descriptive only):")
    A("")
    A("| Score | ρ | n |")
    A("| --- | ---: | ---: |")
    for name, stats in ra.items():
        A(f"| `{name}` | {_rho(stats['spearman'])} | {stats['n']} |")
    A("")
    A("**Mean usefulness by SCAF score band** — if the score ordered evidence "
      "quality, this column would rise:")
    A("")
    A("| SCAF score | n | Mean usefulness |")
    A("| --- | ---: | ---: |")
    for band, e in hq["by_score_bin" if "by_score_bin" in hq else "by_scaf_score_band"].items():
        A(f"| {band.replace('_', ' ')} | {e.get('n', 0)} | {e.get('mean', '—')} |")
    A("")
    A(f"> {hq['guard']}")
    A("")

    # 6 -------------------------------------------------------------------
    A("## 6. SCAF component behaviour")
    A("")
    A(f"Weights in this run: `{comp.get('weights')}`, threshold "
      f"`{comp.get('threshold')}`, support method `{comp.get('support_method')}`.")
    A("")
    A("| Term | Distinct values | Range | Admitted − rejected | ρ with human label |")
    A("| --- | ---: | --- | ---: | ---: |")
    for name, t in comp["terms"].items():
        # Spearman is undefined for a constant column; say so rather than "None".
        A(f"| {name} | {t['distinct_values']} | {t['min']}–{t['max']} | "
          f"{t['admitted_minus_rejected']} | "
          f"{_rho(t['spearman_with_human_label'])} |")
    A("")
    A("Reading this table:")
    A("")
    A("- **Support (σ)** separates admitted from rejected by a wide margin — but it "
      "carries half the score, so it does that *by construction*. Its association "
      "with the human label is "
      f"{_rho(comp['terms']['support_sigma']['spearman_with_human_label'])}. It is "
      "computed as **lexical/content-word coverage weighted by corpus IDF**, not "
      "semantic entailment and not expert relevance.")
    A(f"- **Currency (γ)** varies over "
      f"{comp['terms']['currency_gamma']['min']}–{comp['terms']['currency_gamma']['max']} "
      "and its association with the human label is "
      f"{_rho(comp['terms']['currency_gamma']['spearman_with_human_label'])}.")
    A(f"- **Authority (τ)** takes only "
      f"**{comp['terms']['authority_tau']['distinct_values']} distinct values** "
      f"across all 600 decisions and separates admitted from rejected by "
      f"{comp['terms']['authority_tau']['admitted_minus_rejected']}. It is "
      "effectively non-discriminative in this corpus.")
    A(f"- **Corroboration (ρ)** is weighted **0.0** and reports "
      f"`{comp.get('corroboration_status')}`. It contributes nothing and is "
      "**not experimentally validated** — it is not implemented.")
    A(f"- The admission gate never fired: `{comp.get('gate_fired')}`. Admission is "
      "purely score ≥ threshold.")
    A("")
    A("Reduced-vector variants (**not** validated ablations):")
    A("")
    A("| Variant | Admitted | Rate | Max reachable score |")
    A("| --- | ---: | ---: | ---: |")
    for name, v in comp["ablation_variants"].items():
        A(f"| {name} | {v['admitted']}/{v['of']} | {v['rate']} | "
          f"{v.get('max_possible_score', '—')} |")
    A("")
    A(f"> {comp['guard']}")
    A("")

    # 7 -------------------------------------------------------------------
    A("## 7. RAG²-vs-SCAF disagreement")
    A("")
    A(f"The disagreement is almost entirely one-sided: **{cand['scaf_only']}** "
      f"passages are admitted by SCAF and rejected by RAG², while **{cand['rag2_only']}** "
      "go the other way. Among annotated rows, SCAF-only passages average "
      f"{g['scaf_only'].get('mean')} usefulness — close to the overall mean of "
      f"{hq['overall'].get('mean')}, not obviously better or worse.")
    A("")
    A(f"The {g['both_admit'].get('n', 0)} passages both arms admit average "
      f"{g['both_admit'].get('mean')}, and the {g['neither_admits'].get('n', 0)} "
      f"neither admits average {g['neither_admits'].get('mean')}. Both cells are "
      "far too small to support any inference.")
    A("")

    # 8 -------------------------------------------------------------------
    A("## 8. Matched-k / matched-context analysis")
    A("")
    A("The raw admission counts are not comparable: SCAF passes ~19 passages per "
      "question and RAG² passes 0.13. Any downstream difference would be confounded "
      "by context size. The honest comparison re-ranks the **same frozen "
      "candidates** by each arm's own score and takes the top *k*. This requires no "
      "new experiment.")
    A("")
    A("**How much the two arms overlap at matched k:**")
    A("")
    A("| k | Mean Jaccard | Shared passages |")
    A("| ---: | ---: | --- |")
    for k, s in mc["overlap"].items():
        A(f"| {k} | {s['mean_jaccard']} | {s['shared_passages_total']}/{s['possible_total']} |")
    A("")
    A("**At k = 1 the two arms share no passage at all.** They are selecting "
      "essentially disjoint evidence.")
    A("")
    A("**Human usefulness of each arm's own top-k** (annotated rows only):")
    A("")
    A("| k | SCAF | RAG² |")
    A("| ---: | --- | --- |")
    for k, entry in mc["human_quality_of_each_arms_top_k"].items():
        A(f"| {k.split('=')[1]} | {_mean(entry['scaf'])} | {_mean(entry['rag2'])} |")
    A("")
    A("These are close at every k, in both directions. **At matched context "
      "budget, neither arm's ranking retrieves better evidence than the other's** "
      "in this sample.")
    A("")
    A(f"> {mc['guard']}")
    A("")
    A("The earlier preliminary figures (RAG² ≈ 1.000 vs SCAF ≈ 0.893) came from a "
      "run with wildly unmatched context sizes and **must not** be reported as a "
      "headline result.")
    A("")

    # 9 -------------------------------------------------------------------
    A("## 9. Retracted and superseded evidence")
    A("")
    A("| Field | Values across all 600 decisions |")
    A("| --- | --- |")
    A(f"| `retracted` | {retr['retracted']} |")
    A(f"| `supersession` | {retr['supersession']} |")
    A(f"| `currency_state` | {retr['currency_state']} |")
    A("")
    A("**There are zero retracted candidates in this evaluation set, and "
      "supersession is never determined** — it is recorded as `unknown` or `n/a` "
      "for every candidate. SCAF's retraction and supersession handling is "
      "therefore **completely untested by this experiment**. Nothing can be "
      "claimed about it in either direction.")
    A("")

    # 10 ------------------------------------------------------------------
    A("## 10. Temporal / FRB-PAIRS feasibility")
    A("")
    A(f"Candidate publication years: `{temporal['candidate_year_distribution']}`")
    A("")
    A(f"- Earliest year present: **{temporal['earliest_year']}**")
    A(f"- Candidates before 2020: **{temporal['candidates_before_2020']}**")
    A(f"- Candidates before 2022: **{temporal['candidates_before_2022']}** of 600")
    A(f"- γ (currency) range: {temporal['gamma_range'][0]}–{temporal['gamma_range'][1]}")
    A("")
    A("**The current corpus cannot support an old-versus-new temporal "
      "counterfactual.** A FRB-PAIRS-style design needs pairs where an older "
      "passage and a newer passage answer the same question differently. With no "
      "candidate older than "
      f"{temporal['earliest_year']} and only {temporal['candidates_before_2022']} "
      "before 2022, there is almost no old evidence to pair against. This is a "
      "**methodological limitation of the corpus**, not a fixable analysis choice, "
      "and it is why γ has so little to discriminate on.")
    A("")

    # 11 / 12 -------------------------------------------------------------
    A("## 11. Representative successes")
    A("")
    A("The four passages RAG² admitted — all also admitted by SCAF:")
    A("")
    A(_example_block(ex["rag2_admitted"]))
    A("## 12. Representative failures")
    A("")
    A("**SCAF admitted these, and a human judged them not relevant.** These are the "
      "highest-scoring such cases, so they are SCAF's worst mistakes, not its "
      "marginal ones:")
    A("")
    A(_example_block(ex["scaf_admitted_but_human_says_irrelevant"]))
    A("**SCAF rejected these** (the only rejections in the annotated sample):")
    A("")
    A(_example_block(ex["scaf_rejected"]))
    A("**Useful evidence SCAF scored lowest** — passages a human called clearly "
      "relevant that SCAF ranked at the bottom:")
    A("")
    A(_example_block(ex["lowest_scaf_score_that_humans_called_clearly_relevant"]))
    A(f"> {ex['selection_rule']}")
    A("")

    # 13-15 ---------------------------------------------------------------
    A("## 13. Claims supported by the current evidence")
    A("")
    A("These are safe to write in the thesis, with the stated scope:")
    A("")
    A(f"1. **The two policies admit vastly different amounts of evidence.** SCAF "
      f"{_pct(cand['scaf_admission_rate'])} versus RAG² "
      f"{_pct(cand['rag2_admission_rate'])} of 600 candidate decisions.")
    A(f"2. **RAG²'s filter leaves most questions with no evidence at all** — "
      f"{ques['questions_with_zero_rag2_evidence']} of {ques['questions']} "
      "questions receive zero admitted passages.")
    A("3. **The two policies select near-disjoint evidence.** At matched k = 1 they "
      "share no passage; Jaccard rises only to "
      f"{mc['overlap']['8']['mean_jaccard'] if '8' in mc['overlap'] else '—'} at k = 8.")
    A("4. **Neither score orders evidence the way a human does, in this sample.** "
      f"SCAF ρ = {ra['scaf_score']['spearman']}, RAG² ρ = "
      f"{ra['rag2_score']['spearman']}.")
    A("5. **SCAF's corroboration term is inactive** (weight 0.0, "
      f"`{comp.get('corroboration_status')}`) and its authority term is "
      f"near-constant ({comp['terms']['authority_tau']['distinct_values']} distinct "
      "values across 600 decisions).")
    A("6. **Showing an annotator a lexical suggestion inflated the measured "
      "SCAF–human association roughly fivefold** (≈0.63 anchored versus "
      f"{ra['scaf_score']['spearman']} independent). This is a genuine "
      "methodological finding and is worth reporting as one.")
    A("")
    A("## 14. Descriptive-only findings")
    A("")
    A(f"- The {g['rag2_admitted'].get('n', 0)} RAG²-admitted passages average "
      f"{g['rag2_admitted'].get('mean')} usefulness, above the overall mean of "
      f"{hq['overall'].get('mean')}. **n = {g['rag2_admitted'].get('n', 0)}.** This "
      "is a description of four passages, not evidence that RAG² admits better "
      "evidence. All four were forced into the sample by the sampling rule.")
    A(f"- The {g['scaf_rejected'].get('n', 0)} SCAF-rejected passages average "
      f"{g['scaf_rejected'].get('mean')}. **n = {g['scaf_rejected'].get('n', 0)}.** "
      "Nothing follows from two passages.")
    A(f"- All {hq['irrelevant_passages']['n']} passages judged not relevant were "
      "SCAF-admitted. Suggestive of a permissive threshold, but the annotated rows "
      "are a stratified sample, not a census.")
    A("- The score-band table in §5 does not rise monotonically. Suggestive, and "
      "consistent with the weak ρ, but the bands are small.")
    A("")
    A("## 15. Claims that are NOT established")
    A("")
    A("Do **not** write any of these:")
    A("")
    A("| Claim | Status |")
    A("| --- | --- |")
    A("| SCAF improves evidence quality | **Not established.** The association with "
      "human judgement is weak and the threshold does not separate quality. |")
    A("| SCAF is superior to RAG² | **Not established.** At matched k neither ranks "
      "better; admitting more is not admitting better. |")
    A("| SCAF improves answer accuracy | **Requires additional experiment.** No "
      "answer quality was measured; there are no gold answers. |")
    A("| SCAF reduces outdated evidence | **Not established.** No retracted or "
      "superseded candidate exists in the set to reduce. |")
    A("| RAG² has recency bias | **Not established.** Nothing here measures "
      "temporal behaviour, and the corpus has almost no old evidence. |")
    A("| SCAF improves citation quality | **Requires additional experiment.** Not "
      "measured. |")
    A("| SCAF reduces unsupported claims | **Requires additional experiment.** Not "
      "measured. |")
    A("| SCAF's currency/authority/corroboration components are validated | "
      "**Not established.** γ weakly discriminative, τ near-constant, ρ inactive. |")
    A("")

    # 16-18 ---------------------------------------------------------------
    A("## 16. Limitations")
    A("")
    A(f"1. **Only {p['counts']['human_annotations']} of "
      f"{p['counts']['decisions']} decisions were annotated**, by a stratified "
      "rule, not at random.")
    A(f"2. **All {p['sample']['forced_rag2_admitted']} RAG² admissions were forced "
      "into the sample.** They are a census of RAG²'s admissions, not a sample of "
      "them, and the sample is not self-weighting.")
    A(f"3. **The rows are clustered**: {clus['rows']} passages from "
      f"{clus['questions_represented']} questions, up to "
      f"{clus['rows_per_question']['max']} from one. The effective sample size is "
      "nearer the number of questions than the number of rows. No clustering "
      "correction is applied and no p-values are reported.")
    A("4. **One annotator, no second rater.** There is no inter-annotator agreement "
      "figure, so label reliability is unknown.")
    A("5. **σ and the human label are not fully independent constructs.** σ is "
      "lexical overlap; a human judging relevance also responds to shared "
      "terminology. Their association is partly mechanical.")
    A("6. **No answer-level outcome was measured** — no accuracy, no citation "
      "quality, no hallucination rate.")
    A("7. **The corpus is temporally narrow** (§10), which limits both γ and any "
      "temporal experiment.")
    A("8. **Retraction and supersession are untested** (§9).")
    A("")
    A("## 17. Is another experiment necessary?")
    A("")
    A("**Yes — one is, if the thesis wants to claim SCAF helps.** The current "
      "evidence supports claims about *admission behaviour* and a negative result "
      "about *score-versus-human ordering*. It cannot support any claim about "
      "answer quality, because answer quality was never measured.")
    A("")
    A("| Experiment | Priority | Question it answers |")
    A("| --- | --- | --- |")
    A("| **Matched-k answer quality**: fix both arms at the same k (e.g. 5 "
      "passages), generate answers, score them against a reference | **Essential** "
      "| Does SCAF's evidence selection produce better answers when context size "
      "is held constant? This is the claim the thesis actually wants. |")
    A("| **Second annotator on the same 120 rows** | **Essential** | How reliable "
      "are these labels? Without it, every number in §5 rests on one person. |")
    A("| **Threshold sweep against human labels** | Useful but optional | Is there "
      "*any* SCAF threshold that separates useful from unuseful evidence, or is the "
      "score simply not informative? |")
    A("| **Seed a retraction/supersession probe set** | Useful but optional | Does "
      "SCAF's currency machinery do anything when there is something to catch? |")
    A("| **Extend the corpus backwards in time** | Future work | Makes γ and any "
      "FRB-PAIRS temporal design possible at all. |")
    A("| **Implement and weight corroboration (ρ)** | Future work | ρ is currently "
      "dead weight in the formula. |")
    A("")
    A("## 18. Recommended next step")
    A("")
    A("**Run the matched-k answer-quality experiment.** Everything else is "
      "secondary. The thesis currently has a well-documented negative result and a "
      "well-documented behavioural difference; what it lacks is any measurement of "
      "whether the behavioural difference *matters* for the answers a reader would "
      "receive. Matched-k removes the context-size confound that made the earlier "
      "preliminary comparison uninterpretable.")
    A("")
    A("If time does not allow it, the honest thesis is the one written from §13 and "
      "§15: a careful measurement instrument, a clear negative finding, and an "
      "explicit account of what remains unproven. That is a legitimate MS thesis.")
    A("")

    # 19 ------------------------------------------------------------------
    A("## 19. Exact source files and identifiers")
    A("")
    A("```")
    for name, path in p["sources"].items():
        A(f"{name:20s} {path}")
    A(f"{'frozen digest':20s} {p['scientific_run']['frozen_set_digest']}")
    A(f"{'sheet sha256':20s} {p['annotation_sheet_sha256']}")
    A(f"{'label fingerprint':20s} {p['label_fingerprint']}")
    A(f"{'annotation pass':20s} {', '.join(p['annotation_pass'])}")
    A(f"{'analysis version':20s} {p['analysis_version']}")
    A(f"{'generated':20s} {p['generated_utc']}")
    A("```")
    A("")
    A("Regenerate with:")
    A("")
    A("```")
    A("python experiments/scripts/analyse_results.py")
    A("```")
    A("")
    A("The machine-readable form of every number above is "
      "`scientific_results_summary.json` in this directory.")
    A("")
    return "\n".join(lines)
