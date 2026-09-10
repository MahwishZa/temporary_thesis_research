# RAG² reproduction audit

**Audited:** the reproduction of the original RAG² system supplied as
`rag2reproduction.zip`, now integrated at `rag2/`.
**Against:** Sohn et al., *Rationale-Guided Retrieval Augmented Generation for
Medical Question Answering*, NAACL 2025
([2025.naacl-long.635](https://aclanthology.org/2025.naacl-long.635/); arXiv
2411.00300), read in full including Appendix A, plus the authors' released code
under `architecture/rag2/retriever/` and `architecture/rag2/classifier/`.
**Date:** 2026-09-04.
**Verdict (§9):** READY WITH MINOR FIXES — the fixes are applied.

This document is evidence, not assurance. Where something was checked, it says
what was checked and how. Where something could not be checked in this
environment, it says so and does not soften it.

---

## 1. What the original RAG² system does

RAG² is a four-stage, single-pass pipeline. There is no iteration and no second
retrieval round.

```
question x
  │
  ├─[1] Rationale-based query formulation      (§3.3)
  │      base LLM + chain-of-thought prompt  ──►  rationale r
  │      r REPLACES x as the retrieval query
  │
  ├─[2] Balanced retrieval + reranking          (§3.4)
  │      MedCPT query encoder → FAISS MIPS, k candidates from EACH corpus
  │      pooled ──► MedCPT cross-encoder rerank ──► top-k snippets
  │
  ├─[3] Rationale-guided filtering              (§3.2)
  │      Flan-T5-large scores each (question, snippet) pair independently
  │      keep only [HELPFUL]
  │
  └─[4] Answer generation
         base LLM(question + kept snippets) ──► answer
```

The three claimed innovations, and the one quantity the thesis cares about:

**Rationale as query.** The paper is explicit that the initial query is *not*
concatenated: "We search for document snippets solely using the rationale,
excluding the initial query. Including both the initial query and the rationale
exceeds the maximum length of the retriever and … leads to suboptimal
performance" (§3.3). The same LLM does rationale generation and QA.

**Balanced retrieval.** "This approach extracts an equal number of documents
from each corpus, ensuring that all corpora are represented more evenly" (§3.4).
Four corpora, from Self-BioRAG (Appendix Table A1): PubMed 36.5M docs / 69.7M
passages, PMC 1.1M / 46.3M, CPG 35.7k / 607.0k, Textbooks 18 / 134.0k —
37.6M docs, 116.7M passages, 564.2 GB.

**Rationale-guided filtering, trained on perplexity-derived labels.** A
Flan-T5-large filter (770M — "This smaller model can be trained on a single RTX
3090 24G GPU") emits `[HELPFUL]` or `[NOT_HELPFUL]` for one snippet at a time
("the Flan-T5 model can filter only one snippet at a time due to its limited
context length", Limitations).

**The confidence-derived evidence utility signal — what the thesis studies.**
Equation 4 defines two perplexities and Equation 3 their difference:

```
PPL(x)    = exp( −(1/L) · Σ log P(xᵢ | x<ᵢ) )
PPL(x, d) = exp( −(1/L) · Σ log P(xᵢ | x<ᵢ, d) )
ΔPPL      = PPL(x) − PPL(x, d) ≥ τ
```

τ is "a threshold set to select the top percentage of perplexity differentials",
fixed at the **top 25%** across all experiments (§3.2). ΔPPL feeds Figure 2's
labeling tree, which I transcribed directly from the PDF text layer:

| Correct w/o retrieval | Correct w/ retrieval | ΔPPL ≥ τ | Label |
| --- | --- | --- | --- |
| Yes | Yes | Yes | `[HELPFUL]` |
| Yes | Yes | No  | *discard* |
| Yes | No  | –   | `[NOT_HELPFUL]` |
| No  | Yes | –   | `[HELPFUL]` |
| No  | No  | Yes | `[NOT_HELPFUL]` |
| No  | No  | No  | *discard* |

Reference results (Table 2, accuracy %), verified against the PDF:

| Backbone | | MedQA | MedMCQA | MMLU-Med | Avg |
| --- | --- | --- | --- | --- | --- |
| Llama-3-8B-Instruct | no RAG | 57.7 | 53.5 | 69.5 | 60.2 |
| | + RAG² | **64.6** | **59.4** | **74.8** | **66.3** |
| Meerkat-7B | no RAG | 71.2 | 60.8 | 73.8 | 68.6 |
| | + RAG² | **75.6** | **63.0** | **78.7** | **72.4** |
| GPT-4o | no RAG | 88.5 | 76.7 | 92.8 | 86.0 |
| | + RAG² | **91.1** | **77.2** | **92.5** | **86.9** |

**Nothing in RAG² represents time.** No component reads a publication date. That
absence is the thesis's opening.

---

## 2. Repository implementation inventory

The zip contained two things that must not be confused, and the reproduction
already keeps them apart. Integrated at `rag2/`:

| Path | What it is | Provenance |
| --- | --- | --- |
| `architecture/rag2/retriever/` | MedCPT encoding, MIPS, reranking | **the authors' released code, unmodified** |
| `architecture/rag2/classifier/` | Flan-T5 filter training + metrics | **the authors' released code, unmodified** |
| `architecture/rag2/rag2/` | the reproduction package (31 modules) | this work |
| `architecture/rag2/configs/` | 10 experiment configs (YAML, with inheritance) | this work |
| `architecture/rag2/scripts/` | 8 scripts — one CLI per stage, plus the smoke test and its fixture builder | this work |
| `architecture/rag2/tests/` | 15 test modules | this work |
| `docs/rag2_reproduction.md` | the reproduction's own specification, ~700 lines | this work |
| `THESIS.md` | results template, **blank — nothing run yet** | this work |

Reproduction package layout:

```
architecture/rag2/rag2/
  config.py            typed config; every key tagged [S] specified or [A] assumed
  schema.py            Question / Evidence / CandidateSet / FilterDecision / PipelineResult
  prompts.py           versioned templates, hashed into every run manifest
  rationale.py         stage 1
  retrieval/           encoder.py, index.py, balanced.py, rerank.py — stage 2
  filtering/           base.py (ABC), rag2_filter.py, passthrough.py — stage 3
  filter_training/     perplexity.py, labeling.py, build_labels.py, train.py — ΔPPL + Figure 2
  generation.py        stage 4
  corpora/             base.py, json_corpus.py, thesis_chunks.py (added, §6)
  datasets/            MedQA / MedMCQA / MMLU-Med / generic JSONL loaders
  llm/                 HF, vLLM, OpenAI, stub backends behind one ABC
  cache.py             fingerprinted candidate save/replay
  evaluation.py        answer extraction, accuracy, filter metrics, ROUGE-L / BERTScore
  experiment.py        run manifests and seeding
  pipeline.py          stage orchestration
```

The reproduction's specification document is unusually disciplined: it tags every
claim **[S]** specified / **[A]** assumed / **[U]** unavailable / **[D]** paper
and code disagree, and every **[A]** carries the config key that controls it. I
spot-checked its claims against the PDF rather than accepting them (§3).

Prior thesis work already in the repository, untouched by this task: `pmc/`
(acquisition, parsing, QC, M1–M4 corpus metadata, chunking, MedCPT embedding,
exact retrieval) and `pubmed/`.

---

## 3. Component-by-component fidelity assessment

Classification per the brief. "Verified how" says what I actually did — running
the code, or reading it against the PDF and the released artifacts.

### 3.1 Retrieval — **CORRECT / FAITHFUL**

| Property | Paper / release | Reproduction | Verified how |
| --- | --- | --- | --- |
| Query encoder | `ncbi/MedCPT-Query-Encoder` | same | config default, pinned by test |
| Article encoder | `ncbi/MedCPT-Article-Encoder` | same | config default |
| Reranker | `ncbi/MedCPT-Cross-Encoder` | same | config default, pinned by test |
| Pooling | CLS, `last_hidden_state[:, 0, :]` | same | read `encoder.py` against `retriever/query_encode.py` |
| Dimension | 768 float32 | 768 | config, and the index-alignment check |
| Similarity | inner product, `IndexFlatIP` | inner product; faiss when present, exact numpy fallback | read `index.py` |
| Exact vs approximate | exact (flat) | exact, both backends | read `index.py` |
| Balance | equal quota per corpus | equal quota per corpus | smoke test asserts 3 from each of 4 corpora |
| Truncation | 512 tokens | 512 | config `[S]` |

Three defects in the authors' own retriever are fixed rather than copied:
hard-coded `cuda:7`, batch size pinned to 1, and an O(n²) `np.vstack` rebuilt
inside the encoding loop. None changes retrieval semantics.

Two behavioural discrepancies between the paper and the released code are
**exposed as switches instead of being silently resolved** — the right call, and
the reason I rate this faithful rather than approximate:

* **`retrieval.rerank_query`** (default `initial`). The paper says the initial
  query twice — "a reranker is used to rerank the retrieved snippets by
  cross-encoding the **initial query** and each snippet" (Figure 1 caption; I
  confirmed this string in the PDF) and §3.4 — while `retriever/main.py:124`
  passes the rationale file. Default follows the paper; `rationale` reproduces
  the code.
* **`retrieval.shard_merge`** (default `score`). `retriever/main.py` retrieves
  `top_k` from *each* PubMed shard group; with the 10/10/10/8 grouping the
  release's README documents, PubMed contributes 4×`top_k` against `top_k` from
  every other corpus — a 4:1 over-representation that inverts the paper's stated
  intent. The reproduction shards for memory but merges by score, keeping the
  quota exact. `concat` reproduces the release.

Also corrected: the release overloads one `--top_k` as both per-corpus retrieval
depth and final reranked count; the paper distinguishes them (Figure 3 sweeps the
final k over {1,2,4,8,16,32}). Split into `candidates_per_corpus` and
`final_top_k`.

**Determinism.** Search is exact, so there is no ANN nondeterminism. Reranking
sorts with Python's stable `sorted`, so ties keep pooled order. One residual: the
numpy fallback and faiss may break exact score ties differently. This has never
mattered in practice and is noted, not fixed.

### 3.2 Rationale generation — **CORRECT / FAITHFUL**

The prompt is reproduced verbatim from §3.3 (the paper prints it in full); I
matched it string-by-string against the PDF text.

The brief warns that "a component that merely generates a rationale but does not
use it downstream should NOT automatically be considered faithful". It is used:
`pipeline.run_retrieval` builds `queries = [retrieval_query(rationale, q) …]` and
those strings — the rationale alone, not the question — are what the encoder
embeds. `rationale.retrieval_query` returns the question **only** as a fallback
when the rationale is empty, or when `use_rationale=False` reproduces the paper's
plain-MedCPT baseline row. The rationale is also carried into every
`CandidateSet` and survives the cache.

Option serialisation was under-specified by the paper; the reproduction recovered
it from the authors' released training data. **I verified this by round-trip**:
rendering a question from record 0 of
`classifier/data/medqa/llama3_cot/5%-train.json` through `PromptSet` reproduces
the released string byte-for-byte, including the `A) … B) … C) … D) …` inline
form.

### 3.3 Rationale-guided filtering — **CORRECT / FAITHFUL**

The filter input template is **not** printed in the paper. The reproduction
recovered it from the released artifact, and I confirmed it directly: every
record uses

```
Given the following evidence, determine whether it helps answer the provided question.

Evidence: {snippet}

Question: {question_with_options}
```

with `\n\n` separators and the target `[HELPFUL]` / `[NOT_HELPFUL]`. Rendering
through `PromptSet.render_filter_prompt` reproduces it exactly. Note what this
settles: the filter sees the **initial question**, not the rationale — consistent
with Figure 1, whose filter box reads "Snippet + Initial Query".

Scoring reproduces `classifier/run_classifier.py:696-712`: take the logits at the
first decoded position, softmax over the two label columns *only*, argmax.
Implemented as one forward pass with `decoder_input_ids=[[decoder_start]]`
instead of a `generate()` call per snippet. Training-record schema matches the
release's four fields (`id`, `answer`, `dataset_name`, `question`) exactly.

Per the paper's Limitations, snippets are scored **individually** — `decide()`
renders one input per candidate. Not batched into a joint prompt.

One documentation defect found and fixed here — see §6.1.

### 3.4 Confidence-derived evidence utility (ΔPPL) — **MOSTLY CORRECT**

This is the component the thesis studies, so I read it hardest.

Correct and verified: Equation 4's length-normalised exponential of mean token
log-probability; Equation 3's `PPL(x) − PPL(x, d)`, positive when the document
raised confidence; τ as the 75th percentile with the test `ΔPPL ≥ τ` matching
Equation 3's `≥`; `tau_percentile: 25` matching the paper's fixed top-25%; and
Figure 2's tree transcribed **exactly** — I extracted the figure's node and leaf
text from the PDF and compared it against `decide_label` branch by branch. It
matches on all six paths. `tests/test_labeling.py` asserts the truth table
verbatim.

Not fully determined by the paper, hence *mostly* correct — two documented
assumptions that a reader must know about:

1. **Which tokens are scored.** Equation 4 literally sums over the *query* `x`
   ("The perplexity for the input query x and the document d are calculated as
   follows"). The reproduction scores the **rationale** instead. I checked the
   evidence for that reading and it is strong: the abstract says
   "perplexity-based labels **of rationales**"; §2.1 says "our method measures
   perplexity differences in the **rationales generated by the base LLM**"; and
   Figure 2's own structure — which I extracted — shows the LLM emitting an
   "Output Answer" feeding the two correctness tests and a "Rationale" feeding
   the "Lower Perplexity" node. The literal reading of Equation 4 is almost
   certainly a typo. `filter_training.ppl_target: query` restores it, so the
   choice is testable rather than assumed.
2. **Conditioning context.** Equation 4's first term has no conditioning at all;
   the reproduction conditions on the closed-book answer prompt so that ΔPPL
   isolates the document as the single intervention. Defensible, and the
   alternative is not well defined.

Also assumed and flagged: τ taken globally per split rather than per question
(`tau_scope`); the same no-retrieval rationale scored in both terms, so ΔPPL is
not confounded by two different generations' length and content.

**Exposure of intermediate values — the thing the thesis needs.** Adequate. Each
labeled pair carries a provenance sidecar with `delta_ppl`, the τ that applied,
both correctness booleans, `ppl_without` / `ppl_with` / `num_tokens`, and the
evidence's `source` / `doc_id` / `passage_id`. At inference, `FilterDecision`
carries `P([HELPFUL])` as a continuous `score`, not just the boolean. So the
recency-bias probe can regress admission on date without touching baseline code.

**One unresolved oddity, correctly flagged and not explained away.** The released
artifact is named `5%-train.json` with ids like `llama3_5%_23600`, while the
paper states τ = top 25%. What "5%" denotes is undocumented, and it is not a 5%
subsample in the obvious sense — I confirmed the ids run past 23,000 across only
5 shipped records. The reproduction follows the paper's 25% and records the
conflict. That is the right call; it is also a real gap in knowledge about the
authors' actual labeling run.

**Edge case, minor:** a zero-token scored sequence yields `PPL = inf`, so
`ΔPPL` becomes `inf` or `nan`. `percentile()` filters non-finite values when
computing τ, but an `inf` delta would still pass `≥ τ`. Only reachable with an
empty rationale. Noted, not fixed — a fix would be speculative.

### 3.5 Answer generation — **APPROXIMATION** (unavoidable)

Correct and verified: the same backbone LLM as rationale generation (§3.3);
conditioned on the initial question plus the kept snippets; filtering applied
before generation; greedy decoding at temperature 0 (§A.3); provenance retained
on every kept `Evidence` so citations can be reconstructed.

**The prompt itself was never published.** Neither the paper nor the release
contains it; Figure 1 shows only the structure `[Snippet₁ … Snippet_k] + [Initial
Query]`. The reproduction reuses the paper's own CoT prompt with an evidence
block prepended, on the reasoning that the paper states one LLM does both jobs
and Meerkat was instruction-tuned on that prompt. This is the **single largest
prompt-level assumption in the reproduction**, and the reproduction says so in
those words. It is the first thing to vary if accuracy misses the paper.

The rationale is **not** included in the answer prompt. The paper does not say to
include it, and Figure 1 does not show it.

Answer extraction is also unspecified by the paper. An ordered pattern list is
used, last match wins, unmatched counts as incorrect (never as abstention).
Documented, overridable, and pinned by tests.

### 3.6 Reproducibility — **CORRECT / FAITHFUL**, better than the original

Every run writes a manifest: resolved config, config and retrieval fingerprints,
git commit and dirty flag, prompt version and hash, package versions, hardware
and CUDA state, the full command line, and the seed. Candidate sets persist with
a retrieval fingerprint; replaying a cache built under a different retrieval
config **raises** unless explicitly allowed. That is what turns "only the filter
changed" into a checked claim, and it is precisely the thesis's V3 validity
control.

The original left `--seed` unset, so an exact match to the authors' checkpoint is
impossible in principle. The reproduction seeds Python, numpy, torch and
transformers, and records what it seeded. This is documented, not hidden.

### 3.7 Baseline/thesis separation — **CORRECT / FAITHFUL**

I checked this by grep rather than by reading the claim: no module under
`architecture/rag2/rag2/` references SCAF, recency, currency, supersession, retraction,
contested evidence, abstention, entailment, authority tiers or temporal logic in
executable code. The only hits are docstrings explaining that provenance is
carried and not consulted.

`tests/test_metadata_isolation.py` enforces it with a token-level scanner that
skips comments and docstrings, plus — and this is the part that makes it worth
having — a companion test that feeds the scanner three real violations
(`evidence.metadata['publication_date']`, `compute_recency(evidence)`,
`evidence.pub_date > cutoff`) and asserts it fires on each, and two
documentation-only mentions and asserts it does not. It also proves the pipeline
produces identical output when dates are stripped and when they are permuted.

---

## 4. Known approximations

Ordered by how much they could move a result.

| # | Approximation | Why it exists | Effect |
| --- | --- | --- | --- |
| 1 | **Different corpus.** The paper's 564 GB Self-BioRAG corpus is not redistributable. This repository uses its own Alzheimer/dementia corpus. | Corpus not available | **Dominant.** Accuracy is not comparable to Table 2 in absolute terms. Expect a gap; explain it, do not tune it away. |
| 2 | **Three corpora, not four.** The paper balances over PubMed / PMC / CPG / Textbooks; this corpus has `pubmed-abstract`, `pmc-fulltext`, `currency-pack`. | Domain-scoped corpus; no textbook layer | Balanced retrieval still functions — the quota simply splits three ways. The *mechanism* is intact; the *mixture* differs. |
| 3 | **Reconstructed filter checkpoint.** The paper's is "not available for distribution". | Not distributed | Must be retrained. Different base-LLM generations → different ΔPPL → different labels → a different filter. Unseeded original: bit-identity impossible. |
| 4 | **Answer-generation prompt** (§3.5). | Never published | Directly moves accuracy. First thing to vary. |
| 5 | **ΔPPL scored over rationale tokens** (§3.4). | Equation 4 vs. prose contradiction | Changes labels, hence the filter. Switch exists. |
| 6 | **τ taken globally per split.** | Population unstated | Changes the helpful/not-helpful balance. Switch exists. |
| 7 | **Answer extraction rule.** | Unspecified | Small but real; unparsed counts as incorrect. |
| 8 | **Per-corpus retrieval depth = 100.** | Never published; the release's default | Changes what the reranker sees. |
| 9 | **Meerkat-7B checkpoint id.** | Paper cites the work, not a path | Should be confirmed before a Meerkat run. |
| 10 | **Filter inference scores one window per pair** (truncate, not stride). | Deliberate correction of a release bug — see §6.3 | More correct than the release. |

---

## 5. Missing components

| Component | Status | Reconstructable? |
| --- | --- | --- |
| Trained Flan-T5 filter checkpoint | Not distributed by the authors | **Yes** — rerun labeling + training (`scripts/03`, `scripts/04`). Not bit-identical. |
| The four biomedical corpora + MedCPT embeddings (564 GB) | Not distributed | **In principle**, at substantial cost. Out of scope: this thesis uses its own corpus. |
| `classifier/data/preprocess.py` (the labeling code) | **Empty file, 0 bytes**, upstream too | **Yes** — reconstructed from Figure 2 + §3.2 in `filter_training/labeling.py`. |
| Rationale files (`*_llama_cot.json`) | Not distributed | **Yes** — regenerate with the base LLM. |
| Answer-generation prompt | Never published | **No.** Assumed (§3.5). |
| Sliding-window chunk size / stride | Never published | **No.** This repository sets its own (256 words, 32 overlap). |
| Random seeds | Original unseeded | **No.** |
| Meaning of "5%" in the released filename | Undocumented | **No.** |
| Per-corpus retrieval depth | Never published | **No.** Assumed 100. |
| Optimal top-k for MMLU-Med and GPT-4o | Not recoverable from the figures | **No.** Re-select on validation. |
| GPT-4o snapshot | Not pinned | **No.** Not needed here. |

Nothing on this list is missing *from the reproduction*. Each is missing from
what the authors released, and each is either reconstructed or flagged.

---

## 6. Corrections made

Deliberately few. The reproduction was already sound; the brief's instruction was
not to rewrite what is correct.

### 6.1 A documentation claim that was not true (fixed)

Both `architecture/rag2/rag2/filtering/rag2_filter.py` and `docs/rag2_reproduction.md` §5.6 stated
that `tests/test_filter_scoring.py` "pins the equivalence" between the
single-forward-pass scoring path and the release's
`generate(..., output_scores=True).scores[0]`. **It does not.** That file's only
torch-dependent test pins the two-way softmax helper against `torch.softmax`;
nothing anywhere exercised the two logit-extraction paths. I confirmed this by
searching the whole test tree for `use_generate`, `_first_token_logits`,
`scores[0]` and `decoder_input_ids` — no matches.

The underlying claim is almost certainly true (the first decoded position depends
only on the encoder output and the decoder start token). But an unpinned claim
described as pinned is exactly the kind of thing this audit exists to catch.

Fixed two ways:

* Added `architecture/rag2/tests/test_filter_first_token_logits.py` (5 tests) covering the
  half this repository owns: both branches read decoder position 0, use the
  configured start token (falling back to pad), slice the same two label columns,
  and reach the same `P([HELPFUL])`. It poisons decoder positions 1+ so reading
  the wrong position fails loudly.
* Rewrote both claims to state precisely what is asserted and what is not — that
  `generate().scores[0]` *is* that forward pass is a property of `transformers`,
  not of this code, and `filter.options.use_generate: true` exists to run the
  literal release path if it is ever in doubt.

**These tests require torch and therefore SKIP in this environment.** They are
written to run where torch exists. I have not seen them pass.

### 6.2 The baseline could not read this repository's corpus (fixed)

The brief requires the baseline to "consume the approved retrieval outputs". It
could not. `corpora/json_corpus.py` reads the release's layout
(`*_Articles_*.json` + `*_Embeds_*.npy`); `pmc/` produces `chunks.jsonl` plus
`embeddings.f32` + `index_manifest.jsonl` + `index_meta.json`. No adapter existed.

Added `architecture/rag2/rag2/corpora/thesis_chunks.py`, registered as loader
`thesis_chunks`, plus `architecture/rag2/configs/thesis_corpus.yaml`. It is additive: no
baseline module changed. Design points, each driven by something real in the
existing artifacts:

* **The index manifest carries no chunk text** — `embed_chunks.MANIFEST_FIELDS`
  omits it by design — so text is joined back from `chunks.jsonl` on `chunk_id`,
  and a mismatch between the two files is a hard error naming the first offender.
* **Row order is the alignment contract.** Manifest line *i* describes row *i* of
  `embeddings.f32`. The loader verifies the declared `row` against its position
  and refuses to start otherwise: a silent misalignment would return the wrong
  passage for every hit.
* **A byte-size check** against `rows × dim × 4` catches a truncated or
  half-written index.
* **One instance per `source_category`**, which is what keeps balanced retrieval
  meaningful over three corpora instead of four.
* **`require_production_index: true` by default** — a stub-encoder index is
  refused, mirroring the guard already in `preprocessing/pmc/embed_chunks.py`.
* **Provenance carried, never promoted into text.** Every manifest field except
  identity lands in `Evidence.metadata`, so `canonical_date`,
  `authority_tier_label`, `in_currency_pack` and `retracted` reach the thesis
  layer without any baseline component reading them.
* Vectors are memory-mapped and yielded in bounded shards, so a category holding
  hundreds of thousands of rows does not materialise at once.

18 tests added (`architecture/rag2/tests/test_thesis_chunk_corpus.py`), including an
end-to-end check that `balanced_retrieve` draws an equal quota from all three
corpora through this loader, and a guard pinning `configs/thesis_corpus.yaml` to
MedCPT, 768 dims, `rerank_query: initial` and `shard_merge: score`.

### 6.3 Corrections the reproduction had already made (verified, kept)

Not my work — recorded because they are deviations from the released code and a
reader must know they are deliberate:

* **Filter eval desync.** `run_classifier.py:737` zips per-*feature* predictions
  against per-*example* ids, so when any input overflows 512 tokens the two lists
  desynchronise and reported accuracy is wrong. The reproduction scores one
  window per pair at inference (`filter.overflow: truncate`); `stride` reproduces
  the windowing but aggregates by max helpfulness instead of desynchronising.
* **`token_add.ipynb` loads a T5 checkpoint with `AutoModelForCausalLM`**, which
  is wrong for a seq2seq model. The reproduction uses `AutoModelForSeq2SeqLM`
  throughout, matching `classifier/utils.py:43`.
* The retriever defects listed in §3.1.

### 6.4 What I deliberately did not change

* `architecture/rag2/retriever/` and `architecture/rag2/classifier/` — the authors' release stays byte-
  identical so it remains citable.
* The two `[D]` switches keep their paper-following defaults. They are research
  questions, not bugs.
* `json_corpus.embedding_shards` materialises a non-float32 `.npy` into RAM
  (`np.asarray(memmap, dtype=float32)` copies when the stored dtype differs; I
  verified both branches). Harmless for MedCPT, which is float32 natively, and
  irrelevant to this repository's loader. Documented, not fixed.
* The `PPL = inf` edge case (§3.4). A fix would be invented behaviour.

---

## 7. Tests performed

All commands run from `rag2/`.

**Full suite — `python3 -m pytest`: 176 passed, 1 skipped, 2.05 s** (before my
additions). The skip is `test_filter_scoring.py:42`, which needs torch.

After the additions, on the same environment:

| Suite | Result |
| --- | --- |
| `rag2/` full suite | **194 passed, 2 skipped** (both skips are torch-gated modules) |
| `architecture/rag2/tests/test_thesis_chunk_corpus.py` (new) | **18 passed** |
| `architecture/rag2/tests/test_filter_first_token_logits.py` (new) | 5 tests, **module skipped** — needs torch |
| `pmc/` (7 modules) | **311 passed** — unaffected |
| `pubmed/` (2 modules) | **51 passed** — unaffected |

Environment: Python 3.11.15, numpy 2.4.6, pytest 9.1.1. **No torch, no
transformers, no faiss, no GPU.**

Independent verification I performed against primary sources, not against the
reproduction's own documentation:

* Extracted the paper's text layer and confirmed verbatim: the τ = top-25%
  sentence; "Flan-T5-large, which has only 770 million parameters"; "cross-
  encoding the initial query and each snippet"; "extracts an equal number of
  documents from each corpus"; "solely using the rationale, excluding the initial
  query"; "the same LLM is used both for rationale generation and QA"; "measures
  perplexity differences in the rationales generated by the base LLM"; "filter
  only one snippet at a time"; the Equation 3/4 block; the CoT prompt.
* Reconstructed **Figure 2's decision tree** from the PDF text layer and compared
  it path-by-path with `decide_label`. Six of six match.
* Confirmed **every Table 2 number** cited in the reproduction's §11, and checked
  the Table 3 / Table 4 attributions (which are interleaved in the extraction and
  easy to swap). Both correct.
* Confirmed Table 1's split sizes: MedQA 10,178 / 1,272 / 1,273; MedMCQA 182,822
  / 4,183 / 6,150; MMLU-Med 1,089.
* **Round-tripped the filter prompt and option serialisation** against the
  authors' released `5%-train.json`: byte-identical.
* Grepped `architecture/rag2/rag2/`, `architecture/rag2/retriever/`, `architecture/rag2/classifier/`, `architecture/rag2/configs/`
  and `architecture/rag2/scripts/` for thesis-concept leakage. None in executable code.
* Verified that all six stage CLIs plus the fixture builder load and parse their
  arguments, and ran `smoke_test.py` in full.

---

## 8. End-to-end smoke-test status

`python3 scripts/smoke_test.py` — **PASSED**, ~2 s, no GPU, no downloads.

It exercises the full path question → rationale → balanced retrieval → rerank →
cache → ΔPPL labeling → filter → answer → evaluation, with 23 assertions:

```
stage 1-2   one candidate set per question; balanced pool: 3 candidates from
            each of 4 corpora; every corpus represented; rationales generated;
            ranks assigned by the reranker
stage 2b    cache round-trips every question; provenance survives
            (pubmed-doc4 / 2004-01-01)
stage 3a    labels produced -- 11 of 24 observations (13 discarded by Figure 2);
            labels are the two filter tokens; training records match the release
            schema; filter input uses the released template
stage 3b-4  filtering and generation; one decision per candidate; predictions
            extracted
filter swap passthrough keeps every candidate ('RAG2 w/o filter');
            all-filtered-out falls back to closed-book; keep_top1 restores top-1
stage 5     accuracy computed; evidence report computed
isolation   no publication date appears in any model input -- 6 dates carried
            as metadata
```

**What this proves:** every stage is wired to the next, the data contracts hold,
the filter is genuinely swappable, provenance survives end to end without
reaching a model, and Figure 2's discard branch actually fires.

**What it does not prove, and must not be read as proving:** any accuracy number.
The smoke test uses a stub LLM and a hash-based query encoder, and says so
itself — "stub LLM: the accuracy number above is meaningless by construction".
The 33.3% it prints is an artifact of the stub.

**Blocked components, named exactly.** These need weights this container does not
have, and I did not fabricate a run of any of them:

| Component | Blocked on | Structurally verified? |
| --- | --- | --- |
| MedCPT query encoding / reranking | torch, transformers, `ncbi/MedCPT-*` weights | Yes — code read against the release; stub encoder exercises the same path |
| Flan-T5 filter inference | torch, transformers, a trained checkpoint (not distributed) | Yes — scoring rule pinned by pure-Python tests; new tests cover the logit paths but skip without torch |
| ΔPPL over a real LLM | torch, an 8B backbone | Yes — formula and tree pinned; stub LLM exercises the path |
| Answer generation | torch, an 8B backbone | Yes — stub backend exercises the path |
| faiss `IndexFlatIP` | faiss | Yes — numpy fallback is mathematically identical and is what ran |
| Retrieval over the real index | `indexes/production/` (built separately on Windows, still running) | Loader verified against synthetic fixtures with the real schema |

**Not run, and not runnable here:** the reproduction over real data. There are no
measured accuracy numbers in this repository, and
`THESIS.md` correctly says "**Status: not yet run**" with
every result cell blank and an explicit instruction to "leave a row blank rather
than estimating it". I verified that document contains no fabricated results.

---

## 9. Baseline readiness verdict

## **READY WITH MINOR FIXES** — and the fixes in §6.1 and §6.2 are applied.

The reproduction is faithful to the original RAG² method. Every stage the paper
specifies is implemented and wired; the three claimed innovations are all present
and genuinely operative (the rationale really does replace the query; the quota
really is equal per corpus; the filter really is trained on Figure 2's
perplexity-derived labels); the prompts that could be recovered are recovered
verbatim and I confirmed them against the authors' own artifact; and where the
paper is silent or contradicts its own code, the reproduction exposes a switch
and documents the reading it takes rather than choosing quietly.

It is a defensible baseline for the thesis. Two qualifications, neither
disqualifying:

1. **Nothing has been measured yet.** "Faithful implementation" is what I can
   attest; "reproduces Table 2" is not, and cannot be until the filter is trained
   and the pipeline is run. Read §8's blocked-components table before quoting any
   readiness claim.
2. **The corpus differs from the paper's**, so absolute accuracy will not match
   Table 2. That is expected and is a finding to explain, not a defect to tune
   away.

What must happen before baseline numbers exist, in order: build the MedCPT index
(running separately) → generate rationales → retrieve and cache → build ΔPPL
labels → train the filter → run the pipeline → evaluate → record in
`THESIS.md` with manifest fingerprints.

---

## 10. Remaining limitations

**Unverified in this environment** — no torch, transformers, faiss or GPU:

* Every model-dependent path. MedCPT encoding and reranking, Flan-T5 filter
  inference, ΔPPL under a real LLM, and answer generation have been verified by
  code reading and by stub-driven execution of the same code path, not by running
  a real model.
* The 5 new tests in `test_filter_first_token_logits.py` — written, never seen to
  pass.
* That `generate().scores[0]` equals the forward-pass logits (§6.1). Argued from
  decoding semantics; not asserted.
* Retrieval over the real `indexes/production/`, which does not exist yet in this
  container.

**Unverifiable from here regardless of environment:**

* The reproduction's claim that `architecture/rag2/retriever/` and `architecture/rag2/classifier/` are
  byte-identical to upstream `dmis-lab/RAG2 @ 86add43`. This session's GitHub
  access is scoped to `MahwishZa/thesis_research`, so I could not clone upstream
  to diff. The claim is plausible and internally consistent, but **it is the
  reproduction's claim, not my finding.**
* Whether the authors' undocumented "5%" artifact reflects a different threshold
  than the paper's 25% (§3.4).

**Known gaps in the artifact itself:**

* `THESIS.md` is a blank template.
* The committed `data/corpora/pmc/chunks/chunk_stats.json` records a **partial** container run
  (60,874 chunks over 76 parsed records, most documents abstract-only), not the
  production Windows run reported as 42,964 documents / 781,563 chunks / 773,183
  unique / 8,380 duplicates flagged, digest `da1886b0…`. I did not have the full
  parsed corpus here and did not overwrite the file with a number I could not
  reproduce. Refresh it from the production run.
* No MedQA/MedMCQA/MMLU-Med data is present, so the dataset loaders are exercised
  only against fixtures.
* `configs/thesis_corpus.yaml` leaves `dataset.path` and `filter.checkpoint`
  empty by necessity; both must be set per run.

**Method-level limitations inherited from the original**, worth stating because
the thesis will cite them: the filter judges one snippet at a time with no
cross-snippet reasoning; there is no second retrieval round; τ is a single fixed
quantile; and **no component represents time in any way.**

---

## 11. How the thesis will later extend the baseline

The extension asks whether the confidence-derived utility signal of §3.4 admits
evidence at different rates depending on age — and if so, corrects it. Three
layers, one-directional, described in `experiments/README.md`.

**Layer 2 — recency-bias probe.** Observes; changes nothing. Everything it needs
is already produced and carried:

| Needed | Already available |
| --- | --- |
| per-candidate `P([HELPFUL])`, keep/drop | `FilterDecision.score` / `.keep` / `.label` |
| ΔPPL, both perplexities, the τ applied | `LabeledPair.provenance` sidecar |
| publication date, precision, split | `Evidence.metadata`: `canonical_date`, `date_precision`, `split_june_2024` |
| authority tier, currency-pack membership, retraction | `Evidence.metadata`, carried and unread |
| identical candidate population across arms | `rag2.cache` fingerprinted replay |

That last row is the important one. Validity control V3 requires upstream stages
to execute once and the candidate set to be replayed byte-identically to every
arm. `cache.load_candidates` refuses a cache whose retrieval fingerprint does not
match the current config unless mismatch is explicitly allowed — so "only the
admission policy changed" is enforced, not asserted.

**Layer 3 — SCAF.** Arrives as a new `EvidenceFilter` implementation registered
under its own key, selected by config. The seam already exists and is already
used by two other filters (`passthrough` for the paper's own "w/o filter"
ablation, `no_evidence` for the closed-book row), so SCAF needs **no baseline
file to change**. The baseline filter stays present and selectable, which is what
makes the comparison meaningful.

Ablation A12 treats authority ordering as a tested variable rather than a
hard-coded constant; nothing in the baseline presupposes an ordering, so that
stays open.

**The boundary is machine-checked.** `test_metadata_isolation.py` scans every
module under `architecture/rag2/rag2/` for executable references to publication, recency or
currency fields and fails if one appears — and proves the scanner works by
feeding it real violations. If SCAF logic is ever written into the baseline
instead of layered on top, the build breaks. That is deliberate: the thesis
measures the untouched original, so the original has to stay untouched.

**Not implemented anywhere in this repository, by instruction:** SCAF, recency
weighting, publication-date weighting, authority weighting, currency scoring,
supersession, contested-evidence handling, abstention, recency-bias correction,
temporal counterfactuals, significance analysis. The baseline comes first.
