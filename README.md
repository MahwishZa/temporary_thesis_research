# thesis_research

MS thesis research repository: **does confidence-based evidence admission in
retrieval-augmented medical QA prefer the past?**

Retrieval-augmented generation for clinical questions depends on a filter that
decides which retrieved snippets reach the answer model. RAG² (Sohn et al.,
NAACL 2025) makes that decision with a small model trained on perplexity-based
labels — a *confidence* signal. This project asks what that confidence signal
does with the **age** of evidence, and tests an alternative admission policy
(SCAF) that scores age explicitly instead of implicitly.

---

## Start here

**➡ [`THESIS.md`](THESIS.md) is the canonical research document.** Motivation,
research questions, both systems, the corpus, the experimental design, every
result obtained, what is still pending, and how to reproduce it. This README is
only a map.

The proposal itself is [`docs/MS_Thesis_Proposal.pdf`](docs/MS_Thesis_Proposal.pdf).

## Where things live

| Path | What lives here |
| --- | --- |
| [`THESIS.md`](THESIS.md) | the whole research story — read this first |
| `architecture/rag2/` | the reproduced RAG² baseline (the system under study) |
| `architecture/scaf/` | the SCAF admission policy (the proposed replacement) |
| `experiments/scripts/` | **every runnable entry point — this is what you type** |
| `experiments/analysis/` | libraries those scripts call: ablations, statistics, reporting |
| `experiments/recency_bias/` | the Filter Recency-Bias Probe (primary contribution) |
| `experiments/results/` | generated evidence — never hand-edited |
| `data/`, `preprocessing/`, `indexes/` | the corpus, how it was built, and the MedCPT index |
| `docs/` | the proposal, reproduction references, runbooks, QC history |

Large assets — model weights, PMC XML, embeddings, indexes, caches — are
deliberately **not** tracked. See `.gitignore`.

## Running the tests

```bash
python run_tests.py                 # every offline suite, each in its own process
python run_tests.py scaf analysis   # or a subset
```

Suites: `pmc`, `pubmed`, `rag2`, `scaf`, `analysis`, `recency`. All are offline —
no GPU, no network, no model downloads.

## What is still pending

The infrastructure is built and tested. **The two experiments carrying the
thesis's primary claims have not been executed**, because they need resources
that exist only on the student's machine:

| Pending experiment | Needs | Mandatory? |
| --- | --- | --- |
| Filter Recency-Bias Probe (H1 + H4) | MedChangeQA + the trained filter | **yes** — the primary contribution |
| Matched-k = 5 generation | the real frozen candidate text + local generator | yes, for any answer-level claim |
| Entailment-filter ablation (H2 / A1) | GPU + entailment teacher | yes, for C1's full claim |
| Blind pairwise answer evaluation | the matched-k answers, and a harness | secondary |
| Meerkat-7B replication (H3 / A3) | a second trained filter | secondary |

Exact commands for each: [`THESIS.md` §12](THESIS.md#12-what-remains-and-exactly-where-it-runs).

**Code existing is not experimental evidence.** [`THESIS.md`
§11](THESIS.md#11-status-ledger) is the ledger that keeps *implemented*,
*tested*, *executed* and *reportable* apart.
