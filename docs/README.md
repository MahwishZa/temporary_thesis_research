# `docs/` — what was decided, verified, and found

Documentation is filed by **what kind of question it answers**, not by which
component produced it. If you are looking for a claim, this page says which
document is allowed to make it.

```
docs/
├── architecture/     how the system is put together, and the rules that hold it there
├── reproduction/     what the RAG² paper specifies, what was reproduced, what was verified
├── experiments/      what was run, and what it measured
├── runbooks/         how to actually execute a run on the target hardware
└── quality_control/  dated QC reports over the corpus
```

## The documents

| Document | Answers |
| --- | --- |
| [`architecture/architecture_map.md`](architecture/architecture_map.md) | Where does each piece live, what may import what, and why |
| [`architecture/reorganization_2026-09-08.md`](architecture/reorganization_2026-09-08.md) | What moved when the repository was reorganised, what was verified, and what was deliberately left alone |
| [`reproduction/rag2_reproduction.md`](reproduction/rag2_reproduction.md) | What the paper fixes `[S]`, what is ambiguous `[A]`, what is unavailable `[U]`, where paper and released code disagree `[D]` |
| [`reproduction/rag2_reproduction_audit.md`](reproduction/rag2_reproduction_audit.md) | Component by component: what was verified, how, and what remains unverified |
| [`reproduction/reproduction_results.md`](reproduction/reproduction_results.md) | Measured baseline results vs. the paper. **Deliberately blank — nothing measured yet.** |
| [`experiments/preliminary_rag2_vs_scaf.md`](experiments/preliminary_rag2_vs_scaf.md) | The RAG²-vs-SCAF comparison, its acceptance criteria, and the corpus-design finding that blocks the research question |
| [`runbooks/windows_experiment_runbook.md`](runbooks/windows_experiment_runbook.md) | Every command for the Windows/GPU run, in order, with prerequisites |
| `quality_control/pmc_qc_report_*.md` | Corpus-wide QC over the parsed records, and whether each oddity is a parser defect or source variability |
| `quality_control/pmc_xml_retry_2026-09-02_outcome.md` | Outcome of the retry pass over failed downloads |

## Two rules these documents follow

**A blank result is the honest result.** `reproduction/reproduction_results.md`
stays empty until a real model produces a number. No table here is filled in
with an estimate, a projection, or a value adjusted to close a gap against the
paper.

**Approximations are named, not hidden.** Where a component is a stand-in — the
lexical support scorer, the unimplemented corroboration term, the three-way
rather than four-way corpus quota — the document that describes it says so at
the point of description.

## A note on the dated QC reports

`quality_control/` holds **historical records of runs that already happened**.
The file paths and commands inside them are the paths and commands as they were
on the report's date, before the repository was reorganised around the research
lifecycle. They are deliberately left unedited: rewriting them would falsify the
record of what was actually run.

For the current paths, see the layer READMEs. The mapping is:

| In the dated reports | Now |
| --- | --- |
| `pmc/*.py` | `preprocessing/pmc/*.py` |
| `pmc/fulltext/`, `pmc/parsed/`, `pmc/metadata/`, `pmc/chunks/` | `data/corpora/pmc/…` |
| `pmc/pmc_oa_inventory*.csv` | `data/corpora/pmc/pmc_oa_inventory*.csv` |
| `pmc/index/` | `indexes/production/` |
| `pubmed/` | `preprocessing/pubmed/` (code), `data/corpora/pubmed/` (data) |
| `rag2/` | `architecture/rag2/` |
| `scaf/` | `architecture/scaf/` (code), `experiments/` (configs, scripts, runs) |
