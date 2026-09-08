# Repository reorganization — migration record

**Date:** 2026-09-08 · **Commits:** 6, from `da4ffc6` · **Baseline:** `ed2eee6`

What moved, what deliberately did not, what was verified, and what is still
open. This is the record to consult when a path in an older document does not
resolve.

---

## 1. What this was for

The repository had grown by accretion: ~48 MB of research data tracked inside a
source directory, experiment scripts inside the SCAF package, documentation
split between the repository root and the RAG² container, and a `rag2/rag2/`
nesting ambiguous enough that it produced a false report of deleted files.

The reorganization expresses the **research lifecycle** in the directory tree:

```
data/  →  preprocessing/  →  architecture/  →  experiments/  →  docs/
what we    how it becomes     the system        how we           what we
observed   a corpus           under study       tested it        concluded
```

The intent is that the directory a file sits in answers "what kind of question
does this file address?", and that a future corpus is a sibling rather than a
redesign.

## 2. The commits

| # | SHA | Content | Shape |
| --- | --- | --- | --- |
| 1 | `da4ffc6` | Layer READMEs establishing the structure | 4 files, 173 insertions |
| 2 | `1cc7360` | Corpus and dataset assets into `data/` | 29 files, **all pure renames** |
| 3 | `8063491` | Source components into `preprocessing/` and `architecture/` | 135 files, **all R100 pure renames** |
| 4 | `d0fa0b9` | Imports, configs, scripts, documentation | 65 files, 1176 insertions / 626 deletions |
| 5 | `8c8517b` | Validation and this record | 3 files |
| 6 | *this commit* | The §13 finding, found while pushing, and the `retrieve.py` corrections it forces | 3 files |

Commits 2 and 3 contain **only** renames — no content change — so the diff for
the risky part of the migration is mechanically checkable. All content edits are
isolated in commit 4.

## 3. What moved

| From | To | Why |
| --- | --- | --- |
| `pmc/*.csv`, `pmc/metadata/`, `pmc/currency_pack/`, `pmc/fulltext/`, `pmc/parsed/`, `pmc/chunks/` | `data/corpora/pmc/` | research data does not belong inside a source directory |
| `pubmed/pubmed_results.*`, `search_queries.txt`, `search_log.csv` | `data/corpora/pubmed/` | same |
| `scaf/data/dev_questions.jsonl` | `data/datasets/thesis_questions/` | the question set is data, not code |
| `pmc/*.py` + tests | `preprocessing/pmc/` | acquisition → indexing is a lifecycle stage |
| `pubmed/fetch_pubmed.py` + tests | `preprocessing/pubmed/` | same |
| `rag2/` | `architecture/rag2/` | the system under study |
| `scaf/` (package) | `architecture/scaf/` | same |
| `scaf/scripts/`, `scaf/configs/` | `experiments/scripts/`, `experiments/configs/` | *what the system does* and *how we tested it* are different questions |
| `pmc/index/` | `indexes/production/` | a built index is not source, and a second index needs somewhere to go |
| `pmc/candidates/`, `scaf/runs/` | `experiments/runs/` | run outputs belong with the experiments |
| `docs/*.md`, `rag2/docs/*` | `docs/{reproduction,experiments,runbooks,quality_control,architecture}/` | filed by the question each answers |
| `pmc/pmc_qc_report_*.md` | `docs/quality_control/` | reports are documentation |

## 4. What deliberately did not move

**`architecture/rag2/rag2/` stays doubled.** The container must be on `sys.path`
for `import rag2` to resolve, and the authors' `retriever/` and `classifier/`
must sit *beside* the package rather than inside it to stay byte-identical and
citable. Flattening would break the import or break the audit's claim. Verified
empirically before the migration: with both `architecture/` and
`architecture/rag2/` on `sys.path`, `import rag2` resolves to the real package,
not to the container as a namespace portion — a regular package wins over a
namespace portion regardless of path order. Re-verified after (§6 D/E).

**Tests stay beside their components** (your decision 2). 732 tests were not
moved into a centralised `tests/` tree.

**`preprocessing/pmc/` stays one flat directory.** Its modules import each other
by bare name, which Python resolves only for siblings on `sys.path`. Splitting
into `acquisition/`, `chunking/`, `indexing/` would look tidier and break every
one of those imports.

**`preprocessing/pmc/retrieve.py` is kept** (your decision 3), documented as
SUPERSEDED in its own docstring and in `preprocessing/README.md`, with its 54
tests preserved. Removal is deferred to its own commit — and §13 records that it
has a live consumer on `origin/main`, which makes keeping it necessary rather
than merely prudent.

**Production-scale artifacts stay gitignored and stay where they are on disk.**
No XML, chunk layer, index, embeddings or checkpoint was moved or committed.

## 5. Deviations from the approved proposal

Three, each with its reason:

**1. `data/manifests/` was not created; `chunk_stats.json` moved back beside
`chunks.jsonl`.** The proposal filed it under `data/manifests/`. But
`build_chunks.py` writes it into its `--out` directory alongside `chunks.jsonl`,
and a test pins that. Filed separately, the next chunk rebuild would write the
new stats beside `chunks.jsonl` while the committed copy in `data/manifests/`
stayed silently stale — a scientific-integrity hazard, since that file is the
committed evidence of the production build. Co-locating removes the hazard
without touching the producer. `data/manifests/` held nothing else, so it is not
created; `data/README.md` states the rule.

**2. No root `pytest.ini`.** The proposal offered one that discovers everything.
It is not viable, and the reason is worth recording: importing `scaf` registers
the `scaf` filter with the baseline's filter registry, and
`architecture/rag2/tests/test_filter_scoring.py` asserts that registry does
*not* know `scaf` — the guard that pins the one-way dependency. Running both
suites in one interpreter makes SCAF's import satisfy the guard it exists to
fail against. (`--import-mode=importlib` fails differently, on the same
registry.) `run_tests.py` runs each suite in its own subprocess instead, which is
a correctness requirement rather than a convenience.

**3. `experiments/scripts/__init__.py` removed.** An empty file whose only role
was to make `scaf.scripts` an importable subpackage of `scaf`. After the move
`experiments/` has no `__init__.py`, so `experiments.scripts` is not a package
and the file implied a relationship that no longer exists. Nothing imported it;
the one test that did now reads the runner as text, which is the more robust
check anyway.

## 6. Verification performed

| | Check | Result |
| --- | --- | --- |
| A | Every top-level directory has a README | 6/6 |
| B | No file remains at a pre-migration path | `pmc/ pubmed/ rag2/ scaf/` all gone |
| C | `git log --follow` traverses the moves | 2–4 commits of history on each sampled file |
| D | Every preprocessing module imports | 21 scanned, 0 failures |
| E | `import rag2` / `import scaf` from the root | resolve to the real packages, not the container |
| F | Every path in both configs | resolves, or is a documented gitignored build product |
| G | Every script's `--help` from the root | 13/13 |
| H | Every RAG² stage script's `--help` | 9/9 |
| I | Offline test suites | **732 green**: pmc 355 · pubmed 51 · rag2 194 (+2 skipped) · scaf 132 |
| J | Scientific guards | all three still refuse, with updated command paths |
| K | Relative markdown links | 0 broken |
| L | Backticked repo paths | all resolve except gitignored `indexes/production/` |
| M | Dependency direction | no baseline or preprocessing file imports `scaf` |
| N | Protected files vs. commit 3 | 7/7 byte-identical |
| O | Cochrane snapshot MD5 | `dcb1ac4eaa24b75ab3202f2315c6b2e4` unchanged |
| P | Git LFS pointers | oid `027c43ec…` intact |
| Q | `retry71.csv` | untouched, untracked, unstaged |
| R | No XML / chunks / index / embeddings staged | none |

### The behavioural check that matters most

The comparison runner was executed before and after the migration on the same
frozen candidate set:

| | pre-migration | post-migration |
| --- | --- | --- |
| Admission decisions, 30 questions × both arms | — | **identical** |
| Scientific preconditions | 7/16 | 7/16 |
| Fairness checks | 9/9 | 9/9 |
| `reportable` | false | false |

Per-candidate scores differ by at most **2.2 × 10⁻¹⁶** on 51 of 600 — see §8.

## 7. Two corrections made, both evidenced

**Precondition denominator.** `docs/experiments/preliminary_rag2_vs_scaf.md`
reported "7/15" scientific preconditions in three places. The run manifest it
describes records **7/16**, and `scientific_preconditions()` returned 16 checks
even at the commit that produced the report. The denominator was corrected to
match the manifest; **the measurement itself was not touched**.

**Chunk statistics provenance.** `README.md` claimed the committed
`chunk_stats.json` recorded "a partial container run, not the production one".
It records the production run — 781,563 chunks over 42,964 documents, committed
in `2290062` *"Record production chunk statistics"*. Corrected.

## 8. Finding: SCAF support scores are not bit-reproducible across processes

`SupportScorer.score` sums corpus IDF over a Python `set` of query tokens
(`architecture/scaf/policy.py:201`), so float summation order follows the string
hash seed. Measured on identical code: **56 of 600 scores differ between
`PYTHONHASHSEED=1` and `=2`**, by at most 2.2 × 10⁻¹⁶ (one ULP).

This is pre-existing and **not** an effect of the migration — it reproduces
between two hash seeds on unchanged code. No admission decision changed in any
run observed, and the σ *value* is mathematically identical; only the rounding
of its summation differs.

**Not fixed here.** The one-line fix is to sum over `sorted(q_tokens)` (line 202
already sums over a sorted list). Changing a scoring path is a scientific
decision, not a reorganisation, so it is left for a separate commit you approve.
Until then, pin `PYTHONHASHSEED` for any run whose per-candidate scores are
quoted.

## 9. Risks accepted

| Risk | Mitigation |
| --- | --- |
| A path reference missed somewhere | 297 candidate references swept; residual scan reports only the NCBI API path `pmc/utils/oa/oa.fcgi` (not a repo path) and the dated QC reports (§10) |
| Windows checkout diverges mid-migration | the runbook's backslash commands were updated in the same commit as the moves they describe |
| Git history obscured | `git mv` throughout; commits 2 and 3 are pure renames; `--follow` verified |
| A future contributor runs pytest over the whole tree | it fails loudly on a module-name collision rather than silently contaminating the baseline; `run_tests.py` and §5.2 explain why |

## 10. Dated QC reports are left unedited

`docs/quality_control/*.md` record runs that already happened. Their paths and
commands are as they were on the report's date; rewriting them would falsify the
record. `docs/README.md` carries the old-path → new-path mapping for reading
them.

## 11. What this does *not* establish

The structure enforces that the baseline is untouched, that both arms see the
same evidence, and that the software is internally consistent. **It does not
establish that any scientific result has been produced.**

Unchanged by this work, and still true:

- No accuracy has been measured. `docs/reproduction/reproduction_results.md` is
  deliberately blank.
- The RAG² filter checkpoint has never been trained.
- SCAF's support scorer is a lexical prototype; corroboration ρ is not
  implemented and carries weight 0.0.
- **The corpus is a five-year window; 7 of 43,409 documents predate 2020.** The
  recency question the thesis asks cannot be answered from it. That remains the
  primary open item, and it is a supervisor decision, not a software problem.

## 12. What changed scientifically

Nothing. No retrieval algorithm, model, prompt, threshold, filtering semantic,
SCAF rule, corpus byte or index content was modified. No index or embedding was
rebuilt, no model trained, no GPU work run, no model downloaded. The
baseline → SCAF dependency direction and the controlled-comparison invariant
(same candidates, same reranking, same generator, same decoding, different
admission) are preserved and were re-verified after the move.

## 13. UNRESOLVED: `origin/main` carries a tree this reorganization never saw

**This needs your decision before the branch is merged.**

The forensic audit and the approved architecture were performed against this
branch, whose base is `ed2eee6`. After that base was cut, `origin/main` gained
content from a different branch (PR #16, `claude/thesis-architecture`) that is
**not on this branch and was therefore never audited, never proposed, and never
migrated**:

| On `origin/main`, absent here | Size |
| --- | --- |
| `thesis/` — 20 files: `pipeline.py`, `retrieval.py`, `recency.py`, `provenance.py`, `evaluation.py`, `conditions/{rag2_condition,recency_aware,retrieval_only}.py`, tests | ~129 KB with `configs/` |
| `configs/thesis/` — `architecture.yaml` + three condition configs | |
| `MS_Thesis_Proposal.pdf` | |

### Why it matters, concretely

**1. Merging this reorganization into `main` will break `thesis/`.** Three
verified breakages, all from paths this migration changed:

| `thesis/` code | Breaks because |
| --- | --- |
| `_bootstrap.py`: `RAG2_ROOT = <repo>/rag2` | the container is now `architecture/rag2` |
| `_bootstrap.py`: repo root on `sys.path` "so `pmc` and `thesis` import" | `pmc/` is now `preprocessing/pmc/` |
| `retrieval.py`: `pmc_retrieve_module()`, `from pmc import embed_chunks` | same |

`conditions/rag2_condition.py` imports `rag2.*` only through `_bootstrap`, so it
breaks with it.

**2. `thesis/` overlaps `architecture/scaf/` and `experiments/`.** Both trees
implement the same comparison from different angles:

| `thesis/` on main | this branch |
| --- | --- |
| `conditions/retrieval_only.py` | Arm A with `--rag2-filter passthrough` |
| `conditions/rag2_condition.py` | Arm A with `--rag2-filter rag2_perplexity` |
| `conditions/recency_aware.py` | `architecture/scaf/policy.py` (γ currency term) |
| `pipeline.py`, `run.py` | `experiments/scripts/run_comparison.py` |
| `retrieval.py` → `pmc/retrieve.py` | `experiments/scripts/freeze_candidates.py` → `architecture/scaf/frozen.py` |

**3. It changes the standing of `retrieve.py`.** `thesis/retrieval.py` is a live
consumer of `preprocessing/pmc/retrieve.py`. On this branch the module is
unimported; on `main` it is not. Keeping it (your decision 3) is therefore
clearly correct, and removing it later requires resolving that dependency first.

### What was deliberately not done

`thesis/` was **not** merged in, migrated, or edited. Doing so would mean
reorganizing 24 files that no forensic audit covered and no approved proposal
mentioned, and it would require deciding whether `thesis/conditions/` or
`architecture/scaf/` + `experiments/` is the intended experimental path. Those
two trees are alternative answers to the same question, and choosing between
them is a research decision, not a migration step.

### The options

1. **Adopt `architecture/scaf/` + `experiments/` as the experimental path** and
   retire `thesis/` — it is the newer, guarded implementation (16 scientific
   preconditions, 9 fairness checks, frozen-candidate digests).
2. **Adopt `thesis/`** and retire `architecture/scaf/` + `experiments/scripts/`.
3. **Keep both**, in which case `thesis/` needs its own migration commit
   repointing `_bootstrap.py` and `retrieval.py` at the new paths — roughly an
   hour, mechanical, and testable against `thesis/tests/`.

Option 3 is the minimum required to merge this branch without breaking `main`.
Options 1 and 2 are the real question.

