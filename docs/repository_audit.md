# Repository architecture audit and reorganisation

**Date:** 2026-09-08
**Scope:** the whole repository, excluding the research methodology.
**Verdict:** the architecture **was implemented and connected**; the repository
**did not reflect it**. Structure has been corrected. No research behaviour changed.

Nothing in this document changes the research question, the corpus, the chunking
rule, the retrieval models, the RAG² baseline, or the evaluation protocol. No
recency algorithm was invented, no SCAF component was implemented, and no
experimental result is claimed.

---

## 1. What was checked, and how

| Question (from the brief) | Method |
| --- | --- |
| Is the architecture actually implemented? | read every module; ran the end-to-end smoke check |
| Are the components connected? | traced the import graph and the runtime call path from `run.py` |
| Are responsibilities separated? | checked each module against the single stage it claims |
| Does the structure reflect the architecture? | compared the directory tree to `docs/architecture.md` |
| Baseline vs. proposed distinguishable? | inspected the condition registry and the isolation guards |
| Data / source / experiments / docs separated? | classified all 81 tracked non-`rag2/` files |
| Reproducible from a clean clone? | ran every suite; ran every stage CLI from an unrelated CWD |

**Behaviour baseline recorded before any change**, and required to be identical
after: `pmc` 355 tests, `pubmed` 51, `thesis` 54 (460 total), `rag2` 194 passed
+ 2 skipped, and `python -m thesis.run --smoke` passing.

---

## 2. Findings

Severity: **A** = blocks reproducibility or correctness; **B** = structural
defect with a real cost; **C** = hygiene.

### A1 — The corpus pipeline could only run from inside its own directory

`pmc/` was not a package. Its tests imported siblings by bare name
(`import build_chunks as bc`), two of them inserted their own directory on
`sys.path`, and four test files documented `cd pmc && python3 -m unittest …` as
the way to run them. `pmc/build_corpus_metadata.py` reached the parser by
pushing `<repo>/pmc` onto `sys.path` at call time; `pmc/embed_chunks.py` did the
same to reach `compose_embed_text` — a hazard its own docstring recorded, having
previously executed once per chunk across a 781k-chunk run.

A pipeline whose entry point depends on the shell's working directory is a
reproducibility defect: the same command yields a different result, or no
result, depending on where it is typed.

**Fixed.** One installable package with explicit imports throughout. Every stage
now runs as `python -m thesis.corpus_build.<stage>.<module>` from any directory;
all eleven were executed from `/tmp` to confirm it.

### A2 — Ten independent definitions of "where the repository is"

Ten modules each recomputed the repository root as
`Path(__file__).resolve().parent.parent` and then spelled out their own data
paths. The layout could therefore only be changed in ten places at once — which
is precisely why it had not been.

**Fixed.** [`src/thesis/paths.py`](../src/thesis/paths.py) resolves the root once
(env override → nearest `pyproject.toml` → relative fallback) and names every
data location as a constant. All ten now import from it.

### B1 — Structure did not reflect the architecture

`pmc/` held 27 files at one level: eight pipeline modules, eight test files,
three research-data CSVs, five generated QC reports, a `README_download_pmc_xml.md`
and four data directories. The documented ordering
acquisition → parsing → QC → chunking → embedding → indexing existed only in
prose, so nothing prevented a later stage from importing an earlier one, or the
reverse.

**Fixed.** One directory per stage under `src/thesis/corpus_build/`, with the
ordering stated in the package docstring and the one legitimate cross-stage
import (`embedding` reusing `chunking`'s `compose_embed_text`) made explicit and
one-directional.

### B2 — Data, source, documentation and generated reports were interleaved

`pmc_oa_inventory.csv` (immutable research evidence) sat beside `retrieve.py`
(source) and `pmc_qc_report_2026-09-03.md` (a generated report). Nothing in the
tree distinguished an input the research did not create, an artifact a command
reproduces, and code.

**Fixed.** `data/` holds evidence, `src/` holds code, `docs/` holds prose and
generated reports. [`data/README.md`](../data/README.md) states, per path,
whether it is committed and why. Digests were verified across the move: the
currency-pack XML is still MD5 `dcb1ac4eaa24b75ab3202f2315c6b2e4`, and every
file moved as a git rename, so no content changed.

### B3 — Tests lived in three places under three conventions

`pmc/test_*.py` (colocated, sibling imports), `pubmed/test_*.py` (colocated),
`thesis/tests/` (a package inside the source package). There was no unit /
integration / smoke distinction, and no single command ran everything.

**Fixed.** `tests/{unit,integration,smoke}/`, one command (`python -m pytest`)
running both this suite and `rag2/tests`. `tests/integration/test_retrieval.py`
was renamed `test_corpus_retrieval.py` — its basename collided with
`rag2/tests/test_retrieval.py`, which is what had prevented a combined run.

### B4 — The end-to-end smoke check was not part of the test suite

`python -m thesis.run --smoke` was the only check that fails when two components
stop fitting together, and it ran only when invoked by hand.

**Fixed.** `tests/smoke/test_architecture_smoke.py` runs it in the suite. It
caught two real breakages during this reorganisation.

### B5 — The "`rag2/` is untouched" guard could not express what it meant

`TestRag2TreeUntouched` asserted that *no* file under `rag2/` differs from
`origin/main`. But two files there were written by this thesis and must live
there for mechanical reasons: `rag2/rag2/corpora/thesis_chunks.py` is discovered
through RAG²'s corpus registry, and `rag2/configs/thesis_corpus.yaml` is loaded
by RAG²'s own test. The guard therefore forbade repairing a stale corpus path in
a file the reproduction does not own — and four of its user-facing error
messages did name paths that no longer exist.

**Fixed, and strengthened.** The guard now names the four thesis-authored files
explicitly and fails on any other change under `rag2/`; for the two Python files
among them it additionally proves the executable content still matches
`origin/main` after erasing docstrings and raised-message text. Field names and
config keys are string constants outside `raise` statements, so a rename such as
`chunk_id` → `chunkid` still fails the comparison — asserted by a test of the
comparison itself.

### C1 — Miscellaneous hygiene

Fixed: no `pyproject.toml`; `pubmed/.gitignore` duplicating root rules;
`README_download_pmc_xml.md` (vague name → `docs/corpus/pmc_fulltext_download.md`);
19 unused imports; four module-level imports placed after executable statements.

---

## 3. What was deliberately not changed

| | Why |
| --- | --- |
| `rag2/` layout, tests, `pytest.ini` | certified as a unit by `rag2_reproduction_audit.md`; restructuring would void that for no architectural gain |
| `rag2/retriever/`, `rag2/classifier/` | the authors' released code, byte-identical, pinned by digest |
| The QC reports' absolute Windows paths | they are the record of runs that happened on a specific machine; rewriting them would falsify evidence |
| `chunk_stats.json` (60,874 chunks) | it records a **partial** container run, not the production build; §10 of the reproduction audit already documents this, and the provenance layer marks such runs `reportable: false` rather than assuming |
| Two `F841` unused locals and five style findings | pre-existing, unrelated to structure; changing them would be churn inside an unmodified diff |
| **No `LICENSE` was added** | see §5 |

---

## 4. Verification

| Check | Before | After |
| --- | --- | --- |
| Thesis + corpus suites | 460 passed | 460 passed, +2 guard tests, +1 smoke test = **463** |
| `rag2/tests` | 194 passed, 2 skipped | 194 passed, 2 skipped |
| Combined (`python -m pytest`) | not possible (name collision) | **657 passed, 2 skipped** |
| `python -m thesis.run --smoke` | passes | passes, from any working directory |
| Every stage CLI `--help` | required `cd pmc` | all 11 run from `/tmp` |
| Corpus digests | — | unchanged; all moves are git renames |
| `ruff --select E,F,W` | 21 F-findings | 7, all pre-existing style |
| Stale path references | — | swept to zero outside the QC evidence |

The +3 tests are new guards, not re-counted old ones. Every pre-existing test
survives; git records all 40 source and test moves as renames (lowest similarity
62%, that being `test_condition_isolation.py`, whose `rag2/`-drift guard was
rewritten per B5). Apart from import lines, seven assertions changed: two on a
message string this reorganisation moved, and five that built a protected path
from a module attribute that is now a `thesis.paths` constant. The whole change
is 96 files, +943 / -435 lines.

---

## 5. Open item for the author: no `LICENSE`

The repository has no licence file. One is not added here because the choice is
not a structural decision and is not mine to make: this repository redistributes
third-party content under terms that constrain it — PMC open-access articles
under a mix of CC licences, the Cochrane currency-pack document under CC BY-NC
(non-commercial), and the RAG² authors' released code vendored under `rag2/`.

A licence should be chosen deliberately against those constraints, most likely
by licensing the thesis's own code (`src/`, `tests/`, `configs/`) separately from
the redistributed data (`data/`), and recording the data's terms rather than
relicensing them. Until then `pyproject.toml` declares no licence, so the
metadata does not assert something untrue.
