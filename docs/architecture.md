# Thesis research architecture

How the repository's parts compose into one executable research pipeline, what is
implemented, and what is deliberately not.

The architecture adds a thin orchestration layer (`thesis/`) over components that
already existed and were validated separately. It implements **no research method
of its own**: the corpus, the MedCPT retrieval foundation and the reproduced RAG²
baseline are called, never reimplemented or edited.

---

## 1. The pipeline

```
  research query
    |
    v
  [1] query normalisation                thesis/queries.py
    |                                      one query set, digest-recorded,
    |                                      identical across every arm
    v
  [2] retrieval                          thesis/retrieval.py -> pmc/retrieve.py
    |                                      MedCPT query encoder, exact flat search,
    |                                      equal quota per source category,
    |                                      total ordering, replayable candidate sets
    v
  [3] candidate evidence                 thesis/corpus.py
    |                                      chunk_id, document_id, source_category,
    |                                      text, canonical_date, provenance overlays
    v
  [4] temporal policy                    thesis/recency.py
    |                                      'none' for every implemented arm;
    |                                      declared interface, no algorithm yet
    v
  [5] experimental condition             thesis/conditions/
    |     baseline   retrieval only, temporally blind
    |     rag2       the reproduced filter, called through its own interfaces
    |     recency    inner arm + temporal policy   (INTERFACE ONLY)
    v
  [6] generation                         rag2/rag2/generation.py  (rag2 arm only)
    |
    v
  [7] evidence + provenance record       thesis/provenance.py
    |                                      corpus digest, model ids, config
    |                                      fingerprint, git commit, environment
    v
  [8] evaluation                         thesis/evaluation.py
                                           one protocol across arms
```

Orchestration is `thesis/pipeline.py`; the entry point is `python -m thesis.run`.

---

## 2. Component status

Read this table before citing anything from this repository.

| Layer | Component | Status |
| --- | --- | --- |
| Corpus acquisition | `pubmed/`, `pmc/` (inventory, download, parse, QC) | **implemented and validated** (pre-existing) |
| Corpus policy/metadata | `pmc/metadata/`, `pmc/currency_pack/` (M1–M4 overlays) | **implemented and validated** (pre-existing) |
| Chunk construction | `pmc/build_chunks.py` (256-word windows, 32 overlap) | **implemented and validated** (pre-existing) |
| MedCPT embedding/index | `pmc/embed_chunks.py`, `pmc/verify_index.py` | **implemented** (pre-existing); index is built per machine, never committed |
| Retrieval | `pmc/retrieve.py` — exact, balanced, replayable | **implemented and validated** (pre-existing) |
| RAG² reproduction | `rag2/` | **implemented and audited** (pre-existing); see `docs/rag2_reproduction_audit.md` |
| **Architecture layer** | **`thesis/`** | **implemented by this change; smoke-tested, not yet run on the production index** |
| Baseline condition | `thesis/conditions/retrieval_only.py` | **implemented** |
| RAG² condition | `thesis/conditions/rag2_condition.py` | **implemented**; needs a trained filter checkpoint to run |
| Recency condition | `thesis/conditions/recency_aware.py` | **interface only** — see §5 |
| Temporal policies | `thesis/recency.py` | **interface only**; every named policy raises when applied |
| SCAF / FRB-PAIRS | `experiments/scaf/` | **not started, deliberately** — see §6 |
| Thesis results | anywhere | **none exist.** No experiment has been run |

Nothing in the "interface only" rows is a research contribution. They are seams.

---

## 3. How the existing PMC + MedCPT foundation connects

The corpus is an established asset and this layer does not rebuild, re-chunk or
re-embed any part of it.

* **`thesis/corpus.py`** opens `pmc/chunks/chunks.jsonl` and `pmc/index/` and
  exposes one record shape: text, `chunk_id`, `document_id`, `source_category`,
  `canonical_date` and every overlay the M1–M4 metadata pass established. Fields
  it has never heard of are preserved under `extra` rather than dropped.
* **Identity is verified, not assumed.** `corpus.expected_chunk_digest` is a
  contract: the loader compares it against the corpus on disk and *refuses to
  proceed* on a mismatch. Silently recording whatever digest was present would
  make a mismatch invisible — precisely what provenance exists to prevent.
* **`thesis/retrieval.py`** is a facade over `pmc/retrieve.py`. It adds no
  scoring, ranking or caching of its own; putting retrieval behaviour in two
  places is how two places come to disagree.
* **MedCPT asymmetry is respected.** `ncbi/MedCPT-Article-Encoder` embedded the
  chunks; `ncbi/MedCPT-Query-Encoder` embeds the queries searched against them.
  (`pmc.embed_chunks.get_encoder("medcpt", …)` returns the *article* encoder —
  correct for indexing, wrong for queries — so the facade constructs the query
  encoder explicitly. `test_uses_the_query_encoder_not_the_article_encoder` pins it.)

---

## 4. How RAG² connects without being redesigned

`thesis/conditions/rag2_condition.py` contains no RAG² algorithm. It translates
across a seam and nothing else:

| Needs | Reached through |
| --- | --- |
| the filter | `rag2.filtering.base.build_filter` → `EvidenceFilter.apply` |
| record types | `rag2.schema.Question`, `rag2.schema.Evidence` |
| prompts | `rag2.prompts.PromptSet` — unmodified |
| generation | `rag2.generation.generate_answers` |
| every method-level setting | `rag2/configs/thesis_corpus.yaml` |

Reproduction-critical settings have exactly **one** definition, in the tree that
owns them. The thesis config *references* the RAG² config; it does not restate
filter kind, checkpoint, prompts, thresholds or decoding.
`test_rag2_config_is_referenced_not_duplicated` asserts the thesis layer declares
none of RAG²'s keys.

Only evidence *text* crosses into the filter. Corpus provenance rides in
`Evidence.metadata`, which RAG² carries and never reads — the paper's filter saw
a (question, snippet) pair and nothing else, and feeding it corpus metadata would
change what is being reproduced.

`thesis/tests/test_condition_isolation.py::TestRag2TreeUntouched` fails the build
if this layer modifies `rag2/` at all.

---

## 5. The recency boundary — interface, not method

The thesis investigates recency, and the architecture must not pretend to have
solved it. So:

* `TemporalPolicy` is the interface a policy implements.
* `NullTemporalPolicy` (`policy: none`) is the **control**: dates carried, never
  read. Its behaviour must stay exactly identity — every temporal effect the
  thesis reports is measured against it.
* `recency_weighted`, `currency_three_state` and `supersession` are **registered
  and raise when applied.** A run configured for a policy that does not exist
  fails loudly rather than quietly producing baseline numbers under a recency
  label, which would read as "recency made no difference" instead of "recency is
  not implemented".
* Selecting `condition: recency` with `policy: none` is refused for the same
  reason: that is the control arm wearing a recency label.

The boundary sits **around** the RAG² filter, not inside it. The thesis measures
what the original filter does with evidence of different ages; teaching that
filter about dates would destroy the thing being measured.

To implement a policy: subclass `TemporalPolicy`, register it with
`@register_policy("name")`, name it in config. No other file changes.

---

## 6. SCAF / FRB-PAIRS — not implemented

Not started, deliberately. When built it arrives as a new
`rag2.filtering.base.EvidenceFilter` implementation registered under its own key
— the same seam the paper's own "RAG² w/o filter" ablation uses — selected by
config. No baseline file changes and no restructuring.

That seam already exists, so this change adds **no** new extension point for it.
Nothing in this repository implements recency weighting, authority weighting,
currency scoring, supersession, contested-evidence handling or abstention.

---

## 7. Running it

```bash
# offline wiring check: synthetic fixture, hash encoder, no weights, ~1s
python -m thesis.run --smoke

# what arms and policies exist
python -m thesis.run --list

# resolve config and verify the corpus, then stop
python -m thesis.run -c configs/thesis/conditions/baseline.yaml --dry-run

# a real run (needs the built index and a query set)
python -m thesis.run -c configs/thesis/conditions/baseline.yaml \
    -o queries.path=experiments/baseline/queries.jsonl
```

Prerequisites for a real run, both built on the machine holding the MedCPT
weights and neither committed:

```bash
python pmc/build_chunks.py      # -> pmc/chunks/chunks.jsonl
python pmc/embed_chunks.py      # -> pmc/index/
```

### Configuration hierarchy

```
configs/thesis/architecture.yaml          shared: corpus, queries, retrieval depth,
  |                                       recency policy, evaluation, output
  +-- conditions/baseline.yaml            the control arm
  +-- conditions/rag2.yaml                the RAG² arm
  +-- conditions/recency.yaml             INTERFACE ONLY, not runnable
        |
        `-- rag2_config: -----------> rag2/configs/thesis_corpus.yaml
                                        MedCPT ids, filter, prompts, LLM, decoding
```

Overrides use RAG²'s own parser: `-o retrieval.final_top_k=16`.

### Where generated artefacts belong

| Artefact | Location | Committed? |
| --- | --- | --- |
| chunks, index, candidate sets | `pmc/chunks/`, `pmc/index/`, `pmc/candidates/` | no (gitignored) |
| run outputs | `experiments/{condition}/runs/{name}/` | no (gitignored) |
| `chunk_stats.json` | `pmc/chunks/` | **yes** — committed evidence of the run |
| run records | beside each run's outputs | no; cite the fingerprint instead |

---

## 8. Provenance and what makes a run reportable

Every run writes `run_record.json`: corpus stamp (digest, counts, index encoder,
production flag), model identities, config and retrieval fingerprints, query-set
digest, seed, git commit and dirty flag, package versions.

A run is **reportable** only when the corpus digest was actually verified and the
index is a production MedCPT build. Otherwise `report.json` carries
`reportable: false` and the reasons. A wiring test that looked like a research
result would be worse than one that says what it is.

The record also states whether temporal fields survived retrieval
(`retrieval.temporal_fields_carried`). Candidates come from the index *manifest*,
a separate artefact from the chunk file, so "the corpus has dates" does not by
itself mean "retrieval returns them". A candidate set without dates cannot
support a recency arm, and the record says so rather than the experiment failing
later for unclear reasons.

---

## 9. Known blockers before thesis experiments

1. **The production corpus is not present here.** Committed
   `pmc/chunks/chunk_stats.json` records a *partial* container run (60,874 chunks
   over 76 parsed records), not the production build reported as 42,964 documents
   / 781,563 chunks / 773,183 unique, digest `da1886b0…`. Until
   `corpus.expected_chunk_digest` is set to a digest the corpus on disk actually
   reports, every run record carries `digest_verified: false` and
   `reportable: false`. This is deliberate; see `docs/rag2_reproduction_audit.md` §10.
2. **No MedCPT index is built in this checkout** — `pmc/index/` is gitignored.
3. **No trained filter checkpoint.** The paper's is not distributed; the RAG² arm
   cannot run until one is trained (`rag2/scripts/04_train_filter.py`).
4. **No evaluation query set exists yet.** `queries.path` is empty by necessity.
5. **No recency algorithm.** By design — establish and measure the baseline first.
