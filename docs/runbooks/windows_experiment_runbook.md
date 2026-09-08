# Windows/GPU runbook — the scientific RAG² vs SCAF experiment

Every command here exists in this repository and was checked against the actual
scripts. Run them **from the repository root** in order; each step's output feeds
the next. Where a step needs something this checkout does not contain, that is
stated rather than assumed.

Nothing in this procedure runs in the Linux development container: it has no GPU,
no `torch`, no `transformers`, and no `indexes/production/`. That is why the run happens
here.

---

## 0. Prerequisites

| Requirement | Why | Check |
| --- | --- | --- |
| Python 3.10+ | repository baseline | `python --version` |
| **NVIDIA GPU + driver** | filter training and 8B generation | `nvidia-smi` |
| **torch with CUDA** | `+cpu` builds silently run on CPU | `python -c "import torch; print(torch.cuda.is_available())"` → `True` |
| `transformers`, `accelerate`, `sentencepiece`, `datasets`, **`nltk`** | RAG² filter training | `pip show transformers nltk` |
| `faiss` *(optional)* | exact search; a numpy fallback is used otherwise, identical but slower | `python -c "import faiss"` |
| **`indexes/production/`** | the production MedCPT index (773,183 × 768) | `python preprocessing\pmc\verify_index.py` |
| **`data/corpora/pmc/chunks/chunks.jsonl`** | 781,563 chunks, joined to the index by `chunk_id` | `dir data\corpora\pmc\chunks` |
| MedCPT weights | query encoder + cross-encoder, downloaded on first use | ~1 GB |
| **Llama-3-8B-Instruct access** | rationales, ΔPPL labels, answers | gated: accept the licence, then `huggingface-cli login` |

Install:

```
pip uninstall -y torch
pip install torch==2.4.1+cu121 --index-url https://download.pytorch.org/whl/cu121
pip install -r architecture\rag2\requirements.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

`nltk` matters: `architecture/rag2/classifier/run_classifier.py` imports it at module level and
calls `nltk.data.find("tokenizers/punkt")` at startup, so filter training fails
immediately without it. Its first run downloads `punkt` and needs network access.

**4 GB VRAM (RTX 2050) note.** The MedCPT and Flan-T5 stages fit. Llama-3-8B does
**not** fit in 4 GB at any usable precision — plan for a larger GPU, a quantised
build, or a hosted backend for the generation stages (0, 3, 6). Steps 1–2 and 4–5
are unaffected. This is the one hardware question to settle before starting.

---

## 1. Verify the production index

```
python preprocessing\pmc\verify_index.py
```

Must report **16/16 checks passed**, `production: true`, dim 768, and a
`content_digest`. **Record that digest** — put it in
`experiments/configs/preliminary_experiment.yaml` under `corpus.index_digest`, and cite
it beside every result.

If the index does not exist yet, build it first (hours, resumable):

```
python preprocessing\pmc\embed_chunks.py --device cuda --batch-size 8
```

---

## 2. Retrieve and rerank — the real MedCPT path

```
python architecture\rag2\scripts\02_retrieve.py -c architecture\rag2\configs\thesis_corpus.yaml
```

This generates a rationale per question with the backbone LLM, encodes it with
`ncbi/MedCPT-Query-Encoder`, runs balanced retrieval over `indexes/production/`, reranks
with `ncbi/MedCPT-Cross-Encoder`, and writes a candidate cache under
`cache/candidates/`. Note the path it prints.

Point `dataset.path` in `architecture/rag2/configs/thesis_corpus.yaml` at your question set
first — `data/datasets/thesis_questions/dev_questions.jsonl` holds the 30 fixed Alzheimer questions.

---

## 3. Freeze the candidate set

```
python experiments\scripts\freeze_candidates.py --source medcpt --cache cache\candidates\<name>.jsonl
```

`--source medcpt` is the only source a reported run may use; it stamps
`retrieval_is_medcpt: true`. (`--source lexical-dev` exists for offline
development and is rejected by the scientific gate.)

Writes `experiments/runs/frozen_candidates.jsonl` plus a `.meta.json` sidecar carrying
the **frozen-set digest** and the corpus statistics SCAF's support scorer needs.
**Record the digest.** From here, both arms consume this file and nothing
re-retrieves.

---

## 4. Build the ΔPPL filter labels

```
python architecture\rag2\scripts\03_build_filter_labels.py -c architecture\rag2\configs\thesis_corpus.yaml ^
    -o dataset.split=train --candidates cache\candidates\<train cache>.jsonl
```

Runs the backbone LLM twice per (question, snippet) pair — with and without the
document — to score rationale perplexity, then applies the paper's Figure 2
decision tree. **This is the expensive step**: it is O(questions × snippets)
forward passes through an 8B model.

---

## 5. Train the RAG² filter

The authors do not distribute their checkpoint; it must be trained. This drives
their own `classifier/run_classifier.py` with the Appendix A.3 hyperparameters
(lr 3e-5, 40 epochs, batch 16).

```
python architecture\rag2\scripts\04_train_filter.py -c architecture\rag2\configs\thesis_corpus.yaml --init-tokens ^
    --token-dir runs\filter-base

python architecture\rag2\scripts\04_train_filter.py -c architecture\rag2\configs\thesis_corpus.yaml ^
    --model runs\filter-base ^
    --train-file runs\<...>\filter_train.json ^
    --validation-file runs\<...>\filter_val.json ^
    --filter-output-dir runs\filter-medqa ^
    --select
```

`--init-tokens` adds `[HELPFUL]` / `[NOT_HELPFUL]` as single tokens and resizes
the embedding matrix. `--select` picks the epoch with the best validation
accuracy and records every epoch's score.

Check the argv before spending GPU hours — `--dry-run` prints the exact
`run_classifier.py` command without executing it:

```
python architecture\rag2\scripts\04_train_filter.py -c architecture\rag2\configs\thesis_corpus.yaml ^
    --train-file runs\<...>\filter_train.json --dry-run
```

The result is a HuggingFace model directory containing `config.json` and weight
files — that path is `--rag2-checkpoint` in the next step. Put it in
`experiments/configs/preliminary_experiment.yaml` under `arm_a.checkpoint`.

---

## 6. Run the scientific comparison

```
python experiments\scripts\run_comparison.py --scientific ^
    --rag2-filter rag2_perplexity ^
    --rag2-checkpoint runs\filter-medqa\best ^
    --generator huggingface
```

`--scientific` **aborts** unless every precondition holds. It will not fall back
to passthrough, lexical retrieval, a mock generator, or development mode. The
checks, in the order they fire:

1. `--scientific` with `passthrough` → refused before anything loads
2. `--scientific` with `--generator none` → refused
3. `rag2_perplexity` without a checkpoint → refused, with the training commands
4. checkpoint path is not a directory → refused
5. missing `torch`/`transformers` → refused, naming the install command
6. checkpoint unloadable → refused, naming what a checkpoint must contain
7. then the 16 recorded preconditions (below)

Outputs to `experiments/runs/`: `per_question.jsonl` (full traces) and `manifest.json`.
A run is reportable only when `manifest.reportable` is `true`; otherwise the
label reads `DEVELOPMENT RUN -- NOT A SCIENTIFIC RESULT`.

---

## 7. The 16 preconditions, and where each is satisfied

Software readiness is settled in this repository. Everything else is a property
of **your** machine and data.

| Precondition | Settled by |
| --- | --- |
| Arm A is `rag2_perplexity` | **software** — enforced |
| Arm A is NOT passthrough | **software** — enforced |
| Arm B is SCAF | **software** |
| SCAF currency / authority / retraction gate / abstention active | **software** |
| at least 20 questions | **software** — the 30-question set is committed |
| SCAF support has corpus statistics | **software** — emitted by step 3 |
| a generator is configured | **software** — `--generator huggingface` |
| Arm A has a trained checkpoint | **environment** — step 5 |
| checkpoint exists on disk | **environment** — step 5 |
| checkpoint looks trained (config + weights) | **environment** — step 5 |
| Arm A filter loaded its label tokens | **environment** — needs torch + the checkpoint |
| production MedCPT retrieval was used | **environment** — steps 1–3 |
| no development retrieval stand-in | **environment** — use `--source medcpt` |

In the development container the software rows pass and the environment rows
fail — which is the correct report, not a defect.

---

## 8. After the run

- `manifest.json` → `fairness` (9 checks) and `scientific` (16). Both must be
  fully green before any number is quoted.
- `manifest.arm_a_checkpoint_identity` records which checkpoint ran: resolved
  path, file listing, sizes, mtimes and a digest over that listing. Weight bytes
  are not hashed (multi-GB); the listing digest is what distinguishes a
  re-trained checkpoint.
- `manifest.arm_a_generation` / `arm_b_generation` record the resolved model
  revision and decoding parameters — they are the same object by construction.
- `manifest.scaf_components` shows each SCAF sub-score varies; a constant one is
  inert.
- Transcribe results into `docs/experiments/preliminary_rag2_vs_scaf.md`, **as measured**.

---

## 9. Known limitation, unchanged by any of this

The corpus is a five-year window (2021-08-30 → 2026-08-30) by the approved
PubMed strategy: **7 of 43,409 production documents predate 2020**. The recency
comparison the thesis is about therefore has almost no older stratum, and
FRB-PAIRS cannot be constructed from this corpus. That is a supervisor decision,
not a software problem — see `docs/experiments/preliminary_rag2_vs_scaf.md` §0.
