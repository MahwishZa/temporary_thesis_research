# Thesis recency-bias experiments — the Filter Recency-Bias Probe

**This is the thesis's primary contribution (C1), not a follow-up to the SCAF
comparison.** The proposal's own sequencing makes that explicit: the probe is
Phase 4 and SCAF is Phase 5, and Phase 4 is the declared pivot point —

> "[DES] Phase 4 precedes Phase 5 deliberately. If the primary claim fails, that
> must be discovered in month five, when a pivot is inexpensive, rather than in
> month nine."

The project ran Phase 5 first. That is the single largest sequencing problem in
the repository, and this directory is where it gets corrected.

## What it measures

For a filter `f` over matched temporal-counterfactual passage pairs:

```
Delta = E_pairs[ P(admit | older) - P(admit | newer) ]
```

`Delta > 0` means the filter preferentially admits *older* evidence — hypothesis
**H1**. The pairs are matched on claim, source tier and token length, so age is
the only intended systematic difference.

## Where the pairs come from — and where they must not

Proposal §5.2 puts a **provenance firewall** around the primary claim:

> "[DES] Provenance separation is mandatory and is the single most important
> correction in this revision. An earlier formulation authored evaluation items,
> the supersession table and the currency corpus from the same twenty-odd
> documents, rendering a positive result unfalsifiable by construction. The
> probe's primary material is now drawn from MedChangeQA — externally authored,
> peer-reviewed in provenance, publicly released, and independent of any
> supersession table this thesis builds."

FRB-PAIRS comes from **MedChangeQA** (Vladika et al., *Facts Fade Fast*,
Findings of EMNLP 2025 — 512 changed-verdict pairs). The thesis-curated
Alzheimer material supports replication and case study **only**. The lanes must
not cross, and `load_dataset` refuses a thesis-curated `--provenance` rather
than trusting the operator to remember.

### Correcting a conclusion recorded elsewhere in this repository

`README.md` §12 and `docs/experiments/preliminary_rag2_vs_scaf.md` §0 record that
FRB-PAIRS *cannot be constructed* because the production corpus holds only 7
pre-2020 documents. **That is true of the corpus and irrelevant to the design.**
The corpus was never the intended source; the proposal names an external one, and
requires it. Those documents listed "construct FRB-PAIRS from an external
older-evidence source" as one option among three awaiting a supervisor decision.
It is not an option — it is the specification.

## Status

| Component | State |
| --- | --- |
| Pair construction, matching, exclusion accounting | **implemented** (`frb_pairs.py`) |
| Permutation control (V1 / H4) | **implemented** |
| Paired bootstrap, 10,000 resamples, unit = pair | **implemented** |
| Equivalence test against a pre-specified band | **implemented** |
| Blocking gate: H1 unreportable if V1 fails | **implemented** |
| 36 offline tests, no model or GPU required | **passing** |
| MedChangeQA itself | **not present** — must be downloaded |
| Entailment-labelled second filter (H2, A1) | **not trained** |

## Running it

Pair construction needs no model, so validate it anywhere first:

```bash
python experiments/scripts/run_frb_probe.py \
    --dataset data/datasets/medchangeqa/medchangeqa.jsonl \
    --dry-run
```

If MedChangeQA's column names differ from the defaults, map only the ones that
differ — a wrong mapping fails loudly rather than yielding zero pairs:

```bash
python experiments/scripts/run_frb_probe.py \
    --dataset <path> --dry-run \
    --field-map '{"older_text": "abstract_old", "newer_date": "year_new"}'
```

Then, on the machine holding the trained filter:

```powershell
python experiments\scripts\run_frb_probe.py `
    --dataset data\datasets\medchangeqa\medchangeqa.jsonl `
    --checkpoint architecture\rag2\runs\filter-alz `
    --equivalence-band 0.05
```

**Fix `--equivalence-band` before you look at any output.** Proposal §7.3
requires the band, the hypotheses and the significance level to be timestamped
before any contact with test data. Choosing it afterwards converts an
equivalence test into a formality.

The exit code is 0 only when the permutation control passes; H1 is not
reportable otherwise.

## What this layer may not do

It *observes* the baseline and never modifies it. Nothing here may edit
`architecture/rag2/rag2/**` — if an experiment appears to need a baseline change,
that is a finding to write down, not a patch to apply.
`architecture/rag2/tests/test_metadata_isolation.py` fails the build if the
baseline ever learns about publication dates, because a filter that could see a
date would no longer be the thing this probe measures.

## What it still cannot answer

* **H2** (Δ(perplexity) > Δ(support)) needs a second filter trained on the
  entailment label with the same backbone, capacity and training data. Only the
  perplexity-labelled filter exists. This is proposal ablation **A1**, the
  primary causal claim.
* **H3** (replication on a second backbone) needs a third trained filter.

H1 and H4 alone are what proposal §5.4 calls the Minimum Viable Implementation —
"a complete and defensible thesis should all later phases fail" — and they need
only the filter that has already been trained.
