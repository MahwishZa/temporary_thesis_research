# Corpus policy metadata (M1–M4)

Derived, additive overlays that materialize the approved **Corpus Decision
Specification** (M1–M4). Built by `pmc/build_corpus_metadata.py`.

Nothing here modifies the raw acquisition corpus, the PMC XML, the parsed
records, `manifest.csv`, the inventory, the PubMed pipeline, or
`search_queries.txt`. Every file is keyed by `pmcid` / `pmid` and joins back to
the existing records without touching them. All files are **untracked derived
outputs** — not committed by the implementation task that produced them.

## Files

| File | M-item | Rows | What it is |
| --- | --- | --- | --- |
| `canonical_dates.csv` | M3 | 43,409 | Canonical publication date + explicit precision + pre/post-June-2024 split, per record |
| `corpus_policy.csv` | M4 | 43,409 | `source_category` + `eligibility_status` (+ flags) realizing the 5-tier policy |
| `currency_pack.csv` | M1 | 4 | The four currency-pack anchors with provenance and rationale |
| `cpg_registry.csv` | M2 | 19 | **Seed** of the CPG-AD layer (see the STOP note below) |
| `../currency_pack/` | M1 | — | Externally-added Cochrane record: XML, parsed JSON, provenance |

Row counts cover the full retrieval haystack: the **PMC layer** (25,743
manifest records) plus the **PubMed abstract layer** (17,666 records with no PMC
full text). 25,742 + 1 (Cochrane) + 17,666 = 43,409.

## M1 — currency pack

Four anchors named by the proposal (§1.3, §4.4, §5.1):

| PMCID | Document | In raw OA corpus | Acquisition |
| --- | --- | --- | --- |
| PMC11350039 | 2024 revised NIA-AA/AA criteria | yes | pmc-oa |
| PMC10313141 | Lecanemab Appropriate Use Recommendations | yes | pmc-oa |
| PMC12180672 | Donanemab Appropriate Use Recommendations | yes | pmc-oa |
| PMC13082890 | **Cochrane CD016297 — contested-state anchor** | no | manual-oa-fetch |

**Cochrane ingestion mechanism (correction to the earlier audit).** PMC13082890
*is* in the PMC Open Access subset (`in_pmc_oa_dataset=yes`,
`is_pmc_openaccess=yes`, `has_xml=yes`, CC BY-NC). Its download failed only
because the object is **revised in place**: four distinct MD5s have now been
observed (inventory `b20dcfe9…`, reconciled `ddb1c909…`, and this fetch
`dcb1ac4e…`). It was fetched once from the OA S3 object and **pinned by its
observed MD5**, not verified against the stale inventory MD5. The XML, its
parsed record (12,281 body words, 174 references, CC BY-NC confirmed from its
own `ali:license_ref`), and full fetch provenance are under
`../currency_pack/`. Licence CC BY-NC: local text-mining is permitted;
redistribution is non-commercial only.

## M2 — curated CPG/guidance layer  ✅ COMPLETE (per the CPG Source Policy)

`cpg_registry.csv` holds the **19-document curated Alzheimer/dementia guidance
layer**, every record verified present in the raw OA corpus, with full
provenance (19/19 carry a content hash, acquisition source and licence).
The earlier "~10³ documents" figure was **rejected** by the CPG Source Policy:
the authoritative AD-guidance universe is dozens, not thousands, and balanced
retrieval needs a quota-fillable corpus, not bulk.

Coverage: 4 authority-tier labels across 16 guideline families —
diagnostic criteria (NIA-AA revised criteria), appropriate-use criteria
(lecanemab/donanemab AURs, amyloid+tau PET AUC), Alzheimer's Association
clinical practice guidelines (blood-based biomarkers; DETeCD-ADRD diagnostic
evaluation), ARIA guidance and anti-Aβ MRI guidelines, and national/society
dementia guidelines (Italian, SIGN, Canadian VCI, Swiss neuroradiology, Thai
nuclear medicine, cognitive-disorders best practice).

**Classification rules enforced.** `source_category = cpg` for all of them.
Cochrane and other systematic reviews are **never** classified as CPG — the
Cochrane review sits in the currency pack as `systematic-review-as-guidance`.
The amyloid/tau PET AUC is tagged `appropriate-use-criteria` with its PubMed
type (`Systematic Review`) recorded, not silently relabelled.
`authority_tier_label` is a **label only** — no ordering or weight is encoded,
because authority is a tested thesis variable (ablation A12).

`cpg_external_targets.csv` records the 4 authoritative families **not present in
the OA corpus** (AAN, NICE, IWG criteria, AD-specific EAN) as manual-acquisition
targets. No dates, versions or licence terms are asserted for them — acquiring
them requires per-source licence verification and is an optional strengthening
step, not a freeze blocker.

## M3 — canonical dates

- `canonical_date` = earliest real publication event (electronic first),
  excluding preprint dates, from the JATS `pub-date` set when the parsed record
  is available, else the PubMed date.
- `canonical_date_precision` ∈ {day, month, year} — **never upgraded**. A
  year-only record stays year-only.
- `split_june_2024` ∈ {pre, post, unknown} for the counterfactual index (§6.4).
  A 2024 year-only record is `unknown` unless a real month is recoverable from
  an authoritative source (`recovered_month` + `recovered_month_source`), which
  is used for the split only and never rewrites `canonical_date`.

**Date-source caveat (this container) — REQUIRED WINDOWS STEP.** 25,667 PMC
records use the PubMed fallback here because only the 75-record parsed sample
(+ Cochrane) is present; 76 records use JATS. Current split:
`pre 10,965 / post 13,521 / unknown 1,257` (the 1,257 are all genuinely
year-only-2024 — an honest null, never guessed).

The canonical dates are **not final** until the pipeline is re-run on the
machine holding the complete parsed corpus, where JATS becomes primary for all
25,743 PMC records and may recover months for some of the 1,257 unknowns:

```bash
# On the Windows machine, from the repository root:
python pmc\build_corpus_metadata.py --no-fetch
```

Then confirm: `date_source` should show ~25,743 `jats` (not `pubmed`);
`invented-precision violations` must stay 0; and the `unknown` count should be
≤ 1,257. Freeze the corpus only after this run, so the frozen dates are final.

## M4 — corpus policy

`source_category` ∈ {pmc-fulltext (25,742), currency-pack (1), pubmed-abstract
(17,666)}. `eligibility_status` ∈ {eligible, excluded, manual-review,
externally-added} with the raw tier being every record (nothing deleted).

Key rules (from the spec, not new decisions):
- **Retracted articles stay eligible, flagged** — the currency gate γ rejects
  them at admission; they must be retrievable for that to be measured.
- **Errata / retraction notices → excluded** (metadata, not standalone evidence).
- **No licence / DOI-disagreement / licence-disagreement → manual-review**.
- **Preprints, editorials/letters/comments → eligible, flagged** for low
  authority tier (authority is a tested variable, not a delete).
- **Stubs / body-less → `fulltext_eligible=no`** (abstract-level only).

## Regenerating

```bash
# On the machine with the full parsed corpus (JATS-primary dates):
python3 pmc/build_corpus_metadata.py

# On a container with only the sample (PubMed-fallback dates), point --pubmed at
# the resolved LFS object and skip the one-off Cochrane fetch once staged:
python3 pmc/build_corpus_metadata.py --pubmed <path-to-pubmed_results.csv> --no-fetch
```

Tests: `cd pmc && python3 -m unittest test_build_corpus_metadata` (33 tests).
