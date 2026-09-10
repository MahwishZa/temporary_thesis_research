# INVALID FOR PRIMARY REPORTING -- anchored pilot annotation

This directory holds the FIRST evidence-quality annotation pass, preserved
exactly as it was collected. **It must not be used as independent human
validation of SCAF or RAG2**, and no number derived from it belongs in the
thesis as a human-validation result.

## Why it is invalid

The annotation interface displayed a machine suggestion beside every passage.
The audit found:

* **120 of 120** rows were shown the suggestion (`ai_suggestion_shown` true).
* **120 of 120** human labels matched `ai_suggested_label` exactly.
* **Zero** rows were annotated with the suggestion hidden.

Perfect agreement with a suggestion that was visible on every row means the
labels record agreement with the suggestion rule, not an independent judgement
of the passages. There is no unanchored subset to calibrate against, so the
effect cannot be estimated and subtracted -- it can only invalidate.

This matters specifically, not just in general: the suggestion rule is lexical
overlap between question and passage, and SCAF's support term sigma is *also*
lexical overlap. The reported sigma-versus-human correlation of 0.7292, and the
overall SCAF correlation of 0.6284 that sigma carries, are therefore
substantially a measurement of the interface rather than of SCAF.

## What is preserved here

* `annotation_sheet.jsonl` -- the pilot labels, byte-for-byte
* `pilot_integrity.json` -- SHA-256, label fingerprint, distribution, and the
  anchoring evidence, so the original state stays provable
* `evidence_quality_analysis.json` -- the analysis computed from it, if it
  existed at the time of retirement

Nothing here was edited. The pilot labels are a real record of what happened and
are kept for the thesis's limitations section and for reproducibility.

## What replaces it

`annotation_sheet_v2.jsonl` in the parent directory: the **same 120 annotation
ids**, the same questions and the same passages, with the judgements cleared, to
be annotated with no suggestion generated and none displayed.
