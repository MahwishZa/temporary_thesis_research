# Thesis recency-bias experiments

**Not started.** Placeholder for the Filter Recency-Bias Probe.

This layer observes the baseline; it does not modify it. What it will need is
already produced and carried:

| Needed signal | Where it comes from today |
| --- | --- |
| per-candidate `P([HELPFUL])` and keep/drop | `rag2.schema.FilterDecision` (`score`, `keep`, `label`) |
| ΔPPL, PPL with/without evidence, τ | `rag2.filter_training.labeling` provenance sidecar |
| publication date, precision, split | `Evidence.metadata` — `canonical_date`, `date_precision`, `split_june_2024` |
| authority tier, currency-pack membership, retraction | `Evidence.metadata` — carried, never read by the baseline |
| identical candidate population across arms | `rag2.cache` fingerprinted replay (validity control V3) |

Nothing here may edit `architecture/rag2/rag2/**`. If an experiment seems to need a baseline
change, that is a finding to write down, not a patch to apply.
