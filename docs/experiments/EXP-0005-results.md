# EXP-0005 results — Both distant order and category-specific context matter

Completed 2026-09-20 PDT / 2026-09-21 UTC from clean source `933bade`. [Registration](EXP-0005.md); [measurements and provenance](../../results/EXP-0005/report.json). Existing compact-model seeds42/43/44, no neural retraining, no manuscript test scoring. Runtime8.70 seconds on local MPS, no paid API.

## Larger matched sample changes the earlier interpretation

768 fixed validation targets on24 pages/10 physical groups; four corruption repetitions averaged per target. The sample can overlap prior validation and is not a fresh manuscript holdout. Scores are not directly comparable to EXP-0003's different192 targets.

| Context condition | Mean bits/unit across three seeds |
| --- | ---: |
| Original128 units | 1.916253 |
| Last16 only | 2.032903 |
| Original distant112 shuffled, local16 intact | 1.953455 |
| Same-category random donor, ordered | 2.070328 |
| Same-category histogram-matched donor, ordered | 1.970389 |
| Different-category random donor, ordered | 2.243019 |
| Different-category histogram-matched donor, ordered | 2.098973 |

**Correction to the earlier hint:** distant shuffling now hurts all three seeds by0.050127/0.038303/0.023177 bits (mean0.037202). Their descriptive equal-leaf bootstrap intervals are positive, although limited to10 clusters and uncorrected for multiple comparisons. EXP-0003's near-zero average did not survive the larger sample. We should no longer characterize distant order as having no consistent effect under this perturbation. Scrambling characters can destroy within-word patterns as well as inter-word order; those remain separate hypotheses.

Truncation damage averages0.116650 bits. This is larger than shuffle damage, consistent with older symbol content carrying useful information in addition to order. Do not turn their ratio into an exact decomposition: the interventions differ in length and distribution shift.

## Category and frequency controls

Different-minus-same illustration-category donor loss averages0.172691 bits for random donors and0.128584 after histogram matching, about26% attenuation. The fixed illustration metadata are a proxy for broad manuscript group differences; effects do not establish topics or semantics.

Matching is imperfect. Mean total-variation distance from the original remote symbol histogram:

| Donor | Random | Nearest among up to128 sampled candidates |
| --- | ---: | ---: |
| Same illustration category | 0.267680 | 0.144188 |
| Different category | 0.288182 | 0.162377 |

The different-category matched donors remain less similar by this metric. Therefore the remaining prediction gap cannot cleanly be attributed to higher-order category structure after “controlling away” frequencies. That would overstate what matching achieved. Full distance distributions and donor-ledger digest are retained. Ordered/shuffled donor pairs share exact histograms; their effects are also included, including cases where shuffling improves a replacement context.

## Fixed statistical predictors

All scored on the same768 targets; no strength tuning. Counts come from training pages only.

| Predictor | Bits/unit |
| --- | ---: |
| Five-gram | 2.191565 |
| Five-gram × remote histogram ratio, exponent0.5 | 2.144235 |
| Five-gram × remote histogram ratio, exponent1 | 2.148862 |
| 50/50 global/category five-gram mixture | 2.113815 |

Recent-frequency and category information improve these fixed statistical models, but they do not close the gap to the compact transformer's1.916253. The category mixture receives illustration metadata that the transformer does not receive explicitly, so label it metadata-assisted. These controls test predictive mechanisms, not a historical generator.

## What to test next

The current evidence favors investigating both local ordered forms and adaptation to surrounding vocabulary. A concrete new distinction is whether the order effect lives mostly inside whitespace-delimited transcription groups, between those groups, or in line layout. Group-preserving versus character-level permutations can test that without assuming the groups are plaintext words. Freeze that design before another round. Separately, stronger frequency balancing is needed before claiming an independent category effect.

The raw4.0MB donor ledger stays in ignored `outputs/EXP-0005/report.json` with a tracked digest. The tracked report keeps exact target coordinates, per-target scores, all seed/condition summaries and aggregate donor distances. `scripts/summarize_context_controls.py` audits hashes/source/holdout flags; all donor choices are regenerable from registered seeds and source.
