# EXP-0016b results — copy_mutate falsifier

**2026-09-25 erratum:** The historical `copy_mutate` generator misaligned visible characters and mask labels. The PASS below is a score on malformed synthetic data, not a valid corrected-channel recovery result. See [CHANNEL-ALIGNMENT-0001](CHANNEL-ALIGNMENT-0001-erratum.md).

Registration: `docs/experiments/EXP-0016b.md`. Run only after EXP-0016 PASS. Same exact-count decoder and pass rule. **Not** a decipherment.

## Setup

Same architecture (**5,440,829** params), exact-count decoder, 400 stratified train languages, 270 family holdouts. Fillers: `random_char,periodic,copy_mutate` @ 0.30. Fresh train 4,000 updates MPS.

## Decision: **PASS**

| Macro (all 270) | Value |
| --- | ---: |
| langs beat matched-random | **264 / 270** |
| macro recon | **0.2552** |
| macro matched-random | 0.1923 |
| null_recall | 0.5731 |
| null_precision | 0.5601 |
| pred_null_rate | 0.2969 |
| vocab-filter recon | 0.0569 |
| passer families | 12 (all non-IE) |
| non-IE passer fraction | 1.0 |

Copy-mutate lowers recon vs easy-filler primary (0.315 → 0.255) but the multi-language rule still clears.

Artifacts: `results/EXP-0016b/decision.json`, `per_language_compact.json`, `results.json`.

Voynich label-free ran on the **primary** EXP-0016 checkpoint after this PASS; see `EXP-0016-results.md`. ZL3b test unscored. Not a decipherment.
