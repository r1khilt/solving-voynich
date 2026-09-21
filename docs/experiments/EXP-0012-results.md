# EXP-0012 results — Null-aware escalation (easy fillers)

Registration: `docs/experiments/EXP-0012.md`. New pass rule (null-class metrics). **Not** a Voynich decipherment. EXP-0011 thresholds untouched.

## Setup

- Filler families: `random_char`, `periodic` only; rate 0.30; Finnish holdout; seeds 4011/42/4012.
- Neural: BiLSTM + copy features; **null-weighted BCE** (weight 3.5); checkpoint on val **null F1**; signal threshold calibrated on world-C val to ~0.30 null rate. Params **70,301**. MPS; 3,000 updates.
- Classical: sticky-null Viterbi (path kept; no top-k overwrite); periodic/rarity emissions.

## Finnish holdout scores

| Method | null_rec | null_prec | pred_null | mask_acc | recon_acc | pred_bits_gain | null_f1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Neural | 0.584 | 0.542 | 0.309 | 0.740 | 0.151 | −0.308 | 0.562 |
| Classical | 0.392 | 0.368 | 0.305 | 0.633 | 0.102 | 0.010 | 0.380 |
| Majority | — | — | 0 | 0.714 | — | — | — |
| Matched random | — | — | — | — | 0.204 | −0.034 | — |
| Vocab filter (signal f1) | — | — | — | — | — | — | signal_f1=0.424 |

True null rate ≈ 0.286. Neural signal threshold (val-calibrated) ≈ 0.302.

## Decision: FAIL

Neural **met** null_recall, null_precision, and pred_null_rate gates (delete-nothing trap closed) but failed:

- `recon_acc` 0.151 < matched_random 0.204 + 0.08
- `mask_acc` 0.740 < majority 0.714 + 0.05
- `pred_bits_gain` margin vs matched random

Classical failed null recall/precision, recon, and mask_acc.

## Interpretation

Balanced null loss + rate calibration recovers approximately the right null **rate** and middling null F1 on easy fillers, but predicted null **positions** still do not reconstruct `C(L)` better than matched random deletion. Rate-aware null detection ≠ identifiable latent recovery.

## Voynich

**Not run.** Gate requires PASS.

## Artifacts

- `results/EXP-0012/results.json`, `decision.json`
- `data/manifests/exp0012_data.json`
- Derived/weights under ignored `data/processed/exp0012/`, `outputs/EXP-0012/`
