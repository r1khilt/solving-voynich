# EXP-0021 results — Rate-free threshold keep on frozen EXP-0016

Registration: `docs/experiments/EXP-0021.md`. Prereg pushed before scores (`2771196`). **Not** a decipherment. ZL3b **test unscored**.

## Setup

- **Checkpoint:** `outputs/EXP-0016/model.pt` sha256 `434d84f80f405f375dee5a2108da667b22e4f622dd3688e2111e89975856a359`; **5,440,829** params; **not retrained**; no decode head added.
- **Decode (frozen):** keep iff \(P(\mathrm{signal}) \ge 0.5\). Not tuned on Voynich. No global exact-count rate.
- **Units / bits:** same as EXP-0018 (transcribed characters, PUA stripped, SEQ_LEN=128; bits-gain = full_bits − selected_bits; 20 matched-random masks, seed base 9011; beats margin +0.05).
- **Split:** reused EXP-0018 seed 4018; same SELECT/CONFIRM page ids (`data/manifests/exp0021_split.json`).
- **Command:** `.venv/bin/python -m voynich.rate_free_decode --root . --device auto`

## Implied null-rate distribution

| Half | n_windows | min | median | mean | max | std |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| SELECT (diagnostic) | 99 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| CONFIRM | 109 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

## Select half (diagnostic only)

| Metric | Value |
| --- | ---: |
| mean bits-gain | 0.000 |
| mean matched-random | 0.000 |
| delete-nothing gain | 0.000 |
| fraction_beats_random | 0.000 |

Identical to delete-nothing: under the frozen threshold the model kept every character.

## Confirm half (the result that counts)

| Metric | Value |
| --- | ---: |
| n_windows | 109 |
| mean bits-gain | **0.000** |
| mean matched-random | 0.000 |
| delete-nothing gain | 0.000 |
| Δ vs random | 0.000 |
| fraction_beats_random | **0.000** |
| implied-rate std | **0.000** |
| implied-rate median | 0.000 (outside [0.65, 0.75]) |

Pass needs **all** of: mean_gain > 0; mean_gain > mean_random; fraction_beats_random > 0.50; rate std > 0.05; median outside [0.65, 0.75].

## Decision: **FAIL**

Failed: mean_gain > 0; mean_gain > mean_random; fraction_beats_random > 0.50; implied-rate std > 0.05. Only the median-outside-[0.65,0.75] criterion held (median 0.0).

**Rate-free threshold decode of the EXP-0016 checkpoint does not establish a Voynich null layer on this validation confirm split.** Together with EXP-0018, both fixed-rate exact-count and this rate-free absolute-threshold decode are rejected for that claim.

## Diagnosis (one paragraph)

Post-hoc inspection of keep-scores (not used to retune the rule): on validation windows \(P(\mathrm{signal})\) saturates near 1 (typical min ≈ 0.97–0.999, mean ≈ 1.0), so the preregistered ≥0.5 rule deletes **nothing** and collapses to the delete-nothing reference (gain 0, rate std 0). EXP-0018’s exact-count path could still force deletions via **relative** ranking of near-tied scores and still failed its confirm gate at r=0.70; absolute threshold transfer fails earlier by refusing to delete. Negative result stands. Not a decipherment. Test split untouched.

## Next change and falsifier

Recorded in NOTEBOOK only (no new architecture started here).

Artifacts: `results/EXP-0021/results.json`, `decision.json`.
