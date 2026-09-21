# EXP-0026 results — Ciphertext-only typed inverse (EXP-0023 split)

Registration: `docs/experiments/EXP-0026.md` (written **before** holdout scores). **Not** a Voynich decipherment. Thresholds untouched after seeing numbers.

## Setup

- Same EXP-0023 Finnish holdout / train digests; frozen EXP-0014 gates.
- Selection: first 400 train `WORLD_C` samples; gold masks used **only** for selection metrics.
- Holdout decode: ciphertext + frozen EXP-0023 neural keep-probs only (`used_eval_generator_mask: false`).
- Typed family: EXP-0017 ops + `repetition_splice_detector`; ≤200 candidates; search seed 4026.
- Command: `.venv/bin/python -m voynich.ciphertext_only_inverse --root . --device auto` (~310 s).

## Chosen program

`exact_count_neural`  
(human-readable: keep top round(0.70 L) by neural keep-prob from ciphertext; no eval mask)

Train-select: recon 0.1871; null_rec 0.542; null_prec 0.533; pred_null 0.297.

## Holdout metrics

| Metric | Value | Gate |
| --- | ---: | --- |
| recon_acc | **0.1759** | > 0.2043264147237504 |
| null_recall | 0.5091 | ≥ 0.50 |
| null_precision | **0.4920** | ≥ 0.50 |
| pred_null_rate | 0.2969 | ∈ [0.15, 0.45] |
| vocab_filter mask_f1 | 0.3785 | ≤ 0.55 |

## Decision: **FAIL** (`ciphertext_only_inverse_missed`)

Test recon **0.1759** beside EXP-0024 oracle recon **1.0** and EXP-0023 winner **0.1759**. Ciphertext-only typed search recovered the same exact-count program class and missed the frozen bar; the oracle that used the stored generator mask still clears it.

## Interpretation

- A frozen decoder was **not** obtained.
- Confirms EXP-0024’s diagnosis without using the eval-time mask: the metric can see the true inverse (oracle 1.0), but this typed ciphertext-only family does not find it.
- **Not a manuscript reading.** HYP-005 status unchanged (set by EXP-0025: open / not killed).

## Artifacts

- `results/EXP-0026/results.json`, `decision.json`
- `data/manifests/exp0026_data.json`
- Code: `src/voynich/ciphertext_only_inverse.py`; tests: `tests/test_ciphertext_only_inverse.py`
