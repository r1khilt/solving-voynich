# EXP-0013 results — CTC deletion alignment (easy fillers)

Registration: `docs/experiments/EXP-0013.md`. Pass rule frozen before scores. **Not** a Voynich decipherment. EXP-0011/0012 thresholds untouched.

## Setup

- Filler families: `random_char`, `periodic` only; rate 0.30; Finnish holdout; seeds 4011/42/4012.
- Neural: BiLSTM + copy features; **copy-constrained CTC deletion** (kept subsequence ↔ `C(L)`); light null-weighted BCE + rate penalty; checkpoint on val `recon_acc`; val-calibrated signal threshold. Params **70,301**. MPS; 3,000 updates (~97s).
- Classical: sticky-null Viterbi (unchanged).
- Command: `.venv/bin/python -m voynich.latent_recovery --root . --device mps --n-train 4000 --n-val 400 --n-holdout 300 --updates 3000 --experiment-id EXP-0013 --filler-families random_char,periodic --no-voynich`
- Clean preregistration source: `b83897a`.

## `recon_acc` (primary; unchanged)

Kept subsequence vs `C(L)`: prefix match fraction × min/max length ratio. Secondary: `recon_edit_sim` = 1 − Levenshtein/max_len. Teacher-forced gold-mask recon = 1.0 (sanity).

## Finnish holdout scores

| Method | null_rec | null_prec | pred_null | mask_acc | recon_acc | recon_edit_sim | pred_bits_gain | null_f1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Neural (free-running) | 0.666 | 0.574 | 0.333 | 0.763 | 0.199 | 0.713 | −0.189 | 0.617 |
| Classical | 0.392 | 0.368 | 0.305 | 0.633 | 0.102 | 0.554 | 0.010 | 0.380 |
| Majority | — | — | 0 | 0.714 | — | — | — | — |
| Matched random | — | — | — | — | 0.204 | — | — | — |
| Vocab filter (signal f1) | — | — | — | — | — | — | — | 0.424 |

Teacher-forced recon_acc = 1.0. Signal threshold (val-calibrated) ≈ 0.451. True null rate ≈ 0.286.

## Decision: FAIL

Neural **cleared** null_recall, null_precision, and pred_null_rate gates, and vocab filter stayed weak, but failed:

- `recon_acc` 0.1993 **not >** matched_random 0.2043

Classical failed null recall/precision and recon.

EXP-0013b (copy_mutate falsifier) **not run** (requires PASS). Voynich **not run**.

## Diagnosis (examples)

CTC training produces usable keep-scores (null F1 0.62; edit similarity 0.71; some samples reconstruct `C(L)` exactly), but free-running **global threshold** decode still localizes poorly for the prefix-based primary metric: median first kept-stream mismatch index ≈ 1, so a single early false delete shifts the subsequence and collapses `recon_acc` even when Levenshtein similarity remains high. ~5% of holdout pages catastrophically over-delete (near-all-null → recon 0). Rate gates pass; positions remain the failure.

Exploratory (not scored; not a pass): exact-count deletion of the lowest ~30% keep-probs on the **same frozen weights** yields mean recon ≈ 0.218 (> matched random 0.204); oracle-count ranking ≈ 0.282. Suggests decode/count constraint, not a new encoder, as the highest-EV next step.

## Single best next change + falsifier

Preregister a **decode-only** follow-up on frozen EXP-0013 weights: exact-count (or CTC-best path with fixed blank count) free-running mask; same Finnish easy-filler holdout and EXP-0013 pass rule. Falsifier: if exact-count decode PASSES without retraining, the bug was threshold decode; if it still FAILs, escalate to a position-aware CRF/DP that jointly scores keep runs under the copy-CTC emission (still not a third unrelated architecture).

## Artifacts

- `results/EXP-0013/results.json`, `decision.json`
- `data/manifests/exp0013_data.json`
- Derived/weights under ignored `data/processed/exp0013/`, `outputs/EXP-0013/`
