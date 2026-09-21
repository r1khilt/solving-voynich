# EXP-0018 results — Null-rate grid on ZL3b validation (select / confirm)

Registration: `docs/experiments/EXP-0018.md`. Split frozen before scores (`8051d71`). **Not** a decipherment. ZL3b **test unscored**.

## Setup

- **Checkpoint:** `outputs/EXP-0016/model.pt` sha256 `434d84f80f405f375dee5a2108da667b22e4f622dd3688e2111e89975856a359`; **5,440,829** params; **not retrained**.
- **Units:** ZL3b transcribed characters; PUA stripped; not lowercased; `SEQ_LEN=128` non-overlapping windows (same as EXP-0016 Voynich gate).
- **Bits-gain:** `full_bits − selected_bits` under train-fitted rank-bucket bigram; window beats random if gain > mean_random + 0.05; 20 matched-random masks, seed base 9011.
- **Decoder:** exact-count `n_keep=round((1−r)L)` on frozen neural keep-probs.
- **Split seed 4018:** SELECT 12 pages / 99 windows; CONFIRM 12 pages / 109 windows. Page ids in `data/manifests/exp0018_split.json`.
- **Command:** `.venv/bin/python -m voynich.null_rate_grid --root . --device auto`

## Select half (selection, not confirmation)

Neural exact-count. Rate 0 is a reference, not selectable.

| r | role | mean bits-gain | mean matched-random | Δ vs random | frac beats random | pred null |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 0.00 | reference | 0.0000 | 0.0000 | 0.0000 | 0.000 | 0.000 |
| 0.05 | candidate | −0.0126 | −0.0084 | −0.0042 | 0.152 | 0.047 |
| 0.10 | candidate | −0.0258 | −0.0153 | −0.0104 | 0.202 | 0.101 |
| 0.15 | candidate | −0.0490 | −0.0252 | −0.0238 | 0.202 | 0.148 |
| 0.20 | candidate | −0.0661 | −0.0318 | −0.0344 | 0.253 | 0.203 |
| 0.25 | candidate | −0.0722 | −0.0342 | −0.0380 | 0.152 | 0.250 |
| 0.30 | candidate | −0.0741 | −0.0440 | −0.0300 | 0.212 | 0.297 |
| 0.40 | candidate | −0.0928 | −0.0657 | −0.0271 | 0.232 | 0.399 |
| 0.50 | candidate | −0.1009 | −0.0870 | −0.0139 | 0.384 | 0.500 |
| 0.60 | candidate | −0.1258 | −0.1166 | −0.0092 | 0.374 | 0.601 |
| 0.70 | candidate | −0.1589 | −0.1630 | **+0.0041** | 0.475 | 0.703 |

**Selected rate: 0.70** (highest neural Δ vs matched-random; only candidate with Δ > 0).

Classical sticky (secondary, non-deciding) also peaked at high r on select (r=0.70 Δ +0.076, frac 0.596) but did **not** choose the rate.

## Confirm half (the result that counts)

Selected rate **0.70** only. Neural:

| Metric | Value |
| --- | ---: |
| n_windows | 109 |
| mean bits-gain | −0.1521 |
| mean matched-random | −0.1958 |
| Δ vs random | +0.0437 |
| fraction_beats_random | **0.4587** |
| pred_null_rate | 0.703 |

Pass needs mean_gain > mean_random **and** fraction_beats_random > 0.50.

## Decision: **FAIL**

`fraction_beats_random 0.4587 not > 0.50`. Mean gain did beat matched random at this rate; the page/window majority gate did not.

**Fixed-rate exact-count transfer from the multilingual deletion model is rejected on this validation split.**

This does **not** prove a Voynich null component. EXP-0016 synthetic recovery knew the true deletion count; this run did not. Negative structure transfer under a rate grid is not a decipherment and is not a reading.

Classical at the *neural-selected* rate on confirm is secondary (mean_gain −0.002 vs random −0.095; frac 0.679) and is **not** a registered pass.

## Next change (single) and falsifier

**Do not** retune architecture here. Next registered id should be a **rate-free generative hypothesis** (EXP-0016 note): infer keep/delete without a global exact count, or jointly generate the observed stream. **Falsifier:** on a frozen validation confirm split, kept-stream bits-gain at the *model-implied per-window deletion count* must beat matched random **and** fraction_beats_random > 0.50; if the procedure reduces to crowning a high global rate, treat as the same rejected family.

Artifacts: `results/EXP-0018/results.json`, `decision.json`.
