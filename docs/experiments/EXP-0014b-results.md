# EXP-0014b results — Copy-mutate shift on frozen EXP-0013 + exact-count

Registration: `docs/experiments/EXP-0014b.md`. Run only after EXP-0014 PASS. Same frozen recon gate **0.2043264147237504**. **Not** a decipherment.

## Setup

Frozen EXP-0013 weights; exact-count decoder; new Finnish world-C holdout with fillers `random_char,periodic,copy_mutate` @ 0.30; **no encoder retrain**. Seeds 4011/42/4012. Holdout regenerated (not the easy-filler jsonl).

Command: `.venv/bin/python -m voynich.exact_count_decode --root . --device mps --experiment-id EXP-0014b --decoder exact_count --filler-families random_char,periodic,copy_mutate`

## Decision: **FAIL**

| Metric | Value | Gate |
| --- | ---: | --- |
| null_recall | 0.541 | ≥0.50 (cleared) |
| null_precision | 0.523 | ≥0.50 (cleared) |
| pred_null_rate | 0.297 | ∈[0.15,0.45] (cleared) |
| recon_acc | **0.185** | **not >** frozen 0.2043 |
| recomputed matched-random (transparency) | 0.195 | not the gate |

Null-rate gates held; reconstruction did not. Easy-filler exact-count does not transfer to copy_mutate without retraining this encoder. EXP-0016b (retrained at thousand-language scale) is a different id and is not reopened here.

ZL3b not scored.

## Artifacts

`results/EXP-0014b/results.json`, `decision.json`; `data/manifests/exp0014b_data.json`.
