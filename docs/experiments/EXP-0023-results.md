# EXP-0023 results — Typed decoder transfer on copy_mutate

Registration: `docs/experiments/EXP-0023.md` (written before Finnish scores). **Not** a manuscript reading or decipherment.

## Setup

- Fillers: `random_char,periodic,copy_mutate` @ 0.30 (EXP-0014b set).
- Seeds: data 4011, model 42, finnish 4012, search 4023.
- Train: TinySignalModel + copy-constrained CTC, 70,301 params, 3,000 updates, MPS (~325 s wall).
- Typed catalog search on validation world-C; Finnish unused until one holdout score.
- Frozen recon gate: **0.2043264147237504**.
- Controls: matched-random; no-retrain (`outputs/EXP-0013/model.pt` + `exact_count_neural` on same holdout).

Command: `.venv/bin/python -m voynich.copy_mutate_transfer --root . --device auto`

## Decision: **FAIL**

Winner (val): **`exact_count_neural`** (7/200 feasible; val \(s\) 0.1744 > random-program mean 0.1382).

| Metric | Retrained winner | No-retrain control | Gate |
| --- | ---: | ---: | --- |
| null_recall | 0.509 | 0.541 | ≥0.50 |
| null_precision | **0.492** | 0.523 | ≥0.50 |
| pred_null_rate | 0.297 | 0.297 | ∈[0.15,0.45] |
| recon_acc | **0.1759** | **0.1848** | **> 0.20433** |
| matched-random (transparency) | 0.1950 | — | not the gate |
| vocab-filter mask_f1 | 0.379 | — | ≤0.55 |

Failed: null_precision and recon vs frozen gate. Retrained path did **not** beat the no-retrain control on recon (0.176 < 0.185) and both remain below 0.20433. No-retrain recon matches EXP-0014b exactly on this holdout.

## Interpretation

Under the preregistered TinySignalModel + EXP-0017 typed family, training on the copy_mutate filler mix does **not** clear the frozen EXP-0014 bar. This closes the “maybe we just needed to retrain the typed decoder on copy_mutate” hole for this architecture/family; do **not** retune gates or search new architectures inside this id. `copy_lag` was excluded (separate channel; EXP-0020 FAIL). **Not a decipherment.** HYP-005 still open.

## Artifacts

`results/EXP-0023/{results,decision}.json`; `data/manifests/exp0023_data.json`; checkpoint digest in manifest (weights in ignored `outputs/EXP-0023/`).
