# EXP-0011 results — Finnish holdout latent recovery

Registered pass rule: `docs/experiments/EXP-0011.md`. This is **not** a Voynich decipherment.

## Setup

- Pipeline `L → C(L) → N(C(L))`; worlds A–D; filler rate 0.30; filler families include copy/mutate.
- Train languages: English PG#11, Latin PG#218. Holdout: Finnish PG#7000 Kalevala (unseen alphabet/cipher stream).
- Neural: BiLSTM + mask/recon/world heads; after fix, copy-aware side features; **70,301 parameters**.
- Classical: 2-state score/Viterbi with token-level copy/mutate features.
- Seeds: data 4011, model 42, finnish 4012. Device MPS. 3,000 updates (~50s).

## Run 1 (pre-fix)

| Method | mask_f1 | mask_acc | recon_acc | pred_bits_gain |
| --- | ---: | ---: | ---: | ---: |
| Neural | 0.594 | 0.580 | 0.057 | −0.968 |
| Classical | 0.755 | 0.654 | 0.189 | 0.040 |
| Majority | 0.832 | 0.713 | 0.071 | 0.0 |
| Matched random | 0.713 | — | 0.199 | −0.033 |
| Vocab filter | 0.380 | 0.455 | 0.048 | 1.171 |

**Decision:** FAIL. Diagnosis: mixed-world validation inflated neural scores; world-C null detection did not transfer to Finnish.

## Run 2 (single registered fix)

Fix (frozen after run 1, before rerun scores): world-C oversampling (60% of corpus / 70% of each batch), copy-aware input features shared with classical, checkpoint on **world-C-only** validation mask F1.

| Method | mask_f1 | mask_acc | recon_acc | pred_bits_gain |
| --- | ---: | ---: | ---: | ---: |
| Neural | 0.823 | 0.704 | 0.076 | 0.000 |
| Classical | 0.733 | 0.622 | 0.189 | 0.084 |
| Majority | 0.832 | 0.713 | 0.071 | 0.0 |
| Matched random | 0.713 | — | 0.199 | −0.033 |
| Vocab filter | 0.382 | 0.456 | 0.049 | 1.253 |

Post-hoc diagnostic (exploratory): neural predicted null rate **0.033** vs true **0.287** (null recall ≈0.04) — behaviorally near delete-nothing despite higher signal-F1. Classical null rate ≈0.297 but positions poorly aligned (recon ≈ matched random).

**Decision:** FAIL under the preregistered rule. Winner: none.

## Voynich

**Not run.** Gate requires PASS.

## Artifacts

- Compact: `results/EXP-0011/results.json`, `decision.json`, `results_v1_pre_fix.json`
- Manifests: `data/manifests/exp0011_corpora.json`, `exp0011_data.json`
- Weights (ignored): `outputs/EXP-0011/model.pt`
- Derived corpora (ignored): `data/processed/exp0011/`

## Interpretation

The held-out Finnish benchmark does **not** support transferring supervised null removal to an unseen language/cipher under this generator and tiny model. Vocab-filter baseline stayed weak (good: no obvious vocabulary cheat). Reconstruction of `C(L)` remains near majority/random for both models. A stable, verified null component was **not** established.
