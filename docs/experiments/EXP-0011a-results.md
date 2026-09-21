# EXP-0011a results — Easy-filler ablation (random_char + periodic)

Registered pass rule: identical to `docs/experiments/EXP-0011.md` (thresholds not moved).
Registration: `docs/experiments/EXP-0011a.md`. **Not** a Voynich decipherment.

## Setup

- World-C fillers: **`random_char`, `periodic` only** (no copy_mutate, stateful, shift_phase, pseudoword).
- Same languages/holdout/rate/seeds/models/baselines as EXP-0011.
- Neural: 70,301 params; MPS; 3,000 updates (~50s). Seeds: data 4011, model 42, finnish 4012.

## Finnish holdout scores

| Method | mask_f1 | mask_acc | recon_acc | pred_bits_gain | null_recall | pred_null_rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Neural | 0.804 | 0.731 | 0.096 | −0.213 | 0.331 | 0.173 |
| Classical | 0.717 | 0.599 | 0.185 | 0.009 | 0.319 | 0.297 |
| Majority | 0.833 | 0.714 | — | — | 0 | 0 |
| Matched random | 0.713 | — | ~0.20 | — | — | — |
| Vocab filter | 0.424 | — | — | — | — | — |

True null rate ≈ 0.286.

**Decision: FAIL.** Neural fails mask_acc margin, recon margin, bits-gain margin. Classical fails mask_acc, f1-vs-random, recon, bits. Vocab filter remained weak (not a cheat surface).

Exploratory: neural null recall improved vs EXP-0011 full mix (~0.04 → 0.33) and predicted null rate rose (~0.033 → 0.173), but reconstruction still near/below matched random. Easy fillers alone do **not** unlock the frozen pass rule under this architecture/objective.

## Contingent branch taken

EXP-0011b (copy_mutate falsifier) **not run** (requires 0011a PASS).
Architecture escalation → **EXP-0012** (new id, new metrics closing the delete-nothing F1 trap).

## Voynich

**Not run.** Gate requires PASS.

## Artifacts

- `results/EXP-0011a/results.json`, `decision.json`
- `data/manifests/exp-0011a_data.json`
- Derived: `data/processed/exp-0011a/` (ignored)
- Weights: `outputs/EXP-0011a/` (ignored)
