# EXP-0024 results — Oracle inverse on EXP-0023 copy_mutate split

**2026-09-25 erratum:** The 1.0 oracle score below compared the stored mask's deletion output with a target constructed from the same misaligned mask. It does not demonstrate inversion of the intended copy-mutate channel; `search_missed_inverse` must be retested. See [CHANNEL-ALIGNMENT-0001](CHANNEL-ALIGNMENT-0001-erratum.md).

Registration: `docs/experiments/EXP-0024.md` (written before scores). **Not** a manuscript reading or decipherment.

## Setup

- Same EXP-0023 Finnish holdout (`cd72bf0f…`) and train (`2c824915…`); fillers `random_char,periodic,copy_mutate` @ 0.30; seeds 4011 / 4012.
- Oracle: `oracle_gold_mask_delete` — keep positions where the stored generator mask == 1 (inverse of `insert_nulls`). Uses the stored mask because ciphertext alone does not identify inserted nulls.
- Frozen EXP-0014 gates unchanged. EXP-0023 winner recon reported beside oracle: **0.1759**.

Command: `.venv/bin/python -m voynich.oracle_copy_mutate_inverse --root .`

## Decision: **FAIL** (`search_missed_inverse`)

| Decoder | recon_acc | null_prec | null_rec | pred_null | vs gate 0.20433 |
| --- | ---: | ---: | ---: | ---: | --- |
| EXP-0023 winner `exact_count_neural` | 0.1759 | 0.492 | 0.509 | 0.297 | FAIL |
| Oracle `oracle_gold_mask_delete` | **1.0000** | **1.000** | **1.000** | 0.287 | **PASS** |

Vocab-filter mask_f1 0.379 ≤ 0.55. Oracle cleared all frozen gates; EXP-0023 typed winner did not. Mode **`search_missed_inverse`**: the metric can see the true channel inverse; the typed family/search did not find it.

## Interpretation

Retrain+exact-count failure on copy_mutate is **not** because the frozen bar is blind to a correct null-deletion inverse. The bar is reachable by the generator’s own mask. This does **not** yield a discovered decoder or a manuscript reading. HYP-005 still open.

## Artifacts

`results/EXP-0024/{results,decision}.json`; `data/manifests/exp0024_data.json`; reuses `data/processed/exp0023/` (gitignored bulk).
