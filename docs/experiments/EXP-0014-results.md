# EXP-0014 results — Exact-count decode on frozen EXP-0013 weights

Registration: `docs/experiments/EXP-0014.md` (written before this confirmation score). **Not** a Voynich decipherment. Thresholds untouched. This id was previously superseded as a session bar by EXP-0015/0016; it is the scored confirmation of the EXP-0013 exploratory decode, not a new encoder.

## Setup

- Frozen checkpoint `outputs/EXP-0013/model.pt` (70,301 params). **No retrain.**
- Finnish holdout jsonl digest `31984a91eac2d4fecc44b716805dba11c939eb2c8a0b951090e683e7a3b93740` (not regenerated).
- Decoder: `n_keep = round((1-0.30)*L)`; keep highest \(P(\mathrm{signal})\); ties → lower index.
- Command: `.venv/bin/python -m voynich.exact_count_decode --root . --device mps --experiment-id EXP-0014 --decoder exact_count --filler-families random_char,periodic`
- Pass gate for recon: **frozen** EXP-0013 matched-random mean **0.2043264147237504** (not a post-hoc recomputed threshold). Recomputed matched-random on this holdout was identical: 0.2043264147237504.

## Finnish holdout scores

| Method | null_rec | null_prec | pred_null | recon_acc | recon_edit_sim |
| --- | ---: | ---: | ---: | ---: | ---: |
| Neural exact-count | 0.625 | 0.603 | 0.297 | **0.218** | 0.728 |
| Classical sticky Viterbi (continuity) | — | — | — | reported in JSON | — |
| Frozen matched-random gate | — | — | — | 0.2043 | — |
| Vocab filter (signal f1) | — | — | — | — | f1 **0.424** ≤ 0.55 |

Teacher-forced gold-mask recon = 1.0 (sanity). Exploratory EXP-0013 note (~0.218) is confirmed as a **scored** confirmation, not a blind discovery.

## Decision: **PASS**

All EXP-0014 gates cleared: null_recall 0.625≥0.50; null_precision 0.603≥0.50; pred_null 0.297∈[0.15,0.45]; recon 0.21774 **>** 0.20433; vocab-filter f1 0.424≤0.55.

Interpretation: EXP-0013’s FAIL was **threshold decode**, not CTC keep-scores, on easy Finnish fillers. Still not a decipherment.

## Contingent next

EXP-0014b (copy_mutate distribution-shift, no retrain) required by the registration. EXP-0014c joint keep-run **not run** (only if this confirmation FAILs). ZL3b not scored in this id.

## Artifacts

- `results/EXP-0014/results.json`, `decision.json`
- `data/manifests/exp0014_data.json`
