# EXP-0017 results — Typed decoder-program search (synthetic)

Registration: `docs/experiments/EXP-0017.md` (written before Finnish scores). **Not** a manuscript translation or decipherment. Score is recon/null gates + length penalty, not Latin fluency or EVA likelihood.

## Setup

- Search on EXP-0013 **validation world-C only** (digest `0d516e07…`). Finnish unused until one winner frozen.
- Frozen EXP-0013 keep-probs; catalog of 7 ops; 200 candidates (all length-1 and length-2, then shuffled length-3); length penalty 0.02 per extra op; seed 4017.
- Command: `.venv/bin/python -m voynich.decoder_programs --root . --device mps` (~207 s).

## Search (val)

52 / 200 candidates cleared val null gates. Winner: **`exact_count_neural`** (1 op). Val recon 0.2042; selection score 0.2042 **>** random same-length program mean 0.1247.

## Finnish holdout (frozen winner, once)

| Metric | Value | Gate |
| --- | ---: | --- |
| null_recall | 0.625 | ≥0.50 |
| null_precision | 0.603 | ≥0.50 |
| pred_null_rate | 0.297 | ∈[0.15,0.45] |
| recon_acc | **0.218** | **>** matched-random 0.2043 |
| vs vocab-filter recon | 0.218 > 0.056 | strict |
| vocab-filter signal f1 | 0.424 | ≤0.55 |

## Decision: **PASS**

The typed search selected the already-confirmed exact-count neural decoder and transferred to Finnish under the frozen gates. Homophone collapse and mono-sub rank did not win (length penalty + recon of \(C(L)\)). This is a method test: searchable short programs with a hard recon verifier. It is not a Voynich reading.

## Artifacts

`results/EXP-0017/results.json`, `decision.json`; `data/manifests/exp0017_data.json`.
