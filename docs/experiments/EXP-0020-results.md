# EXP-0020 results — Edge-emitting channel vs transformer joint futures (R4 / P2)

Registration: `docs/experiments/EXP-0020.md` (written **before** fresh-key joint scores). **Not** a Voynich decipherment. Thresholds untouched after seeing numbers. EXP-0018 was not a P2 prerequisite and was not run.

## Setup

- Synthetic alphabet size 4; prefix 32; joint horizon H=3; seq_len 40; data seed 4020; model seed 42.
- Families: `iid`, `cycle_null`, `delayed_parity`, `copy_lag`, `equivalent`. Structured predictive gate uses the middle three only.
- Edge grid K ∈ {1,2,4,8,16}, 3 restarts, smallest K within 0.01 val bits of best.
- Transformer: compact `d_model=64`, 2 layers, 800 AdamW updates on familiar-key mixture; optional per-key logit adapter (200 steps) if val improves.
- Command: `.venv/bin/python -m voynich.edge_emitting --root . --device mps`
- Wall clock: 2164.7 s (~36 min); local MPS/CPU; no paid API; no manuscript text.

## Frozen gates (quoted)

PASS iff all: probability validity (row sums / distributions within 1e-6); update closure \(\|b'-bM_a/(bM_a1)\|_\infty<10^{-5}\); structured macro \(\Delta_{\mathrm{joint}}\ge 0.05\) **and** \(\Delta_{\mathrm{joint}}\ge\Delta_1+0.02\); equivalent-generator mean \(|\mathrm{KL}_{\min}-\mathrm{KL}_{\mathrm{red}}|\le 0.05\) with oracle agreement ≤1e-8.

## Observed metrics (fresh keys, macro over 4 keys)

| Family | mean K | edge joint bits/sym | TF joint bits/sym | Δ_joint (TF−edge) | Δ_1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| iid | 1.0 | 1.9997 | 2.0019 | +0.0023 | +0.0002 |
| cycle_null | 2.0 | 1.8140 | 1.9888 | +0.1747 | +0.1738 |
| delayed_parity | 10.0 | 1.6752 | 2.0037 | +0.3285 | +0.3258 |
| copy_lag | 16.0 | 1.5385 | 1.0437 | **−0.4948** | −0.4340 |
| equivalent | 2.0 | 0.0040 | 0.0000 | −0.0040 | −0.0040 |

Structured macro \(\Delta_{\mathrm{joint}}=0.0028\), \(\Delta_1=0.0219\).

Controls: probability validity **ok**; update closure **ok** (max error 0 on scored models); equivalent oracle max diff **0**; mean |KL gap| **0.0000** ≤ 0.05.

## Decision: **FAIL** (`joint_not_better`)

Structured \(\Delta_{\mathrm{joint}}=0.0028 < 0.05\). First-symbol extra gate not reached. No post-fail retune.

## Interpretation

Validity and equivalent-generator uniqueness controls passed. The explicit channel **does** beat the transformer on fresh-key `cycle_null` and `delayed_parity` joints, but **loses heavily** on `copy_lag` (variable copying), which pulls the preregistered three-family mean below the 0.05 margin. A compact edge HMM is a poor copy-lag model; the transformer reference is not. This is a synthetic method miss, not a manuscript finding. HYP-003 remains unresolved. Not a decipherment.

## Artifacts

- `results/EXP-0020/results.json`, `decision.json`
- `data/manifests/exp0020_data.json` (bulk strings gitignored under `data/processed/exp0020/`)
- Code: `src/voynich/edge_emitting.py`; tests: `tests/test_edge_emitting.py`
