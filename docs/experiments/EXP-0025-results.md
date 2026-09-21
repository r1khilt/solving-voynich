# EXP-0025 results — Harder-iid generator-adversary for HYP-005

Registration: `docs/experiments/EXP-0025.md` (written **before** fitness and held-out scores). **Not** a Voynich decipherment. Thresholds untouched after seeing numbers.

## Setup

- Successor to EXP-0019 FAIL `iid_control_failed`.
- **iid control change:** `iid_uniform_fixedlen` (word length 8, uniform `a–z`) replaces train-unigram `iid_char`.
- Held-out rule unchanged (≥2/3 separate); floors/τ unchanged.
- Search: population 24, generations 20, seed `4025`, elite 4.
- Command: `.venv/bin/python -m voynich.generator_adversary_hard_iid --root .`
- Wall clock: ~28.3 s; numpy only; no paid API.

## Observed metrics

### iid_uniform_fixedlen (optimizable)

Separated on **4 / 4** optimizable metrics (gate requires ≥3). Control passed.

### Evolutionary winner

Optimizable matches: **1 / 4**. Never forced ≥3/4 under this budget/gene ranges (same structural miss as EXP-0019’s secondary note).

Held-out (reported under valid iid; decision mode is surface failure): separated on **1 / 3**.

## Decision: **FAIL** (`surface_unmatchable`)

iid gate cleared. Search could not match ≥3/4 selected surface stats. Per frozen rule: this simplified copy-mutate family cannot force the hoax surface set; **HYP-005 not killed** (`open_not_killed`).

## Interpretation

- Hardening the iid falsifier fixed the EXP-0019 vacuous-control abort.
- Under a valid control, the searched selfcite family still cannot be forced onto ≥3/4 of the optimizable surface set.
- This is **not** evidence that the manuscript is language, and **not** a kill of HYP-005 (other nonsemantic families remain).
- **Not a manuscript reading.**

## Artifacts

- `results/EXP-0025/results.json`, `decision.json`
- `data/manifests/exp0025_data.json`
- Code: `src/voynich/generator_adversary_hard_iid.py`; tests: `tests/test_generator_adversary_hard_iid.py`
