# EXP-0019 results — Generator-adversary search for HYP-005 (R3)

Registration: `docs/experiments/EXP-0019.md` (written **before** fitness and held-out scores). **Not** a Voynich decipherment. Thresholds untouched after seeing numbers.

## Setup

- Data: ZL3b `train` / `validation` only; test present unscored (25 pages).
- Search: population 24, generations 20, seed `4019`, elite 4; 241 unique genomes scored.
- Optimizable (fitness): `zipf_slope_top500`, `adjacent_repeat_rate`, `char_h2_bits`, `top1_word_share`.
- Held-out (once on winner): `line_initial_type_share`, `ab_top50_jaccard`, `word_h2_bits`.
- Command: `.venv/bin/python -m voynich.generator_adversary --root .`
- Wall clock: ~29.4 s; numpy only; no paid API; no neural training.

## Observed metrics

### iid_char negative control (optimizable)

| Metric | G | R (val) | τ | Verdict |
| --- | ---: | ---: | ---: | --- |
| zipf_slope_top500 | −0.7945 | −0.8760 | 0.15 | **match** |
| adjacent_repeat_rate | 0.00245 | 0.00538 | 0.005 | **match** |
| char_h2_bits | 3.863 | 2.175 | 0.25 | separate |
| top1_word_share | 0.0188 | 0.0298 | 0.010 | separate |

iid separated on **2 / 4** optimizable metrics (gate requires ≥3).

### Evolutionary winner (genome `window=15`, `p_cite=0.30`, mutate≈0.616, splice≈0.334)

Optimizable matches: **1 / 4** (only `adjacent_repeat_rate`). Best-of-generation history never exceeded 1/4 across all 20 generations. Defaults reference also 1/4.

Held-out (audit only under this decision mode): separated on **1 / 3** (`line_initial_type_share` only). `ab_top50_jaccard` matched; `word_h2_bits` matched under a large τ driven by |train−val| ≈ 1.64 bits (validation is much smaller than train).

## Decision: **FAIL** (`iid_control_failed`)

Preregistered order: iid sanity failed first. Do **not** interpret the surface-match / held-out-fail pattern from this run.

Secondary (non-deciding) observation: even ignoring iid, the searched selfcite family never forced ≥3/4 optimizable matches under the frozen gene ranges and budget — would have been `surface_unmatched`.

## Interpretation

- The optimizable subset + NB-0022-style floors make Zipf slope and adjacent-repeat tolerant enough that **iid character text matches them** on this validation-sized sample. That breaks the registered sanity control.
- This is **not** evidence for or against historical nonsemantic production (HYP-005 remains unresolved).
- A successor under a **new** id could harden the iid falsifier (e.g. require separation on `char_h2_bits` and `top1_word_share` plus at least one of Zipf/adjacent, or score iid against the full six-metric NB-0022 battery) **without** peeking to weaken held-out gates. That redesign is **not** executed here.

## Artifacts

- `results/EXP-0019/results.json`, `decision.json`
- `data/manifests/exp0019_data.json`
- Code: `src/voynich/generator_adversary.py`; tests: `tests/test_generator_adversary.py`
