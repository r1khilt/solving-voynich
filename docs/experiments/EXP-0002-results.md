# EXP-0002 results — Longer training and architecture comparison

Completed 2026-09-20 PDT / 2026-09-21 UTC. [Registration](EXP-0002.md); [complete compact measurements](../../results/EXP-0002/comparison.json).

## Observations

All 18 runs used clean training source revision `f79e51b85c18bb0708ff76773fc51245595601a2`, identical registered ZL3b data/tokenizer digests, MPS float32, and seeds 42/43/44. No manuscript test score was computed. Summed measured training time: 652.3 seconds (10.87 minutes, excluding separate analysis/startup), 34,300 updates, 61,618,743 sampled eligible targets including repeated exposure. No paid API calls.

| Configuration | Parameters | Mean validation bits/unit | Sample SD across seeds | Best steps, seeds 42/43/44 |
| --- | ---: | ---: | ---: | --- |
| small | 430,720 | 1.839734 | 0.004262 | 2000, 1700, 1700 |
| reference | 1,814,208 | 1.843741 | 0.003607 | 1100, 1200, 1000 |
| mtp | 1,835,712 | 1.837866 | 0.003054 | 1100, 1200, 1000 |
| qk_norm | 1,814,592 | 1.846920 | 0.000384 | 1000, 900, 800 |
| gated_attention | 1,817,280 | 1.840176 | 0.004393 | 1100, 1200, 1100 |
| attention_only | 160,128 | 1.859446 | 0.007109 | 2000, 2000, 1900 |

Lower bits is better next-unit prediction. All whole-validation scores cover the same 25,917 eligible targets on 24 pages. The prior five-gram baseline is 2.087802 bits/unit; all six families beat it. These values include definite spaces, newlines and EOS and exclude unknown/uncertain targets.

## Decision and interpretation

MTP has the lowest mean, but its advantage over small is just 0.001868 bits. The preregistered near-tie rule selects **small**, with 430,720 parameters versus MTP's 1,835,712. The threshold is an engineering preference, not a statistical significance test. Attention-only has the fewest parameters but falls 0.021580 bits behind the minimum, narrowly outside the fixed 0.02 band; this boundary should not be interpreted as a scientific discontinuity.

The selected small checkpoints score training pages at 1.685450 / 1.703267 / 1.706853 bits, versus validation 1.837855 / 1.836734 / 1.844612. The gap indicates some fitting to the seen pages. Most larger-model validation minima occur around 800–1,200 updates; longer optimization under this schedule did not consistently help. This does not establish that all longer schedules, regularizers or architectures are exhausted. Learning curves, selected steps, stop reasons, checkpoint sizes/digests and token exposure are retained in the JSON.

Three seeds vary initialization/sampling, not manuscript provenance. Validation was repeatedly consulted and selected the checkpoints, so these are development scores rather than independent final performance estimates. Predicting transcription more efficiently does not identify a cipher, language or meaning.

## Reproduction and checks

```sh
.venv/bin/python scripts/run_ablations.py --configs small reference mtp qk_norm gated_attention attention_only --seeds 42 43 44 --steps 2000 --device mps --output-root outputs/EXP-0002 --execute
.venv/bin/python scripts/summarize_ablations.py
```

The launcher and summary refuse silent overwrite. Existing checkpoints remain in ignored `outputs/EXP-0002/`; reruns require archiving prior outputs and explicitly changing the experiment destination where applicable. The summary audited identical training-source hashes, clean training trees, frozen corpus hashes, seed coverage and the no-test flag, and separately scored selected training checkpoints. Earlier 200-update EXP-0001 remains available as an initial pilot, not part of this schedule-matched ranking.

Next: registered context/causal tests in EXP-0003 and independent synthetic calibration in EXP-0004.
