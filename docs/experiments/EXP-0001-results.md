# EXP-0001 results — Bounded implementation pilot

Completed 2026-09-20 PDT / 2026-09-21 UTC. Source revision: `4fc90195797eda52afbbe7366d5d8163199495de`, clean working tree for both training runs. [Registered plan](EXP-0001.md). No paid research/training APIs or accelerator were used.

## Validation and execution

Pre-run checks: **222 pytest tests and 23 subtests passed**; Ruff and Git whitespace checks passed. Data cache/source and deterministic preparation rebuild verified checksums; documentation links resolved. Tests cover parser uncertainty/metadata, group integrity, registered corpus drift rejection, causal prefix invariance, fast/explicit forward parity, head summation, horizon boundaries/masks, test-access safeguards, exact CPU interrupted/resumed training, and activation intervention controls.

Runtime: Python 3.12.13, PyTorch 2.14.0, NumPy 2.5.3, CPU float32, four threads per process. Smoke and reference jobs were launched concurrently, so timings are descriptive rather than a hardware benchmark. Each manifest records package/code/data hashes and complete configuration. No training text or checkpoints were committed; compact measurements and checkpoint digests are tracked under `results/EXP-0001/`.

```sh
.venv/bin/python -m voynich.train --config configs/smoke.json --run-dir outputs/EXP-0001-smoke --device cpu
.venv/bin/python -m voynich.train --config configs/reference.json --steps 200 --run-dir outputs/EXP-0001-reference --device cpu
.venv/bin/python -m voynich.evaluate --checkpoint outputs/EXP-0001-reference/best.pt --baselines --device cpu --output outputs/EXP-0001-reference/validation-comparison.json
```

For reproduction use fresh output-directory names; overwrite is intentionally rejected.

| Run | Parameters | Steps | Initial validation bits/token | Best validation bits/token | Reported training elapsed |
| --- | ---: | ---: | ---: | ---: | ---: |
| Smoke, 64-unit context | 96,576 | 40 | 6.773726 | 3.236800 | 0.456 s |
| Reference, 256-unit context | 1,814,208 | 200 | 6.930006 | 1.960671 | 14.528 s |

Both stopped at their registered step budgets, with best checkpoints at the final step. Reference processed 358,387 eligible training targets across sampled windows. These runs use different sizes, contexts and schedules; their relative scores are **not** a controlled architecture comparison.

## Matched validation comparison

All four methods below score the same **25,917 eligible targets across 24 validation pages**, using the reference's nonoverlapping 256-unit windows. Lower is better. Units include definite spaces, locus newlines, and EOS; uncertainty/unknown targets are excluded. This is not bits per manuscript glyph or an estimate of intrinsic entropy.

| Method | Validation bits/token |
| --- | ---: |
| Reference transformer, seed 42, 200 steps | **1.960671** |
| Smoothed five-gram | 2.087802 |
| Local-copy/unigram mixture | 3.377559 |
| Unigram | 3.997425 |

The observed advantage over five-gram is about 0.127 bits/token in this pilot. It shows a working predictor with better aggregate validation loss under these settings. It does not identify which structure supplies that gain, establish seed stability, compare architectural variants, or imply semantic understanding. Baseline hyperparameters were fixed in the implementation, not optimized in this run. Per-page values are preserved in the JSON report; no confidence interval or final test result is claimed.

## Causal plumbing checks

The CLI used illustrative equal-length contexts `qokeedy qokeedy` and `qokedy qokeedyq`, next-token target `y`, seed 42. These are **not** a registered meaningful manuscript behavior or a validated clean/corrupted pair; the chosen target actually has a higher probability in the context labeled corrupted. Their role is implementation validation.

```sh
.venv/bin/python -m voynich.interpret --checkpoint outputs/EXP-0001-reference/best.pt --clean 'qokeedy qokeedy' --corrupted 'qokedy qokeedyq' --target y --site blocks.0.attn.result --head 0 --position -1 --output outputs/EXP-0001-reference/patch-head.json
.venv/bin/python -m voynich.interpret --checkpoint outputs/EXP-0001-reference/best.pt --clean 'qokeedy qokeedy' --corrupted 'qokedy qokeedyq' --target y --site blocks.3.resid_post --output outputs/EXP-0001-reference/patch-residual.json
```

- Identity patches produced maximum logit difference **0.0**.
- Replacing the complete final residual restored the clean target log probability exactly; normalized recovery **1.0**. This is an expected forward-computation identity, not a discovered circuit.
- The selected first-layer head patch changed target log probability by about **+0.007163 nats**. Its normalized recovery was negative, and the shuffled donor also changed the prediction. No special function or meaning is assigned to this head.
- Reverse and shuffled-donor controls, exact contexts, target, patch site, metrics and checkpoint hash were saved.

## Limitations and next work

No substantive plan deviations: the pre-training metadata-only split revision was documented before source publication and outcomes. The smoke/reference runs stayed within the step budgets. **The final test split remains unscored.** Astronomy cannot be represented in all three splits without splitting its two physical groups; train and validation receive one group each. Short shared labels and approximate/topic/hand dependencies remain.

The implementation is ready for a separately registered multi-seed comparison of small/reference/MTP/QK-normalization/gating variants, followed by synthetic generator calibration and prespecified causal behaviors. The 15-run launcher was exercised in **dry-run mode only**. No full sweep, sparse autoencoder training, cipher recovery experiment, or manuscript decipherment has been performed.
