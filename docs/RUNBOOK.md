# Running the research code

Run these commands from the repository root. No service credentials or API credits are needed for this local pipeline. The source download is roughly 0.4 MB; a fresh Python environment downloads considerably larger PyTorch wheels.

## Install and acquire data

```sh
uv sync --extra dev --locked
.venv/bin/python scripts/download_data.py
.venv/bin/python -m voynich.data
```

The downloader validates the pinned source checksum and records provenance. Preparation verifies the frozen split manifest, writes page-separated JSONL files under `data/processed/zl3b`, and fits the vocabulary only to training pages. Downloaded and processed text, environments, and model checkpoints are Git-ignored. Track manifests, code, and compact measurements instead. See [data documentation](research/DATA.md) for exact provenance, permissions, parser choices, and uncertainty handling.

## Validate and run a bounded pilot

```sh
.venv/bin/python -m pytest -q
.venv/bin/ruff check src tests scripts
.venv/bin/python -m voynich.train --config configs/smoke.json --run-dir outputs/smoke-local --device cpu
```

Use a new run directory for every invocation. The default device preference is CUDA, then MPS, then CPU; an explicitly requested unavailable accelerator raises an error. Four CPU threads are used by default and can be changed with `--threads`. The smoke configuration takes 40 optimizer steps; it validates the pipeline and does not rank architectures.

```sh
.venv/bin/python -m voynich.train --config configs/reference.json --run-dir outputs/reference-seed42 --device auto
.venv/bin/python -m voynich.evaluate --checkpoint outputs/reference-seed42/best.pt --baselines --output outputs/reference-seed42/validation-comparison.json
```

The reference defaults to at most 2,000 steps and can stop early on validation patience. `--steps` and `--seed` override the corresponding configuration. The manifest records effective settings, software versions, Git state, corpus digests and device. `initial.pt`, `best.pt`, and `last.pt` retain model/optimizer/RNG state; history and summary JSON hold compact results. Only load checkpoints from trusted sources; the loader requests PyTorch's restricted weights-only deserialization.

Resume an incomplete run into a **new** output directory using the identical config, step budget, seed, corpus and device family:

```sh
.venv/bin/python -m voynich.train --config configs/reference.json --resume outputs/reference-seed42/last.pt --run-dir outputs/reference-seed42-resumed
```

An interrupted run can lose steps after its most recent evaluation checkpoint. Extending a completed schedule is a new experiment; the resume interface deliberately rejects a changed schedule. Checkpoints carry the best state so a resumed run preserves it even without further improvement. Bitwise equivalence is tested on CPU in the pinned environment; it is not promised across hardware/library versions.

## Controlled architecture comparisons

```sh
.venv/bin/python scripts/run_ablations.py --configs small reference mtp qk_norm gated_attention --seeds 42 43 44 --steps 2000
```

This prints a plan without training. After recording the intended comparison, add `--execute` and choose a fresh `--output-root`. Compare primary validation scores and per-page variability; do not choose results based on repeated test-set inspection. Multi-seed ranking and confidence analysis remain research work, not automatic conclusions of the launcher.

## Inspect and intervene

Example inputs below are illustrative transcription-like strings, not a translation claim or a scientifically validated counterfactual:

```sh
.venv/bin/python -m voynich.interpret --checkpoint outputs/reference-seed42/best.pt --clean 'qokeedy qokeedy' --corrupted 'qokedy qokeedyq' --target y --site blocks.0.attn.result --head 0 --position -1 --output outputs/reference-seed42/patch-example.json
```

Clean/corrupted token lengths must match and fit the model's context. The target must be a single known, unambiguous unit. The command records identity, reverse and shuffled-donor controls. Normalized recovery is undefined for a negligible original gap and is not a probability. Use the Python API for multiple contexts, additional intervention sites, head ablations, or gradients.

## Final test evaluation

Only after freezing selection and registering the comparison:

```sh
.venv/bin/python -m voynich.evaluate --checkpoint outputs/reference-seed42/best.pt --baselines --split test --allow-test --output outputs/reference-seed42/final-test.json
```

The extra flag makes test exposure deliberate. Record that exposure in the notebook and avoid representing it as untouched in later adaptive analyses. The code does not automatically perform final test evaluation.
