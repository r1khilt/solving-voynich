# Fresh latent-mechanism investigation

The user's J-space reference is [Anthropic's global-workspace research](https://www.anthropic.com/research/global-workspace). This directory separates the literature, the audited failure of our earlier model, and independently specified new tasks.

- [Methods review](METHODS_REVIEW.md): exact Jacobian-lens estimator, predictive-state geometry, causal abstraction, sparse features, controls, and limits of transfer to small glyph models.
- [Failure diagnosis](FAILURE_DIAGNOSIS.md): reproduced WMD-0001 competence, observation dependence, verifier/search bottlenecks, and identifiability problems.
- [Task design](TASK_DESIGN.md): 440 known-fact/anchored-alias/copy records with disjoint template families and country-pair combinations.
- [JSPACE-0001 registration](../../experiments/JSPACE-0001.md): local 8B-model restricted-lens campaign with explicit numerical, competence, and causal gates.

The investigation has no known historical translations as labels. Successful natural-language interventions could qualify a method; applying it to unknown symbols still requires learned competence, identifiable latent variables, and independent anchors.

## Reproduction

Use the isolated ignored environment `outputs/JSPACE-0001/venv`; exact installed package versions and model/corpus hashes are in `results/JSPACE-0001/inputs.json`. Core packages are MLX 0.32.2, mlx-lm 0.31.3, NumPy 2.5.3, SciPy 1.18.1, transformers 5.17.0. Existing model snapshot is specified in `configs/jspace0001.json`; a different machine must explicitly change that local path before registering its own run. No model auto-download is performed.

```
MLX_ENABLE_TF32=0 PYTHONPATH=src outputs/JSPACE-0001/venv/bin/python -m voynich.workspace.qualify
MLX_ENABLE_TF32=0 PYTHONPATH=src outputs/JSPACE-0001/venv/bin/python -m voynich.workspace.campaign prepare
MLX_ENABLE_TF32=0 PYTHONPATH=src outputs/JSPACE-0001/venv/bin/python -m voynich.workspace.campaign pilot-fit
```

The pilot refuses repeat scoring; preparation refuses overwriting frozen inputs. `fit` resumes from the last complete checkpoint with the same input/source hashes and cumulative fit budget. The input manifest's source revision is the preparation base revision; file hashes identify the exact new source, including files not yet committed at preparation time. Actual launch revision is recorded separately in the notebook/run log. Raw derivative arrays and calibration text are ignored; compact reports are versioned.

Precision engineering found that nominal float32 tensors alone were insufficient on the local M5. See [MLX's documented numerical precision policy](https://ml-explore.github.io/mlx/build/html/usage/precision.html). Early failed diagnostics remain in ignored local files; the final qualification is tracked. The expanded matrices contain the cached quantized values, not recovered original high-precision weights.
