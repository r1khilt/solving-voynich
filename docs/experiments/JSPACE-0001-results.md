# JSPACE-0001: stable directions, weak causal control

**Registered decision: FAIL.** On the untouched final task split, the selected country-coordinate swap redirected17/151 initially correct semantic records (11.26%), below50%. It exceeded random and orthogonal controls by11.26percentage points, and the raw-output-direction swap by7.95points, below the required20points. Copy preservation passed48/48. No historical reading follows.

## Execution and validation

The cached Qwen3-8B-4bit model, expanded into the registered float32 block computation, completed512 independent128-token article windows in8342.94s (2h19m03s), peak MLX allocation34.015GB. This measures32 selected token rows at nine source layers; it is not full-vocabulary J-space. Calibration source hashes remained unchanged from launch revisiona0cb885. Follow-up source56779d4 was frozen before scoring; later commits changed documentation/input audits only.

Actual8B cached-versus-full execution passed: maximum logit discrepancy0.00008297, all tested argmax outputs identical. All6480 development and1440 final intervention records are valid, and every identity text matches its baseline. Maximum measured coordinate-swap error was2.48e-6 development and8.85e-7 final. Independent rescoring from generated strings and frozen accepted labels reproduced every correctness flag. No repeated final run, changed aliases, or adaptive threshold.

The32-article head-row reconstruction sensitivity audit completed in161.82s. Transport correction relative norms ranged0.000380–0.001904, median0.000915, with minimum original/corrected cosine0.9999983. This small geometric difference does not prove identical behavior; primary directions were not replaced.

## Results

Development selected zero-based layer27 (block28), with8/67 successful swaps versus4/67 for raw output directions. Earlier tied-zero layers were not selected; selection maximized the registered advantage score with an earlier-layer tie break.

| Final condition at block28 | Donor answer among initially correct records | Correct copying |
| --- | ---: | ---: |
| Jacobian country swap | 17/151 (11.26%) | 48/48 |
| Matched random | 0/151 | 48/48 |
| Orthogonal random | 0/151 | 48/48 |
| Raw-output-direction swap | 5/151 (3.31%) | 48/48 |
| Full donor residual at this one position | 65/151 (43.05%) | 48/48 |
| Identity | 0/151 | 48/48 |

The unedited model answered151/192 semantic records correctly (78.65%). The all-case swap rate was29/192; it includes cases whose original answer was already wrong and cannot replace the primary denominator. Final currency transfer was0/32; capital5/40, continent1/38, language11/41. By family: indirect facts5/80; anchored aliases12/71. By country pair: France/China7/58, France/Egypt10/50, China/Brazil0/43. This is uneven control, not a reliable portable variable.

Four of1440 final generations reached the12-token cap (62/6480 development). All strings and truncation flags are retained. Country facts, reverse directions, templates and recipient prompts are shared; the three pairs per split are not151 independent discoveries. No population significance claim is made.

## What the geometry tells us

Country directions estimated from independent article halves agree well: minimum cosine among the ten country token rows is0.972 at the earliest tested block and exceeds0.999 at block28. The actual country-difference axes are also stable (minimum0.941 earliest;0.9991 at block28). Therefore more article averaging alone is not a compelling explanation for the failed intervention.

At block28, the source country exceeds its paired donor in the readout on90/96 development semantic records. Yet swapping those coordinates changes few correct answers. **Readable information and a reliable causal control variable are different properties.** Late stability is also partly expected from the residual identity path; it does not independently establish a workspace.

## Limits and next diagnosis

All primary edits were at one layer and the last rendered prompt position. This is a deliberately restricted adaptation. The [reference paper's intervention methods](https://transformer-circuits.pub/2026/workspace/index.html) include swaps across token positions and, for flexible-use experiments, a band of layers. Our failure does not contradict that broader protocol or transfer its claims to this model.

A full donor residual at one position is not a complete donor computation. Earlier prefix states and their attention keys/values remain available, so subsequent computation can recover the original information. The43% donor result and complete early-layer resistance motivate a separate position-coverage diagnostic. That is a hypothesis about bypass paths, not an established explanation. The exposed final split cannot be reused as a fresh confirmation set for that successor.

Artifacts: `results/JSPACE-0001/{fit,development,final,decision,selection,cached-qualification,head-transport-sensitivity,independent-score-audit}.json`. Lossless compact row archives are in `results/JSPACE-0001/audit/`; bulk tensors remain ignored with checksums. Source/input manifests retain exact software, model, corpus, token IDs, seeds and provenance. The finite queue completed all stages without errors or resource stops. No paid services, model download, weight updates or manuscript holdout scoring.
