# TEACH-0011 output-projection numerical amendment

**Written after an invalid first execution and before any valid TEACH-0011 result.** The first report passed base/edited query-head reconstruction (`<=8.05e-7`) and source-sum checks (`2.38e-7`), but the projected full six-source delta differed from the projected complete QKV delta by `1.1920929e-6` in both seeds. The independent auditor rejected it. Invalid artifacts are preserved under `results/TEACH-0011/invalid-projection-*`; the exposed behavioral aggregates are not yet a valid result.

Before rerun, apply the unchanged output-projection matrix to intervention deltas in float64 and cast the projected delta back to model float32. On the invalid tensors this reduces the full-source versus complete-QKV projected discrepancy to `4.7683716e-7`; the pre-projection head-space discrepancy was already `9.5367432e-7`. This is higher-precision evaluation of the same fixed linear map, not a change to any learned weight or scientific intervention.

No seed, split, checkpoint, fixed head set, QKV cell, source mask, selection rule, control or threshold changes. Because the invalid report exposed aggregates, any corrected execution is explicitly non-blind and must pass the unchanged independent auditor.
