# TEACH-0008 scaled-attention numerical amendment

**Written after an invalid first execution and before any valid TEACH-0008 result.** The frozen runner's seed-0 donor-layer reconstruction error was `1.0728836e-6`, narrowly above the preregistered `<1e-6` requirement. The independent auditor rejected the report with `AssertionError: Geometry or numerical mismatch`; no valid verdict or audit artifact exists. Invalid report/rows are preserved under `results/TEACH-0008/invalid-numerical-*`.

The manual decomposition computed attention as explicit query-key matrix multiplication, softmax and value multiplication. Native PyTorch `MultiheadAttention` with `need_weights=False` uses `scaled_dot_product_attention`, whose floating-point operation order differs slightly. Before rerun, replace only the explicit attention kernel with `torch.nn.functional.scaled_dot_product_attention` at the same frozen scale, dropout zero and noncausal setting. Per-head outputs remain exposed before the unchanged output projection. No seed, split, checkpoint, head subset rule, intervention, control or threshold changes.

The invalid report exposed aggregate component behavior, so the corrected run is non-blind. It is a numerical pipeline repair, not a pristine confirmation. The unchanged `<1e-6` gate and independent auditor remain authoritative.
