# TEACH-0009: Q/K/V and source-edge decomposition of the two-head key write

**Preregistration before generating the new suite or loading dense checkpoints.** TEACH-0008 found that zero-based heads 1 and 3 in dense encoder layer 1 jointly write almost all of TEACH-0007's reusable query-slot key. This study asks what those heads actually do: change query-key routing, transport changed values, or require both; and which of the six source slots supply the causal write.

The [QK/OV transformer-circuit framework](https://transformer-circuits.pub/2021/framework/index.html) motivates separating addressing from value/output transport. [Path patching](https://arxiv.org/abs/2304.05969), the [IOI circuit study](https://arxiv.org/abs/2211.00593), [causal abstraction/interchange](https://arxiv.org/abs/2106.02997) and guidance on [activation-patching interpretation](https://arxiv.org/abs/2404.15255) motivate original-model recomputation, fresh confirmation and explicit sufficiency controls. These sources do not predict this model's mechanism, and attention weights alone are not treated as explanations. Prior Voynich co-occurrence and token-similarity work reviewed in `SIMILARITY_THEORY.md` supplies no latent labels or causal anchors; this remains synthetic method qualification.

## Frozen suite and exact decomposition

Generate 256 new cross-G groups with seed `69111`, first 128 discovery and last 128 confirmation. Load the same two frozen TEACH-0004 dense-row checkpoints. Heads `[1,3]`, encoder layer 1 and query slot 5 are fixed from TEACH-0008; no new head or layer search is allowed.

At the input to that layer, explicitly compute projected Q, K and V for every head and slot, query-to-slot softmax weights, and per-source contributions `A(query, source) * V(source)`. Require the sum of six source contributions to match the explicitly reconstructed selected-head output below `1e-6`, and the reconstructed head output to match TEACH-0008's scaled-dot-product head output below `1e-6`. Use native attention only as the residual-stream baseline, as documented by the TEACH-0008 numerical amendments.

For the two fixed heads, run the full eight-cell binary factorial in which query Q, all-slot K and all-slot V independently come from the base (`B`) or the single donor-G0 (`D`) state:

- `BBB`: identity;
- `DBB`, `BDB`, `BBD`: donor Q-only, K-only, V-only;
- `DDB`, `DBD`, `BDD`: donor QK, QV, KV;
- `DDD`: full selected-head donor output.

Every donor object is computed once under G0 and reused with recipient G0/G1/G2. Report donor items, exact three-G groups, base preservation and cross-G non-injection for every cell. `QK-only sufficient` is a descriptive label if `DDB` reaches at least 75% items, 60% groups and 90% non-injection in both seeds. `V-only sufficient` uses the same rule for `BBD`. If `DDD` passes but both fail, label the result `QK/V conjunctive` descriptively. Q-only, K-only, QV and KV cells diagnose interactions and may prevent a simple category; do not force a winner.

## Frozen source-slot screen and confirmation

For each fixed head, decompose the full donor-minus-base query-head change into six additive source-slot contributions: F row 0, F row 1, G row 0, G row 1, marker and query. On discovery, screen all 63 nonempty shared slot subsets. Select the smallest subset whose minimum across both seeds reaches 60% exact groups and 90% non-injection; ties use higher minimum group accuracy, then item accuracy, then lower bitmask. If none qualifies, freeze the maximum-minimum subset and mark discovery unqualified.

On confirmation test the selected source subset, all six sources, the queried F row alone, the other F row alone, both F rows, all non-F slots, a norm-matched Gaussian selected-write delta (seed `69211`) and a cyclic next-group mismatch of the selected source delta rescaled to the target norm. Also record clean base/donor and the two-F-row input positive control. The cyclic mismatch preserves recipient G index but moves each group to the next group.

`SOURCE-EDGE-SUPPORTED` requires discovery qualification and, in both seeds: clean and input-positive controls at least 95%; full six-source selected-head transfer and the selected source subset at least 75% donor items, 60% exact groups and 90% non-injection; the selected subset beats both random and mismatch by at least 40 item-accuracy points; and all numerical errors are below `1e-6`. A passing source subset identifies sufficient finite contributions at this synthetic site, not a unique information-theoretic attribution; correlated or cancelling paths can exist.

Report attention mass descriptively after aligning each example's queried versus other F row. Mass is not a causal criterion. Preserve the full discovery screen, confirmation predictions, QKV cells, source masks, source/checkpoint/suite hashes and numerical checks. An independent compact auditor reconstructs every selection, score and registered decision without rerunning inference.

One CPU execution, 600-second cap. No training, paid API, download, manuscript data or final-test scoring. Any positive result explains a synthetic typed-row circuit only.
