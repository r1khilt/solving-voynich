# TEACH-0011: Q/K/V and G-row edge decomposition of object readout

**Preregistration before generating the new suite or loading dense checkpoints.** TEACH-0010 shows that all four zero-based encoder-layer2 query heads jointly convert a transplanted key plus the recipient G context into an object-specific write. This study asks whether the transplanted key changes query addressing toward the matching G row and whether specific G-row source contributions carry the object write.

The [QK/OV transformer-circuit framework](https://transformer-circuits.pub/2021/framework/index.html), [path patching](https://arxiv.org/abs/2304.05969), [causal abstraction/interchange](https://arxiv.org/abs/2106.02997), [IOI circuit analysis](https://arxiv.org/abs/2211.00593), and [activation-patching interpretation](https://arxiv.org/abs/2404.15255) motivate a prospective Q/K/V factorial and finite source-edge interventions. Attention mass is descriptive, not evidence by itself. None of this prior work establishes a Voynich mechanism.

## Fresh recipient-conditioned suite

Generate 256 new cross-G groups with seed `71111`, first 128 discovery and last 128 confirmation. Load both frozen TEACH-0004 dense checkpoints. Reproduce TEACH-0010's upstream cause: compute the donor-G0 query key after two layers once per group, replace only the cut-2 query slot in each base recipient G0/G1/G2 state, and recompute encoder layer 2. Heads `[0,1,2,3]`, layer 2 and query slot 5 are fixed from TEACH-0010.

Explicitly project all Q, K and V tensors. Use PyTorch's full-query math scaled-dot-product backend to obtain attended head outputs and weights. At query slot 5, both clean-base and edited-recipient head outputs must match TEACH-0010's native-weight head values below `1e-6`; six returned-weight×V source contributions must sum below `1e-6`; their full edited-minus-base projected delta must match the complete QKV delta below `1e-6`.

## Q/K/V factorial

For all four heads, run the full binary factorial in which query-slot Q, all-slot K and all-slot V come from clean base (`B`) or edited recipient (`D`): `BBB`, `DBB`, `BDB`, `BBD`, `DDB`, `DBD`, `BDD`, `DDD`. Only query-slot Q is replaced in donor-Q cells; other query vectors remain base because attention rows are independent. Add each hybrid's output-projected query delta to the native clean attention write and recompute the downstream MLP/layer.

Report donor items, exact groups, base preservation and cross-G non-injection. A cell is descriptively sufficient if it reaches at least 75% items, 60% groups and 90% non-injection in both seeds. Label `Q-only sufficient`, `K-only sufficient`, `V-only sufficient`, combinations, or `factorial unresolved` from those frozen criteria; do not force a unique attribution when multiple cells pass.

## Source-slot screen and confirmation

Decompose the all-head edited-minus-base query output into six source contributions: F0, F1, G0, G1, marker and query. Screen all 63 nonempty fixed physical-slot subsets on discovery. Select the smallest subset whose minimum across both seeds reaches 75% donor items and 60% exact groups; ties use higher minimum group accuracy, then item accuracy, then lower mask. If none qualifies, freeze the maximum-minimum subset and mark discovery unqualified.

On confirmation test the frozen subset, all six sources, both G rows, all non-G sources, the row whose key equals the transplanted donor key, and the row whose key equals the original base key. Also test a norm-matched Gaussian projected delta (seed `71211`) and the selected-source delta from the next recipient G table within the same group, norm-rescaled. Preserve clean base/donor, full upstream key and input-level F-positive controls.

`G-EDGE-SUPPORTED` requires discovery qualification and, in both seeds: clean base/donor, upstream key and F-positive controls at least 95%; selected source subset and all-source delta at least 75% items, 60% exact groups and 90% non-injection; selected subset beats random and cyclic-recipient controls by at least 40 item-accuracy points; and all numerical errors are below `1e-6`.

Align descriptive attention mass to the original-base-key and transplanted-donor-key G rows for every head before averaging. Preserve discovery masks, all confirmation predictions, attention summaries, hashes and numerical checks. An independent compact auditor reconstructs selection, scores and decisions without rerunning inference.

One CPU execution, 600-second cap. No training, paid API, download, manuscript data or final-test scoring. A pass identifies a synthetic G-row readout edge only.
