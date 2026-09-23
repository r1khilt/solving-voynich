# TEACH-0008: attention-head and MLP decomposition of the dense key write

**Preregistration before new suite generation or dense-checkpoint access.** TEACH-0007 independently localized a perfect cross-G key state to the dense transformer's query slot after two layers. At the preceding cut, the query slot had no donor-key effect. This study decomposes the second transformer's update (zero-based encoder layer 1) into its self-attention and feed-forward writes and then into four attention-head contributions.

## Exact decomposition and fresh split

Generate 256 new groups from seed `68111`, first 128 discovery and last 128 confirmation. Load the two frozen TEACH-0004 dense-row checkpoints. For norm-first layer 1, explicitly reproduce:

`post_attention = x + self_attention(norm1(x))`

`post_mlp = post_attention + feed_forward(norm2(post_attention))`.

Also reproduce multi-head attention from `in_proj_weight/bias`, scaled dot-product softmax, per-head value aggregation and `out_proj`. Native layer output, explicit layer output and summed head output must agree below `1e-6` in each seed or the result is incomplete.

At the query slot only, discovery tests donor replacement of: the full attention write; post-attention state; MLP write while retaining base attention; final post-MLP state; each of four attention-head output-projection contributions; and all 15 nonempty head subsets. Every donor state/write is computed once under `G0` and reused with recipient `G0,G1,G2`.

## Frozen subset selection and confirmation

A head subset qualifies on discovery if its minimum across both seeds reaches at least 60% exact three-G groups and 90% cross-G non-injection. Select the smallest qualified subset; break ties by maximum minimum group accuracy, then item accuracy, then numeric bitmask. If none qualifies, freeze the maximum-minimum subset and mark discovery unqualified.

On untouched confirmation groups report:

- selected head-subset write;
- all-head attention write;
- complementary-head write;
- donor MLP-only write;
- donor post-attention and final post-MLP query states;
- norm-matched Gaussian query write, seed `68211`;
- clean base/donor, cut-0 two-F-row positive and exact decomposition controls.

`HEAD-WRITE-SUPPORTED` requires discovery qualification and, in both seeds: clean/positive controls≥95%; selected subset donor items≥75%, exact groups≥60%, non-injection≥90%; selected donor advantage over norm-random≥40 points; and every numerical error<`1e-6`. Report complementary heads and MLP descriptively; a selected subset need not be uniquely necessary because parallel/redundant paths may exist.

Fit the selected query-write difference's discovery key-centroid span and report rank/dimension curve on confirmation, but do not register a new subspace verdict: TEACH-0007 already established the post-layer key subspace, while this study's primary question is which layer components write it. Preserve discovery screen, confirmation rows, source/checkpoint/suite hashes and numerical checks; independent audit reconstructs decisions without rerunning inference. CPU cap 600 seconds, no training, paid API, download, manuscript data or final-test scoring. Any pass is synthetic mechanism evidence only.
