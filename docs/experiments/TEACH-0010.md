# TEACH-0010: component decomposition of downstream key-to-object readout

**Preregistration before generating the new suite or loading dense checkpoints.** TEACH-0007–0009 identify a reusable query-slot key after two dense transformer layers and show that layer 1 heads 1+3 read the matching F-row V stream to write it. TEACH-0007 also shows that one layer later the query state becomes G-specific and answer-like. This study asks which components of zero-based encoder layer 2 convert the transplanted key plus the recipient G table into an object answer.

The [QK/OV circuit framework](https://transformer-circuits.pub/2021/framework/index.html), [path patching](https://arxiv.org/abs/2304.05969), [causal abstraction/interchange](https://arxiv.org/abs/2106.02997), the [IOI circuit study](https://arxiv.org/abs/2211.00593), and cautions on [activation patching](https://arxiv.org/abs/2404.15255) motivate intervening in the original downstream computation and testing complete component controls. They do not predict a particular layer-2 head or establish any Voynich mechanism.

## Fresh suite and recipient-conditioned upstream cause

Generate 256 new cross-G groups with seed `70111`, first 128 discovery and last 128 confirmation. Load both frozen TEACH-0004 dense checkpoints. For every base recipient G0/G1/G2 episode, run through cut 2. Separately compute the donor-G0 cut-2 query key once per group, then replace only the base recipient's cut-2 query slot with that unchanged donor key. This is the frozen upstream cause. Continue the edited recipient run through encoder layers 2 and 3. It must produce the recipient-specific donor answer on at least 95% of items in both seeds or the component study is uninformative.

At encoder layer 2, explicitly reproduce native self-attention, four pre-output-projection head values, the attention residual, MLP write and final layer output using the TEACH-0008 native-residual convention. All head-sum/native-attention and explicit-layer/native-layer errors must remain below `1e-6`.

## Discovery head selection

At query slot 5, compute each head's edited-recipient minus clean-base output contribution. Screen all 15 nonempty head subsets by adding their projected delta to the clean base attention write and recomputing the native MLP and final encoder layer. A subset qualifies if its minimum across both seeds reaches at least 75% donor items and 60% exact three-G groups. Select the smallest qualifying subset; ties use higher minimum group accuracy, then item accuracy, then lower bitmask. If none qualifies, freeze the maximum-minimum subset and mark discovery unqualified.

## Confirmation and controls

On untouched confirmation groups report:

- selected and all-head attention-write deltas from the edited recipient run;
- complementary heads;
- edited-recipient MLP write alone on the clean post-attention state;
- edited-recipient post-attention query state;
- edited-recipient final layer query state;
- the complete upstream cut-2 key patch;
- input-level two-F-row positive control;
- norm-matched Gaussian selected-write delta, seed `70211`;
- selected-write delta from the next recipient G table in the same group, cyclically shifted and norm-rescaled;
- the group's edited G0 selected-write delta repeated across G0/G1/G2.

The cyclic-recipient and repeated-G0 controls test whether the selected write is recipient-specific and answer-like. They are not semantic nulls. Report ordinary donor-item/group accuracy and cross-G non-injection for every condition.

`ANSWER-WRITE-SUPPORTED` requires discovery qualification and, in both seeds: clean base/donor, full upstream key patch and input-level positive controls at least 95%; selected-head items at least 75% and exact groups at least 60%; selected-head cross-G non-injection at least 90%; selected-head item accuracy exceeds both random and cyclic-recipient controls by at least 40 points; and every numerical error is below `1e-6`. Complement, MLP-only, post-attention, final-state and repeated-G0 conditions are descriptive.

Preserve full discovery screens, confirmation predictions, source/checkpoint/suite hashes and numerical checks. An independent compact auditor reconstructs the global head choice, all scores and the registered decision without rerunning inference. One CPU execution, 600-second cap; no training, paid API, download, manuscript data or final-test scoring. A pass localizes a synthetic answer-writing component, not a unique algorithm or Voynich reading.
