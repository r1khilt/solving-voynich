# TEACH-0005: cross-G interchange test of the learned intermediate state

**Preregistration before loading any TEACH-0004 checkpoint for intervention analysis.** TEACH-0004's two-read models achieved perfect registered fresh-family behavior and row-attention argmaxes in both seeds, while its one-read twin failed composition. That establishes behavioral competence and a useful architectural contrast; it does not establish that the post-F-read vector causally represents a reusable key. This study freezes the causal test proposed before TEACH-0004 ran in [BEYOND_DENSE_BINDING_ARCHITECTURES.md](../research/latent-mechanisms-2026-09-21/BEYOND_DENSE_BINDING_ARCHITECTURES.md). It reuses only the two TEACH-0004 two-read checkpoints and performs no training or checkpoint selection.

## Causal question

For each fresh held-out group, a base first table maps query name `n` to `k0`; a donor swaps the assignment so the same `n` maps to `k1`. Three independently remapped base second tables `G0,G1,G2` contain both keys and use six distinct output objects. The donor post-F state is computed once with `G0`, patched into the base computation, and then reused unchanged with every `Gj`. A reusable-key state predicts `Gj(k1)` separately for each recipient table. A donor-answer state instead predicts the fixed `G0(k1)`, which is wrong under `G1,G2` by construction.

This is an interchange intervention in the sense of causal-abstraction work by [Geiger et al.](https://proceedings.neurips.cc/paper/2021/file/4f5c422f4d49a5a807eda27434231040-Paper.pdf) and [Geiger et al. 2022](https://proceedings.mlr.press/v162/geiger22a.html), specialized to the registered synthetic computation. Their results motivate the method but do not predict this outcome. The state site is architecture-defined before analysis: the 128-dimensional weighted F-row value named `first` in `LearnedMemory.forward`, immediately before concatenation with the query and projection into the G-row query. No probe, neuron selection, direction fitting or validation search is allowed.

## Frozen suite and controls

Seeds are suite `65111` and norm-random `65211`; 128 unique groups are generated. Every base F family, same-key-donor F family and each of the three base G families must hash to TEACH-0004 holdout. The same-key donor keeps `n -> k0` but changes the irrelevant second name/key relation. All six G outputs within a group are distinct, so cross-G adaptation cannot pass by retaining one donor object. Both TEACH-0004 replicates are reported separately; groups are shared.

Each group produces three recipient-table items and these frozen conditions:

- clean base and clean donor;
- donor `first` patched into each base recipient;
- same-key/different-distractor `first` patched into base;
- independently sampled Gaussian state rescaled to the donor-state norm;
- zero `first` necessity ablation;
- donor F-attention weights applied to base F-row values, separating selection weights from transported content;
- base `first` restoration after the donor-state corruption;
- complete donor post-G `second` state patched into every base recipient, a positive control for fixed answer injection;
- direct-key and copy tasks with the donor `first` patch, which should not materially degrade unrelated behavior.

The runner records every token sequence, oracle label, prediction and relevant target probability. It verifies the frozen checkpoint hashes from TEACH-0004, exact unpatched recomputation, finite logits, source hashes and the deterministic suite. Results go to a fresh `TEACH-0005` namespace. No manuscript data, paid API, new model download or MPS training is involved.

## Frozen metrics and decision

Primary denominators contain groups whose clean base and donor composed predictions are correct under all three G tables. Report eligibility rather than silently dropping failures. The registered verdict is `CAUSAL-KEY-SUPPORTED` only if **both checkpoint seeds** satisfy every clause:

1. at least 95% of all groups are clean-correct for base and donor across all three G tables;
2. donor-first patches produce all three recipient-specific `Gj(k1)` answers in at least 80% of eligible groups and at least 90% of eligible items;
3. donor-first item accuracy exceeds norm-matched-random donor-target accuracy by at least 40 percentage points;
4. mean donor-target probability rises by at least 0.40 versus the clean base run;
5. on `G1,G2`, donor-first patches avoid the fixed donor `G0(k1)` answer on at least 90% of eligible items, while full donor-second patches produce that fixed answer on at least 80%;
6. zeroing `first` reduces base-answer accuracy by at least 30 points, while restoring the original base `first` restores at least 95% base-answer accuracy;
7. same-key state patches and attention-only patches each preserve at least 90% base-answer accuracy;
8. donor-first patching reduces direct and copy accuracy by no more than five points from their clean values;
9. all logits are finite and exact unpatched state-injection recomputation differs from native logits by less than `1e-6`.

Failure of any clause gives `NOT SUPPORTED`, not proof that no transformed or distributed key representation exists. If fewer than 95% of groups pass clean competence, checkpoint/source/artifact validation fails, or the positive full-second answer-injection control fails, the causal claim is `INCONCLUSIVE` rather than negative. Exact group counts and Wilson intervals are reported descriptively; the thresholds above, not a posthoc significance choice, determine the verdict.

## Interpretation boundary

A pass would show that, in this hand-typed synthetic memory architecture, one frozen internal state has the counterfactual behavior of a reusable intermediate key: its effect adapts to recipient G tables rather than carrying a donor answer. It would not make the representation unique, prove individual neurons have invariant meanings, show that the dense-row model uses the same algorithm, or identify any Voynich latent. The manuscript does not supply row boundaries, task markers, key types or ground-truth interventions. Transfer requires separate known-system and manuscript studies.
