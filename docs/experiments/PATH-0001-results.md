# PATH-0001: a single-block attention route was not established

**Registered verdict: uninformative.** On a fresh two-slot binding task, the selected four attention heads in block 25 (zero-based layer 24) redirected 5/16 eligible confirmation answers. A narrower patch that changed donor value vectors only at the differing record-token positions redirected 3/16. But replacing **all 32 head outputs** in that block redirected only 3/16, below the registered 8/16 positive-control requirement. The study therefore cannot decide whether the proposed value route is sufficient. No Voynich decoding was attempted.

## Inputs and selection

Four discovery bundles and four disjoint confirmation bundles each supplied two invented names, two value words, both queried slots, both value orientations, and unrelated copying controls. Each split contained 16 binding records and eight copy records. Both source and donor clean generations were correct and had different first generated tokens for all 16 binding records in each split. All eight copy baselines were correct in each split. The actual model's attention-head reconstruction differed from native attention output by at most 4.77e-6; clean, zero-edit, and captured logits agreed exactly in the generic qualification. Task-specific head reconstructions and identity generations also passed their runtime gates.

The discovery scan made 2,048 single-head causal interventions over zero-based layers 23, 24, 27, and 28. The frozen ranking chose layer 24 and heads 29, 23, 31, and 5 (in that order). No confirmation outcome entered this choice. The confirmation grid contained 192 generations: 24 records × eight conditions. All figures below use the 16 both-clean-correct semantic records, but the four bundles, reverse orientations, and two queried slots are dependent; 16 is not the independent sample size.

| Condition at selected block | Desired donor answer | Unrelated copy preserved |
| --- | ---: | ---: |
| Identity | 0/16 | 8/8 |
| Best one donor head | 0/16 | 8/8 |
| Four selected donor heads | 5/16 | 8/8 |
| Four heads, source Q/K, donor V at changed value-token positions | 3/16 | 8/8 |
| Four heads, donor Q/K, source V | 0/16 | 8/8 |
| Four heads, source Q/K, donor V at all prompt positions | 3/16 | 8/8 |
| Four random heads, residual edit norm matched | 0/16 | 8/8 |
| All 32 donor head outputs | 3/16 | 8/8 |

The selected-four donor-head successes occurred across all four bundles (1, 1, 1, 2). The restricted donor-value successes occurred only in the last two bundles (0, 0, 1, 2). Mean donor-minus-source first-token margin gain was +20.67 logits for selected four heads, +16.49 for the changed-value V patch, -0.47 for Q/K-only, and -2.03 for random-four. A large margin change did not reliably produce the full desired answer. Whole-head and value-stream patches both changed model outputs without reaching the registered sufficiency bar. Adding all 32 heads was **less effective** than adding the selected four in this panel, consistent with interfering effects and nonlinear continuation; it rules out treating head count as a monotone dose.

## Interpretation and next test

This is evidence that the selected heads can influence these explicit bindings, because selected-four transfer redirected five answers and the norm-matched random-four transfer redirected none. It is **not** evidence of a complete attention circuit, nor a positive registered value-route result. The all-head positive control's 3/16 outcome means a single block's final-position attention write did not reliably transplant the answer, even when every head was changed. The model was competent on all clean cases, so poor task performance is not the explanation. One plausible explanation is that relevant information is distributed over several layers or must be changed at earlier prompt positions, as ROUTE-0001 suggested; this is a hypothesis, not identified by PATH-0001.

A separately registered successor should test direct value-position versus other-position **residual** transport across a fixed layer grid on new binding bundles. Full donor residual replacement across all prompt positions supplies an exact first-token positive control; compare it to changed-value positions, earlier positions excluding those values, the final query position, and matched random fields. That would locate the bypass more sharply without using this exposed confirmation panel as a fresh test. Later attention-path tests should require this support before searching individual heads again.

## Execution and integrity

The finite run completed in 993.92 seconds (16m34s), peak MLX allocation 28.809 GB, under the 45-minute/45-GB caps. Existing cached Qwen3-8B four-bit weights were expanded to dense F32 block computation; no new model, training, paid API, external upload, or manuscript final scoring. Source revision `ce33f96` and hashes are in `results/PATH-0001/inputs.json`. Prompt IDs and rendered tokens are ignored in `outputs/PATH-0001/`; compact baseline, scan, selection, row, summary, qualification, decision, and independent-audit JSONs are tracked. The independent audit replays all 2,048 scan keys, selected ranking, all 192 confirmation keys, correctness flags, aggregate counts, decision threshold, edit-norm match, identities, and source/input hashes. It passed. Two CPU task checks and three tiny live-Qwen numerical tests passed before the run. The original JSPACE/NEURON failures and ROUTE exploratory diagnosis remain unchanged.
