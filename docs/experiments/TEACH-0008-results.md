# TEACH-0008 results: two attention heads jointly write the dense model's key

**Registered outcome after two transparent numerical repairs: `HEAD-WRITE-SUPPORTED`. Independent audit: PASS.** In both independently trained dense transformers, discovery selected the same pair—zero-based heads 1 and 3 in the second transformer layer. On fresh confirmation groups, transplanting only their combined query-slot attention write changed 378/384 and 379/384 outputs to the donor key's recipient-specific answer. The other two heads, the MLP write alone and norm-matched random writes changed 0/384.

## What ran and why the valid result is non-blind

The study generated 256 new held-out cross-G groups from seed `68111`, split 128 discovery/128 confirmation. It loaded the two frozen TEACH-0004 dense-row checkpoints and explicitly decomposed zero-based encoder layer 1 into four self-attention heads, the output projection, the attention residual and the feed-forward residual. Discovery screened all 15 nonempty head subsets; one global subset was selected across both seeds before confirmation scoring.

The first two executions were invalid because one explicit whole-layer reconstruction error was `1.0728836e-6`, barely above the frozen `<1e-6` gate. Both reports and rows are preserved. The first repair matched PyTorch's scaled-dot-product attention kernel. The second separated the explicit head-sum check from the exact native attention residual baseline, preventing sub-ULP attention differences from being amplified by the following MLP. Neither repair changed a seed, example, checkpoint, head rule, intervention, behavioral output or threshold. Because aggregate behavior was visible after the first invalid run, this final execution is a non-blind numerical repair rather than a pristine confirmation.

In the valid run, explicit head sums matched native attention with maximum errors `5.96e-7`–`8.34e-7`; explicit residual-plus-MLP outputs using the native attention baseline matched native layer outputs exactly (`0.0`). Runtime was 0.643 seconds on CPU.

## The two useful heads are jointly sufficient

On discovery, heads `[1, 3]` were the smallest subset meeting the frozen shared criterion. Their minimum across the two seeds was 99.22% donor items, 98.44% exact three-G groups and 100% cross-G non-injection. Neither head worked well alone:

| Discovery intervention | Seed 0 items / groups | Seed 1 items / groups |
| --- | ---: | ---: |
| Head 1 alone | 8.85% / 5.47% | 15.10% / 5.47% |
| Head 3 alone | 25.00% / 13.28% | 15.89% / 8.59% |
| **Heads 1 + 3** | **99.48% / 98.44%** | **99.22% / 98.44%** |
| Heads 0 + 2 | 0% / 0% | 0% / 0% |
| All four heads | 99.48% / 98.44% | 100% / 100% |

The pair's effect is therefore strongly non-additive at the decision level. Each partial write usually remains on the original side of the downstream classifier, while their combination crosses it. This does not prove that heads 1 and 3 implement two clean symbolic subroutines; it shows that their joint output-projected update is the smallest causally sufficient subset under the registered screen.

## Fresh confirmation isolates attention from the MLP and other heads

Each seed had 128 untouched groups and three recipient G tables per group:

| Confirmation intervention | Seed 0 items / groups | Seed 1 items / groups |
| --- | ---: | ---: |
| **Selected heads 1 + 3** | **378/384 / 125/128** | **379/384 / 124/128** |
| All four heads | 378/384 / 125/128 | 382/384 / 127/128 |
| Complement heads 0 + 2 | 0/384 / 0/128 | 0/384 / 0/128 |
| Donor MLP write only | 0/384 / 0/128 | 0/384 / 0/128 |
| Norm-matched random write | 0/384 / 0/128 | 0/384 / 0/128 |
| Full post-attention query state | 384/384 / 128/128 | 384/384 / 128/128 |
| Full final-layer query state | 384/384 / 128/128 | 384/384 / 128/128 |
| Two-F-row positive control | 384/384 / 128/128 | 384/384 / 128/128 |

All selected-head predictions had 256/256 cross-G non-injection in each seed. That means the intervention did not merely copy the donor's final answer: the unchanged recipient G table recomputed the appropriate result from the transplanted key. Clean base/donor controls were perfect.

Heads 0 and 2 can make tiny boundary corrections: all four heads recover four additional items relative to the selected pair in seed 1, but none in seed 0. They are neither independently sufficient nor necessary for the main key transfer on this panel. The MLP is also not a parallel key-writing path at this site. It may still normalize or transform other features during ordinary execution; the claim is limited to the donor-key counterfactual tested here.

## Their write has the same low-dimensional distributed geometry

The discovery key-centroid span of the selected-head query update again had centered rank 11. Projecting the confirmation update into that span retained 373/384 and 376/384 donor items, close to the unprojected pair's 378/384 and 379/384. An 8D truncation already reached 365/384 and 369/384. Lower ranks degraded smoothly:

| Dimensions | Seed 0 items / groups | Seed 1 items / groups |
| ---: | ---: | ---: |
| 1 | 23/384 / 6/128 | 24/384 / 5/128 |
| 2 | 64/384 / 20/128 | 90/384 / 17/128 |
| 4 | 159/384 / 40/128 | 192/384 / 55/128 |
| 8 | 365/384 / 115/128 | 369/384 / 116/128 |
| 11 | 373/384 / 121/128 | 376/384 / 122/128 |

This connects component-level routing to TEACH-0007's state-level result: the second layer's attention heads write almost exactly the low-rank key variable later found in the query residual stream. The 11D ceiling follows the twelve synthetic key classes and should not be universalized to language.

## Verification and meaning

`scripts/teacher0008_analyze.py` independently regenerated the deterministic suite, reproduced the global subset choice and all registered decision clauses, verified frozen source/checkpoint/row hashes, checked every confirmation label and rescored all ten interventions. It reports `audit: pass`. It does not rerun neural inference or independently recompute the hidden-state geometry.

The strongest justified mechanism story is now: early F rows carry the supplied relation; in layer 2, attention heads 1 and 3 jointly route a reusable key into the query slot; later layers combine that key with the current G table and turn it into an answer. This is a causal circuit fragment, not a visualization-based guess.

The task still supplies explicit typed rows, a dedicated query token, known key classes and supervised answers. The result does not identify a Voynich latent, word meaning, language or cipher. It qualifies a sharper method for the real problem: search for compact, causally reusable variables and then decompose which heads write, read and transform them, with cross-context controls that distinguish a reusable latent from final-answer copying.
