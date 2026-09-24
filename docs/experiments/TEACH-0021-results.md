# TEACH-0021 results: second-row selection is the dominant observed error

**Scope:** source-frozen, exploratory inference on the previously exposed
seed-74111 development suite. This is not the interrupted TEACH-0014
confirmation verdict, a valid TEACH-0015/16 entry, or a Voynich
decipherment.

## What was tested

Commit `4e99b039cb508edc9859e29e5e1c3358653c1481` froze the
registration, runner, independent all-row auditor, replay and tests before
checkpoint inference. The two completed 6,000-step public-row oracle
checkpoints were evaluated on 128 items each from
`first_hop_confirm`, `direct_confirm` and
`composed_confirm_confirm` in the previously exposed seed-74111 suite.
The model received exactly parsed visible rows. Each target row was
reconstructed from those visible rows, not the generator's signal-path
metadata.

The intervention substituted the model's **own projected value** of a
correct or wrong visible row after its read attention and before its
learned state update. Identity substitutions reproduced all CPU logits
within the registered 1e-5 bound. Every clean prediction matched the
independently replayed TEACH-0020 archive. The no-model audit checked all
768 episode rows and all sampled logits. A separate checkpoint replay
matched 108 full 2,064-logit vectors exactly on CPU. The runner took
1.378 seconds and wrote a 4,593,399-byte raw archive, under its 20-minute
and 100-MiB caps; the replay took 0.900 seconds under its 10-minute cap.

## Two-hop behavior

| Seed | Clean answer | First target-row attention | Second target-row attention | Gold first value | Gold last value | Gold both values | Wrong both values |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 97/128 | 122/128 | 99/128 | 123/128 | 127/128 | 127/128 | 0/128 |
| 1 | 100/128 | 123/128 | 100/128 | 121/128 | 128/128 | 128/128 | 0/128 |

Among the 31 clean errors in seed 0, the second target row was *not* the
attention argmax in 29; among the 28 clean errors in seed 1, it was wrong
in all 28. First-row argmax was correct in 27/31 and 24/28 of those
clean-error cases. On the paired clean errors, gold-last rescued 30/31
and 28/28 without damaging any clean-correct item. Gold-first rescued
27/31 and 22/28 but damaged one clean-correct item per seed. The wrong
visible-row value control scored 0/128 in both seeds and damaged every
clean-correct answer. The model therefore is sensitive to the supplied
value; a generic intervention does not rescue it.

Both one-hop panels also became perfect under the correct row-value
substitution: first-hop from 118/128 and 113/128 to 128/128 both seeds,
and direct from 119/128 and 110/128 to 128/128 both seeds. Their
wrong-value controls scored zero. Native one-hop target-row attention
was 122/128 and 120/128 on first-hop, and 120/128 and 115/128 on direct.

## Interpretation and limits

The strongest localized observation is **second-read selection**. The
model usually identifies the first row but often attends to a different
second row, and nearly every clean two-hop error accompanies that
second-row miss. Supplying the correct final value almost perfectly
restores the answer, which argues against final output binding being
the dominant failure on this exposed suite. Supplying the correct first
value helps substantially but does not fully repair the second read,
so the learned value-to-state-to-next-address path remains suspect.
Attention argmax alone is descriptive; the value interventions provide
the causal sufficiency result. An argmax miss does not by itself prove
the model never used the correct row's information, and a gold-last
intervention bypasses the attention mixture.

The suite was exposed before the study and the arm selection responded
to TEACH-0020. No statistical significance or confirmatory
generalization claim is made. The original campaign lacks ten of 22
complete runs, and seed 84311 remains unscored. This synthetic
two-operand graph program is an instrument for testing memory/reader
mechanisms, not evidence that Voynich text encodes such programs.

The next model change should target carrying a crisp intermediate value
into the *next* address, with a tied or identity-initialized value/state
path, while retaining the same correctly parsed input and an
equal-compute current-reader control. A fresh graph-family split is
required before treating improvement as generalization.
