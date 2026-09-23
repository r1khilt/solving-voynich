# TEACH-0004 results: parsed relational records unlock perfect fresh-family binding

**Registered verdict: NOT QUALIFIED. Independent artifact audit: PASS.** The verdict is negative only for the preregistered claim that the learned two-read memory would outperform *both* controls by at least 15 percentage points. It did not outperform the parsed-row dense control because both models achieved perfect behavioral scores in both seeds. This is nevertheless the first teacher study in this sequence to eliminate the query-insensitive visible-candidate shortcut on fresh families.

## What ran

Frozen source revision `e6e2aaab2d9413132b38b7da95661e857fd0558f` defined two seeds × three arms, each trained for 5,000 MPS updates at batch 64 on identical per-seed episode streams. `two_read` and `one_read` each had 222,215 parameters. The two-read model used a query-conditioned F-row read as context for its G-row read; the one-read ablation replaced that context with the unconditioned mean of F-row values. `dense_row` had 883,591 parameters and four dense transformer blocks over the same explicitly parsed F rows, G rows, task marker and query.

The source-matched benchmark passed in 2.23 seconds, with a conservative six-arm projection of 467.56 seconds, sampled MPS peak 14,138,368 bytes, finite gradients and zero repeated-forward logit error. The scientific run completed in **301.37 seconds** with the same sampled allocation peak, below the registered 3,600-second/8-GiB caps. No paid API, model download, manuscript training or Voynich final-test scoring was used.

## Exact held-out behavior

Counts below use 256 items for each single-item cell, 128 query-reversal pairs and 128 four-way factorial quartets. `HH` means both F and G families were held out. A pair or quartet counts only when every linked answer is correct.

| Seed | Arm | First-hop item / pair | Direct item / pair | Composed HH | Factorial item / quartet | Copy |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | Two-read | 256 / 128 | 256 / 128 | 256 | 512 / 128 | 256 |
| 0 | One-read | 256 / 128 | 256 / 128 | 113 | 256 / 0 | 256 |
| 0 | Dense-row | 256 / 128 | 256 / 128 | 256 | 512 / 128 | 256 |
| 1 | Two-read | 256 / 128 | 256 / 128 | 256 | 512 / 128 | 256 |
| 1 | One-read | 256 / 128 | 256 / 128 | 110 | 254 / 0 | 256 |
| 1 | Dense-row | 256 / 128 | 256 / 128 | 256 | 512 / 128 | 256 |

The two-read attention argmax selected the constructed F and G rows on every scored first-hop, direct and composed item in both seeds. In the factorial panel, swapping the F assignment changed all 256 required paired predictions and remapping G changed all 256. Its complete quartet rate was 128/128 in both seeds. Thus it did not merely learn a readable pointer: the answer path behaved correctly under every registered counterfactual.

The matched one-read ablation isolated the missing dependency. It still selected the correct F row on all 512 factorial items, but because that F read did not feed the composed G query, G-row argmax accuracy fell to 259/512 in seed 0 and 256/512 in seed 1. Swapping F changed only 4/256 and 6/256 factorial predictions, while remapping G changed all 256. Jointly held-out composition was 113/256 and 110/256 and exact quartets were 0/128. This reproduces the old shortcut in a controlled architecture: the model can read either table alone, but composition fails when the first answer is prevented from controlling the second read.

The dense-row control was also perfect in both seeds. It had 3.98 times as many parameters and a different computation path, so this is not an equal-parameter comparison. It shows that explicit recurrent reads are not uniquely required once the input is supplied as typed relational records. Relative to TEACH-0003, where dense sequence models up to 75.58 million parameters stayed near chance, the strongest changed variable is the parsed row interface plus the resulting relational inductive bias. TEACH-0004 did not isolate row parsing from all other implementation differences, so that explanation remains a comparative inference rather than a single-factor causal result.

## Registered decision and interpretation

The two-read model passed every absolute competence clause and exceeded the one-read control by 55.86 and 57.03 points on jointly held-out composition, and by 100 points on exact factorial quartets. It failed only the two clauses requiring at least a 15-point gain over dense-row, whose perfect scores leave no possible positive margin. The registered verdict is therefore `not_qualified`; it must not be relabeled after seeing the outcome.

The useful conclusion is narrower and stronger than “bigger worked.” A 222k-parameter model with the right interface learned the behavior that 75.6M-parameter raw dense teachers did not. That makes structured representation a much better current lead than indiscriminate scale for this diagnostic. Perfect behavior and attention alignment still do not identify a unique internal variable. The next registered study must patch the proposed post-F-read state across independently changed G tables and test whether the patched answer follows each base G table. That cross-G interchange distinguishes a reusable intermediate key from transplanting a donor answer.

## Independent verification and limits

`scripts/teacher0004_analyze.py` independently regenerated all 3,328 items per arm, verified symbolic labels, family partitions, pair/quartet structure, frozen-source ancestry and exact source blobs, benchmark/config/suite hashes, all checkpoint/loss/prediction archives and every registered decision clause. `results/TEACH-0004/audit.json` reports `pass / not_qualified`. The audit does not rerun neural inference; artifact hashes alone cannot prove that checkpoints generated the saved predictions or that every optimizer update occurred.

The generator provides explicit symbol classes, row boundaries, table types, task markers and known answers. Voynich text provides none of these. This result qualifies a synthetic causal instrument and identifies a promising architectural direction; it supplies no Voynich token meaning, language identification or decipherment.
