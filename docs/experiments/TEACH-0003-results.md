# TEACH-0003 results: hundredfold dense scale did not learn row binding

**Registered verdict: NOT QUALIFIED. Independent artifact audit: PASS.** Increasing the dense teacher from 668,544 to 75,582,720 parameters (**113.1×**) did not materially improve the fresh-family lookup task. Every dense size remained near two-choice accuracy, and every arm scored **0/128 exact factorial quartets** in both seeds. Correct auxiliary row labels caused one seed-specific improvement on direct single-table lookup, but this did not reproduce in the other seed and did not produce first-hop or composed competence.

## What ran

Frozen source revision `53b548f8315501d832043c09675bdb2e45e6f687` defined two seeds × five arms, each trained for 5,000 MPS updates at batch 64 on the same per-seed episode stream. The arms were dense small (668,544 parameters), dense medium (14,196,864), dense large (75,582,720), and the same large backbone with either true or independent-random auxiliary row labels. The row heads read the final answer-position state but did not route the answer at inference.

The source-matched benchmark passed before launch: 20.47 seconds, 1,246,941,184 sampled MPS bytes, zero wrapped/native logit difference, and a conservative ten-arm projection of 9,388.83 seconds. The scientific run completed in **6,223.71 seconds** (about 1 h 44 min) with the same sampled allocation peak, below the registered six-hour/24 GiB caps. Numerical qualification again recorded zero wrapped/native logit difference and finite gradients. No paid API, model download, manuscript training, or Voynich final-test scoring was used.

## Exact held-out behavior

Counts below use 256 items for each single-item cell, 128 query-reversal pairs, and 128 four-way factorial quartets.

| Seed | Arm | First-hop item / pair | Direct item / pair | Composed HH | Factorial item / quartet | Copy |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | Dense small | 140 / 1 | 130 / 1 | 115 | 257 / 0 | 256 |
| 0 | Dense medium | 145 / 1 | 127 / 1 | 119 | 256 / 0 | 256 |
| 0 | Dense large | 134 / 3 | 122 / 1 | 126 | 253 / 0 | 256 |
| 0 | True alignment | 138 / 2 | 130 / 0 | 123 | 255 / 0 | 256 |
| 0 | Random alignment | 144 / 1 | 125 / 3 | 118 | 251 / 0 | 256 |
| 1 | Dense small | 142 / 0 | 107 / 0 | 129 | 257 / 0 | 256 |
| 1 | Dense medium | 136 / 1 | 119 / 0 | 136 | 254 / 0 | 256 |
| 1 | Dense large | 123 / 2 | 113 / 3 | 134 | 262 / 0 | 256 |
| 1 | True alignment | 134 / 7 | 217 / 84 | 130 | 254 / 0 | 255 |
| 1 | Random alignment | 135 / 2 | 110 / 0 | 131 | 253 / 0 | 256 |

`HH` means both F and G table families were held out. Pair counts require both opposite queries over the same tables to be correct. Factorial item counts are out of 512; quartet counts require all four independent F-swap/G-remap answers. Thus roughly half-correct factorial items with zero exact quartets are the signature of choosing a visible second-table candidate without composing both mappings.

The paired predictions make this failure explicit. Dense-small/medium/large models selected a visible candidate on every held-out query-pair item, but changed their answer under query reversal only 0–6 times out of 128 pairs. Across factorial items, changing the first-table assignment changed only 0–9 of 256 paired contrasts, while changing the second-table output mapping changed all 256. Parameter scale therefore did not remove TEACH-0002's query-insensitive candidate shortcut under this grammar, schedule, and objective.

The true-alignment arm's only notable deviation was seed 1 direct lookup: 217/256 held-out items and 84/128 exact pairs, versus 113/256 and 3/128 for its dense-large control. Seed 0 showed 130/256 and 0/128, so the effect failed the two-seed gate. First-hop and composed behavior remained near chance in both seeds. Even the alignment heads themselves were far below the registered 90% thresholds on held-out tasks and jointly held-out composition. This is a fragile optimization difference, not evidence of a reusable row variable or causal mechanism.

All dense final losses remained near the expected visible-candidate shortcut regime (last-100 means 0.656–0.695 nats). Adding the auxiliary loss raised the combined objective, as expected; loss values are not directly comparable across dense and auxiliary arms. No validation-selected checkpoint was used.

## Independent verification and implication

`scripts/teacher0003_analyze.py` independently regenerated all 3,328 evaluation items, checked family partitions and symbolic answers, verified frozen source/config/benchmark hashes, all ten checkpoints/loss/prediction archives, all 5,000 losses per arm, and recomputed every score and decision clause. Its `results/TEACH-0003/audit.json` verdict is `pass / not_qualified`. It does not rerun neural inference; hashes cannot prove that a checkpoint produced its prediction archive or that every optimizer update occurred.

This is strong evidence against the narrow claim that **simply making this dense teacher about 100× larger** is sufficient. It is not a theorem that scale never helps, and it does not compare pretraining, other objectives, more data, or other architectures. The next frozen study, TEACH-0004, tests a different hypothesis: route the answer through two learned key/value reads and compare against controls receiving the same parsed rows. Synthetic success would qualify an analysis instrument, not decipher the manuscript.
