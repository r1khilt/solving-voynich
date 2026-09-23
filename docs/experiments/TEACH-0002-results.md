# TEACH-0002 results: supervised first hops did not establish relational lookup

**Registered verdict: NOT QUALIFIED. Independent artifact audit: PASS.** The curriculum taught both replicates to emit a token from the right *two-item candidate set*, but it did not reliably choose the item named by the query. Consequently, composed lookup remained near 50% and the factorial test never achieved a correct four-way quartet. This is a bounded synthetic method result, not evidence about Voynich glyph meanings or an internal causal circuit.

## What was run and verified

The frozen [registration](TEACH-0002.md) and trainer came from Git revision `a514a36a3e00fee1c0b8c99fe085f053f414a126`. Six MPS arms completed: baseline, curriculum and independent-label null for each of two predeclared seeds, 5,000 updates and batch 128 per arm. Total elapsed time was **908.50 s**, below the 1,800 s cap; sampled MPS allocation peaked at **10,703,360 bytes** (about 10.2 MiB), below the 8 GiB cap. The numerical gate passed: finite loss and gradients; maximum hooked-versus-plain logit difference **3.13×10⁻⁷**, below .002. The sampled allocation is not a system-wide memory high-watermark.

The independent [audit](../../scripts/teacher0002_analyze.py) regenerated the exact 3,328-item evaluation suite per arm from seed `62311`, verified the suite hash and frozen source/config hashes, checked all six checkpoint and compressed prediction-file hashes, and independently parsed every stored label, F/G family partition and family signature. It verified query-reversal and factorial structure, recalculated all scores and Wilson intervals, and reproduced every decision clause from the saved item predictions. Its machine-readable record is [audit.json](../../results/TEACH-0002/audit.json). The [raw report](../../results/TEACH-0002/report.json) and six compressed prediction archives are retained for rescoring. Six synthetic CPU-only audit tests passed; no model inference or new training was used for the audit.

## Exact scores

Each non-factorial item cell contains 256 examples. Pair-exact columns count fully correct two-query pairs out of 128; the factorial column counts fully correct four-way quartets out of 128. `TT`, `HT`, `TH`, and `HH` identify the F/G train or held-out family partition for composed lookup.

| Seed | Arm | First hop train / holdout | First-hop pair | Direct train / holdout | Direct pair | Compose TT / HT / TH / HH | Copy | Factorial quartet |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | Baseline | 14 / 25 | 0 | 122 / 124 | 0 | 122 / 114 / 126 / 125 | 256 | 0 |
| 0 | Curriculum | 123 / 133 | 0 | 132 / 129 | 0 | 113 / 128 / 112 / 130 | 256 | 0 |
| 0 | Null | 124 / 131 | 0 | 125 / 124 | 3 | 15 / 18 / 14 / 22 | 256 | 0 |
| 1 | Baseline | 18 / 22 | 0 | 129 / 130 | 0 | 131 / 129 / 140 / 135 | 256 | 0 |
| 1 | Curriculum | 118 / 136 | 0 | 129 / 127 | 0 | 136 / 139 / 135 / 119 | 256 | 0 |
| 1 | Null | 114 / 131 | 1 | 136 / 125 | 0 | 20 / 22 / 23 / 18 | 256 | 0 |

The curriculum's jointly held-out composition was **130/256 (50.8%)** in seed 0 and **119/256 (46.5%)** in seed 1. Relative to the same-compute baseline, that is only **+5/256 (+2.0 percentage points)** and **−16/256 (−6.3 points)**, far below the registered +20-point gate. Both curriculum factorial scores were **0/128 exact quartets**, as were both baselines. All first-hop and direct accuracy gates also failed. The curriculum-versus-null jointly held-out gain cleared the +40-point gate in seed 0 (**+108/256 = 42.2 points**) but narrowly missed it in seed 1 (**+101/256 = 39.5 points**); that single passed clause does not alter the overall verdict. All null factorial scores met their ≤10% requirement, and all copy controls were perfect. The audit's per-cell Wilson intervals and all twelve Boolean decision clauses per seed are in [audit.json](../../results/TEACH-0002/audit.json).

## Shortcut diagnosis from saved predictions

The held-out paired-query tests keep both tables fixed and change only which row is asked for. A correct lookup must change its answer. These were the curriculum's actual behaviors:

| Seed | Task | First visible candidate | Second visible candidate | Outside candidates | Prediction changed when query reversed | Both queries correct | Distinct held-out families |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | First hop | 118/256 | 138/256 | 0/256 | 0/128 | 0/128 | 116 F families |
| 1 | First hop | 114/256 | 142/256 | 0/256 | 0/128 | 0/128 | 116 F families |
| 0 | Direct | 122/256 | 134/256 | 0/256 | 0/128 | 0/128 | 124 G families |
| 1 | Direct | 125/256 | 131/256 | 0/256 | 1/128 | 0/128 | 124 G families |

Thus the curriculum selected one of the two visible keys/objects on **every** paired item, yet almost never changed its answer when the query changed. The 95% Wilson upper bound for **0/128** query changes or pair-exact successes is **2.9%**; for **0/116** F families with any success it is **3.2%**, and for **0/124** G families it is **3.0%**. These family intervals treat each distinct family as a descriptive unit; repeated pairs within a family and shared atomic symbols still limit independence. Seed 1's one direct prediction change did not make that pair correct. This is direct behavioral evidence for a query-insensitive two-candidate shortcut, not a measurement of neuron-level information flow.

The composed factorial predictions add a useful distinction. Swapping the first-table key assignment changed the curriculum prediction on **0/256 paired contrasts** in seed 0 and **1/256** in seed 1; changing the second table's output mapping changed it on **256/256** in both. The model therefore responded to the visible second-table outputs but almost never to the first-table mapping that composition requires. Each four-way quartet had four distinct correct answers; a fixed choice among its two current second-table outputs can get two items right yet **never** get all four. This explains why curriculum factorial item scores were 256/512 and 257/512 while exact quartet scores were 0/128.

The last-100-step losses also fit the shortcut: **0.642/0.643** for the baselines and **0.650/0.649** for the curricula, close to the roughly 0.624 nats expected if 90% of late-stage examples are unresolved two-choice questions and 10% copy examples are easy. The null's late losses were **1.719/1.721**, close to 0.6·ln(12) + 0.3·ln(2) ≈ **1.699** under its random composed targets. This is a consistency check, not proof that the optimizer learned exactly that strategy. The null's jointly held-out composed scores, 22/256 and 18/256, were near the unrestricted 1/12 object guess rate, while its composed outputs were rarely among the two visible candidates. Its behavior supports the positive-versus-independent-label contrast but does not rescue failed composition.

## Interpretation and bounded next options

The main bottleneck appears **before** two-hop composition: neither first-hop nor direct lookup reliably binds the query token to the matching row. More updates of the unchanged mixture are not the most informative immediate test, because the two-choice behavior was stable across two seeds and by the final loss window. The treatment combines first-hop supervision with a staged schedule, so these data cannot isolate the effect of supervision from ordering; nor can they establish a universal architecture limit.

The following are **unregistered proposals**, not results or permissions to claim a mechanism:

1. Build a positive-control teacher in which single-table relational lookup must pass paired query reversals **before** introducing composition. A row-addressing or key-value retrieval architecture, or an explicit supervised row-alignment head, could test whether the current causal transformer/input format is the barrier. Keep an equal-compute unmodified transformer control and a mapping-independent label null.
2. Make the intended intermediate `F(name)` answer an explicit, separately scored *prediction* during composed episodes, then compare against a same-token-budget teacher with unrelated auxiliary targets. If composition improves only when the intermediate is correct, a later intervention can test whether it is causally used rather than merely correlated with the output.
3. Require fresh-family paired query reversals, F-swap/G-remap factorial contrasts, and positive/null gates in two seeds before mechanistic interpretation. Only then freeze a new causal patching study on unseen families. None of these synthetic outcomes would itself decipher the manuscript.

Artifact auditing cannot prove that every recorded optimizer step occurred, nor that a checkpoint generated its paired prediction archive; the saved hashes establish file identity and internal consistency. Family holdout does not remove recurring atomic symbol relations. The evaluation data are synthetic, and no Voynich manuscript labels were scored in this experiment.
