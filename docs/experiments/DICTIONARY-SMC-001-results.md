# DICTIONARY-SMC-001 — mutations prevent most extinction, stability still fails

2026-10-01. [Frozen registration](DICTIONARY-SMC-001.md) at96562f96d427b32d17d1ba6fba2a4a1abe545e5e was committed/pushed/exact-remote verified before the one run. All144calls and one complete audit finished. ALL THREE arms fail the prospective calibration gate. No threshold, seed, population, stride or mutation-budget retuning occurred.

| Arm | Completed estimates /48 | Largest four-seed log spread | Prefix preferred /16 | Calls with an incremental weight>.5 | Fixed-key evaluations | Summed call wall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| no_move |1|No query had all4positive|0|3|67,072|0.205s|
| row |48|70.337nats|13|25|2,446,848|24.679s|
| joint |46|38.966nats among fully positive queries|10|20|2,425,600|23.231s|

The row arm passes positivity and prefix preference, but fails stability and incremental weight concentration. The joint arm fails allfourclauses, including extinction for case0-root seeds81401/81402. no_move fails allfourclauses. Every fully positive row/joint query has spread>2nats. The maximum incremental weight is1for no_move, .997432397for row and .999151302for joint; corresponding minimum stage ESS1/1.005153475/1.001698837. These are incremental weights, not independent-prior likelihood shares; equal resampled weights cannot erase that collapse.

## What changed

With identical conditional-prior initial populations, no_move loses47of48populations. Allowing dictionary changes yields48of48(row) and46of48(joint)completed populations. This supports using observation-conditioned mapping revision rather than only conditioning/resampling a fixed set of prior guesses. It does not demonstrate recovered keys, adequate posterior mixing or accurate likelihood estimates.

Row accepted505,882of1,631,232proposals, with466,844actual changed mappings; joint accepted416,900of1,616,896, with387,744actual changes. Acceptance includes self proposals, hence actual changes are reported separately. Minimum final distinct full keys71for row and26for joint; differences in unused mappings and common ancestry mean these are neither effective independent samples nor distinct readings. Joint has fewer calls violating the weight criterion and lower maximum spread on its fully positive queries, but loses two populations and fails prefix preference. This budget provides no evidence to prefer the joint mixture overall.

The control is much cheaper because most populations go extinct early. Row and joint have matched prescribed proposal/table budgets for equal completed stages, not matched CPU or work on realized trajectories. Compared with the earlier prior-bank experiment, this is additional computation and a different weight statistic; no equal-cost superiority claim is made.

## Correctness, costs and limits

The one full audit regenerated every call, RNG trajectory, final bank, score, trace, population weight, proposal/acceptance summary, work count and artifact binding. ALL4,939,520fixed-key evaluations—including every prefix and rejected proposal—were checked with alternate outgoing DP. Zero support agreed exactly; maximum log discrepancy3.63797880709e−12nats≤1e−7. [Audit](../../results/DICTIONARY-SMC-001/audit.json), [complete compact result](../../results/DICTIONARY-SMC-001/result.json). Same-author stochastic-core replay plus alternate source calculation is not independent expert review; [exact finite invariant-law checks](../../results/DICTIONARY-SMC-THEORY-001/result.json) separately cover1960target/kernelmatrices and392one-particle evidence expectations.

Actual run48.847906wall/48.569973CPU seconds/845,119,488bytes peak processRSS; audit119.453025wall/119.264684CPU/845,348,864bytes. Both within1800wall/1600absoluteCPU/2GiBhost. Full output1,606,644ignored compressed bytes<128MiB. CPU/BLAS1,0GPU/newtraining/paid/holdout. Maximum registered5,709,312evaluations; actual4,939,520lower because extinction. Realized time was much shorter than the conservative3–20min/stage preparation estimate. Compact144per-call metric/manifest files total100,462bytes, result127,170bytes; full banks/trajectories remain ignored under outputs/DICTIONARY-SMC-001.

Before freezing, full2712tests+23subtestsPASS/13skip163.14s; all22new tests passed, including complete144tinycall transport and outgoing likelihood audit. Newcode scopedRuffPASS; fulltree retains the same5oldfindings. All165frozen paths and original source arrays remain unchanged. No retries/resume/extensions, source-state search, fixed-key contextual reader or neural fits were run.

This targets only the explicitly specified root-IID surrogate on twelve previously exposed artificial queries. Correct-prefix and altered-binding comparisons are known-answer-conditioned/hypothetical diagnostics. The context-aware evaluator has exact small-state verification but no realistic contextual inference qualification. No new-key/null recovery test, neuron intervention or actual manuscript decipherment happened. Voynich remains UNSOLVED; the long-term goal is active.

[Post-outcome next decision](../research/after-dictionary-smc-2026-10-01.md): improve the information in proposals and measure genuine contextual inference before claiming a solver. The present mixture's formal full-key reachability is insufficient evidence of useful mixing.
