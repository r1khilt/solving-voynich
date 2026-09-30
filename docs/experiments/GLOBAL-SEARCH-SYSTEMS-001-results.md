# Global search cost benchmark: exact scoring practical, partial bounds capped

2026-09-30. Engineering PASS on the registered artificial workload. No empirical
cipher case, new reading, dictionary recovery or historical decipherment tested.

Source/protocol b7de43fc4cad4d0d85f8bb84d1e5fc0cebafd685 was pushed and its exact
origin/main hash verified before one run (session81132, exit0). Frozen settings
and artificial inputs were not changed. The alternate arithmetic/accounting
audit also completed once (session49811, exit0). All outputs are bound in
`results/GLOBAL-SEARCH-SYSTEMS-001/result.json` and `audit.json`; bulk score,
event and progress archives are ignored and checksum-bound.

All four dictionary variants and their short/full four-record workloads were
scored. Sixteen short-record native/Python checks agree in support, node/edge
counts and likelihood, maximum4.547473508864641e-13nats. Full four-record groups
took0.003237–0.003841seconds on this workload. These are uniform artificial
strings, not a prediction of Latin ciphertext search costs or a recovery test.

The eight-temperature search completed256iterations/2,048proposals, scored
2,034distinct keys/8,136record marginals, and reused15cached scores. It accepted
260/983replacement,202/538swap and41/527block proposals;80/224replica exchanges.
Nine proposals were self transitions. Search elapsed10.059302seconds; whole
run11.055411wall/11.033629CPU seconds,843,710,464bytes peak host RSS, one CPU/BLAS
thread,0GPU/training/paid. Stopping reason was the registered iteration limit.
Every visited fully scored key was retained; no stationary sample/global optimum
or dictionary correctness claim follows from acceptance or objective gains.

The partial relaxation hit the20,000node/80,000edge caps at0,11and20assigned rows
in both backends. Their native elapsed times were0.022026/0.013693/0.005928seconds.
At23assigned rows it completed in0.001119seconds and matched the ordinary exact
record marginal. Cap outputs are limitations, not approximate bounds or zero
probabilities. The current high-order relaxation therefore cannot be assumed a
practical global A* heuristic even when only three rows remain unassigned.
This finite workload does not prove every partial assignment will hit a cap.

The audit verified11artifact identities, all4,306progress rows, all2,034code
costs/objective sums, every proposal/exchange decision and state transition,
temperature/event schedule, selected score, counters and summary/config binding.
It took0.061225wall/0.060227CPU seconds,249,282,560host bytes. This is root-authored
alternate arithmetic, not completed independent agent validation; the earlier
partial read-only reviews ended at service limits. Full regression2,324tests
plus23subtests passed,13skips; five unrelated pre-existing full-tree Ruff issues
remain, changed source/tests pass.

The next executable branch is bounded multi-temperature dictionary search under
the unchanged strong source, on **all32exposed development cases**. Keep original
positive/shuffle controls and full recovery denominators. Warm starts, candidate
retention, budgets and readings need a separate frozen protocol. Actual decoding
must be evaluated after keys/predictions are published; score improvement alone
does not imply better plaintext. The fresh CONFIRM-002 failure remains a failure.
