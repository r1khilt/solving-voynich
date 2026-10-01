# SHARED-KEY-GUIDE-001 — prior dictionary sampling is not a stable guide

2026-10-01. **All three arms fail the registered calibration.** Preserving dictionary reuse fixes a real approximation, but randomly completing the remaining dictionary produces scarce supported samples and extreme likelihood concentration. Integrating one mapping exactly did not make the estimates usable at the tested budget. No new search, recovered passage or Voynich reading is reported.

[Registration](SHARED-KEY-GUIDE-001.md), [method/source review](../research/shared-dictionary-guide-2026-10-01.md), [closed result](../../results/SHARED-KEY-GUIDE-001/result.json), [full audit](../../results/SHARED-KEY-GUIDE-001/audit.json), [compact calls](../../results/SHARED-KEY-GUIDE-001/compact-calls.json), [static publication/archive audit](../../results/SHARED-KEY-GUIDE-001/publication-audit.json).

## Fixed comparison

Twelve queries from four exposed artificial cases: unbound root, archived correct-prefix state, and one hypothetical changed binding. Four predetermined seeds, three arms,144calls. The source is the same smoothed root IID surrogate, with dictionary reuse retained across both passages. These known-answer-conditioned diagnostics are not actual competing beam states or historical evidence.

| Arm | Dictionary tables per call | Positive estimates /48 | Positive estimates with≥99% contribution from one base sample | Correct-prefix preference /16 | Largest finite seed log spread |
| --- | ---: | ---: | ---: | ---: | ---: |
| MC16 |16|5|5/5|2|None: every query has at least one zero seed|
| MC672 |672|43|42/43|9|186.312 nats|
| RB16 |672, grouped into16 integrated samples|29|28/29|6|167.518 nats|

There were5supported complete dictionary evaluations among768MC16tables,162among32256MC672tables, and171among32256RB16tables. These counts include repeated samples across conditional queries and do not estimate independent-key population frequencies. Maximum contribution shares reached1in every arm. Positive-estimate contribution ESS ranged1..1.466forMC672 and1..1.473forRB16. Different grouping resolution prevents treating those ESS values as a direct same-resolution comparison.

All arms failed all-query positivity, the2-nat stability limit, the0.5maximum-contribution limit, and the12/16prefix-preference requirement. The largest reported spreads apply only to queries with all four positive estimates; queries with a zero estimate have an undefined spread, not zero instability. Exact-zero likelihood here means the finite bank supplied no supported completions, not that the true future is impossible. No gate was changed after observing results.

At the tested work budget, RB's fixed-base variance inequality did not imply better stability than MC with the same number of dictionary evaluations. This does not refute Rao-Blackwellization, the shared-dictionary surrogate family or the assumed historical channel. It rejects these fixed prior-draw configurations as calibrated guide estimates.

## Checks and cost

Frozen registration `1f09214e6c2ec7ac59ab59902d390584b6ae8108` was pushed and exact remote identity verified before ONErun72092/ONEaudit79954, both exit0. All153frozen paths unchanged, including140inherited production/input paths. All144calls and every recorded bank, likelihood group, zero, control, count and decision exactly replayed. An alternate outgoing forward-DP calculation checked all65280fixed-dictionary joint future likelihoods; maximum log difference2.7284841053187847e−12 versus1e−7tolerance. This is same-author replay with an alternate direction, not an independent expert or complete solver audit.

Run stage1.224360625wall/1.218409CPU seconds, peak sampled host RSS842579968bytes. Audit stage3.059392459wall/3.052173CPU seconds, RSS843497472bytes. Source admission is outside the stage wall timer; absoluteCPU caps include startup. The prospectively estimated20–180seconds/stage was conservative. All600wall/500CPU/2GiB/16MiBcaps held. Zero GPU/new training/paid API/manuscript holdout. This was a guide calibration, not a new native source search or fixed-key Markov reader run.

Before execution, full2690tests+23subtests passed/13skipped in162.08seconds, including15newtests and real tiny144-call transport/full alternate replay/admission-tamper checks. Scoped lint passed; five unrelated old full-tree findings remain. Frozen production was not edited after outcomes. Static publication checks recomputed likelihood means/contribution shares/ESS, inventories, decisions and resource bounds without new guide evaluations.

Original144call records total1416656bytes and contain sample arrays; they remain ignored locally. A lossless33896byte compressed archive preserves every original UTF-8 file and SHA. Compact per-call scalar reports115101bytes and a manifest-rich result127479bytes are tracked, along with the archive descriptor. Original files were not rewritten. Archive SHA4f211c47039fbc6e0a14f27fa4360703691ae8f937d04da46ff8141287f315ac. Result SHA709ba1d93f51e762b7b816b0db5c61bdb5ea6b89b9b4731e647d6176d5b321fc; audit SHA8e50bf437e4fc3f950b731de41b21c30a58fcb8611cb9e2c13dc2d319cada2f3.

## Post-outcome prescribed-path calculation

A particular correct future reading requires d still-unknown used rows to match their specified codes. A uniform prior completion meets those constraints with probability42^(−d). For raw N-draw sampling, the probability of seeing at least one such completion is at mostN/42^d. Exact integration of one required row reduces the base draw's constraint count by one, while costing42evaluations. This describes one prescribed path's contribution; it is **not** a lower bound on the variance of the sum over all readings, nor proof that the true path dominates the IID surrogate.

Correct-prefix states still needed14/13/16/16unknown future-used rows in cases0/1/2/3; roots needed19/18/21/20. Across the96root/prefix arm/seed banks, no base sample represented the prescribed correct continuation, even after accounting for the integrated row. The largest union bound across these configurations is5.310415441299052e−19. This arithmetic explains why random full keys cannot be expected to supply a specified correct continuation at these budgets; it does not quantify every alternative good reading.

[Next-method analysis](../research/after-shared-key-guide-2026-10-01.md) proposes incorporating observations progressively while revising whole dictionary rows, and verifies a subtle condition: intermediate glyph prefixes must allow cuts inside two-glyph units. No sampler or new empirical run is claimed there. Actual Voynich remains unsolved.
