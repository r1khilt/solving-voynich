# SHARED-KEY-GUIDE-001 — conditional future-guide stability and cost

Prospective exposed-development calibration, 2026-10-01. [Method, derivation, primary-source read depths and limitations](../research/shared-dictionary-guide-2026-10-01.md). Question: is a shared-whole-dictionary IID guide numerically stable enough to justify putting it inside expensive source/key search? No new recovery experiment or manuscript scoring is included.

## Frozen inputs and queries

Four old SOURCE-GUIDE Markov-generated cases, seeds75501..75504; two passages per case,64/64/224/224 source letters per passage.23source rows,6glyphs,42one/two-glyph units with duplicate row mappings, iid uniform dictionary prior, rho1/225. Latin counts19566187bytes SHA9769415ac460c56778a461af7646635abf5906fa726484ba84d98d4f401815f6. Original fixtures45672bytes SHA6372436be35865f6dde6f2fe2bd94b8918936970266ea5ca56523bba8fc0fd9e. Reuse the exact closed SOURCE-PRUNE-DIAG-002 result/audit and four wide-cell archives, with their bound hashes; no new source or gold-path search.

Cases0..3 in order, each root/prefix/altered_binding. Prefix is the previously archived wide first-loss state when there is a loss; for the surviving case0 it is the first archived state with consumed-glyph layer at least10. Keys/offsets are taken from the30-entry archived target state; supplied contexts are recorded but unused by the IID surrogate. Alteration changes the lowest-index assigned row from k to(k+1)%42, leaving offsets and supplied contexts fixed. It need not be a reachable prefix under the actual full model; this is a hypothetical conditional-binding diagnostic. Root has all23rows unknown and offsets0. These twelve queries are selected prospectively before any new guide evaluation, but are explicitly exposed and partly known-answer-conditioned.

## Fixed estimates and controls

Seeds81401..81404. Each seed draws one NumPy uniform42 bank of shape672×23, reused across all twelve queries. Override assigned rows, preserve each completed dictionary throughout both records and every future occurrence. Three arms: MC16 first16draws, MC672 all672draws, RB16 first16base draws with highest-root-probability free row integrated over all42units, lowest-index tie. Each call builds at most672 fixed-dictionary IID tables, max128MiB conservative combined bank/scratch envelope, chunks64; no floor. Same normalized root+epsilon1e-8 source and rho as old renewal guide. The old occurrence-renewal guide is evaluated once per query as a descriptive reference.

Fixed144calls=12queries×4seeds×3arms;65280fixed-dictionary joint future likelihoods. No output-selected seeds/samples/row integration/queries. Tables are logical whole-dictionary evaluations with all record factors, not old per-key cache counts. Record all base group log likelihoods/nulls, positive dictionaries/groups, arithmetic log mean, contribution ESS/max share and elapsed time. MC672/RB16 have matched dictionary table count, not identical CPU; no whole-search cost comparison.

Exploratory calibration is supported PERarm iff all12queries have positive estimates at all4seeds; every query's seed log spread≤2nats; every call's largest base likelihood contribution≤0.5; and correct-prefix conditional estimate exceeds the altered-binding estimate in at least12of16case/seed comparisons. If only the altered estimate is zero and prefix positive, count preference as true; if prefix is zero, count it false. No correctness or rare-event accuracy follows from passing this diagnostic. Partial-binding source masses are conditional and no prefix probabilities are compared. Failure does not refute the surrogate family, assumed language or historical channel.

## Closure, complete audit, resources

Publish all explicit runner PATHS before execution; preserve all inherited140production/input paths. No new compiler, source refit, reader, fixed-key Markov scorer, training, GPU, paid API or manuscript holdout. Save each call exclusively; no empirical retry/resume/extension/retuning. ONErun, ONEcompleteaudit. Audit fully reconstructs all queries/banks/estimates/controls/stats/decisions/resources/hash/version, plus all65280alternate outgoing forward-DP joint likelihoods within1e-7log tolerance; zero support must agree exactly. Same-author replay, not an independent agent/full solver review.

One CPU process, BLAS1. EACHstage600wall/500absoluteCPU/2GiBsampledhost/16MiBcompactoutput; scratch128MiB per guide, bounded672tables per call. Estimate20–180seconds/stage from65ktables across existing short/long glyph observations plus independent forward work; upperstage envelope10minutes. Source admission precedes stage wall timer, CPU cap includes startup; no live balance assumptions. Original source arrays/hash/checkpoints unchanged. Hard stop records failure and closes this ID without a second run/audit.

```
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.:src \
.venv/bin/python -u scripts/run_shared_key_guide001.py --freeze <registration>
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.:src \
.venv/bin/python -u scripts/audit_shared_key_guide001.py
```

Actual Voynich decipherment remains unresolved. Goal remains active.
