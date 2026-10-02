# Admit the trained stochastic path law before claiming source-informed correction

The [joint-reading MH derivation](joint-reading-independence-2026-10-02.md) is finitely qualified. The [full-size diagnostic](../experiments/SOURCE-ACTION-DIAG-001-results.md) shows first dictionary bindings remain weak despite learned guided reuse. Neither supplies actual trained stochastic candidate coverage, its floating density, or its cost. This memo registers that missing admission, without changing the ongoing training program.

## Method review and boundaries

The primary decipherment/MH/Voynich review and reading depths are in the linked joint derivation: selected Chiang2010 §3.2/5, Neklyudov2020 §2/3.1, Reddy–Knight2011 §2.2/2.3/8–9 and Hauer–Kondrak2016 introduction/§5.3–5.4. Our restrictive shared one/two-unit synthetic family and known source alphabet remain distinct from the manuscript.

[Wingate, Stuhlmüller and Goodman2011 §2/Algorithms2–3](https://proceedings.mlr.press/v15/wingate11a/wingate11a.pdf), selected sections read, develops trace-space MH by naming/replaying random choices and accounting for forward/reverse probabilities. This motivates preserving the complete literal action trace and its law rather than assigning a path's probability to a whole dictionary. It does not prove our neural cache, coverage or speed.

[Liang et al.2020/2021](https://arxiv.org/abs/2010.12128), **abstract only**, uses learned proposals for MH in a declarative graphical-model system. Its stated logistic-regression/n-schools results are not decipherment evidence or a validation of this architecture. Learned corrected inference is existing methodology; novelty is not claimed here.

Existing repository components reviewed: original `propose_cached`, full causal reference packing, fixed-geometry memory qualification, trained0/1000 input diagnostic, original compact order12 source/dense adapter and known-key source-point checks in TEMPERED-RECOVERY-DIAG-001. Reuse their laws and identities; no new language model, source prior, length normalization or success refill.

## Fixed workload and law

Use binding92403 checkpoint4000, all64 fixed EXPOSED Pliny development observations and their existing33 controls. One stochastic path per observation/control, temperature1, seeds92501+case-index, all97 attempts retained. No Gold/source score enters proposal features or action selection. No sampled path is repaired, refilled or used to adapt another seed. Failure has zero terminal target mass.

An observation-only sampler selects source row and one/two-glyph emission at each action, constructing used bindings coherently. Its density is the product of actually normalized legal-action probabilities; variable action lengths do not introduce Gold-length features. The fixed total observed-glyph horizon bounds paths. Unused rows remain unbound; duplicated emission units remain allowed.

Record the original cached sampler's actual legal logits with a local read-only instrument. Rebuild literal states/masks and the PCG random choices from those logits/seed, check the reported path log probability within1e−9, and detect any visited probability underflow. Then score the entire SAME sampled prefix using the independent full causal decoder: every legal logit delta≤.002 and whole prefix log-density delta≤.01. Dead prefixes are legitimate reference-prefix workloads, never successful teaching paths. This admits visited CPUfloat32 queries, not every possible floating state.

Separately score every Gold path with the CPUfloat32 reference, match all64 saved MPS path probabilities≤.01, and report their original source point scores. Gold is restricted to reference/recovery diagnostics; it is never an initializer or proposal repair. This is an exposed development investigation, not fresh confirmation or an unknown-author transfer test.

## Original source point law

Rebuild the immutable original compact source with the same selected smoothing and check counts, provenance/selection audit, alphabet and both dense-array digests against TEMPERED-RECOVERY-DIAG-001. For each whole reading X, reset source history per record and retain geometric EOS rho=1/225:

    log Q(X) = sum_records [log rho + len(record) log(1-rho)
                           + sum_letters log Q(letter | reset preceding history)].

Compare direct dense transitions/probabilities with lazy sparse probability rows indexed by independently rebuilt string histories; total log-score difference≤1e−9. Repeat only for whole completed candidates/Gold records. No full-key likelihood lattice, source training, posterior key bank or native search.

For complete stochastic paths report logQ−visited_rows log42−actual_path_logq. This is an unnormalized joint reading importance weight; it is not a dictionary marginal, posterior probability, confidence interval or decipherment score. The fixed canonical observation's orbit factor is common to candidates and is omitted. **No evidence estimator or MH chain is run in this admission.** Failures carry explicit zero target weight and remain in every denominator.

## Decisions and resources

The engineering PASS requires all97 literal/seed/mask/density/source controls and all64 Gold reference checks to complete within1hwall/5000absoluteCPU seconds/4GiBpeakRSS/64MiBowned private outputs, CPUthreads2/$0. Initial timing projection is several-to-tens of minutes, based on the earlier full-size CPU diagnostic; sequential sampling time is unknown and will be measured, not presented as a speedup. Any mismatch/bound stops this namespace and preserves partial receipts without retry. Existing GPU training remains unchanged.

The exploratory success/failure/recovery/weight/cost results have no retrospective positive gate. Even a numerical PASS with zero useful completions is a negative candidate-coverage result. One exposed model/one draw percase cannot certify support, posterior mass, unseen tails, effective sample size, mixing or historical identifiability. A separately registered source-informed correction/revision campaign and comparison with existing search would still be necessary.
