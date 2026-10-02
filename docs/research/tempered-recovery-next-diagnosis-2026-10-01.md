# Separate search coverage, inventory and reading errors

Planning during the single frozen TEMPERED-RECOVERY-001 campaign; no additional source calls, oracle scores or new fitting are performed here. The campaign's predictions must all close before ground-truth inspection, and its one full audit must remain unchanged. This note introduces no replacement seeds, revised success criteria or post hoc success label.

The previous [source-revision analysis](source-key-revision-2026-10-01.md), [dictionary permutation review](dictionary-permutation-orbits-2026-10-01.md) and [joint auxiliary-path qualification](auxiliary-key-swap-2026-10-01.md) separate source-model preference from search failure and inventory changes from label changes. The earlier artificial failures do not establish the cause on these four fresh keys. The next known-answer diagnostic should use all four positive generation units, every predefined seed and arm, and every final particle from every complete positive call. Failed calls stay explicit. It is exploratory on an exposed generator family, not fresh confirmation or historical decipherment.

## Inventory repair has a measurable ceiling

Let c_u count actually used source rows whose generating code is u, and m_u count occurrences of u in a final full dictionary. Any permutation of dictionary rows can match at most sum_u min(c_u,m_u) used rows. The deficit sum_u max(0,c_u-m_u) is the minimum number of code replacements necessary before some permutation could match the used mapping, ignoring intermediate likelihood and support constraints. This previously derived oracle ceiling must be evaluated across all particles, retaining duplicates for empirical frequencies but using distinct dictionaries for support-mass calculations.

If no final particle has a compatible inventory, the newly qualified joint label-swap move cannot recover the true used mapping starting from that bank without a separate inventory-changing move. If inventories are compatible but labels are wrong, that supports testing label exploration; it does not establish that swaps mix well or that the correct labels maximize the model objective. Inventory compatibility is necessary, not sufficient. Source-label relabeling preserves the multiset of codes even when accompanied by a posterior source path.

## A known reading supplies a model-evidence lower bound

Own extension to the earlier [source-particle gap diagnostic](../../scripts/source_particle_gap_diag.py), now for full-dictionary posterior support rather than a bank of source/used-key joint leaves. Let R=23, U=42, pi(K)=U^-R, Q(X) be the complete original-source probability including geometric EOS and a reset for each record, and L_K=sum_X Q(X) I_K(X,C). The intended terminal key posterior is P(K|C)=pi(K)L_K/Z, where Z=sum_K pi(K)L_K. Keys unable to encode all observations have zero L_K; restricting initialization to full support does not change this terminal posterior.

For one known compatible reading X* and its M used source rows, fix those M codes to their generating values. Each of the U^(R-M) completions of unused rows still contributes at least Q(X*) to its marginal likelihood. These are distinct full dictionaries. Consequently

    Z >= Q(X*) / U^M.

This bound requires one verified used mapping, not uniqueness of the mapping or that the model prefers X*. Additional compatible mappings can increase Z; no such unknown multiplicity is assumed. Fixed-length generation differs from the inference model's geometric EOS; score X* under the unchanged inference Q, rather than its generating design probability. A separately measured full generating-key marginal also gives Z >= L_K* / U^R. The maximum of these two lower bounds is valid; adding them would double-count overlapping events and is not justified.

For the set S of distinct full dictionaries in a final bank,

    P(S|C) <= min(1, sum_{K in S} L_K / [Q(X*) U^(R-M)]).

Use each distinct K once in this numerator. Multiplicities define the empirical particle law but do not create new posterior events. All L_K must be complete, unchanged whole-record scores under the same source, channel, EOS and reset rule. No partial, best-path, length-normalized or capped scores can substitute. Calculate the sum in log space; preserve a very negative log bound even when exponentiation underflows. This is a mathematical inequality with floating implementation checks, not an interval-arithmetic certificate.

Any approximate law supported entirely on S has total variation from P at least 1-P(S). Its forward KL to P is at least -log P(S), by conditioning P on S and the nonnegativity of KL. These statements require no independent-particle or mixing assumption. They concern the declared full-key model. They do not bound reading-marginal total variation: different keys can imply the same reading, and full dictionaries differ on unused rows. They do not prove the generating key is globally optimal, that the true reading has high posterior mass, or that a historical manuscript follows this channel.

An exploratory exact-Fraction sanity check used the existing artificial Source, KEYS, RECORDS and literal_record_law in [the finite interface checker](../../scripts/check_posterior_key_reading001.py). Across all450 two-source/two-record panels and2642 distinct known-reading events, enumerating all36 dictionaries verified each unused-row completion count and evidence lower bound. For each event, singleton-first, singleton-last and alternating-key banks gave7926 exact posterior-mass/TV inequalities with no failures. This checks the algebra on finite cases; it is not an actual source/Gold diagnostic, a resource qualification, or an empirical posterior-coverage result. No claimed KL numerical test or historical conclusion follows.

## Next bounded empirical registration

After the original run and audit close, register a separate diagnostic with their immutable result/audit/seal and archive identities. Reuse stored complete final scores to measure distinct-key bank mass bounds and inventory ceilings. Point-score all four known source tuples once under the original source with independent scalar checks; no fitting or selective case choice. If a full generating-key marginal and oracle conditional reading are added, declare their separate fixed source/reader/reference caps before use and preserve failures. Audit every transformation, literal re-encoding, score identity, count and inequality on finite positive and null controls.

First decide whether search coverage, code inventory or conditional reading is the measured bottleneck. Then separately register an actual dense-source posterior-path cost probe and a terminal coupled-label comparison if justified. A cheap invariant kernel is not a competent inverse solver; a larger network is not evidence of recovered rules. No new training, GPU, paid call, historical holdout or additional original-source diagnostic is launched by this note.

**Scheduling amendment before diagnostic use.** [TEMPERED-RECOVERY-DIAG-001](../experiments/TEMPERED-RECOVERY-DIAG-001.md) now permits bounded independent diagnosis after the original fit and64-prediction seal, while its unchanged full audit runs. Original run/audit failure blocks admission; final empirical publication still requires both audits. This changes the earlier unlaunched scheduling proposal, not the original recovery design, inputs, limits, outcomes or thresholds. The separate design fixes allfour oracles/all32banks, exact likelihood/reference/reader caps and600wall/500absoluteCPU/2GiB/8MiB limits. No diagnostic has run at this amendment's preparation checkpoint.
