# BLIND-CHANNEL-DEV-003: systematic unknown-unit search

Started 2026-09-27, frozen after validation on 2026-09-28. **Adaptive exposed development**, continuing the two B positive and
two paired-shuffle cases of [DEV-001](BLIND-CHANNEL-DEV-001-results.md). This
registration freezes the next intervention; it does not make reused answers
unexposed. Final authors, stateful family C and Voynich remain untouched.

## Question and rationale

Can a systematic local search over unknown literal units find the good readings
that the earlier generic search missed? DEV-001 demonstrated better admissible
models but did not find them blindly. [DEV-002](BLIND-CHANNEL-DEV-002-results.md)
closed the simple-substitution gap by changing both search coverage and family
restriction. This follow-up extends systematic coverage to variable-length
units; it does not assume their true dictionary, boundaries or letter mappings.

The [method review](../research/blind-channel-next-inference-review.md) separates
optimization, source-model, segmentation and identifiability failures. Primary
papers rechecked before this intervention: [Nuhn et al. 2013](https://aclanthology.org/P13-1154.pdf)
distinguish search and source-model adequacy; [Berg-Kirkpatrick and Klein 2013](https://aclanthology.org/D13-1087.pdf)
show initialization dependence in supplied-unit decipherment; [Chiang et al. 2010](https://aclanthology.org/N10-1068.pdf)
describe normalized finite-state estimation with a specified parameter inventory.
None establishes recovery for our unknown-unit setting. This is a derived,
restricted optimizer, not a replication of their algorithms. Prior Voynich
applications and their unsupported semantic leaps remain reviewed in the
[blind-recovery design](../research/blind-channel-recovery-design.md).

## Fixed inputs, family and objective

Use unchanged `data/manifests/blind_channel_dev001.json`, SHA-256
`8587ae6010de94996bdd57d74ea2e1eee3f71940cdb13c1cc3125adfdfc9acea`, and selected
source SHA-256 `7de1c233b3ea69eb632247062f97db86040666daa13ba73bba28bbb51b3493b0`.
Use only names starting `B-`, in alphabetical order. Each case has four fitting
records and two held-out records; all are already exposed development material.
Fit reads only pinned source and fitting ciphertext/context, never answers,
transfer records, prior fitted models or gold-assisted witnesses.

The source stays first-order Latin, Caesar/Virgil-trained with source-only
selected smoothing. Known source alphabet, six-glyph alphabet, record boundaries,
maximum emission length two and fixed stop probability `1/225` remain assistance.

Search only one-state deterministic channels, one nonempty unit per source
letter. Different letters may share units; there is no injectivity restriction
and no supplied count of two-glyph units. This family contains the positive B
generators but is narrower than DEV-001's two-state, three-alternative grammar.
Every string of length one or two over the declared glyph alphabet is eligible:
six singletons plus 36 pairs, including unobserved pairs. This removes the
earlier observed-substring exclusion without supplying a true dictionary.

Keep the original coding context and actual finite binary description of each
candidate, including its unit lengths/literals. Do not use a cheaper new family
code. Minimize `model_bits - log2 p(fit ciphertext | source, channel)`; sum all
compatible segmentations, not only the best path. Duplicate units retain
different source contexts. Batched scoring changes computation, not the model.
Independently test it with rational enumerations, source zeros, record resets,
mixed lengths, batch independence and rare-context underflow. Recompute risky
candidate/record pairs in log space. Rescore every final selected channel using
the original forward engine and actual code; stop on any disagreement.

## Search and resource bounds

For each case use seed `52101 + 101 * alphabetical_case_index`, 16 restarts,
at most 80 sweeps per restart, batch size 256, 300 seconds cooperative search
deadline, pool cap 256 and improvement tolerance `1e-8` bits. Restart zero uses
the source-start-frequency/ciphertext-frequency singleton initializer. Other
starts randomly assign every glyph singleton to a different source letter,
then independently draw each remaining letter's unit uniformly from all 42
units. This gives starting parse coverage for the positive source; it does not
force coverage or a length count in later candidates. The initializer is
explicitly restricted to source alphabets at least as large as glyph alphabets.

Each sweep considers every nonidentity pair swap and every replacement of one
letter's unit with another eligible unit. Stable ordering and deduplication
remove repeated identical candidates. At most 253 swaps plus `23 * 41 = 943`
replacements occur per sweep. Choose the best actual score. A complete sweep
with no improvement beyond tolerance permits only a local certificate for its
scanned mapping; partial sweeps cannot certify local optimality. Retain the
best completed candidate even when time expires. No beam, annealing, extra
perturbation, oracle warm start or post-result budget extension enters this run.

Run at most two CPU workers, single-thread numerical libraries. Nominal search
budget 20 CPU-minutes across four cases; allow atomic-batch/final replay and
serialization overhead. Each child additionally has a 360-CPU-second limit and
a 420-second parent wall timeout. Track campaign wall time, child process CPU,
RSS, completed starts/sweeps/neighbors and numerical fallbacks. Stop on any
numerical error or failed process; never replace the case. No paid compute or
API use. Compact records tracked; bulk traces/predictions ignored and hashed.
Each compressed trace has a 200MiB cap; exceeding it is a recorded failure.

## Evaluation, comparisons and interpretation

Freeze code, tests, protocol and runner before fitting; push and verify remote.
Freeze all four selected models, full-trace hashes and run metadata before
evaluation; push and verify again. Evaluate each unchanged channel on fit and
transfer, using both the original engine and the independently authored scalar
forward/max-path/distance helper from DEV-001 on all 24 record evaluations.
Check literal predictions, edit counts, likelihoods, model costs and artifacts.
Shuffles have no plaintext answer and no accuracy metric.

Report each positive's transfer character-edit count/rate, decoded length,
exact records, fit/transfer likelihood and full code cost against frozen DEV-001.
The earlier 10% error diagnostic remains a descriptive reference, not a fresh
confirmatory gate. Report the full panel even if only one case improves. Keep
correct-channel and gold-assisted witness scores clearly labeled as already
exposed evaluator diagnostics; do not let them select new configurations.
Null likelihood alone does not certify semantic rejection.

Also report the selected dictionary's number of distinct units and unit-length
counts, and the conditional best-reading surprisal from already audited record
scores: `(log p(y) - log p(best path, y)) / log(2)`. In this one-state
deterministic family each plaintext determines exactly one path, so this is
posterior min-entropy of the plaintext conditional on the selected channel and
source. It is not Shannon entropy, model uncertainty or a correctness guarantee;
a confidently wrong model can have a concentrated conditional posterior. No
threshold or claim is selected using this diagnostic.

This comparison changes candidate family, initialization, unit pool, neighborhood
coverage and budget. It cannot isolate a single cause. A completed local sweep
is not exhaustive global search. Failure with a worse objective than known
admissible witnesses still establishes a search gap; failure after finding a
better-scoring wrong reading motivates objective/ambiguity analysis. Success
would establish only development competence under the stated assistance; fresh
authors/keys and harder families would still be required before manuscript use.

Commands, after the respective verified freezes:

```sh
PYTHONPATH=.:src .venv/bin/python scripts/run_blind_channel_dev003.py campaign --freeze SOURCE_COMMIT
PYTHONPATH=.:src .venv/bin/python scripts/run_blind_channel_dev003.py evaluate --freeze KEY_COMMIT
```
