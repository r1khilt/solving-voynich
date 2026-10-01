# Shared-key source particles: probability law before a larger search

2026-10-01. Theory and finite rational checks after the source-prefix systems
result. This is a proposed inference method, not an implemented full-size
particle solver, registered recovery experiment or decipherment claim.

## Why change the search

SOURCE-PREFIX-SYSTEMS-001 found no completed candidate for either varied-source
fixture under either priority. Even the progress heuristic with zero queue
evictions failed to finish within 5,000 expansions. Those measurements do not
prove a particular replacement will help. They justify testing a method that
maintains multiple possibilities while applying shared-key constraints from
both records early, rather than relying on one depth preference.

The underlying source/key law should remain fixed when comparing inference.
Otherwise a faster method could simply be solving a different problem.

## Primary-source review and differences

[Doucet and Johansen, tutorial](https://www.stats.ox.ac.uk/~doucet/doucet_johansen_tutorialPF2011.pdf),
§§3.4–3.6 and4.2, develops importance weighting, unbiased offspring counts and
locally adapted proposals. The same tutorial discusses loss of trajectory
diversity. Our variable-length shared-dictionary state is a project adaptation,
not one of its demonstrated decipherment applications.

[Loula et al., ICLR2025](https://proceedings.iclr.cc/paper_files/paper/2025/file/a2d537e69a6c6638a3630eef835f07de-Paper-Conference.pdf),
§2 andAppendixB, distinguishes local constraint normalization from a global
conditioned distribution, retains completed variable-length sequences and
corrects local proposals through importance weights. Its language-generation
benchmarks do not establish historical decipherment or performance on our
23-row duplicate-allowing emission family.

[Hauer, Hayward and Kondrak2014](https://aclanthology.org/C14-1218.pdf),
§§4–5, applies corpus-guided key changes and MCTS to substitution decipherment.
That is a useful search alternative, but MCTS node values are not automatically
posterior probabilities or unbiased evidence estimates. Its monoalphabetic
pattern-equivalence operations cannot be imported unchanged into our variable
one/two-glyph, duplicate-allowing channel.

## A deterministic action schedule preserves the target

Let state s contain source prefixes x_r, ciphertext offsets o_r, closed-record
flags and one partial shared dictionary k. A source row emits one or two glyphs.
Choose exactly one open record to extend, by a deterministic function of s and
the observed ciphertext. Two candidates are first-open-record and least
fraction of ciphertext consumed, with record index breaking ties. The latter
needs ciphertext lengths, which are observed, not hidden source lengths.

Do not branch on which record to extend. Doing that without correction counts
the same reading/key assignment once per possible interleaving.

For the selected record, define the nonnegative action coefficient a(s,u):

- Assigned source row j, with its unit matching the next ciphertext substring:
  `(1-rho) p(j | x_r)`.
- Unassigned row j, bound to the next one/two-glyph substring:
  `(1-rho) p(j | x_r) / (g+g²)`. These are distinct unit choices.
- Exhausted, open record: one EOS action with coefficient `rho`.
- All records closed: one absorbing action with coefficient1.

Incompatible actions have coefficient0. Each branch binds a row once and
retains its value across all records. Per-record source histories remain in
their own order and reset independently. For any completed source/used-key
leaf, the product of action coefficients is

`[product_r p_source(x_r) rho (1-rho)^len(x_r)] / (g+g²)^M`,

where M is the number of observed source rows. Unused row priors integrate to1.
Every positive completed leaf has exactly one path under either deterministic
schedule. Thus scheduling changes exploration, not the reading/key probability
law. Different used dictionaries still need marginalization per reading.

No valid path needs more than `T = sum_r len(cipher_r) + number_of_records`
actions: each letter consumes at least one glyph, and each record has one EOS.
Absorb completed paths with coefficient1 until T. Do not repeatedly charge
EOS, discard early completions or condition on one observed generation length.
At T, every surviving path is complete; failed states have zero target mass.

## Local proposals need their normalization correction

Let `G(s) = sum_u a(s,u)` and propose `q(u|s) = a(s,u)/G(s)` when G>0.
For one proposed path, its importance weight relative to the unnormalized
target is `W = product_t G(s_t)`. EOS contributes rho and absorption contributes1.
When G=0, retain an explicit extinct outcome with weight0. Silently replacing
it with a new lucky guess changes the proposal unless its law is accounted for.

This is an immediate-constraint proposal. It does not include the unknown
probability that later ciphertext remains satisfiable. Treating its completed
outputs as exact posterior samples would be wrong.

Fixed exact counterexample: root choices A/B both have coefficient1/2; their
only final actions have coefficients1/4 and3/4 respectively. Leaf target masses
are1/8 and3/8, so evidence is1/2 and posterior P(A)=1/4. The uncorrected local
proposal chooses A with probability1/2. Importance weights1/4 and3/4 restore
the correct unnormalized measure in expectation.

## Population resampling and what its estimates mean

A fully adapted population step can compute G for all N equally weighted
parents, multiply the evidence estimate by their mean G, draw parent indices
proportional to G using an unbiased resampling scheme, and draw a child from
that parent's q. Starting with evidence estimate1, induction through this
Feynman–Kac recursion gives

`E[Z_hat_T * (1/N) sum_i f(leaf_i)] = sum_leaf target_mass(leaf) f(leaf)`.

In particular, `E[Z_hat_T] = P(cipher)` in exact arithmetic under the declared
law. This argument requires correct coefficients, supported proposals,
unbiased offspring counts, fixed horizon and no uncorrected refill/selection.
It is not a claim about an unimplemented production solver's numerical error.

The normalized finite-population posterior estimate is **not** generally
unbiased. In the fixed counterexample, N=2 first-stage A counts are binomial
with probabilities1/4,1/2,1/4. The expected final A fraction is3/8, not1/4;
yet expected evidence stays1/2 and expected evidence times A fraction is1/8.
Also, log(Z_hat) is not an unbiased estimate of log evidence. A single large
population, narrow final weights or many duplicate particles do not prove
coverage, convergence or recovery.

## Exact checks completed

`scripts/check_source_particle_theory001.py` enumerates all one-particle local
proposal outcomes using rational arithmetic, including extinction. On all36
tiny two-record ciphertext pairs, both deterministic schedules agree exactly
with the separate full-dictionary/source-string enumerator on reading support
and every unnormalized reading mass. Their used-key leaf counts also agree
with each other. Proposal mass including failure is exactly1. Across72 schedule
checks:2,016 enumerated nodes and464
leaves. The fixed two-particle counterexample is checked algebraically.

These are finite mathematical fixtures, with no random sampling, trained-model
prediction, empirical cipher panel, GPU work or recovery claim. They do not
establish finite-population adequacy on large cases. Compact source-bound
results are in `results/SOURCE-PARTICLE-THEORY-001/check.json`.

## Guidance and architecture without changing the meaning of a score

A positive future-completion function h_t(s) can guide proposals, provided its
ratios remain in the importance correction. With final h_T=1, use
`q_h(u|s) proportional to a(s,u) h_(t+1)(child)` and increment
`G_h(s) = sum_u a(s,u) h_(t+1)(child) / h_t(s)`.
Initialize the evidence estimate to known h_0(root). The ratios telescope back
to the original leaf target. A heuristic h can help or hurt variance. Setting
h=0 for a branch is valid only if its completion mass is actually zero;
uncertain predictors need positive support. The exact completion function
would give exact conditioned actions, but computing it is the original hard
inference problem, not free supervision.

An eventual action/value model could see both ciphertexts, both source-prefix
states and partial assignments, then propose where to spend effort. Such a
model's proposal confidence is separate from the Latin prior. Current95M
whole-key training neither supplies this completion function nor demonstrates
good recovery. Mechanistic interpretation would need qualified behavior first,
then interventions on shared-key conflicts, second-record evidence and history,
with matched controls. No new architecture or training is queued here.

## Implementation envelope and prospective checks

The existing statistical source has a sufficient retained-context state, so
particles can carry integer source states rather than copying complete prefix
tuples. Output reconstruction still needs ancestry. For N=4096,T=900, a planned
8-byte ancestry/action entry per particle per step uses29,491,200bytes; source
probability/transition tables use399,571,824bytes. This arithmetic excludes
temporary arrays, retained counts and interpreter overhead; it is a proposed
layout estimate, not measured total memory or throughput.

A future systems qualification must compare scalar and vectorized coefficients,
EOS/absorbing behavior, reset state and key binding; verify literal full-leaf
scores and ancestry; check tiny posterior/evidence behavior across independent
seeds and keep every extinction. Only afterward should a frozen recovery study
compare schedules/budgets on unused controls, reporting exact text, edit error,
used-key recovery, null acceptance, restart stability and genealogical diversity.
Matched-source simulations and real held-out Latin answer different questions.
No large particle run, budget extension or new holdout has been launched.

Better posterior approximation under a wrong historical alphabet, source or
channel can still produce a wrong reading. These improvements address inference
adequacy; they cannot establish that Voynich is generated by this family.
