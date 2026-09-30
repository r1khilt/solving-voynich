# Global search is now the primary fitting bottleneck

2026-09-30. Method review and prospective derivations after the fresh failure
and fixed DIAG-A. No new optimizer, annealing campaign or relaxed-channel
inference is implemented or run in this note.

## Evidence that changes the priority

The [fresh failure](../experiments/BLIND-CHANNEL-CONFIRM-002-results.md) has two
broadly wrong dictionary banks, responsible for723/789errors. The
[all-case diagnosis](../experiments/BLIND-CHANNEL-CONFIRM-002-DIAG-A-results.md)
shows the generating keys beat the selected fits under every used source.
Sixteen greedy restarts fail on each; every endpoint is a local optimum under
the tested low-order moves. Rescoring those endpoints finds no good dictionary.
This is a direct witness of missed solutions, not a global optimum certificate.
For the other14cases, smaller source/decision errors remain. Changing the reader
is still a separate research branch, with lower immediate impact on this panel.

## Primary work revisited

[Berg-Kirkpatrick and Klein2013](https://aclanthology.org/D13-1087.pdf), sections
3.2–3.3 and figures1–3, find rare useful basins in large restart populations.
They also show that small objective gains near high-scoring solutions need not
track accuracy. Their HMM/EM homophonic substitution setting differs from our
deterministic variable-length units and fixed statistical source. The result
supports treating restart coverage as a real problem; it does not authorize a
million-run campaign or make likelihood a truth oracle. The paper's historical
statement that Zodiac340 was unsolved is a2013 statement, not a current claim.

[Corlett and Penn2010](https://aclanthology.org/P10-1106.pdf), section3, relax
unassigned substitution mappings to bound candidate scores in an A* search.
Their character-level bijection, Viterbi objective and known-space assistance
differ from our unknown emission segmentation and path-marginal fit objective.
Their exact-search guarantee and reported runtime cannot be transferred to our
family by renaming variables. A Viterbi upper bound is not automatically a
bound on the sum of all segmentation paths.

[Nuhn et al.2013](https://aclanthology.org/P13-1154.pdf) and
[Kambhatla et al.2018](https://aclanthology.org/D18-1102.pdf), reviewed in the
existing qualification and neural-prefix notes, offer partial-key beam and
neural rest-cost precedents. Heuristic beam ordering needs explicit pruning
records and cannot be called exact. The prior Voynich applications and scope
review remain [here](blind-channel-fresh-qualification-review.md). None of this
establishes Voynich's alphabet, plaintext language or memoryless coding family.
A newer coupled-annealing article was found in a search, but its primary text
was blocked by a browser challenge; no result or design choice here relies on it.

## Candidate1: traverse basins under the strong objective

Keep H(K)=fit_log_likelihood(K)-literal_code_bits(K)*log(2) fixed. A symmetric
replacement/swap/block proposal at inverse temperature beta can use

```text
log_acceptance = min(0, beta * [H(K_new)-H(K_old)]).
```

Unlike greedy descent, it sometimes accepts a worse fit. For nonsymmetric
proposals, the reverse/forward proposal ratio is required; ignoring it changes
the target. Unsupported keys have H=-infinity and require explicit handling,
including the undefined infinity-minus-infinity case. A random perturbation
followed by greedy refinement is a different algorithm, not an MCMC sampler
unless its transition probabilities are accounted for.

For two replicas with scores H_i,H_j and inverse temperatures beta_i,beta_j,
a symmetric state exchange has

```text
log_acceptance_swap = min(0, (beta_i-beta_j)*(H_j-H_i)).
```

These are elementary Metropolis ratio derivations, not empirical results or a
claim of fast mixing. Keep the best fully scored dictionary independently of
the chain's final state. Record acceptance, distinct states, swaps, repeated
basins and exact work counts. Multiple temperatures, nonlocal row cycles and
restarts have different compute costs; compare them at equal scoring budgets.
No chain convergence or optimum claim follows from a fixed number of moves.

## Candidate2: a valid partial-dictionary marginal bound

Let sigma assign some source letters to fixed legal glyph units. For unassigned
letters, permit every legal nonempty unit independently at each occurrence,
with coefficient1 rather than a normalized emission probability. Assigned
letters retain their fixed units. Use the exact same normalized causal source
and geometric length law. Call the resulting finite observed-record path sum
Z_relax_sigma(y). It is an optimization relaxation, **not a generative channel**;
its rows need not sum to1 and its total weight can exceed1.

Every complete dictionary K extending sigma contributes a subset of those
relaxed paths, with identical nonnegative path weights. Consequently

```text
Z_K(y) <= Z_relax_sigma(y).
U(sigma) = sum_records log Z_relax_sigma(y)
           - minimum_completion_code_bits(sigma)*log(2)
```

is an upper bound on H(K) for every completion. Unknown unit lengths can use
their minimum legal length in the description-cost lower bound. A fully
assigned sigma recovers the exact ordinary marginal and actual code cost.
The argument sums all legal segmentation paths; replacing that sum by a best
path destroys this particular bound. The relaxation allows assignments to
vary across occurrences and records, but only in the bound; a returned
dictionary must remain globally consistent across every observation.

The relaxation may be disastrously loose. Unassigned letters can explain almost
any local glyph sequence, and a high-order source may create enormous state
graphs. A* then remains impractical despite a correct theorem. Low-order scores
cannot silently substitute for a high-order bound. Node/queue/work caps must
report a failure or remaining bound. Keep the derivation separate from claims
of practical performance or numerical interval certification.

## Required next program, still prospective

First validate transition ratios and partial-key bounds by complete enumeration
on tiny controlled models, including deliberate local traps, unsupported states,
multiple segmentations, duplicate unit rows and model-code costs. Measure strong
native scoring and relaxed-graph costs on fixed artificial workloads. Freeze
bounded all-case development comparisons against the existing pipeline; retain
the32positive/shuffle cases and full recovery denominators. No answer-guided
proposal or seed selection. Test actual readings as well as objective gains.
Only after that should another unused-key/passage qualification be constructed.

Restricting search to the generating B family's injective rows and fixed6/17
unit-length counts would provide additional side information. If tested, label
it as a separate assisted family and retain the original unrestricted control.
Do not assume those restrictions for Borg or Voynich.

For mechanistic work, keep the already qualified neural sources and study the
wrong *available* candidate preferences with matched context/state interventions.
Such interventions can explain source errors; they cannot repair absent bank
support without a new key-search mechanism. A latent plot, neuron response or
next-letter readout rank is not evidence of a deciphering algorithm.
