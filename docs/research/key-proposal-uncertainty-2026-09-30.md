# Proper joint key learning, supervision masks and symbol symmetry

2026-09-30. Prospective mathematical refinement of the
[learned proposal design](learned-global-key-proposals-2026-09-30.md).
The examples/proofs below are project derivations, checked with exact rational
arithmetic; they are not empirical neural results or a historical decipherment.
No new network/training/cipher-panel inference is implemented here.

## A supervision mask can change the posterior being learned

Let X be latent plaintext, Y its observed cipher, K the random dictionary, and
M_a(X) indicate whether source letter a occurs in the training plaintext.
Ordinary population negative log likelihood for q(K|Y) is a proper objective:
with adequate capacity its optimum is the generating conditional distribution.
An autoregressive whole-key model can express that distribution without
assuming independent row posteriors.

The earlier design suggestion to mask unused source-letter row targets needs
a qualification. For a row predictor trained with

```text
E[ M_a(X) * -log q_a(K_a | Y) ],
```

the optimum at an input y with positive mask probability is

```text
q_a*(k | y) = P(K_a=k | Y=y, M_a=1),
```

not generally P(K_a=k|Y=y). The mask depends on latent plaintext and can select
a different conditional distribution. An unused target being difficult does
not by itself justify deleting its loss and claiming calibrated key beliefs.

Exact counterexample: X is one letter, P(a)=1/4 and P(b)=3/4. Independently draw
each key row uniformly from singleton units {x,y}. Given observed Y=x, the
posterior P(K_a=x|Y=x)=5/8. If the loss for row a is computed only when X=a,
its optimum instead assigns probability1 to K_a=x. The true mixture is
(1/4)*1 + (3/4)*(1/2)=5/8. Enumeration of all four dictionaries/two possible
source letters reproduces these rational values exactly.

Primary future training should therefore keep the declared complete joint-key
loss, or explicitly learn presence and both conditional distributions and
combine them with their inferred mixture weights. Predicting only the used-row
conditional is a possible proposal heuristic, but must be named and compared
as such. For iid row priors/independent plaintext, an actually unused row has
its prior conditional on that event; injective or fixed-unit-count generation
couples rows and invalidates that shortcut. A gold plaintext presence mask is
training supervision; it is unavailable at inference.

Constant positive row weights do not have this input-dependent selection issue
for separate marginal objectives, though the shared finite network still has
capacity/optimization tradeoffs. A calibrated marginal row predictor alone
does not constitute a calibrated whole-key distribution.

## Correct marginal rows can combine into poor whole keys

Take source strings ab/ba with probability1/2 each, two letters, singleton
units x/y, and four iid equally likely legal dictionaries. Observing xy leaves
only keys (x,y) and (y,x), each posterior1/2. Both row marginals are uniform.
Independent sampling from those perfectly correct marginals nevertheless
spends half its probability on (x,x)/(y,y), which cannot emit the observation.
All four keys are legal in the duplicate-allowing family; this is a posterior
correlation issue, not a justification for silently imposing injectivity.
Exact enumeration reproduces the two supported keys and1/2 wasted product mass.

A row-query network followed by independent categorical samples need not solve
this problem merely because the queries exchanged information. Their features
can be correlated while the conditional output law remains factorized. Use a
declared autoregressive/refinement latent joint distribution or an explicitly
scored beam of complete assignments; verify every whole key with the unchanged
native fitting objective. The positive toy identifies a representational
limitation, not the unique cause of current large-case search failures.

## Absence of a cipher glyph differs from absence of a source letter

Observed-glyph first-occurrence canonicalization is exact under consistent
renaming of the observed symbols. Declared glyphs never present in Y cannot be
ordered by first occurrence. Let G permute these unobserved glyphs while fixing
all observed ones. If the channel and prior are invariant under G, then
P(Y|gK)=P(Y|K). More parameters cannot identify names that the observation and
prior leave exchangeable. This is a conditional invariance claim; no current
catastrophic case is diagnosed as having this ambiguity.

An exact equivariant proposal law can average an arbitrary canonical proposal:

```text
q_sym(K | Y) = (1/|G|) * sum_{g in G} q(g^-1 K | canonical(Y)).
```

Reindexing that finite group proves normalization and equivariance. Sampling a
canonical key and then an independent uniform residual permutation realizes
this law. That is distributional equivariance; one fixed random seed need not
produce pointwise equivariant samples. Set-valued orbit outputs can instead
provide exact pointwise set equivariance. A deterministic named key need not
exist as an equivariant choice when its stabilizer forbids one.

Permutation multiplicities and overlapping orbits must be accounted if claiming
proposal probabilities. For a single uniform group action, each distinct orbit
element has the same stabilizer multiplicity; independent bases can contribute
additional mass to the same key. These proposal masses are separate from the
reader's fit-only weights. The existing retained reader bank deduplicates keys
and recomputes their literal objective once; repeated proposals must not create
an additional prior. Enumerating residual orbits adds real native scoring cost
and belongs within the same key-score budget.

## Consequences for a larger conditional model and causal analysis

Primary synthetic training should explicitly draw iid duplicate-allowing keys
from the complete legal unit pool. Training only on bijective keys with fixed
singleton/digram counts would supply an informative generator prior even if
the output head has no hard bijection constraint. Such an assisted-prior arm is
legitimate only when separately declared. Original B-family exposed tests and
an iid-key training distribution then have a declared distribution mismatch;
future unused qualification should include both and preserve all null controls.

Large-model comparisons must separate capacity from these supervision/prior
choices, use the measured local throughput and report complete training and
proposal/search costs. A shared-key autoregressive query decoder is a plausible
implementation of joint inference, not evidence that a global workspace or
language concept has emerged.

Mechanistic analysis must also separate intentionally labeled source-letter
queries from discovered features. A row-label probe can read supplied identity
without showing causal use. Test interventions in the network's memories with
matched norm/history/incorrect-donor controls, downstream complete-key changes,
literal behavior, query-order/rename controls and two independent fits. Decoder
behavior caused by manually changing an already emitted dictionary tests the
explicit decoder, not a causal mechanism inside the proposal network.
