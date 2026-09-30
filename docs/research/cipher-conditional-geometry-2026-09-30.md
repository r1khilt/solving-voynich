# Why source prediction and cipher reading can disagree

Status: mathematical analysis and a four-outcome arithmetic illustration, written
before NEURAL-READER-001 target access. No Latin decoding result, new fitted
model, intervention, or qualification claim. This does not change the registered
reader gates or its decoding objective.

## Source basis and scope

[Gneiting and Raftery (2007)](https://sites.stat.washington.edu/people/raftery/Research/PDF/Gneiting2007jasa.pdf)
explain proper probabilistic scoring, including the logarithmic score. Proper
source scoring is useful, but the distribution being scored matters.
[Kambhatla et al. (2018)](https://aclanthology.org/D18-1102/)
use neural language scores in substitution-cipher search. Their known-language
substitution setting does not establish our variable-unit search quality or a
Voynich source language. The project's earlier reader failures and source/length
diagnoses provide the immediate empirical motivation, not evidence for the
theoretical examples below.

The following equations are standard probability identities derived here for
our setting; no novelty claim. Assume a fixed, correct deterministic channel
`y = C(x)`, and positive normalized source probabilities where the true source
has support. A normalized length distribution is part of the source law for
variable-length strings. Unknown keys and uncertain transcription would require
additional latent variables and cannot be silently conditioned away.

## 1. Better source loss need not mean better posterior loss

Write the set of plaintexts producing `y` as `F_y`. Let `P` be the true source
law and `Q` a model, with `Q_Y(y) = sum_{x in F_y} Q(x)`. Then

```
log P(x)/Q(x)
  = log P_Y(y)/Q_Y(y) + log P(x | y)/Q(x | y).

KL(P_X || Q_X)
  = KL(P_Y || Q_Y) + E_{y~P_Y} KL(P(X|y) || Q(X|y)).
```

This follows by substituting `P(x) = P_Y(y) P(x|y)` and the corresponding
factorization of `Q`, then taking expectations. Source cross-entropy can improve
because the model predicts the *observed ciphertext distribution* better while
its ranking of alternatives inside a single `F_y` worsens. The nonnegative
decomposition gives an upper bound on conditional KL, but it does not preserve
the ordering of two imperfect models. Our observed Pliny losses are additionally
finite-sample, checkpoint-selected scores under a different author distribution;
they are not the population quantities in this identity.

A complete counterexample has four outcomes, with channel groups `{a,b}` and
`{c,d}`:

| Law | a | b | c | d | Source KL, nats | Conditional KL, nats | MAP error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| True P | .45 | .05 | .30 | .20 | 0 | 0 | .25 |
| Q_A | .09 | .01 | .54 | .36 | .510825624 | 0 | .25 |
| Q_B | .20 | .30 | .30 | .20 | .275330624 | .275330624 | .65 |

Q_A gets the two group masses wrong but the within-group posteriors exactly
right. Q_B improves total source log loss while reversing the first group's
decision. MAP error rises from 25% to 65%. These are exact finite toy laws, not
simulated cipher results. Direct double-precision summation verified the KL
decomposition to less than 1e-12 nats on 2026-09-30.

## 2. What can be checked without an exact neural normalizer?

For a correct observed record, let `s(x)` be the joint path score including the
fixed length prior. The gold path's negative log posterior is

```
NLL_gold = logsumexp_{x in F_y} s(x) - s(gold).
```

Exact finite-state inference can compute this quantity. A neural beam generally
cannot: its retained candidates may omit substantial probability mass. For any
**distinct** verified subset `A` containing gold,

```
NLL_gold >= logsumexp_{x in A} s(x) - s(gold).
```

Even one wrong candidate gives a useful lower bound `softplus(s(wrong)-s(gold))`.
If that candidate outscores gold by `m>0`, gold's posterior is at most
`1/(1+exp(m))`. This is a bound under the specified model, not a bound on the
probability that the historical reading is true. It requires comparable full
path scores, exact channel support, and deduplication. Beam width cannot turn a
subset-normalized probability into an exact posterior.

The registered reader already compares returned and true-path scores. A higher
gold score certifies a missed better candidate; a higher wrong-path score shows
a source/decision problem that exhaustive MAP search alone cannot remove.
MAP optimizes exact-string 0–1 loss, not edit distance. A more probable wrong
string does not establish optimal edit-risk behavior or absolute ambiguity.

## 3. A more relevant geometry for mechanistic interpretation

Let `s_theta(x)` be differentiable scores for a fixed channel group and write
`w_x = softmax_{F_y}(s_theta(x))`. If `g_x = grad_theta s_theta(x)`, then

```
grad_theta log Q_theta(x | y) = g_x - sum_z w_z g_z
F_conditional(y) = Cov_{x~w}(g_x).
```

Directions that move every compatible reading's score equally disappear.
Ordinary source sensitivity may spend most of its magnitude on exactly those
directions. A decoder-focused intervention should instead measure *relative*
scores among compatible readings and the downstream decision. At the population
level, for a parameter-independent channel and regular differentiable models,
the score's conditional mean is the ciphertext score, giving the total-variance
identity

```
F_X = F_Y + E_{y~Q_Y} F_{X|Y=y}.
```

These expectations use the model law, unlike the earlier true-law KL identity.
They must not be estimated by substituting unrelated teacher-forced corpora
without labeling the change of measure.

For a recurrent-state perturbation at a shared pre-branch history, `theta` can
be that state and the scores can be continuation scores. On a **fixed finite
candidate list** the covariance formula is exact for the list-normalized model,
but not for the full cipher posterior unless that list is exhaustive. Its rank
is at most `number_of_candidates - 1`; low rank alone is therefore not evidence
of a discovered global workspace. Reselecting beam candidates after each
perturbation also changes the measured function and may introduce discontinuity.
These caveats complement [the broader geometry note](behavioral-latent-geometry-2026-09-30.md).

## Consequence for subsequent experiments

First finish the untouched source audits and fresh reader. If it exposes stable
wrong preferences, a subsequent explicitly exploratory study can localize
correct-versus-wrong score margins, then test whether state interventions change
those margins selectively across seeds and matched controls. Any training on
observed errors makes this panel development data. A repaired reader needs a
new allocation before qualification. Improving source loss, finding a sensitive
latent direction, and recovering an unknown historical key remain separate
claims.
