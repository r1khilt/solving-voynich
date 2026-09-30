# Delayed acceptance: optional cost filter, not a better decipherment objective

2026-09-30. Prospective mathematics/method review only. No implementation or
measured surrogate cost/benefit. Keep GLOBAL-KEY-SEARCH-001's direct exact target
unchanged; do not retrofit this filter during that run.

[Sherlock, Thiery and Golightly](https://arxiv.org/pdf/1506.08155), section2.1,
describe a cheap first acceptance test followed by an exact correction. Their
later efficiency/tuning results concern continuous random-walk high-dimensional
regimes and specified surrogate errors. They do not supply an optimal acceptance
rate, mixing time or decipherment guarantee for23discrete variable-unit rows.
[Sherlock and Lee](https://arxiv.org/pdf/1706.02142), sections1–2.1, distinguish
invariant target from efficiency, showing that delayed acceptance can reduce
acceptance/movement while saving expensive evaluations. Their infinite-state
variance/ergodicity counterexamples should not be copied onto this finite family.
The existing Voynich/cryptographic scope and dictionary search precedents remain
in [the global method review](global-dictionary-search-after-confirm002.md).

## Independent derivation for our finite dictionary target

For a symmetric proposal q, let H be the unchanged exact high-order fitting log
marginal minus literal description cost, h a fixed cheap order1or3 surrogate,
and beta=1/T. Use independent uniform draws at two stages:

```text
r1 = beta * (h(new)-h(old))
r2 = beta * ((H(new)-H(old))-(h(new)-h(old)))
alpha_DA = min(1,exp(r1)) * min(1,exp(r2)).
```

Evaluate H(new) only when stage1passes. At either rejection keep the old state
and its already exact H. Since min(1,exp(r))/min(1,exp(-r))=exp(r), the forward
versus reverse acceptance ratio is exp(r1+r2)=exp(beta*deltaH). With symmetric q
this establishes detailed balance for the **strong** target. The full move's
acceptance probability is at most ordinary Metropolis's min(1,exp(beta*deltaH)).
A cheaper step is therefore not automatically more useful exploration per second.
For nonsymmetric q, its reverse/forward ratio must be included in stage1; it
cancels from stage2 after correcting the actual accepted surrogate proposal.
Greedy optimization after a random perturbation is not automatically symmetric.

Keep strong scores for every replica; adjacent exchanges use the original strong
H ratio, never h. Changing or fitting the surrogate during a run needs an
adaptive-kernel argument/registration. Keep source/model-code terms explicit;
calling a low-order fit an upper bound on high-order fit is invalid.

## Support and what can fail

With strictly positive normalized source rows and legal nonempty units, support
of any fixed observed record is determined by existence of an emission path,
independently of source order. The original sources thus have the same support
for fixed dictionaries, observations and length law. Unsupported candidates can
be rejected before exact scoring, but a cheap arithmetic/resource failure cannot
become zero probability. Start with finite h and H and retain both scores.
A thresholded surrogate or a pruned graph can falsely remove positive strong
states and break the argument. If a different surrogate has such zeros, use a
well-defined positive floor/mixture and account for its actual density instead.

Every current family contains finitely many keys (42^23 before support tests).
On a finite irreducible aperiodic positive-support component, asymptotic variance
is finite and convergence is geometric. Those facts say nothing useful about
the enormous constants or whether registered local/block proposals connect all
supported keys. Do not claim an infinite-variance pathology from the continuous
paper, or global irreducibility without proving this specific support graph.
An all-row uniform proposal would give every supported key positive direct move
probability, but that formal property alone says nothing about practical success.

An elementary two-state example exposes the efficiency tradeoff: strong weights
(.1,.9), cheap weights(.9,.1), symmetric flip proposal. Ordinary acceptance is
1 in the first direction and1/9back. Delayed acceptance becomes1/9and1/81;
both have the same strong stationary weights but substantially different motion.
Stationarity therefore cannot be used as a runtime or discovery argument.

## Decision and a bounded future comparison

The artificial large-source benchmark found exact four-record fitting affordable,
but did not measure a cheap-source comparison or real Latin/hot-key costs.
First run the already frozen direct search. If exact costs dominate, separately
register equal-wall/equal-exact-call comparisons, stage1/2acceptance and actual
recovery across all controls. Cache hits and rejected cheap keys must be counted.
An optional constant-probability mixture with ordinary exact Metropolis preserves
the same target and provides a bypass for a bad filter, but still needs a cost
and recovery test. No pruning/acceptance/temperature choice should use answers.
Do not count frequent visits as extra key prior mass or infer confidence from
finite chain output. Better computation only helps decipherment if recovered
readings and held-out predictions improve under a defensible channel family.
