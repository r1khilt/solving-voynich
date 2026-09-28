# Finite-state channel: independent mathematical checks

2026-09-27. This audits the bounded inference foundation in
`src/voynich/finite_state_channel.py`, using hand-specified rational examples.
No source corpus, benchmark, model fitting, or decipherment is involved. The
[blind-channel design](blind-channel-recovery-design.md) supplies the broader
research motivation and primary-source review. The derivations below concern
the implemented constant-stop, order-zero/order-one source model only.

## Probability and termination

Let the constant stopping probability be `0 < rho < 1`. Every source row,
initial-state distribution, and complete channel-emission row sums to one.
After summing every source/state/emission choice, the probability of exactly
`n` source characters is therefore `rho * (1-rho)^n`, including `n=0`.
Summing over all `n >= 0` gives one. This normalization does not depend on
emission ambiguity, source-context dependence, or channel-state dependence.

Nonempty emissions ensure at most `len(y)` source steps for a fixed observed
string `y`; exact exhaustive inference is finite. Empty emissions would break
that acyclic argument and are rejected. Entire emission chunks must match:
the observation cannot truncate a longer emission.

For single-glyph emissions, the total mass of all complete observed strings
with length at most `L` is exactly `1-(1-rho)^(L+1)`. For lengths from one to
`M`, that cumulative mass lies between
`1-(1-rho)^(floor(L/M)+1)` and `1-(1-rho)^(L+1)`. This is a cumulative mass over
terminated strings, not the probability that an ongoing stream starts with a
particular glyph prefix.

Empty-string marginal likelihood is `rho`, because initial-state mass sums
to one. Its highest-probability joint path instead has mass
`rho * max_s pi(s)`. Matching an observed glyph never licenses renormalizing
the complete emission row over only its matching alternatives.

## Joint paths, plaintext, and posterior counts

For a one-state source with `p(a)=3/5`, `p(b)=2/5`, `rho=1/2`, let `a` have
two separately labeled alternatives, both emitting `x` with probability
`1/2`; let `b` emit `x` deterministically. For observation `x`, the two `a`
paths each have mass `3/40`, while the `b` path has mass `1/10`. The joint
Viterbi path therefore decodes `b`, but summed plaintext mass is `3/20` for
`a` versus `1/10` for `b`: MAP plaintext is `a`. Duplicate latent alternatives
must be counted separately in sums and posterior transition counts.

For an iid one-letter source, `rho=1/2`, and equal emission probabilities for
`x` and `xx`, observation `xx` has total mass `5/32`: one-character mass
`1/8` plus two-character mass `1/32`. Posterior expected source length is
`6/5`, although the joint Viterbi path has length one. Expected counts of the
`x` and `xx` alternatives are `2/5` and `4/5`; their sum equals expected source
length. Impossible observations have no conditional posterior, rather than a
posterior with invented zero length.

## An exact ambiguity that no likelihood optimization resolves

Take an iid uniform source on `a,b`, one state, and the same geometric stop.
Channel I maps `a -> 0`, `b -> 1`; channel S maps `a -> 1`, `b -> 0`.
For **every** finite binary observation `y` of length `n`, both channels have

```text
p(y) = rho * ((1-rho)/2)^n.
```

Each channel has one compatible plaintext and thus a point-mass plaintext
posterior. For every nonempty `y`, those two point masses disagree. Consequently
exact equality of ciphertext distributions does not imply equality of decoded
messages. The formula proves equivalence for all finite strings; enumeration
through length six is only an implementation check of that proof. Equal-cost
symmetrically encoded channel descriptions would also have equal two-part
code scores. Distinguishing the readings requires information that breaks this
symmetry, such as an asymmetric source constraint or an external anchor.
This is a mathematical control, not a claim about Voynich or any manuscript.

## Independent verification scope

`tests/test_finite_state_channel_independent.py` enumerates exact paths with
Python `Fraction`, independently of the engine's forward/backward routines.
It compares rational marginal likelihoods, maximum joint paths, conditional
transition counts, and source lengths against the public engine API. It also
checks state renaming/reordering, zero-mass and impossible paths, JSON model
round trips, geometric length mass, untrimmed emissions, and long-stream
log-space stability. The engine computes with floating point, so numerical
comparisons use explicit tolerances; these tests do not certify arbitrary
weighted-automaton equivalence or infer unknown channel parameters.
