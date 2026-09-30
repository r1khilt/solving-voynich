# Behavioral latent geometry: avoid mistaking a readout bottleneck for a workspace

2026-09-30. Derived research notes while the paired source models train.
No learned-model geometry measurements, interventions, new cipher panels or
manuscript conclusions have been produced by this note.

## Source review and scope

The user's [Anthropic J-space reference](https://www.anthropic.com/research/global-workspace)
links to [Gurnee et al.2026](https://transformer-circuits.pub/2026/workspace/index.html).
The paper averages linearized effects across contexts and future outputs and
then uses interventions to test functional roles beyond an immediate readout.
Its workspace claims concern much larger transformer language models with
word/subword vocabularies and additional behavioral tests. A23-letter Latin
predictor does not inherit those properties. Our earlier failed mechanistic
qualification is not changed by building a larger recurrent model.

[Martens2020](https://www.jmlr.org/papers/volume21/17-678/17-678.pdf), sections5–6,
9,11–12, reviews Fisher geometry, matrix-vector products, empirical-Fisher
limitations and reparameterization. We adapt local distribution geometry to
recurrent **state**, not optimizer parameters; we are not proposing another
natural-gradient training run. The derivations below are our application of
these mathematical identities, not empirical findings from that paper.

## Why cosine and immediate Jacobian rank are insufficient

Let s collect both hidden and cell states from both recurrent layers:
D=2*2*768=3072coordinates. Let z(s) have23next-letter logits, p=softmax(z),
and J=dz/ds. The immediate output-sensitive quadratic form is

F(s)=J^T [diag(p)-p p^T] J.

For small delta, KL(p(s)||p(s+delta)) = .5 delta^T F(s) delta + O(||delta||^3),
assuming smoothness and finite strictly positive probabilities.
The softmax covariance has rank at most22, so **rank(F)≤22 regardless of what
this model learned**. Observing a22-dimensional next-letter-sensitive space
inside3072state coordinates is therefore an algebraic consequence of the
readout. It cannot by itself establish a global workspace, conceptual reasoning,
compressed semantics, or a discovered manuscript mechanism.

Cosine in raw state coordinates is also not intrinsic. Under an invertible
coordinate relabeling s'=A s, with downstream computation reexpressed
accordingly, distributions stay unchanged but ordinary cosine can change.
Here F'=A^(-T) F A^(-1), delta'=A delta, so the local behavioral quantity
delta'^T F' delta' equals delta^T F delta. This compares a function under a
coordinate change; arbitrary A is not asserted to preserve the standard LSTM
parameterization with its elementwise gates.

Eigenvalues of F alone are **not** invariant under that congruence transform.
Calling its largest eigenvectors universal mechanisms would still import a
coordinate choice. With a full-rank reference state covariance C, C'F'=A(CF)A^-1,
so eigenvalues of CF, equivalently C^(1/2)F C^(1/2), have an invariant meaning
relative to that chosen state distribution. In practice C may be singular;
restricting to empirical support, truncation and ridge regularization must be
reported. Adding lambda*I does not preserve arbitrary-coordinate invariance.

For two log-probability gradient covectors g_a and g_b, g_a^T C g_b similarly
measures shared response to reference-distributed state changes and is invariant
under consistent linear coordinate changes. It is a candidate replacement for
raw token-embedding cosine, not proof that two tokens mean the same thing.
Reference distribution, centering, normalization and conditioning matter.

## Observe memory over future behavior, not one output alone

Let T(s,u_<t) evolve the state through a **common** fixed probe continuation,
and let J_t be the derivative of its t-th logits with respect to the initial
state. Define a probe-weighted multi-step operator

G_H = E_(u~r) sum_(t=1..H) J_t^T [diag(p_t)-p_t p_t^T] J_t.

For one probe its rank is at most22H; averaging probes may enlarge the span.
This is a local measure of which state changes become observable through
future predictions. A direction invisible immediately can become visible later.
Compare horizons1,4,16,64 and matched random-initialized/memorizing/generalizing
models before interpreting apparent compression. These are proposed comparisons,
not frozen empirical runs or measured results.

A crucial distinction: when r is fixed recorded Latin, G_H is a **probe-weighted
conditional sensitivity**, not automatically the Fisher information of the
model's generated H-letter joint distribution. The exact chain rule is

KL(P_s(Y_1:H)||P_s'(Y_1:H)) =
E_(Y~P_s) sum_t KL(p_s(.|Y_<t)||p_s'(.|Y_<t)).

On-policy averaging yields the sequence Fisher at equality. Teacher-forcing
external text generally uses a different weighting. Either can be useful, but
they answer different questions and cannot be silently conflated. A finite
probe set also cannot prove two states interchangeable for all continuations;
it does not justify exact merging of neural states during cipher search.

Large Jacobians need not be materialized. JVP/VJP products or controlled finite
differences can estimate quadratic responses on fixed directions; validate
these on a small exact-Jacobian model and local KL scaling first. Numerical
conditioning, saturation and finite perturbations must be measured. A local
linear approximation is not a valid large activation intervention by default.

## Tie causal analysis to an actual reading question

After a reader qualifies on fresh known-key cases, select ambiguous decisions
using a frozen rule and ask whether distant context changes the **correct versus
incorrect complete-candidate margin**. Compare full context, short context,
state deletion, matched resampling, equal-magnitude random directions and a
behavior-matched alternative source. Use both initialization seeds. Evaluate
finite interventions on held-out examples; do not fit directions and declare
success on those same examples. Changes in predictions alone are insufficient.

Off-distribution state patches can damage any model. A successful claim needs
selective loss/recovery of the target behavior while suitable controls remain
intact, and must survive the search-error checks in
[the reader notes](recurrent-reader-search-notes-2026-09-30.md). It still identifies
a model mechanism, not the historical Voynich generator.

Finally, our23outputs are letters, not a vocabulary of concepts. Multi-letter
probe strings can be scored, but overlapping or variable-length strings are
not automatically a normalized lexical distribution. Specify mutually exclusive
outcomes and a stopping/boundary convention before calling their similarities
word structure. This is why immediate character-gradient clusters should not be
presented as a semantic dictionary or translations.
