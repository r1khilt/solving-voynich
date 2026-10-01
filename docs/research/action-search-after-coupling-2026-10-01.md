# A repair model must reason beyond immediate score gain

2026-10-01, AFTER SOURCE-COUPLING-001. Derived implications and proposals;
no policy, architecture, new training, sampler or manuscript search launched.

## What the intervention now establishes

On these four exposed artificial keys, focused one-row changes can improve
the score but usually do not correct a used mapping. Correcting all wrong
used rows is strongly beneficial, while small known corrections often harm
the same likelihood. Pair/triple gold-advantage signal failed. We should not
claim that a generic two/three-row policy solves the newly observed problem.

This rejects a particularly simple local surrogate: separate per-row gains
computed once at the selected key and then summed. For an exactly factorized
likelihood L(K)=constant*product_i f_i(K_i), a positive base and supported
single changes imply delta(M)=sum_i delta(i). If a corrected single has zero
likelihood, f_i(corrected)=0 and the full corrected endpoint must have zero
likelihood too. The observed positive endpoint contradicts that model in12
cells with a zero single correction. Four others have finite singleton sums
that disagree with the endpoint by472.36..1831.99nats. These are algebraic
consequences of the fixed-score observations, not a posterior factorization
or identifiability theorem for all ciphers. Conditioning on ciphertext and
summing compatible paths creates mapping dependencies even under IIDsource.

Exact conditional probabilities computed after each change could behave very
differently from a fixed additive predictor. These data do not invalidate all
autoregressive key models, Gibbs algorithms or Bayes inference. They also do
not prove that a particular action order, block size or outside-face detour
is impossible. The original key prior itself remains iid; the observation
likelihood is the coupled factor.

## A concrete world/action-model role

Use an explicit state (ciphertext,current whole key,declared source and any
retained search history), legal key-edit actions and the exact scorer as a
verifier. A model could predict consequences of sequences of edits or propose
a coherent alternative source/key hypothesis. The empirical lesson is that
an immediate-score critic alone may rank correct early repairs poorly. It
needs testable multi-step predictions, not only a larger embedding or a claim
that attention discovers language.

One candidate is a policy over legal coupled edits conditioned on a current
reading/constraint graph, with every proposed dictionary verified before use.
Another is a source-first proposal that constructs a mutually compatible
reading/key pair, then revises it coherently. Neither may use the generating
key, source length, gold presence mask or true segmentation at test time.
Such proposals need fresh positive/null controls and equal work budgets
against the existing search. World/action terminology is inspiration; this
does not train a robot VLA or recover meanings from images.

## Delayed returns and potential shaping

For a supported finite-score path K_0,...,K_T and undiscounted rewards
r_t=J(K_{t+1})-J(K_t), direct algebra gives

    sum_t r_t = J(K_T)-J(K_0).

A temporarily harmful correction can still lead to a valuable endpoint.
Greedily maximizing r_t does not maximize that whole-path sum. For discount
gamma<1, ordinary J_next-J_current instead yields

    sum_t gamma^t r_t = -J(K_0)
        + (1-gamma)*sum_{t=1}^{T-1} gamma^(t-1)*J(K_t)
        + gamma^(T-1)*J(K_T).

It explicitly rewards intermediate scores as well as the endpoint and is not
the same optimization problem. Terminal/budget handling and variable horizons
must therefore be specified rather than hidden in an RL training loop.

[Ng, Harada & Russell1999](https://ai.stanford.edu/~ang/papers/shaping-icml99.pdf)
provides the primary potential-based shaping precedent. The primary indexed
abstract was checked; its PDF text extraction was corrupted, so no detailed
theorem proof is claimed read. Our finite-path identities here are derived
directly. With shaping gamma*Phi(next)-Phi(current), the discounted sum is
−Phi(K_0)+gamma^T*Phi(K_T). To make it path-independent relative to a declared
terminal objective, terminal potential conventions must be handled explicitly
(for example zero terminal potential in an augmented terminal state). This
does not automatically fix exploration, correctness of J or latent support.

Unsupported keys have J=−infinity. Differences between two such values are
undefined; the environment needs an explicit support/dead-end/action protocol.
An arbitrary floor would change the target and may create spurious routes.
Coherent supported proposals, retained valid candidates and separately declared
surrogates are legitimate search choices; they are not an invariant posterior
sampler unless the actual proposal law/reverse ratio is accounted for.

## Gate before a new expensive campaign

Finish the unchanged four neural fits and their registered completion audit.
Its observed small/first-large failures prevent assuming it already learned
a useful deciphering circuit. A later action/source-pair comparison should
freeze test-time observations, whole-key/reading objective, trajectory length,
source exposure, real scoring budget, restart retention and delayed-outcome
controls. Causal interpretation then asks which intermediate representations
actually enable verified repairs. No cheaper proxy is allowed to replace
correct held-out readings or become a Voynich translation claim.
