# An inverse must learn useful structure, not just receive more parameters

2026-10-01. Project derivations and proposed next methods, following
JOINT-KEY-ORDER-001. No new auxiliary model, action policy, sampler, training run
or decipherment is implemented by this memo. The live four-fit training protocol
remains frozen. The completed first control has poor key recovery and does not
pass its exploratory suffix-order diagnostic. That is evidence about this fit,
not a verdict on the pending95M fits or the information in the ciphertext.

## What the order experiment can and cannot identify

Let C be ciphertext and Z retain the counts, lengths, canonical names and fixed
ordered prefixes preserved by both controls. Any predictor of the form
`q(K|C)=f(K,Z(C))` is exactly invariant under our transformations: Z is unchanged.
The small measured contrasts are consistent with this explanation, but do not
prove it. A network can use ordered prefixes, some changed features can cancel
in an average, and only one fixed shuffle per case was tested. No class of all
permutations was exhaustively checked. Input intervention is not identification
of a neuron or unique algorithm.

Even a separately trained count-only baseline would not directly measure
intrinsic information. For unrestricted Bayes models,

`H(K|Z) - H(K|C) = I(K;C|Z) >= 0`.

For finite fitted models, instead,

`R(q_Z)-R(q_C) = I(K;C|Z) + e_Z-e_C`,

where the e terms are expected conditional KL approximation errors. Their
unknown difference prevents interpreting a finite-model risk gap as mutual
information. Our uniform suffix shuffles also are not samples from the true
conditional natural-language distribution given Z. The observed contrast is an
input-sensitivity statistic, not a mutual-information estimator.

Here R and the entropies use whole-key NLL. Reporting a per-row risk divides
the entire identity, including the mutual information and KL terms, by23.

## A source-backed auxiliary objective, with a precise limitation

[Kambhatla, Born and Sarkar2023, §5.1/8](https://aclanthology.org/2023.findings-eacl.160.pdf)
compares causal ciphertext-reproduction-plus-decoding against other objectives
and attention patterns in recurrence-encoded homophonic substitution. Their
result motivates testing an encoder that must learn ciphertext structure, rather
than receiving only dictionary labels. It does not prove an auxiliary loss will
help our variable-unit inverse. [ALICEv1, §2/5](https://arxiv.org/html/2509.07282v1)
uses explicit bijective structure and intermediate decoding; those assumptions
cannot be imposed on duplicate-allowing keys merely for interpretability.

A candidate is a shared encoder with the existing complete-key conditional
head and a separately specified causal ciphertext or plaintext-sequence head.
Use a positive constant weight on each full sequence NLL, with EOS and record
resets. The encoder must not see a target source length or unit boundary. Gold
targets are legitimate training supervision, not inference inputs. Training
language, channel and dictionary prior remain supplied assumptions.

The existing full-cipher encoder is bidirectional. Merely adding a head that
predicts its input glyphs can be solved by copying: it is not causal sequence
modeling, and a loss decrease would not demonstrate useful structure. A causal
ciphertext objective needs a separate forward pass with the same encoder
parameters and a mask at **every** layer that excludes the target and its
future, including through cross-attention or auxiliary features. At position t,
only the declared prefix is available. Alternatively predict record B given
complete record A and B's prefix; complete B cannot be encoder memory for its
own prediction. Padding/length conditioning must be declared rather than
accidentally revealing the target's end. A plaintext decoder may legitimately
attend to the full observed cipher, but only its preceding plaintext targets.
Target-copy and future-token perturbation checks precede any training. These
are prospective requirements; no existing encoder mask was changed.

The two probability heads each have proper conditional log loss. In an
unrestricted class capable of simultaneously expressing both true conditionals,
their weighted population risk is a sum of conditional entropies plus positive
KL terms and has a jointly Bayes-optimal solution. A finite shared network can
trade off those terms. Therefore neither a proper primary loss nor an auxiliary
head guarantees better key inference; matched no-aux controls, fresh keys and
actual reading remain necessary.

There is a subtle trap when source lengths vary: dividing a whole-sequence loss
by the **gold** length generally changes the learned conditional distribution.
At a fixed input c, a target-dependent weight w(x)>0 gives

`q*(x|c) = P(x|c) w(x) / E[w(X)|c]`.

For equally probable source strings of lengths1 and2, length normalization
learns probabilities2/3 and1/3, instead of1/2 and1/2. This is the sequence analogue
of the presence-mask bias established in the proper-key memo. Length-normalized
evaluation may be descriptive; do not silently claim that its training objective
is proper for the original whole-sequence conditional. This statement concerns
a whole-sequence model; ordinary token-sampled language modeling defines a
different population objective and needs its own sampling/conditioning account.

## A more direct alternative: learn actions in a verified key-search environment

The native source/channel evaluator already supplies an executable environment:
state is a legal whole dictionary, action replaces one row's emitted unit,
transition is exact, and reward is a declared change in its literal fitting
objective. A neural policy or world/action predictor can propose repairs while
the evaluator verifies them. This is a concrete route to learning iterative
cryptanalysis. Calling it action-model-inspired does not make it a trained robot
VLA or a recovered linguistic meaning.

Compared with one-shot key prediction, this changes the task: learn useful
updates given a current candidate and ciphertext, rather than reconstruct all
23 arbitrary rows in one output. The current inverse can provide initial
candidates; existing statistical/search warm starts are essential controls.
The action state, available observations and objective must be explicit. Gold
key distance may supervise a synthetic policy, but cannot be a manuscript reward.
A learned reward model can be wrong; actual acceptance must use the verifier.

Two scientifically different uses must remain separate:

* **Search/optimization:** repair proposed keys, deduplicate, score with the
  unchanged evaluator, and report recovery and support within the visited bank.
  Visit counts do not create a prior; no posterior/evidence/mixing claim follows.
* **Posterior sampling:** specify an actual normalized stochastic transition
  and its reverse probability. If pi(K|C) is proportional to p(K)L(C|K), a
  proposed K' with transition Q has Metropolis acceptance
  `min(1, pi(K'|C) Q(K|K',C) / (pi(K|C) Q(K'|K,C)))`.
  A row/action policy can make these probabilities computable. For distinct
  one-row moves, Q is the row-choice probability times the unit-choice
  probability. Self-transition probability sums all actions that leave K
unchanged. Connectivity, support, cost and mixing still need testing.

Legal row replacements alone do not guarantee irreducibility on the target's
**positive-likelihood** support. An exact counterexample has two source letters,
two singleton glyphs, iid uniform key rows, and source strings ab/ba equally
likely. Observing xy leaves only keys (x,y) and (y,x). Each single-row move
between them passes through (x,x) or (y,y), whose likelihood is zero, so a
one-row Metropolis chain starting at either supported key never reaches the
other. A positive proposal probability for every row/unit does not fix this.
Block moves or a full-support independent proposal can connect these states;
their real cost and mixing remain unmeasured. Temperature changes cannot revive
zero likelihood unless a different intermediate target is explicitly introduced.

The action environment must also handle unsupported keys explicitly: a literal
log likelihood can be minus infinity, making subtraction of two such scores
undefined. Do not quietly convert this to a finite reward or train on NaNs.
Declare a support flag, valid transitions and an exact acceptance rule; any
surrogate used to guide search remains separate from the unchanged final score.
Even an exact native score embodies the supplied source/channel assumptions
and can prefer the wrong reading, as the existing source diagnostics show.

An independent neural whole-key proposal instead uses the ratio
`pi(K')q(K|C)/(pi(K)q(K'|C))`. The conditional neural q is a proposal, **not** an
additional historical prior. Multiplying the fitting target by q would change
the target and can count source-language evidence twice.

Deterministic greedy repair is especially dangerous for probability claims.
If R maps a proposed K to a repaired key, its output probability is
`q_R(k|C) = sum_{K:R(K)=k} q(K|C)`.
It is generally neither q(k|C) nor the probability of the sampled starting key.
Treating repaired samples as unmodified q draws gives invalid importance weights
or Metropolis ratios. A repair that maps every key to one candidate destroys
support entirely. Such repair remains a valid search heuristic when reported
as search, with every repaired key independently rescored and deduplicated.

Likewise, neural importance sampling can estimate a finite target normalizer
with weights `p(K)L(C|K)/q(K|C)` only with the required proposal support and
correct actual proposal probabilities. Very diffuse or mismatched q can make
variance catastrophic. Truncation, unseen-glyph symmetrization, temperature,
deduplication and repairs all change the law and cannot be ignored. Numerical
probability checks are necessary, and a finite visited-bank reading is not an
exact whole-space posterior or evidence calculation.

## Exact toy checks and next decision

Exact rational arithmetic checked the length-tilt example above and a two-state
Metropolis example. With pi=(2/3,1/3), q=(1/4,3/4), A→B acceptance is1/6 and
B→A acceptance is1; both stationary flows are1/12. Taking unnormalized weights
(1/3,1/6) gives an importance expectation1/2 under q. Mapping both states to A
by greedy repair gives output law(1,0), demonstrating lost support. These are
derivations, not neural or historical experiment results.

Enumeration of the four two-row dictionaries also checks the disconnected
support example: likelihoods are(0,1/2,1/2,0), posterior masses(0,1/2,1/2,0),
and neither positive-mass key has a positive-likelihood single-row neighbor.
This blocks the tempting claim that a learned local repair policy automatically
provides a correct, exploring posterior sampler.

Complete the unchanged four-fit comparison and full audit. Then register one
bounded architecture/objective comparison with unchanged key priors, source
exposure, actual compute measurements and exact evaluator controls. The
order-negative first fit makes auxiliary structure learning or learned verified
repair more justified than neuron hunting on its current unsuccessful outputs.
Whether the larger fits change that decision is still open. No next expensive
training run is launched or silently queued by this memo.
