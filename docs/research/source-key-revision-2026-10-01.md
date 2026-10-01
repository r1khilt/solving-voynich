# Revising the shared key after source-particle search

2026-10-01. Method review and proposed exploratory test, written before
SOURCE-REVISE-001. Voynich remains unsolved. This is a restricted artificial
decoder investigation, not evidence that the manuscript uses Latin or this code.

## Why change the inference procedure

SOURCE-GUIDE-001 improved all eight paired searches but none of its sixteen
banks contained the generating reading or used mapping. Once a particle binds
a letter, its later descendants cannot change that binding. Better proposals
therefore still inherit early mistakes. The previous known-answer diagnostic
also established that some much higher-probability readings were missed; it
did not establish that the generating reading is globally optimal.

The intervention here is whole-key revision using the existing exact native
source marginal and tested replica search. It is a new connection between
existing components, not a novel MCMC algorithm. There is no architecture
change and no gradient training in this experiment. The separately frozen
neural campaign continues unchanged.

## Primary literature and prior applications

The prior Voynich/ML/decipherment review remains
`docs/research/PRIOR_WORK.md`, especially its direct Voynich neural studies and
Hauer & Kondrak's 2016 anagram/substitution assumptions. Prediction, saliency,
or a proposed Hebrew reading does not qualify this different channel.

[Hauer, Hayward & Kondrak (2014)](https://aclanthology.org/C14-1218.pdf),
sections 4–5, combines character/word information with key modification and
Monte Carlo tree search for monoalphabetic substitution. Its permutation and
word-pattern constraints differ from our duplicated, variable-length units.
The transferable idea is to change a complete mapping and score its
consequences across the text. We reuse our earlier symmetric proposals rather
than transferring their constraints or their accuracy claims.

[Lindsten, Jordan & Schon (2014)](https://jmlr.org/papers/volume15/lindsten14a/lindsten14a.pdf),
sections 3.1–3.2 (equation 3) and 6.2 (equations 22–23), gives a principled
particle Gibbs/ancestor-sampling construction. Reference paths are part of an
extended probability law. For non-Markovian paths, ancestor weights depend on
the candidate past and reference future. Their truncation analysis needs a
decay condition. Our shared dictionary can constrain arbitrarily distant
positions, so local source order alone does not justify truncating dictionary
dependence. This is our inference from their construction, not a theorem about
our code from that paper. Simply forcing one reference particle through the
existing locally adapted resampling is not the derivation we need. We defer
PGAS rather than claiming a valid kernel without that derivation.

Existing project math in `key-inverse-next-objectives-2026-10-01.md` demonstrates
that a positive-support single-row move graph can be disconnected. Swaps and
multirow replacements can bridge some such examples. Our fixed blocks of
2/3/6 rows do not prove irreducibility on all 23-row positive-support keys.
Finite search is an optimizer; its bank is not a posterior sample or coverage
certificate. See the existing `tempered_unit_search.py` tests for proposal
symmetry and exact finite transition controls.

## Probability model and decision rule

Let U be the 42 single/double strings over ABCDEF, K in U^23 have iid uniform
rows, and two records have independent reset source draws conditional on K.
For record c and fixed K:

    L(K,c) = sum_{x: encode_K(x)=c} rho (1-rho)^len(x) p_source(x)
    J(K) = -23 log(42) + sum_records log L(K,c), rho=1/225.

The order-12 source/state automaton and native lattice sum ALL compatible
source paths. Real zero support stays negative infinity. Numerical or graph
resource failures raise errors; they cannot become zero likelihood.
Unobserved rows remain explicit in full keys. The constant full-key prior
matches the artificial iid generator; the earlier GLOBAL-KEY literal MDL
length penalty is deliberately not imported. Searching J is not equivalent
to summing every compatible dictionary for a plaintext reading.

For each exposed particle bank, select its highest original literal leaf,
break ties lexically, preserve every bound row, and fill unbound rows iid once
with a registered seed. This filled key may allow readings outside the old
leaf. Decode it with exact fixed-key Viterbi. Then revise the entire key and
decode the best visited key by exactly the SAME Viterbi rule. Compare those
two predictions. The previous particle modal-reading errors are a different
decision rule and cannot be used as the before-revision baseline.

Eight replicas at temperatures 1,2,4,8,16,32,64,128 target exp(J/T) in ideal
arithmetic. Symmetric replacement/swap/block mixture weights are 2:1:1.
Accept a key change with min(1, exp((J_new-J_old)/T)); exchange neighboring
replicas with min(1, exp((1/T_left-1/T_right)(J_right-J_left))). These invariant
kernels do not imply that a finite run is stationary or finds the global
maximum. Keep every scored key and the earliest best key, including warm start.

## Separating possible failure modes

Archive the fitted search and both predictions before consulting gold. Then
score the generating FULL key, check its used mapping against the bank, and
decode with the supplied generating key using the same exact reader.

- Generating-key score above the selected score demonstrates an available
  better key that search missed; it does not prove global optimality.
- Errors with the generating key supplied reveal ambiguity/source/decision
  limitations even when dictionary inference is bypassed. Forced fixture
  lengths are not the decoder's unconditional geometric-length distribution.
- Higher J with unchanged or worse true-text error is possible: key-marginal
  likelihood, fixed-key Viterbi and actual recovery are different objectives.
- Improvement on these four already exposed keys is a development result.
  It needs fresh positive/null controls and broader mechanism families before
  a recovery claim, and independent historical constraints before decipherment.

This test can tell us whether repairing early bindings is useful. It cannot
alone tell us which architecture, language or encoding the manuscript used.
