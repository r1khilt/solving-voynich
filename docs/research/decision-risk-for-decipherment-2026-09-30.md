# Decision error after exact search: a possible next branch

2026-09-30. Theory/prior-work note written while KEY-BANK-READ-002 runs, before
its answer evaluation. No risk decoder, posterior sampling experiment or accuracy
improvement is claimed here. The active MAP experiment stays unchanged.

## What the literature establishes

[Kumar and Byrne, NAACL2004](https://aclanthology.org/N04-1022.pdf), sections2.1and3,
formulate minimum Bayes risk (MBR) with a loss chosen for the task, including word
errors. In practice their decision uses an estimated distribution and a restricted
hypothesis set. This provides the decision-theoretic rationale; their supervised
translation setting does not validate our unknown-key model. Read depth: targeted
fulltext sections2.1,3 and surrounding explanation, not an independent replication.
Some extracted equation glyphs are malformed; equations below are derived directly
for our declared probability model, not transcribed from the extraction.

[Tromble et al., EMNLP2008](https://aclanthology.org/D08-1065.pdf), sections2–4,
separate the evidence space used to estimate risk from the space of permitted
outputs. Their efficient lattice treatment relies on a locally decomposable
n-gram gain and an approximation to BLEU. It is not a ready-made exact algorithm
for expected Levenshtein distance on our shared-key graph. Read depth: targeted
fulltext sections2–4, not all experimental tables.

[Freitag et al., arXiv2111.09388v3](https://arxiv.org/abs/2111.09388v3) report MBR
using unbiased samples and neural translation metrics. Read depth: abstract only.
This motivates inspecting sampling assumptions; it does not justify importing a
semantic similarity metric into decipherment or expecting their gains here.

Targeted searches `Voynich "minimum Bayes risk" decipherment` and
`decipherment "minimum Bayes risk" cipher` did not return a directly applicable
primary Voynich demonstration. This narrow search is not evidence that none
exists. The repository's prior Voynich and cipher reviews still govern historical
claims. We have no known plaintext-language identification from these methods.

## Our distinct three problems

A bank can miss every key supporting the true text. Inference can miss a better
candidate inside that bank. Even perfect inference under a complete bank can
choose the wrong text because the source/key probabilities or the decision loss
are unsuitable. READ001's separated bounds address the second issue within its
banks; they do not resolve the first or third.

For a globally consistent text tuple `x`, let

```
f(x) = Q(x) sum_i w_i 1{C_i(x)=y},
Z = sum_x f(x) = sum_i w_i product_r Z_i(y_r),
p(x|y,bank) = f(x)/Z.
```

MAP chooses the largest `p(x)`. It minimizes model-expected whole-tuple0–1loss.
Our recovery metric instead counts `d(a,x)=sum_r Levenshtein(a_r,x_r)`.
A risk decision minimizes `R(a)=sum_x p(x)d(a,x)` over a declared action set.
These are different objectives even under the exact same probability model.

A simple mathematical example, not an experiment: posterior probabilities
`000:.40, 111:.35, 110:.25` yield expected edit losses1.55,1.45,1.15 respectively.
MAP selects000; the best of these three risk actions is110. This establishes a
possible decision mismatch, not evidence it causes any current empirical error.

## Preserve the channel constraints

Use only whole tuples that re-encode under at least one positive-weight **single
shared key**. Independent record medians or characterwise majority votes can
violate this condition. Freely rewriting a passage to sound natural is not
admissible decipherment. A bounded candidate action set gives a restricted
optimum; do not call it the global edit-risk optimum.

No gold, known true length, new language choice or posthoc transcription repair
may enter prediction. A literal character edit loss has a transparent relation
to exact recovery; a learned semantic metric might reward fluent paraphrases
that cannot explain the ciphertext. Better expected model risk is not evidence
that the model posterior is historically calibrated.

## Bounds from the already available exact evidence

The following are elementary derivations for this project, not novelty claims.
Let a distinct enumerated setS have absolute probabilities `p(x)=f(x)/Z` and
mass `m=sum_S p(x)`, with omitted mass `epsilon=1-m`. Do not normalize only overS
and pretend its missing mass is zero. Deduplicate whole strings across keys and
include their combined compatible key weights.

For our nonempty emissions, each plaintext record length is at most its observed
glyph count `n_r`. Thus for supported actions/references, the summed edit loss is
at most `N=sum_r n_r`. The unnormalized truncated risk
`R_S(a)=sum_S p(x)d(a,x)` satisfies

```
R_S(a) <= R(a) <= R_S(a) + epsilon*N.
```

That interval may be too loose. Metric reverse triangle inequality gives the
stronger comparison bound

```
D_S(a,b) = sum_S p(x) [d(a,x)-d(b,x)],
R(a)-R(b) in [D_S(a,b)-epsilon*d(a,b),
             D_S(a,b)+epsilon*d(a,b)].
```

Proof: each omitted reference contributes a difference between `-d(a,b)` and
`d(a,b)`; its probabilities sum toepsilon. This can certify which of two nearby
readings has lower full-model risk even when substantial mass lies outsideS.
A winner certified against every member of a fixed action set is optimal in
that action set under the full posterior, not necessarily outside it.

A second useful sufficient condition: if a candidate has posterior mass greater
than1/2, it uniquely minimizes expected metric distance over all distinct actions.
For candidatea with massp and anyb, its own contribution to `R(b)-R(a)` is
`p*d(a,b)`; the remaining mass contributes at least `-(1-p)*d(a,b)`.
Therefore `R(b)-R(a) >= (2p-1)*d(a,b)>0`. If this applies, switching to edit risk
cannot change that output. At exactly1/2 it gives non-strict optimality only.

A separated MAP upper bound limits the score of **one** unseen tuple. It does
not bound the **sum** of all unseen probability mass. Consequently the existing
k8certificate is not a risk certificate. All these probability/mass calculations
also need numerical-error handling; floating separation is not an interval proof.

## Sampling alternative and what to test first

An exact-in-model sampling construction is available in principle: sample one
key with probabilities proportional to `w_i product_r Z_i(y_r)`; conditional on
that key, sample each source path from its normalized constrained forward/backward
lattice. Keep that same key for all records. Marginalizing the sampled key produces
the correct whole-text posterior, including multiple keys yielding the same text.
Uniform key sampling or treating topkstrings as unbiased samples does not.

For a future implementation, first verify complete small-support risk enumeration,
the shared-key constraint, deduplication, tail comparison bounds, the half-mass
lemma and empirical sampling frequencies against an analytic distribution. Then
freeze candidates, sampling seed/budget, loss and decision scope before empirical
answers. Estimate pairwise edit computation cost; naive action×reference scoring
can dominate inference. No new fullsource fit is required merely to test the
loss mismatch. If the model strongly favors the wrong text or the right key is
missing, source improvement and key search remain the substantive problems.

Current priority remains completing READ002's actual recovery/control evaluation.
A risk branch is justified only after its support and posterior diagnostics, and
still requires fresh end-to-end unknown-key qualification before historical use.

## Exact expected edit distance is a separate inference problem

A possible exact finite-state construction follows directly from the edit-distance
recurrence. It is an unimplemented derivation, not a computational feasibility or
novelty claim. For a fixed proposed recorda of lengthm, carry the Levenshtein DP row
`v[j]=distance(generated prefix,a[:j])` alongside the source state and ciphertext
offset. Initialize `v[j]=j`. On emitting source characterc:

```
v_new[0] = v[0]+1
v_new[j] = min(v[j]+1,
               v_new[j-1]+1,
               v[j-1]+1{a[j-1] != c})  for j=1..m.
```

Merge only states with identical source state, observed offset and whole DP row.
The row is a deterministic sufficient summary for future edit distance to fixeda.
At termination sum each path's probability times `v[m]`, then divide by the exact
per-key record evidence. This computes expected **minimum** edit distance, rather
than summing alternative alignments or swapping expectation with minimization.
Nonempty emissions preserve acyclicity. Distinct DP rows can destroy most of the
source-state compression, so the state space can grow exponentially; explicit
caps and a small analytic validation are essential before any empirical run.

Additivity offers one legitimate simplification. With the posterior key weights
conditioned on **all** transfer records, the risk of a whole candidate tuple is

```
R(a) = sum_r sum_i p(K_i|all y) E[d(a_r,X_r)|K_i,y_r].
```

Thus per-record expected distances can be cached even though decisions must still
choose a jointly admissible tuple. The conditional expectation uses the exact
key-specific posterior record distribution. This algebra does not permit picking
independent convenient keys for different records: their marginal weights all
come from the same all-record posterior. It suggests comparing a capped DP-row
method, complete small-case enumeration and posterior sampling before choosing
an implementation, rather than assuming that the existing MAP lattice makes
expected edit distance free.
