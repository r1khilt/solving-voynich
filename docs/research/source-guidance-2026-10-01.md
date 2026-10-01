# Future evidence without changing the decipherment target

2026-10-01. Project derivation and implementation choice, not a historical finding.
Read with `source-key-particles-2026-10-01.md` and the negative particle results.

## Why this question

The prior DIAG-A measured a large separation between known generating leaves
and every visited Markov leaf. The target has better available explanations;
local proposals fail to visit them. More particles did not fix that fixed grid.
This motivates proposals informed by the remainder of BOTH ciphertext records.
It does not prove that a particular lookahead approximation will work.

Primary review: [Heng et al., Controlled SMC](https://www.stats.ox.ac.uk/~doucet/HengBishopDeligiannidisDoucet_controlledSMC.pdf),
sections 3.1–3.2 and equations 11–17, were read for positive twisting,
proposal corrections and unchanged terminal normalizer. The paper's learned
iterative control policies are not implemented here. Its applications and
assumptions do not establish cipher recovery or Voynich results.
[Zhao et al., ICML 2024](https://proceedings.mlr.press/v235/zhao24c.html)
abstract/publisher record was read as a learned-language-guide precedent; PDF
retrieval failed, so do not claim a full-method replication.
Existing prior-work review (especially Hauer/Kondrak's language-model flexibility
warning) and earlier Hauer 2014 MCTS review remain relevant: fluency is not proof.
No direct prior Voynich result using this exact variable-unit guide was established
by this limited review. No novelty claim is made.

## A computable surrogate

Let the real source probabilities be p(row | source history). The guide uses
only the existing source's root row, smoothed by 1e-8 per letter and normalized.
For a partial key K, define iid surrogate emission mass

    w_K(u) = sum_{assigned rows r: K[r]=u} p0[r]
             + sum_{unassigned rows r} p0[r] / (g+g^2).

Unassigned units are redrawn at EVERY occurrence. Assigned rows retain their
fixed units. This is an explicitly different channel, not a shared-key posterior
and not an upper bound. In particular it penalizes repeated use of an unknown
row anew, while the real prior pays once. The approximation can rank good keys
poorly. The fresh comparison measures this risk.

For each observed record C and position i, compute backward log sums using

    D[len(C)] = rho
    D[i] = (1-rho) * (w_K(C[i]) D[i+1]
                     + w_K(C[i:i+2]) D[i+2])

with the double term absent at the final glyph. This integrates all legal
segmentations under the surrogate. Unfinished records contribute log D[offset];
closed records contribute zero. The total log guide is floored at -2000.
Thus h is positive in exact arithmetic even where the surrogate has no path.
All complete states have log h=0. Logs avoid underflow in the recurrence.
The full remaining ciphertext is consulted on every potential first binding.
Source context is deliberately absent from this cheap guide.

## Target correction and finite precision

The original action mass a includes the true contextual source probability,
geometric continuation/EOS, and once-only iid shared-key prior. For each child:

    b(parent, child) = a(parent, child) h(child) / h(parent)
    G(parent) = sum_child b(parent, child).

Select a parent proportional to G and its child proportional to b. Initialize
log Zhat to log h(initial), then add log mean G at each fixed-horizon step.
The h ratios telescope to h(terminal)/h(initial), and h(terminal)=1, leaving
exactly the original leaf target. Literal leaf scores ALWAYS accumulate log a,
never log b. In exact arithmetic the usual unbiased unnormalized SMC measure
law applies; finite posterior frequencies and log Zhat remain biased.

Categorical sampling uses blocked log Gumbel-max rather than exponentiating
large negative weights. Independent E=-log U exponential races select the
minimum E/weight with probability weight/sum weights; equivalently maximize
log weight-log(-log U). This avoids explicitly dropping tiny categories during
softmax exponentiation. Finite floating RNG, subtraction and logaddexp are still
approximations; no exact support or interval certificate is claimed. ESS uses
exponentiation only as a diagnostic. Both experimental arms use this same
transport/RNG implementation; constant h=1 is the matched local control.

## Implementation and evidence limits

Frozen original particle/training modules remain unchanged. New guide tables
are keyed by the full partial dictionary and bounded by an LRU cache. Each table
includes all records/offsets. Build batches have at most 64 keys; retained tables
are copied to prevent a view retaining a whole batch. Allocation/work caps stop
the experiment rather than extending it. Candidate state arrays include every
original positive action; no gold source, length, key or segmentation is input.

Independent tiny string/fresh-unit enumeration checks the surrogate recurrence.
All 36 tiny ciphertext pairs under both schedules exhaustively check proposal
normalization including extinction, h telescoping, and original reading masses
against the alternative full-key/source enumerator. These are meaningful law
checks, not proof that finite particles find good full-size keys.

The next registered comparison uses four fresh artificial keys/source pairs,
two population seeds and two guidance arms. Same forced-length caveat as prior
systems work: lengths are not supplied to the solver, but this is not an
unconditional geometric-source population. No historical manuscript channel,
new language, recovery qualification or mechanistic interpretation is earned.
