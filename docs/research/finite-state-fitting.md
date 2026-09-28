# Learning emission probabilities and paying for encoding rules

2026-09-27. Development foundation for the [blind-channel design](blind-channel-recovery-design.md).
No sealed benchmark, corpus fitting, or Voynich decoding is reported here.

Before implementation, reviewed [Chiang et al.2010, section2](https://aclanthology.org/N10-1068.pdf):
forward/backward yields fractional transition counts and normalizes competing
transitions; the paper also documents local optima and dependence on supplied
transition inventories. We use this ordinary fixed-support EM calculation, not
its approximate Bayesian sampler. Its success on specific tasks does not
establish recovery of an unknown variable-length historical encoding.

For each observed record y, compute posterior expected counts of every
emission alternative j in row r=(state, source-character). Sum counts across
records before normalizing: q_new(r,j)=sum_y E[N_rj|y]/sum_y E[N_r|y].
Keep zero-occupancy rows unchanged. The source, stop law, initial-state law,
state count, emission support and glyph strings are fixed during these updates.
No smoothing or probability floors are added. Zero probabilities cannot revive;
positive alternatives may become zero. Exact EM improves observed likelihood
up to floating-point error but neither escapes all local optima nor guarantees
the right plaintext. Updates return the last completely scored model on a
deadline; checks occur between records, so one exact record evaluation can
overrun the requested limit. No truth or chosen parse is a fit input.

For later structural comparisons, floating weights are explicitly quantized
onto a shared denominator D. Duplicate (destination,string) alternatives merge,
exact zeros are removed, and each retained alternative receives a positive
integer count. Initial-state weights may be zero. Deterministic residual-based
rounding enforces sum D. This operation is separate from EM, may lower
likelihood, and is not advertised as optimal quantization. All final candidate
data scores must be recomputed using their actual coded grid weights.

The implemented binary model description is conditional on a fixed context:
ordered source/glyph alphabets, candidate source list, denominator, maximum
states/alternatives/emission length, and fixed stop law. These cannot be changed
for free while comparing candidates. It writes:

1. A fixed-width candidate-source index and state-count header.
2. Initial-state weights using the lexicographic rank of a nonnegative integer
   composition of D; there are binomial(D+S-1,S-1) possibilities.
3. For each state/source-character row, a bounded alternative-count header.
4. Every alternative's next-state index, bounded glyph-string length, and each
   glyph's index in the observed alphabet.
5. Row weights using the rank of a positive composition of D into m parts;
   there are binomial(D-1,m-1) possibilities.

A field with K possibilities uses ceil(log2 K) bits; singleton fields use zero
bits. Headers fix all later field lengths and the number of rows, so a complete
description is self-delimiting under the shared context. Unused binary field
values are reserved. Arbitrary state names have no meaning and are represented
by ordinal indices. Distinct state/support orderings may describe equivalent
models; shortest-found syntax is not proof of global minimality.

Candidate score is actual description length minus log2 p(observed records),
where p sums every latent path with the quantized channel. It charges literal
emission strings and weights rather than a tunable parameter-count coefficient.
This is an explicit conditional two-part code, not a Bayes factor, not optimal
universal coding, and not a semantic truth criterion. The caller must associate
the chosen source index with the predeclared source model; the numerical API
does not contain the full external source catalogue.

Tests independently cover exact small EM updates, aggregate count weighting,
monotonicity, unchanged unvisited rows, zero support, integer composition ranks,
prefix-free complete tiny model families and actual score accounting. Such
checks qualify the implementation only. Structural moves, empirical source
calibration, throughput estimates, sealed author/key/family splits and negative
control decisions remain necessary before blind-recovery claims.
