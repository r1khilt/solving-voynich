# Read messages from posterior key particles without conditioning twice

Own derivation and bounded finite interface qualification, developed while TEMPERED-RECOVERY-001 runs. The running experiment, its selected-key reader, frozen source and all prospective criteria remain unchanged. This supplies an interface for a separately registered future message-level comparison; it does not claim improved empirical recovery or a Voynich decipherment.

## Prior work and what is already implemented

The [shared-key mixture decoder](shared-key-mixture-decoding-2026-09-30.md) already sums compatible key mass for an entire tuple of records. The [decision-risk memo](decision-risk-for-decipherment-2026-09-30.md) already distinguishes key coverage, inference, source inadequacy and the decision loss. Repeating those implementations would miss the actual interface problem: a posterior particle population is different from a prior-weighted candidate bank.

[Kumar and Byrne 2004, §3](https://aclanthology.org/N04-1022.pdf) motivates choosing outputs under task loss; practical hypotheses and probability estimates remain restricted. [Ravi and Knight 2011, §2.2](https://aclanthology.org/P11-1002.pdf) uses Bayesian decipherment and then a final fitted-channel Viterbi decision. Its word-substitution/English-bigram setting does not validate our variable-length, duplicate-allowing dictionary or posterior mixture. [Ichihara et al. 2025, §§3.2–3.3 and Appendix B](https://aclanthology.org/2025.acl-long.793.pdf) separates sampling error from model-distribution mismatch. Its main theorem assumes an embedding inner-product utility and smoothness; iid reference assumptions also matter. We do not import its rate as a guarantee for edit distance or correlated SMC particles. Read depth: these targeted sections, not independent replication of the reported experiments. Previous Voynich method/scope review remains in the frozen [fresh recovery memo](tempered-recovery-2026-10-01.md).

## The same-observation interface

Let C be the entire observed record tuple, Q(X) its declared source/stopping/reset law, and I_K(X,C) the indicator that one shared dictionary K re-encodes every record. Define L_K = sum_X Q(X) I_K(X,C). Let μ_K be empirical mass in an equally weighted final resampled population: multiplicity(K)/N. Duplicate particles carry mass and must not disappear during deduplication.

A conditional reader driven by this empirical posterior must represent

    P_μ(X|C) = sum_K μ_K Q(X) I_K(X,C) / L_K.

The existing shared-key decoder instead accepts weights w and conditions them on C, giving Q(X) sum_K w_K I_K / sum_K w_K L_K. Therefore use

    A = sum_K μ_K / L_K,
    w_K = (μ_K / L_K) / A.

Then its derived evidence is 1/A and its decoded conditional distribution is exactly P_μ in ideal arithmetic. This evidence is a diagnostic identity of the derived finite mixture, not the original-family evidence estimate. Every likelihood must cover the same whole records and exactly the same source, stopping, resets and emissions as the reader. Future empirical admission must bind those identities; the pure conversion helper cannot verify a caller's source/cipher hashes.

Using w=μ reconditions the key population to μ_K L_K / sum_J μ_J L_J. If μ were the exact π(K|C), this would weight the original prior by L_K squared. Deduplicating and assigning equal mass changes μ differently. Neither shortcut preserves the intended empirical posterior. Conversely, a bank of prior proposals or optimization visits is not automatically a posterior particle population and does not qualify for this adapter. For new transfer records D, the correct law contains L_K(D), with μ from C; dividing by the old likelihood and pretending C equals D is wrong.

The new bounded helper retains first occurrence and multiplicity, requires finite nonpositive complete log likelihoods, refuses duplicate-score disagreement, and uses a minimum-score anchor before adding log masses. This avoids erasing multiplicity when every score has a huge common negative offset. Numerical normalization is still floating arithmetic, not an interval certificate. Population size is 1–128 and dictionaries 1–64 nonempty rows. It accesses no source, ciphertext, fitted archive, Gold or neural weights.

## A concrete joint-record witness

Two source letters a,b have probabilities 2/3,1/3, geometric stopping 1/3, two records both equal glyph0. Keys K1=(0,1), K2=(1,0), K3=(0,0) have full-record likelihoods 16/729,4/729,4/81. An artificial six-particle population has multiplicities1,2,3; it is a defined empirical measure, not a sampled-calibration claim.

Correct conversion gives decoder weights3/31,24/31,4/31. Its message probabilities are aa:7/18, bb:7/18, ab:1/9, ba:1/9. Feeding multiplicity as a new prior instead gives aa:16/33 and bb:5/33. This is a wrong distribution even with perfect inference and the same language model. The test checks this through the real existing decoder and independent literal rational enumeration.

## What additional samples can and cannot repair

These are project derivations, not new theorems attributed to the cited papers. The conditional kernel H_K(X)=Q(X)I_K/L_K is a Markov kernel. For exact model key posterior ν, total variation contracts:

    TV(P_μ, P_ν) ≤ TV(μ,ν).

Proof: interchange sums and use sum_X H_K(X)=1 in half the L1 norm. For any valid whole-tuple action a, summed edit distance lies in [0,G], where G is total observed glyph count, because every emission is nonempty. Hence |R_μ(a)−R_ν(a)| ≤ G·TV(μ,ν). A risk winner under μ has at most 2G·TV(μ,ν) excess model risk over the best action in the same fixed set. This bound does not make the unknown TV small; poor key coverage can dominate any decision improvement. A mismatched source/channel creates an additional unmeasured distribution error.

Conditional on a fixed empirical bank μ and an independently constructed finite action set of M admissible tuples, iid references sampled by drawing K~μ and then exact conditional paths give, by Hoeffding and a union bound,

    ε = G sqrt(log(2M/δ)/(2R)),
    max_a |estimated R_μ(a)−R_μ(a)| ≤ ε

with probability at least1−δ for R independent reference tuples. The sampled action winner then has excess ν-risk at most2ε+2G·TV(μ,ν) within that action set. Original SMC particles need not be iid: conditional sampling must use fresh independent draws from the fixed empirical measure. This certifies Monte Carlo estimation under that measure, not full-posterior or historical calibration. Candidate and reference streams must be separate. An action must re-encode under one shared key; independent record medians are not necessarily admissible.

## Finite checks and prospective use

The finite checker fixes both order-zero and contextual positive two-letter sources, all36 dictionaries from six one/two-glyph units, and every ordered pair of the15 binary records of length0–3:450 cases. It literally enumerates all compatible source tuples, calculates exact likelihoods/conditional laws with Fractions, retains deterministic unequal particle multiplicities, checks inverse-likelihood reconditioning and normalization, verifies TV contraction against the complete-model posterior, and compares the existing floating mixture decoder's maximum and evidence at tolerance1e-12. Empty records, duplicate emissions, overlapping units and shared cross-record assignments are included. It reports naive double-conditioning discrepancies rather than claiming each case must differ. 120wall/100absoluteCPU/512MiB/CPU1/$0, no original corpus/model or empirical campaign archive is opened. The saved receipt binds this helper, checker, tests, existing decoder and memo. One save; no empirical rerun.

The live recovery experiment stays the registered selected-key baseline. A future post-fit read comparison needs its own frozen prediction plan, bounded original-source costs, all-case failure accounting and new fresh qualification after adaptive choices. Compare selecting one key, posterior-mixture text MAP and raw-edit-risk actions while preserving one shared key, then diagnose missing support versus wrong source preference. We have not implemented an order-12 posterior-path sampler or run this adapter on original-source particles. The finite interface result does not establish feasibility at that scale or solve the missing-key problem.
