# Contextual row conditionals: exact local inference, possible global barriers

2026-10-01. Source review, own mathematics and prospective rationale for [CONTEXTUAL-GIBBS-001](../experiments/CONTEXTUAL-GIBBS-001.md). No actual contextual sampling outcome is known at registration.

## Evidence motivating the comparison

[CENSORED-CONTEXT-001](../experiments/CENSORED-CONTEXT-001-results.md) qualifies exact observation-prefix likelihoods for fixed dictionaries under the original1,447,724-state contextual source. Closed IID banks have substantial context-induced ranking inversions and concentrated correction factors. Those observations motivate targeting contextual likelihoods throughout search; they do not guarantee that random initialization or local updates will recover a key.

Primary review: [Ravi and Knight2011](https://aclanthology.org/P11-1025.pdf), selected§3.1 especially type-sampling paragraphs on PDFpages6–7, revisited2026-10-01. Their type updates consistently change every occurrence of a cipher type, avoiding inconsistent per-token choices; their source combines English letter and word information. Our variable-emission direction, all-source-path marginalization, uniform dictionary prior, Latin contextual source and absence of known word boundaries differ. Their reported decipherments or sampling speed cannot be transferred. No exchangeability-based incremental score or dictionary, sparse channel prior or annealing is borrowed here.

The earlier [primary SMC review](dictionary-smc-2026-10-01.md) revisits DelMoral/Doucet/Jasra2006 §3.3.2.3eq30–31: invariant mutation is compatible with the specified sequential target when incremental weights are evaluated before mutation and support requirements hold. We use shrinking observation-prefix support and the already-tested censoring bridge. Earlier [Voynich prior-work caveats](native-censored-context-2026-10-01.md) remain: word/anagram and language-score plausibility cannot establish a decipherment. Read depths are selected sections, not full-paper audits. An attempted Neal1993 report PDF open returned a tool InternalError; its text was not used as evidence.

## Own conditional kernel and finite checks

For a positive dictionary K at a fixed stage, choose a free source row r uniformly. Under the uniform conditional dictionary prior, enumerate ALL legal codes u and sample with

q(u | K_except_r) = G_t(K_with_r=u) / sum_v G_t(K_with_r=v).

The old positive code is in the denominator, so it is positive. Row selection is state-independent; known bindings never change. For two positive keys differing only at r, both denominator and row-selection probability are shared. Thus G_t(K)P(K→K') = G_t(K')P(K'→K), including the mixture over possible self transitions. This is target invariance, not exact full-key sampling. Negative-infinite true likelihoods have zero conditional probability; graph caps abort and never become negative-infinite likelihoods.

Preparation tests enumerate all49two-row partial dictionaries across the contextual observation schedule.124positive exact Fraction target/kernel matrices and8,349detailed-balance pairs are checked, with every row sum1 and weighted normalizer transport at every positive stage. Actual sampled conditional choices are independently checked against source-string sums; known rows, complete-key likelihoods, fixed-key telescoping, same initial bank/first weights, work guards and literal extinction are tested. Additional complete16-call toy-source run/audit and forced-native-cap controls validate transport, scoring and failed-gate semantics. These are finite mathematical and implementation checks, not large-corpus recovery or convergence evidence.

## A concrete barrier

Take two source letters, two single-glyph closed observations0and1, and legal emissions of one or two glyphs. Exactly two dictionaries support both observations: (0,1) and (1,0). Holding either row fixed makes the other row's conditional a point mass. Random-row Gibbs cannot swap the two rows, even though both dictionaries have positive likelihood. Independent exact enumeration and an actual update test verify this witness. With23rows there may be spare rows and alternative paths; the witness does not prove that our large-case support is disconnected. It does disprove a generic irreducibility or guaranteed mixing claim for this algorithm.

Observation-prefix SMC can retain alternatives before later constraints harden, but finite resampling can discard them. The experiment records incremental ESS, maximum weight, distinct parents, mapping changes and conditional positive-code counts. These algorithmic mechanisms are relevant to search; they are not neural circuits or meanings learned by a competent linguistic model.

## Work matching and restrictions

One Gibbs row update costs42fixed-key target tables per particle, including recomputation of its old code. The control performs42uniform-code random-row MH updates per particle. Both spend43tables per particle/stage including observation weighting when complete. They change different numbers of rows and consume different source graphs, random draws and CPU time. A success would establish improvement at matched target-table counts only, not matched CPU or a universal Gibbs preference.

All four input keys/strings were exposed in earlier development. A fully unknown initial dictionary is an improvement in the current task setup; it does not make these fixtures fresh or blind. Best-key selection is by contextual likelihood alone. Ground-truth used-row matches are scored afterwards, not supplied to fitting. This comparison cannot establish language identification, semantic reading, neural competence, historical validity or Voynich decipherment. Global joint proposals/source-path data augmentation would require their own inference law, finite controls and registration if row conditionals fail.
