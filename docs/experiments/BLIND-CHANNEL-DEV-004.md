# BLIND-CHANNEL-DEV-004: separate source context from decoding decisions

2026-09-28. Registered before new source-selection or cipher scores. This is
an **adaptive diagnostic on the exposed DEV001/003 B cases**, not confirmation,
new key discovery, language identification or a manuscript reading.

## Question and rationale

DEV003's learned B2 channel beats the generating description by 30.779 fit bits
but has 32 rather than 21 transfer edits. Increasing search alone need not fix
that preference. Hold the dictionaries fixed and separately change (a) the
source model's context and (b) the final reading decision. This can distinguish
useful interventions without hand-correcting exposed letters.

[Nuhn et al. 2013](https://aclanthology.org/P13-1154.pdf), especially the abstract
and experiments, shows that exact optimization of a weaker character model can
read worse than approximate search under a stronger one. Its supplied-unit
substitution task differs from our ambiguous unknown-unit family. We test this
direction, not their numerical result or an assumed Voynich cipher.
[Kumar and Byrne 2004](https://aclanthology.org/N04-1022.pdf), section 3, separates
the highest-probability whole output from minimum expected task loss. Their
translation system differs from this cipher; our loss is literal character
Levenshtein distance and our posterior comes from an explicit channel.
[Reddy and Knight 2011](https://aclanthology.org/W11-1511.pdf) supplies the existing
Voynich statistical context, not evidence for a Latin source or this codebook.
See also [the blind-channel design](../research/blind-channel-recovery-design.md)
and [DEV003 results](BLIND-CHANNEL-DEV-003-results.md).

## Fixed inputs and isolation

Use the unchanged `data/manifests/blind_channel_dev001.json`
(SHA-256 `8587ae6010de94996bdd57d74ea2e1eee3f71940cdb13c1cc3125adfdfc9acea`),
old source (`7de1c233b3ea69eb632247062f97db86040666daa13ba73bba28bbb51b3493b0`),
and the four DEV003 learned dictionaries frozen at
`8e6e5c8d17bf4bdd568f22a2e1df22d12972b1ac`.
The two positive generating channels are evaluator-only diagnostic arms.
Preserve the two paired within-record glyph shuffles. Use all four fit and two
transfer records per case, unchanged rho=1/225, alphabet, and code costs.
No dictionary refit, search, seed reroll, rare-letter correction or selective
record exclusion is permitted in this diagnostic.

Train source candidates on the pinned first 50,000 eligible Caesar letters;
select on the pinned first 50,000 eligible Virgil letters. Preserve body
boundaries and reset source context at them. Never read Cicero plaintext for
source fitting or selection. Never open Sallust/Tacitus or generate stateful C.
Source-only validation has been used previously: it is development, not a new
independent measure of language-model generalization.

## Source-context arm

Candidate order list is 0, then orders 1, 2, 3, each with total Dirichlet mass
tau in [0.25, 1, 4, 16, 64, 256]. Base character counts receive +0.5 each.
For each nonempty context h, recursively interpolate its next-character counts
with the shorter suffix row using total mass tau. Count only within records;
unseen contexts back off completely. All probabilities stay normalized and
positive. At a record start use the unigram; grow the context to the maximum
order. Order 1/tau64 must reproduce the old source after refitting on Caesar
plus Virgil, within numerical tolerance.

Report every candidate's Virgil bits/letter. Stable listed-order ties. Choose
the best configuration per order and the global best solely on that score,
then refit the selected per-order models on Caesar+Virgil (100,000 letters).
The primary source-context comparison is old source versus global winner;
order-specific scores are a declared sensitivity panel. Preserve all selected
models and hashes before any new cipher inference. The actual source-model
tables may stay in ignored outputs with a tracked manifest.

For each fixed learned/positive-gold channel and each old/per-order-selected
source, compute the exact unpruned marginal and most-probable plaintext.
Deterministic units give one path per plaintext here. Report fit data bits,
unchanged literal model bits, transfer likelihood, edit distance, decoded
length, exact records and conditional MAP surprisal. Source catalogues are
fixed before inference: comparisons are conditional sensitivity analyses, not
an uncharged search over sources or Bayesian language evidence.
The unchanged one-bit selector distinguishes the channel family from the old
glyph baseline; it is not payment for choosing among these source models.
No source is chosen using target ciphertext scores.

## Decision-rule arm

Under the old source only, compare its exact MAP reading with an approximate
minimum-edit-risk decision. Draw 32 candidate plaintexts and an independent
256-reference risk bank from the exact fixed-channel posterior using backward
likelihoods and forward conditional sampling. Add MAP to the unique candidate
set. Choose the candidate minimizing average unnormalized character edit
distance; preserve multiplicities in the risk bank and deterministic ties.
This gives a candidate-set Monte Carlo decision, not an exact global Bayes
optimum or an unbiased selected-risk estimate. Posterior risk need not track
true errors when the source/channel is wrong.

Use seed 61101 +1000*case_index +100*split_index +record_index, case order
`B-key1`, `B-key1-shuffle`, `B-key2`, `B-key2-shuffle`, split order fit/transfer.
Add 10000 for positive-gold arms. Derive disjoint candidate/risk RNG streams
deterministically inside the decoder. No reruns to pick a favorable seed.
Report both decisions, all candidate risks, bank hashes/counts, and actual
edits after prediction freeze. Nulls receive no accuracy or semantic threshold.

## Answer-phase dictionary support diagnostic

Before results, additionally fix an evaluator-only lower bound: the minimum
Levenshtein distance between the true text and **any** plaintext that exactly
encodes to the observed record under each fixed dictionary. Use the acyclic
product graph of ciphertext offset and true-text offset. Edges delete a true
character, emit a compatible source character, or do both with match/substitution
cost. All units are nonempty. Exhaustive tiny plaintext enumeration must agree.
An unsupported ciphertext returns no bound, not a finite error count.

This diagnostic reads answers only after prediction freeze and never fits a
dictionary or chooses a source. A positive minimum isolates errors that no
reading decision under that dictionary can avoid. A zero minimum establishes
support, not that any particular language model should rank truth first.
The bound ignores source probabilities; here the trained sources are positive,
but for arbitrary sources with zeros it would be a relaxed support bound.

## Interpretation rules fixed before results

- Better oracle readings with richer source context support a source/decoder
  limitation in the old oracle; this is not a change to the learned key.
- Better learned readings with richer context show a useful decoding repair
  conditional on that existing dictionary, not new blind recovery.
- A generating channel regaining better fit score under a source chosen without
  answers reverses the previously observed pairwise objective mismatch. It does
  not establish its global optimality or guarantee a new search finds it.
- Lower true error under the sampled-risk decision supports that intervention
  on these cases; lower estimated risk alone is an optimization tautology.
- Poorer results or unchanged errors remain reported. No new pass threshold,
  language choice, model choice or manuscript claim follows from this panel.

## Bounds, validation and publication

At most two single-thread CPU processes, 8 GiB aggregate planning bound,
45 CPU minutes for selection, predictions, evaluation and independent replay,
and 20 minutes wall deadline per inference stage;
no paid compute/API use. Enforce disposable-process CPU limits, finite model
order/context caps, bounded draws and no output overwrite. Report actual CPU,
wall and peak RSS; stop on hash mismatch, inference mismatch or resource failure.

Before empirical work, compare higher-order marginals and MAP with independent
tiny exhaustive sums, order1 with the existing engine, and posterior sampling
with enumerated probabilities. Verify bit-vector edit distance against a simple
independent recurrence. Independently audit source estimation and record resets.
Freeze code/protocol, then source selections, then predictions before reporting
new correctness. Full predictions/banks stay ignored; track hashes and compact
results. Publish negative findings too, update the notebook, and retain all
earlier failed and successful development results.
