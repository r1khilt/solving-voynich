# Shared-prefix neural proposals: implemented core and qualified tiny cases

2026-09-30. Engineering development while the separately frozen global
fitting campaign runs. **No trained Latin model, empirical cipher panel,
transfer/answer file, historical transcript or new model training is used.**
The current GLOBAL-KEY-READ-001 comparison remains unchanged.

## Why this component is necessary

The completed KEY-BANK-NEURAL-002 exposed comparison found that neural
rescoring can prefer useful texts, but cannot select true tuples absent from
the statistical proposal union. The new component lets the neural source
construct its own literal-channel-consistent candidates within a supplied,
frozen dictionary bank. It does not discover a missing dictionary, identify a
language or fix a misspecified source. Its scored object is one text tuple with
summed compatible-key mass, not sum over all text tuples or exact evidence.

Prior method/Voynich scope was reviewed before implementing the existing
[derivation](neural-shared-prefix-search-2026-09-30.md); primary PDFs were
rechecked during this implementation and before its publication.
[Nuhn et al. 2013](https://aclanthology.org/P13-1154.pdf) provide substitution
beam-search precedent. [Kambhatla et al. 2018](https://aclanthology.org/D18-1102.pdf),
section3.2, augment partial decipherments with sampled neural completions as
a score estimate; that is not our admissible bound. Their section4 uses a
4096-unit mLSTM, English Gigaword plus Zodiac correspondence and much larger
beam budgets. Those data/assistance/scales do not transfer to our Latin task
or establish an accepted Voynich decoding.
[Huang et al. 2017](https://aclanthology.org/D17-1227.pdf), section3, justify
completion stopping because future probabilities cannot increase a prefix's
score, with optimality conditional on prior beam pruning. Section4 changes the
bound for a bounded length reward. We retain our fixed geometric length law,
keep every dropped branch's bound, and make no interval-arithmetic certificate.
The original [qualification review](blind-channel-fresh-qualification-review.md)
records corpus/source assistance, ambiguous emissions and historical limits.

## Implemented invariant and mathematical score

New `src/voynich/recurrent_shared_prefix.py` takes distinct dictionaries with
normalized finite log weights, deterministic nonempty units, observed records
and a normalized positive full-history causal provider. A hypothesis retains
completed record strings, exact current record prefix, source log factor and
source-offset/key-mask groups. Each key occurs at its unique emitted offset;
keys intersect on each letter and remain shared across every record.

For a full text tuple x, the scored quantity is

```text
S(x) = source_log_probability(x with independent record resets)
       + total_source_letters(x)*log(1-rho) + R*log(rho)
       + logsumexp(log_weight[k] for globally compatible k).
```

A minimum-length dynamic program independently checks channel reachability
and minimum remaining source letters for every key/current observed suffix,
plus future records. Impossible suffixes remove keys exactly. Unfinished
hypotheses use the derived bound

```text
U(h) = source_prefix_log_factor + R*log(rho)
       + logsumexp(log_weight[k] + min_remaining_letters[k]*log(1-rho)).
```

Future source probabilities are at most1; actual remaining length for any
compatible descendant is at least that minimum. Bound each compatible key
contribution separately before summing. The result is no greater than the
basic bound omitting minimum length. This elementary argument is ours under
these assumptions, not a theorem copied from the cited decipherment papers.

Each source-letter depth performs complete zero-letter boundary closure before
beam pruning, retaining simultaneous end-record and extend-record branches
when different keys permit both. BOS resets, including empty records, add no
extra source-letter likelihood; stopping factors enter exactly R times. The
same completed text may have several compatible keys, whose mass is added
once. The beam ranks the tightened bound with deterministic textual ties.
Every dropped prefix's bound contributes to a retained maximum. Expansion,
channel-cell and source-cache caps raise explicit errors, never produce zero
probability or a fake bound. With nonempty emissions total source depth is at
most total observed glyph length. No new dominance pruning is implemented.

## Representation equality is stronger than cosine or next-token agreement

Source transitions are cached only for identical **full current record
prefixes** under the same fixed, reset provider. Completed earlier records do
not affect this source after reset. Different text tuples are still distinct
candidates; sharing a source transition does not merge their probabilities.

A new positive counterexample supplies histories `a` and `b` with exactly the
same next-letter distribution(.5,.5), but the same following input `a` makes
their subsequent distributions differ. The winning four-letter tuple begins
with the initially less probable `b`. Thus equal one-step readouts cannot
justify merging histories or establish a shared latent algorithm. In general,
cosine ignores magnitude: collinear states can enter different nonlinear
transition/output regimes. Neither fact proves anything about the trained
Latin models' actual neurons. A causal representation claim still needs
matched state interventions, controls and seed stability on those models.

## Completed engineering checks and limits

Independent enumeration checks all120two-key banks drawn from16binary
emission dictionaries, all16two-record observations including empty/impossible
records:1,920problems. All complete scores, best tuples and compatible keys
agree. Separate exhaustive full-text enumeration verifies every issued
retained/discarded bound for narrow and wide beams, including mixed unit
lengths and duplicate rows. A neural greedy trap makes beam1 return a wrong
candidate with an unseparated bound; a wide beam reaches the exact tiny best.
Shared-key impossible records, weight−10000 support, reset/stop accounting,
malformed provider distributions and all resource-cap failure paths are tested.

Two independently whole-sequence-scored tiny random LSTMs (seeds71021/71029,
embedding5/width7/two layers) agree with cached stepwise search within2e-6nats
and produce the exhaustive best. These are CPU random-weight arithmetic tests,
not learned-language recovery, MPS memory qualification or 7.4M-model evidence.
The next-token equality counterexample is a constructed source, not a causal
intervention on a trained model. No semantic mechanism is claimed.

Full-size hidden-state cache cost and variable-batch Metal allocation remain
unmeasured. Channel preprocessing scales with key count, record lengths and
unit inventory; the tighter bound can itself dominate runtime. Before trained
Latin use, register a bounded artificial systems benchmark measuring exact
counters, memory, repeated-prefix sharing, fixed-shape alternatives and
whole-sequence/CPU score agreement. Then freeze all cases/both neural seeds and
reading limits for one exposed comparison with explicit candidate-coverage and
ranking diagnostics; publish predictions before answers and require fresh
unused qualification afterward. No trained run or speed/recovery claim has
been authorized by this engineering result alone.
