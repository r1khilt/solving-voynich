# SUFFIX-READER-001: longer source context with exact ambiguous-unit decoding

2026-09-30. Prospective development comparison, registered before source
selection or new cipher scoring. The preceding CONFIRM001 remains failed and
is not accessed or rerun. This experiment supplies the correct keys to isolate
reading competence; it does not qualify blind key discovery or Voynich reading.

## Question, previous work and architectural choice

CONFIRM001 made142/3,584 character edits even when its generating keys were
supplied. Can a stronger source reduce this component of error? Existing
DEV004 showed that extra source context can help, but its dense history table
stopped at order3. This experiment raises the possible history to12 letters
while preserving exact sums and MAP readings under each frozen source.

Reviewed primary work:

- [MacKay and Peto1995](https://www.cs.toronto.edu/pub/gh/MacKay%2BPeto-1995.pdf),
  source paper model/Dirichlet sections: motivates count-dependent interpolation
  toward a lower-order distribution. Our fixed recursive plug-in estimator is
  not their full hierarchical inference or integration over uncertain parameters.
- [Kambhatla et al.2018](https://aclanthology.org/D18-1102.pdf), methods and
  experiment sections: richer language models can improve supplied-symbol
  substitution search. Their4096-unit mLSTM uses English Gigaword plus Zodiac
  letters and approximate beam scoring. That evidence does not qualify an
  unknown-unit Latin or Voynich decoder. Here the intervention isolates context
  with exact finite-state inference before adding neural proposals or search.
- [Hauer and Kondrak2016](https://aclanthology.org/Q16-1006/), abstract and prior
  project review: source-language screening and anagram/substitution hypotheses
  were applied to Voynich. A favored language score or plausible decoding does
  not establish the language or reading. This test assumes Latin explicitly.

An attempted full-text retrieval of Cleary/Witten1984 timed out; the retrieved
abstract is background only. No PPM algorithm or new PPM-specific claim is
implemented. See the existing
[inference review](../research/blind-channel-next-inference-review.md) for
related cipher methods and the distinction between stronger scoring and search.

## Model and exact-state argument

Count contexts only where a following character exists, resetting at each
declared corpus body. Retain every observed context of length0..D. The count
inventory is prefix- and suffix-closed. At the root use add-one-half counts.
For nonempty context h use

`p(c|h) = [count(hc) + tau * p(c|suffix(h))] / [count(h) + tau]`.

An unobserved context has no counts and exactly equals its shorter suffix.
The finite state is the longest observed suffix of the full plaintext history.
After a character is appended, take the longest observed suffix again. This
state preserves future probabilities: a longer newly matched context would
have an observed prefix before appending, which contradicts the previous
state being longest. Prefix closure is required and checked. This is an exact
representation of the specified source, not a beam or approximate history merge.

Decode with sum-product and max-product dynamic programming over observed
offset × source state. Nonempty deterministic units make the graph acyclic.
Retain geometric stopping probability1/225, including its length preference;
the actual224-letter length is not supplied as a decoding constraint. Duplicate
units, impossible records and empty strings are supported by the implementation
and tested, although the chosen generator provides all six singleton units.
Known-key deterministic emission gives one observed string per plaintext, so
the best path is a best plaintext here. No claim for ambiguous latent-path
stochastic channels follows.

## Frozen data, choices and execution order

Use the unchanged pinned `blind_channel_development_corpora.json`:50,000
letters each of Caesar, Virgil and Cicero, alphabet`abcdefghiklmnopqrstuxyz`,
existing normalization and body-boundary resets. Source identities are checked
against the manifest. No acquisition, additional text or target-author training.

1. Publish/remote-verify this protocol, source, runner, references and tests.
2. In one preparation invocation, train16 candidates on Caesar: maximum history
   D in3,5,8,12 crossed with tau in4,16,64,256. Score Virgil bits/character.
   Select the minimum, ties favor lower D then lower tau. Report all candidates.
   Refit just that selection on Caesar+Virgil. Preserve the old order3/tau256
   source unchanged as baseline; compare its complete table to the new sparse
   representation. Record source selection before accessing Cicero.
3. Prepare16 new B keys with seed88129+104729*i, i=0..15, using the frozen
   original generator:6singletons plus17different digrams over ABCDEF, randomly
   permuted onto23letters. Correct dictionary supplied. Each key gets two
   Cicero windows, starts40000+512*i and40256+512*i, length224. All fall within
   the third body[35839,50000); verify no overlap with previous DEV001 cipher
   windows. No redraw, dropped record, alternate passage or source retuning.
4. Publish/remote-verify source selection and the entire known-key panel manifest
   before decoding. Run both old and selected sources on all32records; preserve
   completed per-case archives, caps/failures and total resources. Compare all32
   selected readings with a separately written reverse recurrence and direct
   path re-encoding/scoring; compare32baseline results to its existing independent
   reference. The new reference is root-authored, not a second researcher.
5. Publish/remote-verify predictions before one accuracy evaluation. Retain all
   cases, support, decoded lengths, raw edit counts, exact-record counts and
   source scores. Source/model cap or timeout is an explicit experiment failure;
   do not prune, silently replace the model or report a selected subset as passed.

The panel uses new keys and new scoring windows for this cipher pipeline, not
a newly acquired author or globally unread plaintext. Cicero's entire pinned
text was already processed for corpus provenance. Isolation is procedural;
public seeds and known-key construction are not secrecy. This is development
evidence on one target author, not16 independent language/corpus replications.
Do not inspect or adapt to CONFIRM001/Sallust/Tacitus results in this experiment.

## Metrics and prospective decision

Total7,168 true letters. Report every key's448-letter aligned character error
and both exact-record counts. All three reader gates must pass: at least25%
relative edit reduction from the frozen baseline, overallCER≤2%, and each
keyCER≤5%. Use integer arithmetic for boundaries. A baseline of zero errors
requires zero selected errors; it does not imply a positive relative gain.
Unsupported readings would retain full-deletion penalties; with this supplied
complete dictionary and positive source they also indicate an implementation
failure. Never reinterpret a failed gate as confirmation by dropping a key.

This known-key comparison has no mechanism discovery or language-identification
claim. It does not establish rejection of structured nonsense, transcription
robustness, support protection for learned keys or historical cipher validity.
Those remain required before any manuscript application. A lower source
validation loss alone is insufficient; actual new-cipher errors decide the
declared reading gate. If it fails, preserve the failure before another method.

## Bounds, checks and records

One CPU thread. Preparation1200CPU/1800wallseconds; prediction1800CPU/2700wall;
evaluation120CPU/180wall. Total hard CPU limits3120seconds, under oneCPUhour.
Planning peak memory8GiB;1.2million source-context cap and500,000 lattice nodes
per record, fail without pruning. No paid APIs, neural training or new downloads.
Record actual wall/CPU/RSS. No resource extension or automatic rerun.

Before empirical execution, test sparse-state equivalence against uncompressed
rational history probabilities, all short observed strings under complete
plaintext enumeration, the old dense source at orders0..3, record boundaries,
normalization, aliasing, empty/impossible records, malformed inputs and cap
failures. Check independent counts by fixed-length substring counters. The
empirical reference independently constructs string states and recurses from
the end; direct full-history scoring verifies returned paths. Agreement
tolerance1e-7; no approximate pruning. Keep raw/large models and predictions
ignored, compact checksums/results tracked. Update notebook/memory, run relevant
checks and full regression, commit and remotely verify coherent checkpoints.
