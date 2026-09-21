# Recovery measurement audit and additive diagnostics

Date: 2026-09-21. Applies to `codex/research-audit-fixes`, based on main
`3e956c8`, whose EXP-0012/0013 are the null-aware/CTC experiments. The separate
`codex/episodic-rule-recovery` branch currently uses the same IDs for different
studies; these changes do not concern that running campaign.

This is an evaluation implementation correction, not a new scientific result.
Historical `recon_acc`, `pred_bits_gain`, default random draws and pass functions
are preserved. No old FAIL becomes PASS, and no manuscript is scored here.

## Review before implementation

Reviewed the supplied external audit, `latent_recovery.py`, its tests, the
EXP-0011–0015 registrations, project protocol and prior-work review.
The prior-work review distinguishes known-language alignment/decipherment
methods from unknown manuscript recovery; a plaintext character LM cannot
meaningfully score fresh randomly assigned ciphertext IDs without a mapping.

Method references checked on 2026-09-21:

- [NIST SCTK alignment documentation](https://raw.githubusercontent.com/usnistgov/SCTK/master/doc/sclite.htm)
  explains dynamic-programming alignment with insertion, deletion and
  substitution costs. Our existing character metric uses unit costs and
  normalization by maximum string length; it is not NIST's weighted word-error
  scoring. Alignment avoids cascading positional errors but can also match
  repeated glyphs; it does not prove that the correct positions were selected.
- [Jurafsky and Martin, N-gram Language Models, draft August 19, 2026](https://web.stanford.edu/~jurafsky/slp3/3.pdf)
  explains conditional normalization, sequence probability and separate
  training/evaluation. We use positive additive smoothing over a fixed support
  and an initial-symbol distribution. Without an end/length model, normalization
  is over strings of each fixed length, not over all variable-length strings.

## Finding 1: positional recovery is offset-sensitive

The audit correctly identifies a mismatch between positional agreement and
approximate sequence recovery. Dropping `c` from `abcdefghij` gives legacy
positional score about 0.20 but edit similarity 0.90 and exact recovery false.
The positional calculation is defined and reproducible; calling its arithmetic
broken is too broad. It is a poor sole description of an insertion/deletion
task. Edit similarity already existed as secondary `recon_edit_sim` before this
audit, while the random baseline omitted it. Exact-sequence recovery was
mentioned in the original EXP-0011 registration but was absent from its reports.

The implementation now reports all three notions consistently:

- `recon_acc`: original prefix agreement multiplied by length ratio, unchanged.
- `recon_edit_sim`: mean `1 - unit_cost_edit_distance / max(lengths, 1)`;
  unchanged for ordinary reports and now also present for random controls.
- `recon_exact_match_rate`: fraction of selected strings exactly equal to the
  target, including correct empty-string handling.
- `reconstruction_diagnostics`: sequence-mean edit distance, original length,
  target length, retained/deleted counts, aligned similarity and exact recovery.

Both ordinary and random reports use the same diagnostic helper. Random
summaries average all sequence/seed rows; since every seed visits every sample,
this equals averaging within seed then across seeds. Replicates are not new
independent tasks and no inferential interval is asserted.

Exact surface recovery still does not identify a unique mask when characters
repeat. For example, either half of `aaaa` can recover `aa`. Keep mask accuracy,
null precision/recall and retention rates alongside string metrics.

The historical `matched_random_baseline` actually deletes the **gold** number
of null positions; it does not match each model prediction's count. Its default
is preserved and labeled `matched_count_source="gold_signal_mask"`.
The new optional `reference_masks=` argument matches each supplied prediction's
retained count and labels that different policy. Do not substitute the new
policy inside an old gate or describe gold-count controls as prediction-matched.

## Finding 2: selected-string representation and deletion bias

The legacy rank bigram is already fitted on separate training strings. It does
not learn its bigram probabilities from each selected evaluation string.
The real issues are:

1. Re-ranking after deletion changes the encoding being scored. Original
   `aaaabbbc` restricted to `abbbc` has frozen codes `[0,2,2,2,5]`; re-ranking the
   retained string changes them to `[2,0,0,0,5]`.
2. Deletion can select frequent/repeated symbols. Lower retained-text surprise
   alone can reward content destruction. Fewer observations also increase
   variability; lower average surprise is not an automatic arithmetic consequence
   of shortening every string. The old `<4`-character penalty already prevents
   the most trivial empty-string win, but does not resolve selection bias.
3. The legacy model stores smoothed probabilities only for observed bigrams,
   then uses a different fallback for absent pairs. Their completed rows are
   generally not normalized. Preserve these historical scores as proxies;
   do not reinterpret them as proper raw-symbol log likelihoods or code lengths.

## New opt-in fixed-representation proxy

`fit_fixed_rank_bigram(training_texts, training_source=...)` explicitly fits a
separate `FixedRankBigram`. It requires an identified training source/split,
retains its content digest and supports serializable immutable probability
tables. The caller remains responsible for genuine training/evaluation
separation; a source label is provenance, not proof of independence.

The representation has eight rank buckets plus one space bucket by default.
Each training string is rank-encoded independently. For an evaluation input,
compute its map **once on the complete original input**, before any selection.
The candidate and every deletion control select positions in this same encoded
sequence. Ties use first occurrence, so bijective renaming of non-space symbols
preserves the encoding. Arbitrary space redefinition, homophony or segmentation
changes are not covered by that invariance.

Dense initial and transition counts receive the same positive pseudocount
(default 0.1); every row normalizes over the entire fixed vocabulary, including
unseen buckets. Sequence score includes the first symbol and records the number
of symbols and transitions. An empty sequence has zero total negative log
probability under the length-zero distribution, but undefined per-symbol score
(`null`), not a low-score success.

`fixed_rank_deletion_diagnostic(text, mask, fitted_model)` reports:

- Full and selected encoded-string scores and explicit input/retention counts.
- Twenty deterministic uniform random subsets with the candidate's **exact
  retained count**, preserving original order.
- Twenty controls that additionally preserve its **encoded bucket histogram**.
  These expose improvements due solely to selecting frequent/easy buckets.
- Every control score, mean, standard deviation, distinct sampled masks and
  `random_minus_selected` difference. Positive differences are descriptive
  comparisons, never a recovery or significance decision.

Selections shorter than four symbols, unchanged whole inputs and sampled
controls with fewer than two distinct masks are marked ineligible for this
descriptive comparison. These eligibility labels are not historical pass rules.
Histogram controls can have no score variation even with several distinct masks;
that is retained, not presented as independent evidence.

The map uses the full original string and is therefore **offline side
information**, not a prefix-causal predictor of raw ciphertext. Its map, mask and
length costs are not charged. A normalized LM over these encoded strings does
not make this a normalized likelihood over original strings or a valid complete
compression scheme. Neither control matches every layout, run or copy property;
neither corrects adaptive selection of masks or metrics after seeing results.
Correct recovery still requires independent reference/mask evidence and hard
null controls. No new threshold is proposed from these diagnostic fixtures.

Example API, with independently acquired/frozen inputs supplied by the caller:

```python
from voynich.recovery_metrics import fit_fixed_rank_bigram, fixed_rank_deletion_diagnostic
from voynich.latent_recovery import evaluate_masks, matched_random_baseline

proxy = fit_fixed_rank_bigram(train_world_b_texts, training_source="dataset/version/train-world-B")
diagnostic = fixed_rank_deletion_diagnostic(observed_text, predicted_mask, proxy)

# Existing report fields and pass functions are unchanged; the proxy is opt-in.
report = evaluate_masks(samples, predictions, legacy_rank_model, fixed_rank_model=proxy)
prediction_count_control = matched_random_baseline(
    samples, legacy_rank_model, reference_masks=predictions)
```

## Verification and limits

Tests use constructed strings only. They cover insertion/deletion/substitution,
empty strings, ambiguous repeated symbols, original-rank stability, exact
normalization by enumerating all strings of small fixed lengths, unseen
vocabulary entries, same-count controls, a frequent-symbol deletion trap,
determinism and no model mutation. Regression fixtures reproduce every legacy
random draw and scalar and show that high new diagnostics cannot rescue an old
failed gate. They do not train a model or score a historical holdout.

Observed validation: 43 tests passed across `test_recovery_metrics.py` and
`test_latent_recovery.py` on CPU; Ruff for the new module/tests and whitespace
checks passed. Source comparison verified that twelve legacy metric,
pass-decision and threshold functions remain byte-identical to the branch base.

Current changes do not select a new winner, repair existing model masks, or
prove that any earlier failure was only a scoring artifact. A future primary
alignment metric and its baselines, thresholds and independent holdout need a
new registration. EXP-0012 also failed an accuracy margin, so changing alignment
alone would not satisfy its old complete rule.
