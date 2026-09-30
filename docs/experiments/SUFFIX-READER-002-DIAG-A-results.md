# Exact letter count does not repair the current reader

2026-09-30. **The answer-informed length intervention increases character
errors from229to243**, while exactly correct records increase from2to5of32.
The original reader qualification remains FAIL. This rules out simply revealing
length as a sufficient repair for this source/decision rule on these cases.
It does not prove that length information is useless to a different reader.

The model leaves substantial posterior uncertainty even with the true key.
These are controlled artificial ciphers of Latin, not Voynich readings.

## Procedure and provenance

The [diagnostic plan](SUFFIX-READER-002-DIAG-A.md), forward expectation-moment
implementation, separate raw-count backward reference and tests were pushed
and remotely verified at `48c8efc1d5260e75f260d35704e5aa352b40ac5a` before the
single diagnostic run. That checkpoint also preserves SUFFIX002's failed
result. No source parameter, dictionary, record, threshold or previous
prediction was changed.

Every existing arm/record gets an exact posterior calculation:128in total.
For calibrated12only,32additional exact calculations restrict source length
to224. The source-step count is added to the sufficient state; only impossible
length states are removed. No beam, posterior sampling or selected-case rerun.
All128unconstrained readings exactly reproduce their frozen texts and scores.

## Remaining uncertainty under the fixed model

| Arm | Character edits | Exact records /32 | Model-implied expected exact records | Mean posterior entropy, bits |
| --- | ---: | ---: | ---: | ---: |
| Fixed 3 |282|1|1.0371|18.5275|
| Fixed 12 |254|1|1.3855|16.9948|
| Calibrated 12 |229|2|2.2028|15.2378|
| Calibrated 3 |258|1|1.1521|17.9829|

The expected exact count sums each chosen complete text's conditional
probability. Its agreement with observed counts on32dependent development
records is descriptive, not evidence of population calibration. A low
probability of getting every character right does not imply that most
individual characters are uncertain.

Calibrated12's chosen full-record probabilities range from0.000207to0.508367;
mean0.0688385. This model does not assign near-certainty to any complete
passage. Entropy ranges from3.5189to29.1571bits. True-text posterior surprisal
averages14.3921bits; all true texts remain supported. These posterior quantities
condition on a particular small Latin source and supplied dictionary. They
are not natural-language impossibility bounds or probabilities of a historical
interpretation being true.

## Supplied true length: complete outcome

| Key | Ordinary calibrated reader edits | True-length reader edits |
| --- | ---: | ---: |
| 1 |8|11|
| 2 |20|16|
| 3 |17|24|
| 4 |15|11|
| 5 |20|24|
| 6 |6|6|
| 7 |4|2|
| 8 |27|20|
| 9 |27|34|
| 10 |19|19|
| 11 |8|8|
| 12 |5|10|
| 13 |9|5|
| 14 |8|13|
| 15 |28|30|
| 16 |8|10|
| Total /7,168 |229|243|

Five keys improve, three tie and eight worsen. By record, eight improve,
ten tie and fourteen worsen. The constraint changes26MAPscores. Exactly
correct records improve2→5, but total character error worsens3.1948%→3.3901%.
Different metrics need not move together: choosing the most probable complete
string optimizes exact-string loss under the model, not expected edit distance.
On a finite natural-text panel with a misspecified source, extra information
also need not improve realized errors. This experiment does not separately
identify those two effects.

Before intervention, the model's mean expected length is223.3278, versus a
mean MAP length222.25and true224. It gives the true length24.7849%posterior
mass on average. Revealing length reduces mean entropy15.2378→14.1521bits,
leaving most of this model's uncertainty unresolved. Conditioning on a
particular length can increase entropy for an individual record; there is no
claim of monotonic reduction in every case.

The length is explicit answer-derived assistance. No new reader is qualified,
and no length-dependent rule is proposed for Voynich.

## Validation and actual resources

All160forward/backward comparisons pass. Maximum discrepancies: log mass
1.0801e-12, MAPscore7.9581e-13, entropy2.299e-10bits, expected length7.549e-11,
length variance1.459e-8. Exact integer path counts match. Every new reading
re-encodes, has exactly224letters, and has an edit count verified by a second
implementation. The unrestricted aggregate errors equal the original evaluation.
All34diagnostic/source/input bindings and all30original confirmation bindings
remain unchanged. References are separately written but root-authored; no
independent researcher replication is claimed.

Full regression before execution: **1,913tests and23subtests passed**,8skipped,
132.29seconds. Seventeen new tests include ten complete rational enumerations.
New code lint passes; the same five unrelated full-tree findings remain.
Measured diagnostic CPU18.292706seconds, wall18.311851seconds, peakRSS834,256,896
bytes. Maximum length-augmented lattice65,383states, below500,000cap.
One CPU thread, zero paid services, no cap failures or reruns.

A subsequent artifact-audit helper used a relative path where its return
manifest expected an absolute path. All assertions and the audit-file write
had completed before that auxiliary exception. The saved audit was read back
and its diagnostic hash checked; no diagnostic or inference was restarted.

Evidence: [complete diagnostic](../../results/SUFFIX-READER-002-DIAG-A/diagnostic.json),
[artifact audit](../../results/SUFFIX-READER-002-DIAG-A/artifact_audit.json),
and the [unchanged failed comparison](SUFFIX-READER-002-results.md).
Counterfactual plaintext archive3,566bytes, SHA256
`5d9659089efbd3e2e1a371db1f2b5bce81b449cf81e6c72e232da2d28a7f6bc8`,
stored outside Git. Complete diagnostic SHA256
`dcb4f918e634b3532a4c8c8dff5e3de1b189c2b8afcbcb2a20075a77092c3cf9`.

## Consequence for the next model

Do not spend more new cipher windows on nearby smoothing or length fixes.
The next substantial source comparison should add broader training text and
richer linguistic patterns, with unchanged-data controls separating those two
changes. Neural rescoring is a candidate, not an established solution: it loses
the current exact finite-state history merge, so search error needs its own
measurement. Previously failed sampled edit-risk selection is also not erased
by the present MAP/edit-loss distinction. Any revisit needs a specific new
control and must retain the earlier negative result.

The useful advance is a narrower diagnosis: source weighting and longer
context help, but correct keys plus correct length still leave incorrect
preferences. Blind key search, unknown language, transcription and historical
meaning remain separate unresolved problems.
