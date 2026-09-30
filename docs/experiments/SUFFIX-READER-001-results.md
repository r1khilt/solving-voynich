# Longer context reduces errors, but the known-key reader still fails

2026-09-30. **Registered reader gate FAIL.** With the correct key supplied,
the new source makes191/7,168 character edits (2.6646%), versus227 (3.1669%)
for the old source. The15.8590% improvement misses the required25%; overall
error exceeds2%; two keys exceed the5% per-key limit. No case was discarded.
This tests reading ambiguous synthetic encrypted Latin, not finding its key,
identifying a language, or deciphering Borg/Voynich.

The implementation supports much longer context with exact inference. Its
larger state inventory helped modestly; it did not establish reliable reading.

## Ordered experiment and source selection

The [registration](SUFFIX-READER-001.md), new implementation, references and
tests were pushed and remotely verified at
`063b8ac6feccd7308325bcc5bd74e96b84d7b764` before preparation. All16 declared
source choices trained on50,000 Caesar letters and were scored on50,000 Virgil
letters. The winner was maximum history12/tau256,3.434327bits/character versus
3.452566 for order3/tau256. This is about0.53% better source prediction, not
a cipher-reading metric. Refit on both authors contains688,558 observed contexts.
Independently reconstructed counts match; the sparse representation of the old
order3 source matches its complete probability table exactly.

Source selection was saved before accessing Cicero. The prepared source and
all16new known-key cases were pushed/remotely verified at
`d2b8da210ef607610077128651ce4cea233049a0` before prediction. Each key has two
224-letter windows from the fixed allocation[40000,48160), with intervening
gaps, entirely inside one corpus body. They do not overlap the earlier DEV001
cipher windows. This author had already been acquired and processed; the
experiment is development with a new scoring allocation, not unread-author
confirmation. No CONFIRM001 ciphertext, answer or learned key is used.

All64readings, per-case archives and numerical comparisons were then
pushed/remotely verified at`fec709a7561f9dafbca95ef6ba80e23d0aeb06e0` before
the one accuracy evaluation. Commands used the registered `prepare`, `predict`
and `evaluate` stages of `scripts/run_suffix_reader001.py`, respectively, with
those freeze hashes and single-thread environment settings recorded in NB235–238.

## Complete outcomes

Every row has448 true letters. Error is aligned character edit distance;
decoded length need not equal the true length. Both arms receive the exact
generating dictionary; the actual text length is not imposed on decoding.

| Key | Previous edits | Longer-context edits | Longer-context CER |
| --- | ---: | ---: | ---: |
| 1 |16|18|4.0179%|
| 2 |14|12|2.6786%|
| 3 |9|4|0.8929%|
| 4 |9|9|2.0089%|
| 5 |10|8|1.7857%|
| 6 |11|9|2.0089%|
| 7 |19|13|2.9018%|
| 8 |10|10|2.2321%|
| 9 |12|8|1.7857%|
| 10 |14|8|1.7857%|
| 11 |7|10|2.2321%|
| 12 |19|14|3.1250%|
| 13 |31|27|6.0268%|
| 14 |13|10|2.2321%|
| 15 |4|4|0.8929%|
| 16 |29|27|6.0268%|
| Total /7,168 |227|191|2.6646%|

Eleven keys improve, three tie, two worsen. Sixteen of32readings change.
Exact records improve only2/32→3/32. All records remain supported; no missing
symbol or dictionary search can explain these residual errors because the
complete true dictionaries are supplied. Exact inference rules out a beam
discarding a higher-scoring path under this model. It does not prove that the
model's preferred reading is the historical or human-correct one.

## Posthoc inspection: nominal context length differs from effective weighting

After the failed accuracy result, a separately labeled descriptive calculation
decomposed the unchanged interpolation into its context-depth contributions.
This does not refit a source, decode again, select a new candidate or change
the qualification. At each history, the empirical next-letter distribution
has mixture weight`count/(count+256)`, with the remaining mass going to its
suffix. Recursing gives a normalized mixture over history depths. After
conditioning on the next letter, its normalized component contributions give
the corresponding depth responsibilities. These are a mathematical
decomposition, not neurons, attention weights or a causal percentage of errors.

At all7,168 true-text contexts, mean weights are:

| History depth | Mean mixture weight |
| --- | ---: |
| 0 (smoothed unigram) |1.9591%|
| 1 |22.7545%|
| 2 |43.9392%|
| 3 |22.0170%|
| 4 through12 combined |9.3301%|

Conditioning on each true next letter increases the combined4–12 responsibility
to12.8615%. Repeating the descriptive calculation on the7,130 decoded letters'
own histories gives9.3950% mixture weight and12.9123% responsibility. This
similarity is descriptive, not another independent sample. Every mixture
reconstructs the actual source probability within3.34e-16. Two hand-derived
fraction/unseen-history tests validate the decomposition.

This identifies a concrete limitation of the present scaling attempt: most
probability-mixture weight remains on histories no longer than three letters.
It does **not** show that9% of decisions use long context, that forcing larger
weights would improve accuracy, or that added context is intrinsically useless.
The strong common interpolation mass was selected on Virgil. Sparse counts,
domain shift, the shared mass across depths and source-family misspecification
remain possible reasons; this experiment does not causally separate them.

## Validation, resources and next decision

The new model has31artificial tests: rational full-history probabilities,
complete tiny plaintext enumeration, dense-source equivalence, long-history
state sufficiency, boundary resets, normalization and explicit cap failures.
Full pre-execution regression:1,879passed plus23subtests,8skipped,130.69seconds.
Two later decomposition tests also pass. New code is lint-clean; the same five
unrelated pre-existing full-tree lint findings remain.

All32new-model records have reverse-recursion/direct path checks, and all32old
records have the prior independent reference checks; maximum score difference
6.26e-13. Separately re-encoded all64returned texts, checked all16per-case
archives against the complete archive, and independently replayed all64edit
counts. The new reverse recurrence and descriptive helper are root-authored;
algorithmic agreement is not a separate researcher replication.

Preparation18.541555CPU/18.566864wallseconds; prediction3.738624CPU/3.771052wall;
evaluation0.014559CPU/0.047077wall; posthoc decomposition3.202045CPU/3.204717wall.
Total measured phaseCPU25.496783seconds, excluding unit tests and auxiliary
artifact checks. Largest reported processRSS917,454,848bytes. Zero paiduse.
Fixed-known-key speed does not establish feasible blind-search speed.

Compact evidence: [source selection](../../results/SUFFIX-READER-001/source_selection.json),
[evaluation](../../results/SUFFIX-READER-001/evaluation.json),
[descriptive supplement](../../results/SUFFIX-READER-001/descriptive_supplement.json)
and [edit-count audit](../../results/SUFFIX-READER-001/descriptive_audit.json).
Source archive SHA256`5fcea148591aa23446b9cff3bd7ebf88eea55799bd70b585dca307d05ee2ba4d`;
prediction archive SHA256`02cd14c1cfd46228171360a0f216ef1a35a9edeed938785f5599995f4f63f5e2`.
Large/raw artifacts remain ignored and all original pipeline files unchanged.

Do not promote this source as a qualified reader or retrofit thresholds.
The next source-method comparison should test how longer patterns acquire
useful weight, including depth-dependent smoothing or stronger source families
and more varied training text. Select using source-only material and evaluate
on a new declared allocation; do not tune these32passages. Dictionary support
uncertainty, blind search, unknown language and historical transcription are
separate unresolved requirements. No new historical reading is proposed.
