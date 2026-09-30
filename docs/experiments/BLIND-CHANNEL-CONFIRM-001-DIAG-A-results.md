# One missing unit caused whole-record failure; correct identity is a separate issue

2026-09-30. This is the fixed, answer-informed diagnostic registered after
CONFIRM001 failed. It isolates a model failure on synthetic encrypted Latin;
it is **not blind recovery or a repaired qualification**.

The selected key6 cannot parse one new passage at all. Restoring its missing
singleton F to either of two **wrong letters** makes that passage readable
with one error. Assigning F to the correct letter makes it exact. Thus the
catastrophic failure is mainly loss of coverage, not hundreds of independently
wrong letter assignments. All three interventions were specified before this
diagnostic ran, including the incorrect-letter controls.

## Frozen intervention and outcomes

The [plan](BLIND-CHANNEL-CONFIRM-001-DIAG-A.md), runner and original failed
evaluation were committed, pushed and remotely verified at
`99d3d137763c4b792e5650b4c787244c03ff8be7` before the single invocation:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.:src \
  .venv/bin/python scripts/diagnose_blind_channel_confirm001_support.py \
  --freeze 99d3d137763c4b792e5650b4c787244c03ff8be7
```

Every arm uses the same fixed source, stopping law, four fitting records and
two transfer records. Each intervention replaces one singleton by singleton F;
the literal dictionary cost stays184bits. No optimizer runs. All k/y/z letters
are absent from the896 true fitting letters. The original transfer contains
one y, no k or z. These exposure facts and the correct row are answer-informed.

| Fixed dictionary arm | Supported transfer records /2 | First / second transfer edits | Total edits /448 | Exact transfer records /2 | Fitting code increase, bits |
| --- | ---: | ---: | ---: | ---: | ---: |
| Saved learned key | 1 | 224 /8 | 232 | 0 | 0 |
| y→F, correct-letter repair | 2 | 0 /8 | 8 | 1 | +0.858186 |
| k→F, wrong-letter control | 2 | 1 /8 | 9 | 0 | +0.023557 |
| z→F, wrong-letter control | 2 | 1 /8 | 9 | 0 | +0.028362 |

The original224-deletion penalty remains in the failed confirmation. It is
not a claim that the old model emitted224 individually wrong characters:
it emitted no reading for that unsupported record. Restoring support allows
the already mostly correct dictionary to work on the rest of the passage.

All four arms preserve the same fitting MAP readings and18 fitting edits.
Their fitting marginal likelihoods nevertheless differ, because those sums
include alternative plaintext paths, not just the MAP path or the true text.
The current fitting objective prefers the original key to all three repairs,
and prefers each incorrect-letter repair to the correct one. No complexity
change explains this preference. This is a finite-data/source-model effect
on these specific candidates, not proof that no ciphertext-only method could
infer the unseen row or that the optimizer found a global optimum.

## What changes in the research direction

The diagnosis separates three questions: can a key explain an observation at
all; which letter owns an uncertain unit; and which of several possible
readings the source model chooses. Counting correct literal rows alone hid
the catastrophic difference between the original20/23-correct key and these
one-row interventions. Key uncertainty must therefore be evaluated through
new-record support and plaintext accuracy as well as assignment accuracy.

There is a simple structural fact behind the failure. With nonempty emissions,
a positive stopping law and a source that gives every finite plaintext positive
probability, a deterministic dictionary supports every finite string over a
glyph alphabet **if and only if** its unit set contains every singleton glyph.
Sufficiency follows by concatenating singleton units. Necessity follows by
considering each one-glyph record: longer units cannot emit it. This is a
deduction about the model family, not a new empirical result. The generating
B family supplied all six singletons; the fitted family did not preserve that
property. The observed failure is on a longer string, where a missing singleton
need not cause failure universally but did cause failure here.

A future synthetic comparison can explicitly declare and test that structural
assumption. It must label the help supplied by knowing the family. It must not
impose “all singleton glyphs are legitimate units” on Borg or Voynich without
evidence. An alternative is to retain competing **whole keys** with shared
assignments across a document. Choosing an unrelated substitute at each token
would change the cipher model and can erase the very consistency we need to
recover. Adding a generic noise escape can avoid zero likelihood, but finite
likelihood alone is not correct decoding; corruption and structured-null
controls would be required.

These are proposed directions, not implemented improvements. They complement
the [existing inference review](../research/blind-channel-next-inference-review.md)
on rare-unit pruning, support changes, finite-state inference and the limits
of larger neural source models. The failed fresh test also has independent
search failures and142 true-key transfer edits out of3,584. This diagnostic
does not resolve those. Source-model quality must be tested with known keys
before interpreting a blind solver's failure, and all future qualification
must use a declared unused allocation. No exposed confirmation is rerun here.

## Verification and limits

The [saved evaluation](../../results/BLIND-CHANNEL-CONFIRM-001-DIAG-A/evaluation.json)
preserves every arm, record score, error count, decoded length and resource
measurement. Existing independently authored reference algorithms check the
24 record outcomes; maximum score difference is8.53e-13. The baseline metrics
match the original evaluation exactly, and all arm edit counts have independent
replays. This is computational verification, not an independent researcher
replication or a blinded intervention.

A separate root [artifact audit](../../results/BLIND-CHANNEL-CONFIRM-001-DIAG-A/artifact_audit.json)
also checks24 source-independent Boolean parse supports and exactly re-encodes
all23 supported readings. The two control passages differ from the true text
only by replacing its single y with k or z, respectively. All fitting readings
and the second transfer reading are identical across arms. Archive hashes,
the original evaluation and all30 frozen pipeline files were rechecked.

The single run used0.444775CPU seconds and0.748377wallseconds, reported peak
RSS100,368,384bytes, zero paid services and no new data. Prediction archive
SHA256`b28518246fb2021abfda1fb521f756534ec9cc4a1a47dc6a16e7a319e48acde9`;
original evaluation SHA256
`c91ccc5281c4e618585c29248f652b13a44d4227809ff735aa82d5931e99899d`.
The original30pipeline source files remain frozen. The underlying eight-key
qualification still fails. No historical plaintext or meaning is established.
