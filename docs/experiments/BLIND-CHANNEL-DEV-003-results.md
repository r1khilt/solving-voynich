# Unknown-unit search produces mostly correct readings in both development cases

2026-09-28. The [frozen systematic search](BLIND-CHANNEL-DEV-003.md) reduced
held-out errors from 477 to **4**, and from 430 to **32**, each over 448 source
letters. It inferred variable-length units from ciphertext without their true
dictionary, boundaries or mappings. Independent decoding replay agrees.
These are two reused synthetic cases under declared assistance, not fresh
confirmation or a Voynich decipherment.

| Positive case | Earlier transfer edits /448 | New edits /448 | New edit rate | True-channel oracle edits /448 | New fit edits /896 |
| --- | ---: | ---: | ---: | ---: | ---: |
| B key 1 | 477 | 4 | 0.89% | 4 | 19 |
| B key 2 | 430 | 32 | 7.14% | 21 | 91 |

Both now meet the earlier **10% development-error diagnostic**. No positive
transfer record is completely exact: zero of four. Both learned readings have
446 letters versus 448 true letters across their two transfer records; edit
distance also counts substitutions and insertions. The original failed
[DEV-001](BLIND-CHANNEL-DEV-001-results.md) results remain unchanged.

## What this establishes

Previously, a good reading could be obtained only by supplying the true channel
or constructing a gold-assisted witness. The new runs discover near-correct
readings without either. This supports the diagnosis that the earlier search,
initialization and coverage were major barriers. It establishes development
competence within a restricted unknown-unit family, rather than merely finding
a statistical pattern or lowering a prediction loss.

A post-evaluation comparison of the preserved prediction strings finds that
**B key 1 exactly matches the true-channel oracle's reading on all six fit and
transfer records**. Matching that oracle is not perfect plaintext: its source
model and ambiguous code still make errors. B key 2 remains worse than its
oracle on both fit and transfer. Full predictions and their hashes are preserved
in the [evaluation](../../results/BLIND-CHANNEL-DEV-003/evaluation.json).

The learned dictionaries are not the complete generating keys. Both have 20
distinct units across 23 letter rows, with eight singleton rows and 15 two-glyph
rows. Gold has 23 distinct units, six singletons and 17 pairs. An explicitly
post-evaluation literal comparison finds 20/23 matching rows for key 1 and
17/23 for key 2. Key 1 differs at `k,y,z`; key 2 at `b,h,k,x,y,z`.
These are exact row matches, not an identifiability or equivalence analysis.
The [reading diagnostics](../../results/BLIND-CHANNEL-DEV-003/reading_diagnostics.json)
separate these posthoc checks from registered quantities. No diagnostic was
used to refit or select a new model.

## The remaining objective problem

| Case | Earlier learned fit bits | Generating-channel fit bits | New learned fit bits | New model bits |
| --- | ---: | ---: | ---: | ---: |
| B key 1 | 4059.605341 | 3280.876477 | 3273.838397 | 184 |
| B key 2 | 3610.535134 | 3277.618275 | 3246.839272 | 184 |

Lower is better under the unchanged source and literal model code. Both new
models beat the generating description and the earlier admissible gold-assisted
witnesses. Therefore the old evidence of a missed better reference no longer
applies to these selected winners.

For key 1, a shorter/different dictionary retains the oracle's observed readings.
For key 2, the objective prefers the learned model by **30.779 bits** while its
reading is worse. Only six of those bits come from the shorter model description;
the remaining 24.779 come from marginal likelihood. This is a concrete
score-versus-recovery mismatch, not proof of intrinsic nonidentifiability or a
globally optimal wrong key. Further search under the same objective is not
guaranteed to fix it. Stronger source context, ambiguity and finite-sample effects
deserve separate controls before more compute is treated as the answer.
Marginal-likelihood fitting, most-probable-whole-reading decoding and character
edit loss are also different objectives. This result alone does not distinguish
source misspecification from finite-sample preference or a decoding decision
rule that is poorly matched to edit error.

## Conditional ambiguity and shuffled controls

For each record, the registered diagnostic
`(log p(ciphertext) - log p(best joint path, ciphertext)) / log(2)` measures the
surprisal of its best reading under the selected model. In this deterministic
one-state family, one plaintext determines one path, so this is conditional
posterior min-entropy. It is not Shannon entropy, character error, model
uncertainty, or a probability that a historical decipherment is correct.

| Case | Transfer best-reading surprisal, two records in bits | Transfer log likelihood | Distinct learned units |
| --- | ---: | ---: | ---: |
| B key 1 | 4.35, 4.60 | -1107.390130 | 20 |
| B key 1 shuffle | 238.75, 227.94 | -1432.033814 | 8 |
| B key 2 | 16.37, 20.30 | -1042.164143 | 20 |
| B key 2 shuffle | 205.65, 209.06 | -1362.202682 | 9 |

The paired shuffles preserve each record's glyph counts and lengths. They
receive no plaintext accuracy and no new rejection threshold. The marked
concentration difference is descriptive evidence on this small panel, not a
general distinction between language and nonlanguage. A wrong model can also
be conditionally confident, and whole-record surprisal depends on record length.
An immediate counterexample is the bijective A family: each supported string has
only one reading, so this surprisal is zero even for shuffled nonsense. The
diagnostic measures conditional ambiguity, not meaningfulness.

## What changed and what remains supplied

Compared with DEV-001, the new search uses the narrower one-state deterministic
family, different initialization, all 42 possible one/two-glyph units rather
than observed substrings, complete single-row/pair-swap neighborhoods, and a
larger time budget. The source probabilities and actual finite model code are
unchanged. This bundled intervention does not isolate speed, family restriction,
pool expansion, initialization or budget as the sole cause.

Latin, its 23-letter alphabet, the six-glyph alphabet, record boundaries,
length bound two, the stop law and a small channel grammar remain supplied.
Both keys use the same development author and source-selection pipeline.
There are only two positive keys. Reused fitting and transfer cases make this
adaptive development. No final author, stateful C case, historical Borg decoder
or Voynich reading was evaluated. A local certificate cannot establish a
globally best key; the [separate theory review](../research/unknown-unit-search-alternatives.md)
provides exact counterexamples and conditional alternatives, not an extra run.

## Resources and verification

All four fits completed all 16 starts and 64 local optima in total, with
2,021,878 neighbor evaluations across 1,707 complete sweeps. Campaign wall time
was **403.105 seconds** using two workers and one numerical thread per worker.
Summed core-search time was 765.457 seconds; recorded process CPU through trace
creation was 781.834 seconds. No time cap or log-domain fallback triggered.
The two largest process RSS peaks sum 1,859,321,856 bytes, a conservative bound
for the measured two-worker peaks, not a host-wide memory measurement. No paid
compute or API use.

Independent [trace accounting](../../results/BLIND-CHANNEL-DEV-003/search_summary.json)
verifies every archived move and neighborhood, score arithmetic/literal cost,
seeded random initialization, parent trajectory, selected minimum, counters,
certificate ownership and archive hashes. It does not numerically recompute all
two million empirical likelihoods or replay the first frequency ranking.
Independent scalar forward/max-path/edit replay verifies all 24 final record
evaluations, literal predictions and final model costs; maximum difference
**5.684341886080801e-13**. No new full independent end-to-end data-builder audit
is implied. The original source and data audits remain the input provenance.

Ordered checkpoints, each pushed and remotely verified before its next stage:

- Source, protocol and implementation tests: `830a53f79f318d90e933fe67b4e4c559bd875cdb`.
- Four selected models and trace auditor: `8e6e5c8d17bf4bdd568f22a2e1df22d12972b1ac`.
- Evaluation SHA-256: `f820d84b41d78300a8b64afbb171b6fbf8ca3f45c618753bf4597406afd25de1`.
- Full predictions SHA-256: `8756c02f656d8db5e510d88d257bb1dd358b75425d6f160c31653f19b9474185` (17,218 bytes).
- Trace summary SHA-256: `9cc1915b6f68d59f2f46f50fb9775aa18fd31f755401ee41f4bfab1cf078589e`.

The unchanged data/source hashes and exact fitting command are in the
registration and NB-209–210. Evaluation command:

```sh
PYTHONPATH=.:src .venv/bin/python scripts/run_blind_channel_dev003.py evaluate --freeze 8e6e5c8d17bf4bdd568f22a2e1df22d12972b1ac
```

Before fitting, the full repository suite passed 1,495 tests plus 23 subtests,
with eight skips. The new trace auditor adds 45 passing artificial/adversarial
tests. After integration and evaluation, the full suite passed **1,540 tests
plus 23 subtests**, with eight skips, in 196.07 seconds; log
`outputs/validation/2026-09-28-unit-results-pytest.log`. Changed-file Ruff passes;
five pre-existing unrelated lint findings remain. Bulk text, predictions and
traces stay ignored and hashed.

Next: preserve these development successes, investigate the source/objective
mismatch without hand-fixing the exposed keys, and freeze a fresh-key/author
qualification before any manuscript inference. The broader stateful recovery
and language-selection problems remain open.
