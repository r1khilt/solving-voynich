# Dictionary refinement reaches the true-key transfer baseline

2026-09-29. The [registered refinement](BLIND-CHANNEL-DEV-005.md) reduces B2
transfer errors from **22 to 8 per448 letters**, while B1 remains **0/448**.
Both learned dictionaries now produce exactly the same transfer strings as
decoding with the generating key. Independent numerical references agree.
This is adaptive, warm-started synthetic development with known Latin and a
restricted channel family, not a fresh-key result or a Voynich reading.

## Recovery and remaining errors

| Dictionary | Before: transfer edits | After: transfer edits | Exact transfer records | After: fit edits /896 | After: minimum possible transfer edits |
| --- | ---: | ---: | ---: | ---: | ---: |
| B1 learned | 0 | **0/448** | 2/2 | 13 | 0 |
| B1 generating oracle | 0 | 0/448 | 2/2 | 13 | 0 |
| B2 learned | 22 | **8/448** | 1/2 | 43 | **0** |
| B2 generating oracle | 8 | 8/448 | 1/2 | 34 | 0 |

Source and decoder are unchanged from DEV004's selected order3/tau256 arm.
Only learned dictionaries were refined, using fitting ciphertext and their
own DEV003 starting dictionaries. No plaintext, generating row or transfer
record entered fitting. Every source-letter row was eligible for the same
swaps and replacements; none was manually repaired.

B2's previous learned dictionary excluded exact truth: even an answer-aware
best reading needed at least20 fit and12 transfer edits. Both floors are now
**zero**. Search removed that obstruction without using answer-based floors
as its objective. A zero floor means truth is representable; it does not mean
the source ranks it first. B2 still makes8 transfer edits even with the true
dictionary.

Post-evaluation, hash-checked string comparison confirms both learned transfer
records equal their true-key counterparts for each positive case, not just
their error counts. B1 also matches all4 oracle fitting predictions; B2 matches
2/4. B2 fit errors improve70→43 but remain above the oracle's34. Both learned
dictionaries match20/23 literal generating rows, with20/21 distinct units for
B1/B2. Exact transfer does not establish a complete recovered key. These
comparisons are descriptive, not a basis for another refit.

Development transfer errors were477/430 in DEV001,4/32 after DEV003's broader
search,0/22 after DEV004's source repair, and now0/8. Methods and assumptions
changed, and cases were reused. This is an engineering history, not independent
replications or a clean attribution of the whole gain to one intervention.

## Objective and search

| Learned case | Starting fit bits | Selected fit bits | Neighbors | Complete sweeps | Stop |
| --- | ---: | ---: | ---: | ---: | --- |
| B1 | 2999.280905 | 2999.228927 | 3,574 | 3 | Local optimum |
| B2 | 3070.025033 | 3041.215795 | 7,161 | 6 | Local optimum |
| B1 paired shuffle | 4293.349433 | 4281.050802 | 1,377 | 1 | Time limit |
| B2 paired shuffle | 4106.130564 | 4091.557370 | 1,159 | 0 | Time limit |

Scores exclude the shared legacy one-bit selector used in DEV004 tables.
Positive final dictionaries have184-bit literal codes, versus190 for the
generating ones. Learned B1/B2 beat their generating fit objectives by
6.661841/20.084710bits. B2's worse fit reading despite its better objective is
a residual score/recovery mismatch. Local optimality does not prove global
optimality or a historical interpretation.

Both shuffled controls kept their best completed candidates at the registered
300-second cooperative limit. They have no local certificates, no plaintext
accuracy, and no newly chosen semantic rejection threshold. All6 records per
null remain supported. Their log-likelihoods and decoded lengths are in the
evaluation; a supported reading is not evidence that random glyphs contain
a message. No timeout was extended or case replaced.

## Ordering, replay and artifacts

Code/protocol `d8d61bc8986f9d7ca6ac5b1052af716369823201` was pushed and remotely
verified before fitting. All four selected dictionaries and campaign metadata
were pushed and verified at `b073ca212c545e44d73b593739cb0e1520f876fa` before
evaluation. Exact commands:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.:src .venv/bin/python scripts/run_blind_channel_dev005.py campaign --freeze d8d61bc8986f9d7ca6ac5b1052af716369823201
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.:src .venv/bin/python scripts/run_blind_channel_dev005.py evaluate --freeze b073ca212c545e44d73b593739cb0e1520f876fa
```

Root-written trace accounting checks13,271 neighbors, move order, neighborhood
counts, literal model cost, score arithmetic, parent trajectory, best-candidate
retention and certificate ownership. Input, warm-start and compressed trace
hashes were rechecked. It does **not** numerically recompute every neighbor.

Previously independently authored backward inference replays every final fit
score before key freeze and all36 record predictions afterward. All literal
MAP strings agree; maximum score difference is **1.4779288903810084e-12**.
Independent ordinary edit-distance and shortest-path references agree on all24
positive/oracle record edits and dictionary floors. The new optimizer/runner
is root-authored and has no new independent agent review: those agents reached
usage limits after delivering DEV004 references. Exhaustive artificial controls
and reuse of independent references do not erase that review limitation.

Compact outputs:

- [Campaign](../../results/BLIND-CHANNEL-DEV-005/campaign.json) and four adjacent selected-key freezes.
- [Evaluation](../../results/BLIND-CHANNEL-DEV-005/evaluation.json), SHA256 `879b17b4e3db4d69cbd5086bcf7f61b7122b60174a2d5066210056a9e3b8d768`.
- [Post-evaluation comparisons](../../results/BLIND-CHANNEL-DEV-005/reading_diagnostics.json), SHA256 `bdf22b13e2be6544b6d3f99ed29e3c2ba90cc1c78084c2c187f3030fdab59dc0`.

The ignored5,365-byte prediction archive has SHA256
`237e50d90790c49bc329a0b311875c09a31f659df58a4536130de0e231f4275c`.
Four ignored full search traces are individually hashed in the key freezes.
The post-evaluation comparison helper wrote complete JSON and then raised
because a relative path was passed to an absolute-path metadata formatter.
The existing file was parsed and hash-verified without overwriting or rerunning
a fit/evaluation. This operational error did not affect results.

## Resources and next decision

| Stage/case | Wall seconds | CPU seconds | Peak process RSS bytes |
| --- | ---: | ---: | ---: |
| B1 fit, checks and trace write | 55.108 | 54.694 | 100,990,976 |
| B1 shuffle fit, checks and trace write | 301.096 | 297.604 | 116,490,240 |
| B2 fit, checks and trace write | 164.319 | 163.119 | 101,597,184 |
| B2 shuffle fit, checks and trace write | 300.806 | 296.901 | 123,633,664 |
| Evaluation and reference replay | 3.243 | 2.856 | 122,224,640 |

Two single-thread fit workers; campaign wall520.613seconds. Summed measured
fit/evaluation CPU815.173seconds, no paid compute/API use. Sum of the two
largest worker peaks240,123,904bytes is a conservative worker-memory figure,
not whole-machine usage. No hard resource cap failed. The final full suite
passes **1,718 tests and23 subtests,8 skipped**, in213.91seconds, with numerical
threads limited to one. Log: `outputs/validation/2026-09-29-dev005-final-pytest.log`.
All changed Python files pass Ruff, and result hashes, JSON, report links and
diff hygiene pass. The five previously recorded unrelated full-tree lint
findings remain. See NB-218 for the completed checkpoint.

Stop tuning these exposed cases. Next freeze the complete pipeline, run from
scratch on fresh independent keys, fit on a new author and transfer to another,
retaining failures and matched controls. The method still assumes a Latin
source and deterministic, memoryless one/two-glyph emissions. Stateful familyC,
historical Borg and Voynich remain unresolved. The concrete gain is discovery
of transfer readings as good as the true-key decoder on two development
ciphers; fresh qualification must establish how far that capability carries.
