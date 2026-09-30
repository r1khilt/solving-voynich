# BLIND-CHANNEL-CONFIRM-002: fresh unknown-key end-to-end qualification

Registered 2026-09-30 before opening the unused reserved passages, generating
the registered keys, or running any empirical fitting. Confirmatory for this
restricted family. All earlier exposed outcomes remain development results.

## Question and prior evidence

Can the current three-stage unknown-key pipeline recover new ciphertexts from
scratch, rather than merely improve the eight already exposed CONFIRM-001 keys?
READ-002 had 43 errors in 3,584 letters, versus 309 for its original point
dictionaries. That encouraging
development result is not a fresh qualification. All true tuples eventually
entered the candidate banks, but the source model preferred wrong tuples in
seven cases. More exhaustive optimization cannot fix an incorrect objective.
This experiment freezes the existing method rather than adapting it further.

The method review is [fresh qualification review](../research/blind-channel-fresh-qualification-review.md),
updated by the actual [READ-002 results](KEY-BANK-READ-002-results.md),
[known-key neural qualification](NEURAL-READER-001-results.md), and
[decision-risk analysis](../research/decision-risk-for-decipherment-2026-09-30.md).
The review inspected [Nuhn et al. (2013)](https://aclanthology.org/P13-1154.pdf),
[Berg-Kirkpatrick and Klein (2013)](https://aclanthology.org/D13-1087.pdf),
and [Reddy and Knight (2011)](https://aclanthology.org/W11-1511.pdf). The original
PDFs were revisited for this registration (abstract/sections 3 and 6;
sections 1–3; introduction/section 2, respectively). Their context/restart/ambiguity findings
motivate source controls, actual recovery measurements, and constrained claims.
They establish neither an unknown-unit Latin solution nor a Voynich language.
No new architecture is introduced; neural rescoring and edit-risk decisions
are deliberately separate future tests. The current source and fixed-k reader
have already passed the exact-numerical comparisons recorded in READ-002.

## Fixed panel and data isolation

Sixteen new B keys plus sixteen paired within-record glyph shuffles. Each key
has four Nepos fitting passages and two Apuleius transfer passages, each 224
normalized source letters: 64 fit and 32 transfer windows, 7,168 positive
transfer letters. These are **new keys and passages, not new authors**.
The fixed author pair, nearby windows in a work, and artificial generator limit
generalization. There is no randomly sampled population of languages/authors.

Use only the already acquired LATIN-SOURCE-001 reserved-author archives:

| Input | SHA-256 |
| --- | --- |
| Corpus manifest | `fcade8e816fac499284c103808be3b3e7c88e66711cbb6f426be089ec8ecc9c5` |
| Nepos `phi0588.json.gz` | `56cf1b48f809e222131f37a1438945d3fb93149ad18bb3364debab161691ef4e` |
| Apuleius `phi1212.json.gz` | `bb10bf87640c9e8fd734342746876d25bef5b793bceb3fb2e874e0bf8abf9416` |
| Large compact source counts | `9769415ac460c56778a461af7646635abf5906fa726484ba84d98d4f401815f6` |

Original XML provenance, text extraction, alphabet normalization, rights, and
record hashes are inherited unchanged. No acquisition or source retraining.
The source manifest assigns both authors `reserved_reader_author`; the frozen
whole-author overlap audit must pass with zero 64-letter training/validation
overlap before construction. Its protection includes the complete reserved
authors, not just the new windows.

Exclude **whole original records** containing any NEURAL-READER-001 window.
Enumerate remaining record-contained 224-letter windows at stride 512;
shuffle the slot order with seeds 671031 (Nepos) and 671288 (Apuleius).
Greedily accept the first 64 and 32 eligible windows in that order, excluding
any 64-letter overlap with (a) those whole previously used records,
(b) complete old Caesar/Cicero/Virgil/Sallust/Tacitus normalized editions,
or (c) any earlier selected fresh window. The entire eligible order is fixed
before scoring. This is a mechanical duplicate filter, not selection based
on ease, letter exposure, oracle accuracy, or key fit. Insufficient text stops
preparation; no shorter windows, replacement authors, or fallback allocation.
The builder independently scans literal substrings as well as the allocator's
BLAKE2b-128 signatures. This exact check cannot detect shorter/paraphrased reuse.

Key i, zero-based, uses seed `269129 + 104729*i`, the unchanged B generator:
all six singleton glyphs plus 17 distinct digrams assigned to the 23 Latin
letters `abcdefghiklmnopqrstuxyz`. Stop if a key repeats any of the previous
58 B keys or another new key; do not redraw. Key i receives windows
fit `[4*i:4*i+4]` and transfer `[2*i:2*i+2]`. The matched null shuffles each
encoded record using one RNG seeded `key_seed+7000000`, consuming fit records
then transfer records. It preserves every individual length/glyph histogram.

Permute the 32-case inventory using seed 670003, then assign opaque
`case-01` through `case-32`. Fitting search seed is `73101+257*opaque_index`.
The fit panel exposes only fitting-artifact identity and this search seed.
The separate full panel contains transfer/answer artifact descriptors and
positive/pair labels. The constructor necessarily handles plaintext and keys
to audit generation. The fitting process parses only its fit panel and fitting
ciphertext; the predictor opens transfer ciphertext and labels only after all
fits are frozen, and never opens answers. The evaluator opens answers only
after all prediction outcomes and independent audits are published.
Hashes of full-panel bytes during fit freeze checks do not parse its contents.
Public seeds and accessible local files provide procedural isolation, not
cryptographic blindness. No LLM supplies guesses to the decoder.

## Frozen inference pipeline and assistance

Each positive and shuffle receives exactly the same fitting algorithm/budget:

1. Original order1/tau64 Caesar/Virgil source; frequency initialization plus
   random restarts, at most 16 starts, 80 sweeps/start, 300 cooperative seconds,
   batch size 256, complete 42-unit pool, tolerance 1e-8 bits. No old key is used.
2. Original order3/tau256 source, initialized by that case's selected first
   stage; 20 sweeps, 300 cooperative seconds, tolerance 1e-8 bits, retaining the
   best fully evaluated candidate. Compare each stage only under its own source.
3. Current large order12/minimum-count4/tau64 statistical source; four complete
   best-improvement neighborhoods, tolerance 1e-6 nats, as EXPAND-001. Retain
   every unique candidate from every completed neighborhood, including those
   with zero fit support. No incomplete neighborhood yields an admitted bank.

Use the unchanged literal channel code (`denominator=32`, `max_states=2`,
`max_emission_length=2`, `max_alternatives=3`, one source), known Latin/alphabet,
glyphs ABCDEF, record boundaries, and geometric stop probability 1/225.
No true record length constraint or true emission boundaries are provided.
Generating keys are injective as rows but **not uniquely decodable**: each
digram equals a concatenation of two available singleton codewords. The search
family also allows duplicate rows. Positive probabilities do not establish
identifiability, global search, or correct reading.

Within each finite, fit-selected candidate bank, normalize
`fit_log_likelihood - literal_model_bits*log(2)`. These weights are restricted
model weights, not full-family Bayesian evidence or calibrated confidence.
Preserve every finite log weight, including masses too small for exponentiation.
Each transfer tuple shares one unchanged key across both records.

After all 32 terminal fit/audit records are published, compute exact native
transfer evidence for all cases and compare with the quantized glyph-iid model
estimated only from its four fitting records. For positives, save four fixed
large-source decisions: stage2 parent point, expanded fit-best point, joint
key/text MAP, and primary marginal-text MAP using unchanged fixed k=8 ranked
lists and their floating upper bounds. No best-k selection; no widening after
seeing results. The shuffled controls receive evidence only, by design.
No answer is used to rank candidates. There is no new neural training or scoring.

Also save the original frequency initializer's **order3** reading before answers.
This `frequency_order3` comparator is weak and uses a different source from
the four primary arms. It is continuity with the original qualification, not
a clean same-source ablation. The parent and expanded point arms *are*
same-large-source decision comparisons. After prediction freeze, independently
decode the true key with the same large source to measure oracle excess.

## Prospective decisions, failures, and reporting

For all sixteen positive keys, primary recovery must satisfy every clause:

| Requirement | Threshold |
| --- | --- |
| Complete audited fitting and prediction outcomes | All 32 cases |
| Supported positive primary records | All 32 |
| Mean primary CER | At most 2% of 7,168 letters |
| Individual key CER | At most 5% of its 448 letters |
| Each key's primary CER minus large-source true-key CER | At most 2 percentage points |
| Each key's primary edits versus `frequency_order3` | Strictly fewer |

Use literal Levenshtein edits, not positional matches. No upper cap on CER.
An unavailable positive reading costs all 224 true letters in that record.
Retain all missing/failed cases in denominators and report conditional metrics
separately. A perfect frequency baseline makes the strict improvement clause
fail; poor oracle performance does not relax absolute gates.
Missing/null fit failures are never counted as successful noise rejection.

The separate evidence screen passes only when all 32 audited pipelines
complete, all sixteen positives have predictive gain **strictly greater than
zero bits** over their frozen iid model, and all sixteen shuffles have gain
at most zero. This is the READ-002 predictive-only screen, not the different
old CONFIRM-001 two-part fit/transfer threshold. It cannot establish rejection
of structured gibberish or semantic confidence. Even 0/16 false flags has a
one-sided 95% binomial upper bound about 17.1% under independent draws, an
assumption weakened here by shared passages, authors, and methods.

Report all arms, each key, support, exact records, decoded lengths, edit
counts, same-source oracle comparisons, literal row matches for parent/best
keys, source-letter exposure, true-tuple bank support and objective preference,
floating bound separation, evidence and controls, failures, resource use, and
all stage stop reasons. Bound separation is reported, not a recovery gate;
it certifies only this finite-bank floating-score objective within the stated
tolerance. Gates are engineering tolerances, not significance tests.

## Execution, audits, and bounds

Four CPU workers, numerical thread pools capped at one per child. Local only,
no paid APIs/GPU runs. Python reports 18 logical CPUs on this host. Plan at most
4 GiB sampled peak RSS per worker (16 GiB aggregate planning, not measured
whole-host usage); preparation 3 GiB. CPU/wall limits are enforced separately
in disposable processes; sampled memory checks are not an OS hard memory cap.

| Stage | Wall / CPU seconds | Outer timeout |
| --- | --- | --- |
| Preparation, once | 900 / 600 | Parent observation does not restart it |
| Each of 32 fits | 1500 / 1440 | 1510 seconds |
| Each successful fit's audit | 650 / 600 | 660 seconds |
| Each of 32 predictions | 1200 / 1100 | 1210 seconds |
| Global prediction audit | 2100 / 1800 | No selective retry |
| Post-freeze evaluation | 1500 / 1200 | No selective retry |

Worst-case component allocation, including 900 CPU seconds for validation,
is 104,980 CPU seconds (29.16 CPU hours), below a **30 CPU-hour cap**; with
four workers the corresponding outer wall allocation is about nine hours.
Expected empirical wall time from prior costs is one to two hours, not a
guarantee. The component ceilings leave margin for parent/child overhead;
record actual consumption separately. No cap increases after
observing slow/hard cases. No selective retries, overwritten files, discarded
controls, or silently normalized partial banks. The source freeze itself does
not authorize paid use.

Artificial pre-data checks cover window exclusion/insufficiency, distinct
nonregistered key seeds, literal encoding, no-overwrite and input allowlists,
real two-letter three-stage inference, full trajectory/weight audits, 32-case
prediction/evaluation orchestration, exact reference scores, withheld gold,
late failures and full denominators, and every gate's failure conditions.
The independent fit auditor reconstructs all neighborhoods/costs/normalization
and replays selected order1/order3 scores plus a fixed sample of large-source
backward lattices. It does not replay every neighbor's likelihood. Prediction
auditing checks every evidence sum and weight, every saved point and frequency
baseline, every native versus ranked-list marginal, fixed sampled independent
backward lattices, and sampled full-bank candidate supports. Gold evaluation
uses a second edit algorithm and independent true-key score/encoding replay.

Stage order, with numerical environment `OPENBLAS_NUM_THREADS=1`,
`OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `PYTHONPATH=.:src`:

```text
.venv/bin/python scripts/build_blind_channel_confirm002.py --freeze SOURCE_COMMIT
# Audit and publish the prepared panel before fitting.
.venv/bin/python -u scripts/run_blind_channel_confirm002.py campaign --freeze PANEL_COMMIT
# Publish every terminal fit/audit/failure before transfer.
.venv/bin/python -u scripts/predict_blind_channel_confirm002.py campaign --freeze FIT_COMMIT
.venv/bin/python -u scripts/audit_blind_channel_confirm002.py predictions
# Publish all predictions and audit before opening answers.
.venv/bin/python -u scripts/evaluate_blind_channel_confirm002.py --freeze PREDICTION_COMMIT
```

All source dependencies are frozen through `confirm002_common.PATHS`; compact
manifests include hashes of ignored ciphertexts, traces, candidate banks and
reading archives. Exclusive attempt markers prevent accidental repetition.
Verify Git remote identity after each stage checkpoint. Record failures and
actual numbers in a separate results document and the notebook.

An eventual pass qualifies this known-Latin artificial memoryless family at
this budget. It does not identify Voynich's language, mechanism, glyph units,
or meaning. The next bridge requires a separately frozen harder family or
historical cipher, with structured nulls and transcription uncertainty.
