# KEY-BANK-READ-002: expanded-bank reading and predictive noise controls

Prospective **exposed development**, registered before new predictions. Previous
READ001 mixture45/3,584edits is the fixed comparison, not a fresh holdout. No
Voynich claim, no repair of original confirmation failure.

## Model and question

Use all16audited KEY-BANK-EXPAND-001 banks unchanged, including zero fitting masses
in the inventory. No fitting, new candidates, source training, language selection
or gold-informed repair. The exact normalized conditional predictive model is
`P(transfer|fit,selected bank)=sum_i w_i product_r Z_i(record_r)`.
One key is shared across both records. This is proper prediction within a
fit-selected finite bank, not full-family Bayesian evidence. Prior model code is
already present in fitting weights; do not charge it again on transfer.

The prior decipherment/weighted-automata review and derivation in
[shared-key-mixture-decoding](../research/shared-key-mixture-decoding-2026-09-30.md),
[bounded lists](../research/bounded-key-list-mixture-2026-09-30.md), and
[search alternatives](../research/unknown-unit-search-alternatives.md) apply.
Native exact marginal inference passed all-case numerical qualification; expansion
improved fitting objectives, including every null. Neither establishes recovery.
This experiment tests whether wider bank coverage helps held-out text and whether
a flexible model discriminates encrypted language from frequency-matched noise.

All original CONFIRM001 identities, two transfer records per case, source order12/
minimum4/tau64 and rho1/225 remain fixed. Source alphabet, Latin, emission family,
record boundaries and original learned parents are supplied assumptions. The
record plaintext length is not supplied to inference. Hash-bound inputs inherited
from the expansion, source and READ001 manifests; no new random seed/allocation.

## Fixed outputs and decisions

**All16cases:** exact native per-key transfer evidence, full bank mixture evidence,
original point and expanded fitting-best point evidence. Compare to the original
fitting-only quantized geometric glyph-iid baseline. Report gain in bits and
positive-minus-paired-shuffle gains. Fixed screen: all16evidences available,
all8positives strictly above iid, and no shuffled case above iid. This is a
weak-null development screen, not evidence against all structured nonlanguage.
No plaintext MAP on nulls is scheduled; they have no intended plaintext truth.
This scope is chosen before execution, not after timing/results.

**All8positives:** unchanged READ001 four-arm reader: original point, expanded
fit-selected point, joint key/text MAP, marginal-text candidate at fixedk8 with
unseen-score bound. Every positive fitting weight is retained. Fixed improvement
rule: all8complete,16supportedrecords, strictly fewer than45mixture edits and every
key at most5%of448. Always report allarms and compare joint versus marginal
separately; missing outputs incur full true-length error with separate availability
and conditional-completed metrics. Report allper-key errors/exactrecords.
After prediction freeze, diagnose true-tuple support and model preference with the
unchanged independent edit/truth diagnostics. Widerk/restarts are prohibited here.

## Integrity and staging

Publish/remoteverify source, protocol and admitted audited banks before one run.
Prediction never opens answer artifacts. Save all-case evidence before positive
MAP; preserve points before expensive mixture inference. Late MAP failure cannot
erase evidence, and unsupported cases remain explicit. Full progress/arrays are
hashed ignored archives with compact tracked reports. Exclusive writes prevent
silent reruns.

Compare every positive key's native evidence against its independent Python k-best
backward evidence, tolerance1e-7nats. Existing reader checks every listed tuple's
encoding/source path and sampled independent forward evidence. The no-gold audit
reconstructs all evidence/weight/bound arithmetic, full inventory/progress,
independent baseline counts, sampled string-context backward inference at original,
best, middle-active,last-active keys for all16cases, and sampled literal whole-bank
candidate support for positive MAP. Record finite numerical deltas and support
agreement; this is not interval arithmetic or exhaustive independent score replay.

Publish/remoteverify all terminal predictions/failures and audit before opening
old answers. The cases remain historically exposed despite this procedural seal.
Reproduce previous original point predictions exactly; use two edit implementations.
No criterion changes after seeing results. Scientific interpretation must separate
bank coverage, inference error and source/objective preference.

## Resources and stop conditions

TwoCPUworkers, numerical threads1, noGPU/paid services. Perworker1200wall/1100CPU
seconds,1210outer timeout,4GiBsampledRSS;500knodes/2Medges/50kprefix expansions
unchanged. Eightpositivebanks are roughly3–4times prior size; expect10–30minutes,
worst-case campaign ceiling~162minutes under per-case outer caps. No cap extension,
subset retry or partial-bank normalization. Audit900seconds, evaluation600seconds.
Allfailures remain in accounting. Existingsource/protocols stay unchanged.

Commands use `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.:src`:

- `.venv/bin/python -u scripts/run_key_bank_read002.py campaign --freeze <published-source>`
- `.venv/bin/python -u scripts/audit_key_bank_read002.py`
- After prediction publication: `.venv/bin/python scripts/evaluate_key_bank_read002.py --freeze <published-predictions>`
