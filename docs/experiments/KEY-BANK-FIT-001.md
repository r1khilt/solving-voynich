# KEY-BANK-FIT-001: complete fitting-only local key banks

Exploratory development on all8positive and8glyph-shuffle cases from the already
exposed BLIND-CHANNEL-CONFIRM-001. No fresh qualification, key repair from answers,
or transfer reading in this stage. Original FAIL remains. The source-versus-key
diagnostic motivates stronger scoring and uncertainty but does not validate them.

Prior method and Voynich review: [fixed-key diagnostic](KEY-SOURCE-DIAG-001.md),
[shared-key mathematics](../research/shared-key-mixture-decoding-2026-09-30.md).
Ravi/Knight motivates globally consistent key uncertainty; Nuhn etal motivates
stronger language priors; neither establishes this local bank as a calibrated
posterior or a Voynich model. Use the already audited compact order12 source,
minimum support4, Pliny-selected tau64, with all6.498M eligible training letters.
No source retraining, tuning, neural beam score or new passage acquisition.

For each case, read only the four original fitting ciphertexts and original
learned key. Enumerate the parent, every nonidentity row swap and every one-row
replacement from all42literal units over ABCDEF. Exact deterministic deduplication;
all23rows are eligible, including rare rows. No glyph-coverage repairs, answer
matching, frequency filters, transfer support filters, or candidate removal by
small weight. The same policy applies to positives and nulls. Expect at most
1,197keys/case; duplicates in parents reduce the swap count.

Compute exact per-record path-marginal probabilities with the frozen statistical
source and original rho1/225. Log weight = sum of four fitting log likelihoods
minus ln2 times the unchanged literal dictionary code. Normalize across the
complete bank only. Truly unsupported fitting keys retain entries with zero
mass (`null` log weight); numerical/resource failures stop the case rather than
being treated as unsupported. Save every scored candidate and progress trace,
then bank SHA/bytes, best point key, support count, normalized maximum weight and
resource usage. The normalizer belongs to this data-dependent restricted bank;
it is not evidence for the whole channel family or a free model-selection score.

No gold, transfer record or true-key objective is opened in `fit`. A separate
freeze must precede any use of bank keys/weights on transfer. Future comparisons
can separate unchanged point key, fit-selected point key and shared uncertainty;
they are not executed or chosen by this protocol. There is no quality PASS at
this stage. Completion requires the entire declared bank for every case; preserve
incomplete cases and do not discard null failures from a campaign claim.

Resources: two CPU worker processes, numerical thread counts1, no GPU or paid
services. Per case650wall/600CPU seconds, outer660second timeout,500kstates per
record, sampled4GiB RSS/worker (8GiB combined planning bound). At most10,400case
wall seconds/9,600CPU seconds; two workers give at most88minutes before small
launch overhead. Planning cap90minutes. The source-generated benchmark projected
12–33seconds per complete bank, but empirical learned dictionaries (especially
nulls) have much larger lattices; budget conservatively for10–90minutes. Resource
ceilings are not authorizations to increase budgets after observing failure.

Source/protocol and all16parents are committed and remotely verified before
launch. Exclusive started/results paths prohibit accidental replacement. Preserve
every partial trace and process return code. No restart in this namespace.

```
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.:src \
  .venv/bin/python scripts/run_key_bank_fit001.py campaign --freeze COMMIT
```
