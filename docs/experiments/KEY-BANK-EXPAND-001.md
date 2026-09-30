# KEY-BANK-EXPAND-001: bounded multi-round key search, full alternative retention

Prospective fitting-only exploration on all16already exposed CONFIRM001cases.
Original confirmation FAIL and later development recovery gain retain their
meanings. No fresh qualification, transfer reading or answer inspection here.

## Why this change

KEY-BANK-READ-001 found exact true text outside the one-move banks for two cases,
while the other residual mistakes concern score preferences. Its separated
within-bank bounds do not justify increasing reading-list size. Complete a
bounded sequence of key neighborhoods before considering coordinated macro
moves. [The prior search review](../research/unknown-unit-search-alternatives.md)
explains local barriers and why an improved objective need not improve recovery.
The weighted-automata/source/Voynich review linked in NATIVE-SUFFIX-001 still
applies. The new native scorer changes execution, not source or key family.

NATIVE-SUFFIX-001 must pass its frozen all-case equivalence/speed gate and audit
before admission. It measured15.34×aggregate acceleration relative to dense full
Python, and12.04×relative to dense marginal-only Python. Point samples project
roughly3–5seconds for a positive one-move bank and17–33seconds for a null bank;
these are sampling-based estimates, not guarantees for later keys/neighborhoods.

## Fixed algorithm and model

For every original positive/shuffle case, start from its original CONFIRM001
learned dictionary and four fitting ciphertext records. Use the identical large
order12/minimum4/tau64 source, supplied23-letter/6-glyph alphabets, literal units
of lengths1–2, duplicate units allowed, unchanged dictionary code and rho1/225.
No training, gold-directed row choice, rare-letter repair, coverage filtering,
source selection, new source corpus or transfer data.

Perform at most **four complete neighborhood rounds**. A neighborhood contains
the current center, every unequal-row swap and every one-row replacement from
all42units, deterministically deduplicated. Cache identical keys across rounds;
each distinct key gets exact native marginals for allfourfitrecords. Score is
sum of log likelihoods minus ln2 times the unchanged literal dictionary code.
Choose the first highest-scoring key in that neighborhood's fixed order. Move
the center only for improvement>1e-6nats; otherwise stop. This tolerance exceeds
the benchmark's roundoff but does not establish interval-certified ordering.

Keep the **entire union of all completed neighborhoods**, including rejected
alternatives and keys with zero fitting mass. At most4,785distinct keys/case.
Normalize positive log weights after the run completes. No posterior cutoff,
selection by future accuracy or retention by visitation frequency. The fitting
normalizer is restricted/data-dependent, not whole-family evidence.

Record every scored key, all per-record likelihoods/nodes/edges, code costs,
round memberships, accepted gains, terminal center and best bank key. A tolerance
stop certifies only the named center's examined neighborhood up to that tolerance.
The best bank key can differ by a sub-tolerance amount; do not transfer that
certificate to it. Four accepted moves do not certify the final center's next
neighborhood. No global optimality claim. Numerical failure or interrupted
round retains progress but cannot yield a completed normalized bank.

## Checks, staging and resources

Source/algorithm/protocol, original parents and passing native benchmark/audit
are committed/pushed/remoteverified before one campaign invocation. Use the
hash-bound library built for that benchmark. Full frozen source/path identity
guards remain active. Artificial tests cover distant improvements, complete
candidate retention/deduplication, exact first-round correspondence to the old
bank, local barriers, tolerance scope, zero mass, numeric errors, interruption,
and stable normalization under large common log-score offsets.

After fitting, independently check complete round membership, cache/trace
identities, selection/gain/weight arithmetic and every retained case. All first
neighborhood rows for positives can be compared with the old completed banks;
new rows and nulls need sampled independent Python replay. No historical partial
null trace is a complete reference bank. Preserve every failed case.

Two CPU workers with one numerical thread each. Each case650wall/600CPU seconds,
outer660seconds,500kstates/2Medges per record, sampled4GiB/worker. Full90minute
planning ceiling; current samples suggest5–20minutes, with substantial uncertainty
from changed dictionaries. No GPU or paid services, no resource extension,
smaller-bank fallback, rerun, or substitution for an old failed namespace.
Further decoding/evaluation requires a new freeze after audits; do not inspect
answers in this fitting stage or convert fitting gains into recovery claims.
