# KEY-BANK-EXPAND-001: all sixteen expanded key banks complete and pass audits

The wider fit-only search completed for **all eight positive examples and all
eight scrambled controls**. It evaluated71,430case-key candidates, deduplicated
within each case, across62complete neighborhoods.65,838candidates have positive
fitting mass; all zero-mass entries and rejected alternatives remain inventoried.

This closes the missing-control-bank computation gap from KEY-BANK-FIT-001.
It does not yet establish better reading accuracy or meaningful-text detection:
no new transfer predictions or answer access occurred in this fitting stage.
The source, code penalty, original warm starts and fitting inputs are unchanged.

## Positive searches

| Case | Bank size | Complete rounds | Stop | Fit objective improvement over original, bits |
|---|---:|---:|---|---:|
|1|4642|4|Center's local tolerance stop|5.395225|
|2|3491|3|Center's local tolerance stop|0.420141|
|3|4640|4|Center's local tolerance stop|0.010605|
|4|4629|4|Round limit|299.963325|
|5|4638|4|Center's local tolerance stop|0.013278|
|6|3491|3|Center's local tolerance stop|1.370613|
|7|4637|4|Center's local tolerance stop|22.606932|
|8|4637|4|Round limit|123.850354|

The previous one-move bank found182.354070and77.971154bits of fitting improvement
for cases4and8. Repeated expansion now reaches299.963325and123.850354bits.
Those are concrete further improvements in the fixed fitting objective; they
are not percentages of text recovered. These two searches reached their fixed
four-round limits, so the next neighborhoods have not been certified optimal.

Six other positive centers reach the registered1e-6nat local tolerance stop.
A certificate applies only to that center and the examined swaps/replacements,
not to a global optimum or every key in its retained bank. No coordinated
multi-row move beyond successive accepted single moves was attempted.

## Scrambled controls remain part of the experiment

| Control | Bank size | Complete rounds | Fit improvement over original, bits | Fit wall seconds |
|---|---:|---:|---:|---:|
|1|4592|4|46.283997|127.354534|
|2|4566|4|66.462593|146.799947|
|3|4638|4|35.589037|95.887315|
|4|4567|4|32.191985|102.042428|
|5|4531|4|46.232952|69.739517|
|6|4558|4|61.859651|97.568311|
|7|4561|4|42.133294|131.552505|
|8|4612|4|106.797735|96.061709|

All controls reach the round limit, with further fitting improvements still
possible. The fact that noise improves too is precisely why fitting gains alone
cannot identify a language or encoding system. These complete banks now permit
an all-case predictive comparison that was previously unavailable.

The old600CPU-second failures remain untouched. The new namespace uses the
separately validated native scorer and a different, prospectively registered
four-round policy. It is not a relabeling or hidden restart of those failures.

## Validation and cost

Source/protocol`accca6d7b07920ac6f25d491e1db087349bab2a5`was pushed and remotely
verified before the sole16-case invocation. All16worker processes exit0; no
budget extension, smaller-bank fallback, missing case, or partial normalization.

All16bank audits pass complete independent neighborhood enumeration, candidate
ordering/deduplication, round membership, selection/gain, code-cost and posterior
weight arithmetic, archive identities and the complete progress streams.
Independent string-context backward inference replays416sampled records,
including each accepted center and the selected/middle/end bank samples;
maximum likelihood discrepancy1.1369e-12nats. Every first-neighborhood candidate
on the positives is compared with the old full-bank records:38,168per-record
scores/graph counts, maximum likelihood difference1.5917e-12nats.

These checks do not independently recompute all285,720new per-record scores.
The post-run campaign audit additionally checks94frozenbindings, all16process
outcomes, compact round summaries, bank/audit identities and resource totals.
Seven adversarial audit tests reject corrupted memberships, scores, weights,
selection, terminal centers and code costs. Full2,185tests plus23subtests pass;
10skip. Changed-file lint and whitespace checks pass. Full-tree lint retains
exactly five previously documented findings in unrelated files.

Campaign527.621113seconds(8.79minutes), two CPU workers. Summed worker time
976.359902wall/975.212119CPU seconds, maximum individual RSS843,808,768bytes.
Independent audits add353.291360summed wall/352.618073CPU seconds, overlapping
some fitting and one another; their largest individual RSS is1,555,759,104bytes.
Do not add summed worker times to elapsed campaign time. No GPU or paid service;
$0research compute spend.

## Next unresolved question

Freeze these audited banks before using new ciphertext. Compare predictive
probabilities on all sixteen positive/control cases, and test whether the
larger banks improve the old45-edit reading result. For the controls, exact
predictive evidence is available from the native marginal scorer without
inventing a plaintext accuracy target. Any more expensive MAP-reading choice
must be declared separately, with resource failures retained.

Retaining more possible keys may repair missing support, or it may let the
source model prefer a more convincing wrong reading. A fresh end-to-end
unknown-key qualification remains necessary after development. No Voynich
reading, source-language identification or historical cipher family is established.
