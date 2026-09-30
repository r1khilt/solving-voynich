# KEY-BANK-READ-002 results: modest reading gain, complete controls, clearer failure cause

The expanded bank reduces **45→43character edits out of3,584** on the old exposed
panel: a4.44%relative reduction, not a breakthrough. All8positivecases now have
at least one positive-weight key capable of encoding the entire true text tuple.
The remaining wrong outputs are preferred by the model despite that support.
The complete8positive/8shuffle predictive screen passes. Fresh end-to-end
unknown-key qualification and actual Voynich decipherment remain unresolved.

## Fixed comparison

Source/protocol `583b71190f452d5182d6443a93b534fd85e4717d` and prediction/audit
checkpoint `adafb316fc235d1bb96674076c7f4ab59a9a12f7` were each pushed and remotely
verified before their dependent execution. One campaign, no retries/extensions.
All16processes exit0. All16exact predictive evidences and all8scheduledpositive
four-arm readings complete. Null plaintext MAP was outside the prospectively
registered scope; all noise controls are included in the evidence screen.

| Decision | Previous READ001 edits | Expanded READ002 edits | Exact records | Supported records |
|---|---:|---:|---:|---:|
| Original large-source point key |309|309|4/16|15/16|
| Fitting-best point key |269|264|5/16|15/16|
| Joint key/text MAP |46|45|6/16|16/16|
| Marginal-text MAP |45|43|6/16|16/16|

Allarms use the same3,584-letter denominator. Expanded marginal error rate1.200%;
worst key17/448=3.795%. The registered development clauses allPASS: complete,
16supported, fewer than45edits, everykey≤5%. This modest threshold was frozen
before prediction; PASS is not fresh qualification or statistical significance.
Marginalizing instead of joint key/text selection saves only2edits, keys1/5.
The earlier recovery of key6's unsupported passage is retained, not a new gain.

| Key | Old marginal edits | New fit-point | New joint | New marginal | True-tuple supporting keys |
|---|---:|---:|---:|---:|---:|
|1|8|8|9|8|376|
|2|6|6|6|6|294|
|3|7|7|7|7|398|
|4|8|2|1|1|206|
|5|2|2|3|2|376|
|6|0|224|0|0|2|
|7|2|2|2|2|423|
|8|12|13|17|17|210|

Sixkeys tie their previous marginal errors, one improves, one worsens. Key4's
seven-edit improvement is partly offset by key8's five-edit deterioration.
Showing only the improved case would misrepresent the result.

## All-case predictive screen

Bits gained over the original fitting-only quantized geometric glyph-iid model:

| Key | Encrypted language | Paired shuffled control |
|---|---:|---:|
|1|733.616|−159.293|
|2|786.991|−188.203|
|3|837.632|−171.193|
|4|867.306|−158.232|
|5|1,016.544|−189.432|
|6|787.959|−141.219|
|7|623.132|−117.430|
|8|557.530|−161.282|

All8positives beat iid; no shuffle does. The fixed screenPASS is now available
because the wider fitting campaign completed every control bank. It rejects these
frequency-matched shuffles, not all structured nonsense or alternative historical
processes. The proper conditional predictive mixture uses one shared key and
normalized fitting weights; its data-dependent bank is not full-family evidence.
The original CONFIRM001 failure and READ001 missing-control results stay unchanged.

## What now limits recovery

All8floating MAP bounds separate; the smallest margin is.830068nats. Subject to
numerical checks, increasing the fixedklist cannot make the true tuple beat the
selected tuple under this same objective/bank. Keys4/8's prior support gaps are
closed. One key's whole text is exact; allsevennonexact cases have the true tuple
supported but ranked below a wrong one. This distinguishes source/key scoring
from a missing true candidate that could win the current objective.

Posthoc algebra decomposes the true-minus-selected log score into language-source
and fitting-key-support contributions. The language source prefers the wrong
output in **allsevennonexact cases**. Key-support mass favors the truth forkeys1/4/8
and is effectively tied for2/3/5/7. In key8, the source disadvantage is36.132nats;
a7.014nat advantage from fitting-key support cannot offset it, leaving29.118nats
against the true tuple. This is an observed scoring error under the declared model,
not evidence that the historical Voynich text has this encoding.

The independent no-gold posterior diagnostic gives selected wholetext mass
.0080–.4563; none meets the>half-mass sufficient condition for unchanged
metric-risk choice. Key8 has.9678mass on one key but only.0118mass on its selected
wholetext. Key uncertainty and reading uncertainty are different. These are
restricted model probabilities, not calibrated historical confidence. A decision
rule targeting edit loss could be tested, but no such benefit has been measured.
See the [theory note](../research/decision-risk-for-decipherment-2026-09-30.md).

## Validation and cost

- Full2,199tests+23subtestsPASS,10skips; changedRuff/diffhygienePASS. Full-treeRuff
  retains exactlyfivepreviousunrelated findings.
- All153frozen source/input paths checked.65,838positive-fittingcase-keys produce
  131,676native per-record transfer scores. All29,377positivekey total evidences
  agree with independent Python k-best backward inference within2.956e-12nats.
- All213,112listed tuples re-encode and replay source scores;1,860sampledforward
  records agree within2.502e-12.128independentstring-contextbackwardrecords agree
  within1.137e-12. The latter is a sample, not all131,676recordreplays.
- Allinventory/progress/baseline/bound arithmetic checked;522sampledtuples produce
  1,917,083whole-bankliteralcompatibilitychecks, not exhaustive candidate rescoring.
- After prediction publication:64independentrecord edit checks,16exact original
  point-output replays, independent totals/denominators/gates and eight score
  decompositions PASS. No gold-informed repairs or new predictions.
- Prediction893.900elapsedseconds(14.90min); summedworkers1,626.842wall/
  1,625.599CPU seconds, maximumworkerRSS876,773,376bytes. Audit96.928wall/
  96.806CPU seconds,813,989,888bytespeak. Evaluation1.414wall/1.373CPU seconds,
  968,376,320bytespeak. Diagnostic/accounting and test overhead are separate;
  worker sums are not elapsed campaign time. NoGPU/paid compute.

Compact authoritative records are `results/KEY-BANK-READ-002/evaluation.json`,
`audit.json`, `prediction-accounting.json`, `evaluation-audit.json`,
`posterior-diagnostic.json` and `score-decomposition.json`. Hashed bulk predictions
remain under ignored `outputs/KEY-BANK-READ-002/`.

Next: stop treating broader local search as the main remedy for this panel.
The existing complete pipeline needs fresh unknown-key evaluation. Source quality
and an explicitly loss-aware reading rule are separate, testable follow-ups;
neither should be tuned until these eight exposed examples look perfect and then
called a decipherment. Historical controls and Voynich model adequacy remain ahead.
