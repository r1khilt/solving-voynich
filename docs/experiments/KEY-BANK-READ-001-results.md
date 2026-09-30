# KEY-BANK-READ-001: retaining alternative keys repairs an unreadable passage

**Character errors fall309→45of3,584 on the eight exposed development ciphers.**
The fixed positive-case engineering gate passes. This is a substantial local
recovery improvement, not fresh unknown-key qualification or a Voynich reading.
The eight scrambled controls still lack complete fitted banks.

| Decision | Character edits /3,584 | Edit error | Supported records /16 | Exact records /16 |
|---|---:|---:|---:|---:|
| Original learned key, improved source |309|8.622%|15|4|
| Best key using fitting passages only |269|7.506%|15|4|
| Best key/text using the new ciphertext too |46|1.283%|16|5|
| Sum support across compatible keys |45|1.256%|16|5|

The last arm reduces edits85.44%relative to the original key and83.27%relative
to the fitting-only choice. All8positive inferences complete, all16records are
supported, and the worst key has12/448=2.68%error, below the registered5%ceiling.
These rules were frozen before predictions. There is no significance or
fresh-generalization claim from this small, adaptively explored panel.

## What actually produced the gain

The largest improvement is key6: one previously unsupported224-letter passage
becomes exactly correct. Both records for that key now read perfectly. The
winning dictionary change was not hand-picked from the answer: it was among
every one-row replacement/swap of every original key, fitted before this
follow-up opened its transfer records or answers.

A post-evaluation inspection of saved probabilities finds that the eventual
key6choice, `y: C → F`, ranked18th on fitting data with2.681%conditional bank
weight. After conditioning on the new ciphertext, its key probability rises
to82.758%within the restricted bank; only55of952positive-fitting keys support
both new records. Its selected whole-text tuple has21.484%of the model's
conditional text probability. These are model probabilities, not calibrated
confidence in an unknown historical translation. No keys or predictions were
changed by this inspection.

This is the useful mechanism: **retain plausible alternatives until additional
observations distinguish them**. Committing to the best fitting key discarded
the required mapping. A stronger language model alone could not repair that
missing dictionary support.

Do not attribute the entire gain to summing over keys. Joint key/text choice
already reaches46edits; summing keys improves just one character, on key5.
Of264fewer errors versus the original arm,224come from key6. Excluding that
case, errors still fall85→45, largely from the stronger fitting search. The
summed-key arm and fitting-only arm have identical aggregate errors on those
seven other keys. Key1 actually worsens6→8; the aggregate gain is not universal.

## Every positive case, including residual failures

| Key | Original | Fit-only key | Joint key/text | Summed keys | True whole text available within positive-weight bank? |
|---|---:|---:|---:|---:|---|
|1|6|8|8|8|Yes; model prefers wrong text|
|2|6|6|6|6|Yes; model prefers wrong text|
|3|7|7|7|7|Yes; model prefers wrong text|
|4|18|8|8|8|No|
|5|2|2|3|2|Yes; model prefers wrong text|
|6|224|224|0|0|Yes; exact chosen text|
|7|20|2|2|2|Yes; model prefers wrong text|
|8|26|12|12|12|No|

For keys4and8, the limited one-move bank excludes every key capable of producing
the exact true text tuple. A more thorough reader over this same bank cannot
fix that support failure. The20observed errors in these cases are not claimed
to be an irreducible20-edit floor; only exact recovery is proved impossible
within the bank.

For keys1/2/3/5/7, the true tuple is supported but loses to the chosen wrong
tuple by4.216/4.352/4.037/3.602/1.283nats under the summed-key objective.
All8search bounds are separated with floating checks; smallest best-minus-unseen
margin0.624289nats. Thus a wider list using the unchanged objective/bank would
not beat the current winner in exact arithmetic if the checked bounds hold.
These are not interval-arithmetic proofs. Improving the source, key weights or
decision criterion is a different question from searching harder.

Even near-certain key weights can be misleading when the bank excludes the
truth. Keys4and8 concentrate almost all bank mass on one candidate while their
true text has zero support. The new posterior diagnostic makes that limitation
explicit; no posterior threshold is adopted from these observations.

## Controls, staging and validation

- Source, complete fitted banks, unavailable-control inventory and protocol
  committed/pushed/remote-verified at`99847635e9ca54eb8dfc659669aa9d2d739d9352`
  before the sole prediction campaign.
- All predictions and the answer-free audit committed/pushed/remote-verified
  at`eff63841c0e1c29aab2172740dbe33617f7faaab`before the sole evaluation.
  Historical exposure of these cases remains; these freezes do not make them fresh.
- All16prediction processes exit0. The eight nulls retain original-point
  readings and explicit unavailable-bank status. Their missing mixture results
  prohibit an all-case or null-discrimination success claim.
- All8,287positive-weight keys are retained.58,792listed tuples re-encode and
  replay their path scores;540independent forward-record checks agree within
  2.5012e-12nats. No k increase, positive-weight pruning or restart.
- Independent audit checks complete inventories/progress, archive hashes,
  list ordering, union counts, joint optimum and mixture-bound arithmetic.
  It directly checks539,607candidate/key pairs across521sampled distinct tuples,
  including every winner. It does not independently re-score every union tuple
  against every key.
- All64positive record edits match an independent implementation. All32original
  baseline records reproduce the prior report exactly, including scores and
  graph counts. Post-evaluation audit rechecks89bindings, every denominator,
  compact aggregation, support/exact counts and the unchanged engineering gate.
- Full2,136tests plus23subtests pass;10skip. Changed-file lint and whitespace
  checks pass; full-tree lint retains exactly five previously documented,
  unrelated findings.

Prediction campaign209.260069seconds. Summed workers358.205722wall/
357.846601CPU seconds; maximum per-worker RSS882,098,176bytes. Answer-free
audit3.644472seconds; evaluation0.861377seconds. Two CPU workers, no GPU or
paid services; $0. Prior fitting cost/failures remain in the separate fit report.

## Next decision

A fresh end-to-end unknown-key test is still required. Before spending that
allocation, address both identified bottlenecks: a search that can move beyond
one edit of an old key, and source/decision errors despite adequate support.
Preserve uncertain alternatives through new-ciphertext reading. Do not spend
on larger k for these already-separated within-bank objectives.

The slow null-bank fitting is also an engineering gap. Benchmark an exact
likelihood-only/native lattice scorer against the audited Python inference
before increasing search budgets. The existing fitting loop computes Viterbi
text/backpointers for every candidate although bank weights need marginal
likelihood only; removing that work is an unmeasured hypothesis at this point.
The validated dense source tables are available without changing probabilities.
A separate bounded registration is required before any further empirical run.
