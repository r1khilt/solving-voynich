# TEACH-0022 initialization check and first-read attention mass

**Status:** exploratory analysis on exposed development data only. The
TEACH-0022 six-run source-frozen MPS training was already launched; this
check did not change it, select a checkpoint, or open seed 84511.

The linked architecture ties the map applied to a query symbol and a
row-left symbol. That creates a same-symbol similarity prior even with
random weights. Before viewing any new trained checkpoint, both fixed
initialization seeds were run without an optimizer step on four
128-item panels of the already exposed seed-84411 development suite.
The unlinked architecture was tested at the same initialization seeds.
The compact raw counts are in
`results/TEACH-0022/init-check.json`, SHA-256
`df7a6e34caf967c74ac2ac3212cb57a85533e05b089a03f80acfe7ce77c02277`.

| Architecture | Seed | Copy | First-hop answer | Direct answer | Two-hop answer | First target-row argmax on two-hop | Second target-row argmax |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Unlinked | 0 | 128/128 | 0/128 | 0/128 | 0/128 | 6/128 | 3/128 |
| Unlinked | 1 | 128/128 | 0/128 | 0/128 | 0/128 | 4/128 | 8/128 |
| Linked | 0 | 128/128 | 13/128 | 7/128 | 4/128 | 128/128 | 15/128 |
| Linked | 1 | 128/128 | 2/128 | 5/128 | 7/128 | 128/128 | 16/128 |

The linked model's perfect **first-row argmax at initialization is an
architectural prior**, not learned graph execution. It does *not*
produce good first-hop or composition answers. Its attention can put
the correct row first while still mixing enough other row values to
give the wrong transported symbol. The unlinked model has no analogous
first-row preference before training. Both models copy perfectly at
initialization because the query and tied answer embedding make copy
an unusually easy control; this should not be interpreted as trained
reasoning.

The actual first-row **attention probability** makes the distinction
sharp. On these 128 two-hop development episodes, linked means were
0.062677 and 0.062675 across seeds; unlinked means were 0.062499 and
0.062500. There are 16 visible rows, so uniform attention is 0.062500.
The linked model is ranking the right row first while assigning it
almost exactly uniform *mass*. The second target-row means were
0.063525 and 0.063443 linked, also near uniform. This supplemental
mass check was computed after the registered initialization count
check and is descriptive.

There is a simple source-based calculation for this behavior. At
initialization each symbol-coordinate has variance 0.02²; the shared
512×512 linear map's default weight variance is approximately
1/(3×512). Hence the expected same-symbol key self-dot is about
512×0.02²/3 = 0.0683, and the attention's division by √512 makes its
expected score advantage roughly 0.0030. For 16 otherwise near-equal
rows, the corresponding softmax mass is
`exp(0.0030)/(15+exp(0.0030)) ≈ 0.06268`.
This approximation ignores task-vector cross terms, correlations
and the learned update; it predicts the observed *first*-read mean
closely without fitting any data. It explains why an apparently
perfect argmax can coexist with useless value transport.

A separate, post hoc read of the already audited TEACH-0021 old-reader
archive reinforces the need to inspect attention **mass**, not only
argmax. On clean-correct two-hop episodes, mean probability assigned to
the true first row was 0.4614 and 0.4600 in seeds 0/1; on
clean-wrong episodes it was 0.3404 and 0.3130. Median probabilities
were 0.4281/0.4377 versus 0.3310/0.2943. The second true-row
probability was high when the answer was correct (means 0.9781/0.9798)
and near zero when wrong (means 0.0705/0.0362). Those values come
from the source-frozen TEACH-0021 per-item archive and its independent
audit; the conditional comparison was selected after seeing outcomes
and is descriptive.

This refines the earlier localization. The visible failure manifests
at the second read, but an imprecise **first value mixture** can be an
upstream cause even if the first target row wins argmax. The
gold-first-value intervention in TEACH-0021 rescued many errors,
consistent with that possibility. No single attention statistic proves
causation; the correct/wrong value interventions carry the stronger
causal evidence. The ongoing fresh-suite result must report both
answer behavior and native row selection, and a later mechanism assay
should retain attention mass, state update and wrong-value controls.
