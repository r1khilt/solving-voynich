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
