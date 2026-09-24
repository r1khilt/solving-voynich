# TEACH-0024: native hard-read result

**Status:** exploratory causal result on previously exposed synthetic development data, 2026-09-24. The prospective conditions and fixed inputs are in [TEACH-0024-soft-read-localization.md](TEACH-0024-soft-read-localization.md). Source was frozen at `a0e79964596d27b676564459f45f2c42acdbf2f6` before trained inference. Neither the older reserved seed84311 nor TEACH-0022's new confirmation seed84511 was scored here.

## Inputs and validation

The assay used the two completed step6,000 TEACH-0014 v3 `oracle_rows_workspace` checkpoints (SHA-256 `114e03863529fba2048925e843b76da6a9621422072a99bae18cc221ae091a7c`, `a40a3616527fe2c1a4b66fd1125ba9ad1b7aab214e26d9f8f89309dd0c82fdcd`) and all128 `composed_confirm_confirm` episodes in each run from the exposed seed74111 suite (canonical SHA-256 `09930778461abea7618db8d31eb9aa82c610be0e05ca4d87493a7ff9834c98af`). Each condition patched the existing post-attention read-value site, leaving the checkpoint and episode unchanged. The first/second *native* hard conditions used the model's own clean-run attention argmax; gold conditions used the correct visible row.

The full repository suite passed1,041 tests,8 skipped and23 subtests before source freeze; changed-file Ruff and diff hygiene passed. The CPU assay completed in1.214s and archived2,191,449B. The independent no-model all-row audit passed and cross-checked every clean and gold prediction against TEACH-0021. The separate CPU checkpoint replay passed48 sampled full-logit vectors, all condition predictions and both native attention vectors with maximum absolute logit error0.0. Compact audit SHA-256 is `c34366773ad95cdd66c61c80c0ce28312896581a8840ba91248b514e3c313c56`; replay audit SHA-256 is `edb7e17dc6966703c64e3254fd2ab47e73cb533d8feef54cae21d81800cc80ac`. Raw per-item rows (SHA-256 `a46b8fae152315a8706636658592854dfbc687574d21366d60d87e3c30b5aebe`) remain ignored.

## Paired result

| Condition | Seed0 correct /128 | Seed1 correct /128 | Seed0 errors rescued /31; correct damaged /97 | Seed1 errors rescued /28; correct damaged /100 |
| --- | ---: | ---: | ---: | ---: |
| Clean, identity | 97 | 100 | 0; 0 | 0; 0 |
| **Native hard first** | **117** | **117** | **23; 3** | **19; 2** |
| Native hard second | 99 | 100 | 2; 0 | 0; 0 |
| Native hard both | 99 | 100 | 2; 0 | 0; 0 |
| Gold first | 123 | 121 | 27; 1 | 22; 1 |
| Gold last | 127 | 128 | 30; 0 | 28; 0 |
| Wrong first | 4 | 4 | 0; 93 | 0; 96 |

The first target row was already the native attention argmax on27/31 seed0 and24/28 seed1 clean errors. Native hard-first repaired23/27 and19/24 of those cases (42/51 pooled), and the same23 and19 cases were rescued by gold-first because the replacement value is identical when native and gold row agree. Across all clean errors, native hard-first rescued42/59 while damaging5/197 previously correct cases: net+37/256 exact answers. The absolute two-hop accuracy changed from197/256 (77.0%) to234/256 (91.4%) across the two runs. This is paired behavior within this particular exposed panel, not a confidence interval over manuscripts or graph families.

Clean-wrong items assigned mean attention mass0.340/0.313 to the correct first row, versus0.461/0.460 on clean-correct items; their first-read effective row counts were8.29/8.57 versus6.78/6.54. The correct second-row mass collapsed to0.0705/0.0362 on clean-wrong items, versus0.978/0.980 on clean-correct items. Forcing the clean-run *second* argmax value did little: most failed second selections were confidently wrong. The `hard_both` condition also used that old second argmax, so it overwrote benefits of hard-first; it is not a new two-step rollout after hardening.

## Interpretation and limits

The model frequently had enough information to identify the first correct row by top rank, but its soft read mixed many values. Replacing that mixture with its own top-row value often repaired the full two-hop answer. This is direct behavioral evidence that first-read value mixing contributes materially to second-step failure in these trained synthetic models. The remaining errors and the gap to gold-last show it is not the entire failure. The patch is a designed intervention at a known read-value site, not a discovered neuron, a unique mechanistic circuit, autonomous learned decipherment, or a Voynich interpretation. Since the panel was previously exposed during model diagnosis, an untouched fresh suite is required for any generalization claim; TEACH-0022's registered confirmation comparison remains separate and in progress.
