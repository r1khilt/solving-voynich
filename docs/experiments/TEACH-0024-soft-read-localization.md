# TEACH-0024: first-read mixture versus wrong-row choice

**Status:** prospective exploratory causal assay, 2026-09-24, on
already exposed seed-74111 development data. The unrelated
TEACH-0022 source-frozen MPS training is active and unchanged. No
TEACH-0014 reserved seed-84311 or TEACH-0022 fresh confirmation
seed-84511 prediction is accessed here.

## Question and prior evidence

TEACH-0021 showed that most two-hop errors have a wrong second-row
attention argmax, but its first-row argmax was usually correct even on
those errors. Post hoc first-target attention mass averaged only
0.340/0.313 on wrong examples versus 0.461/0.460 on correct ones.
TEACH-0022's linked random reader gave the correct first row top rank
128/128 while assigning it nearly uniform mass and barely solving
two-hop queries. Can the *model's own* top first row yield better
composition if its value is carried crisply rather than as an
attention-weighted mixture?

This is a narrower causal alternative to saying “the second read is
wrong.” It uses the current model's own finite row selection, not a
gold route oracle. The repository's mechanistic protocol and
TEACH-0021's finite value-intervention controls motivate it. The
global-workspace/J-space literature reviewed locally warns that a
readable direction or attention view is insufficient without
behavioral interventions; this assay does not implement Anthropic's
averaged Jacobian lens or claim a workspace.

## Fixed inputs and conditions

Use only both completed step-6000 public-row oracle checkpoints
from TEACH-0014 v3 and the 128
`composed_confirm_confirm` episodes of the exposed seed-74111
manifest (canonical SHA-256
`09930778461abea7618db8d31eb9aa82c610be0e05ca4d87493a7ff9834c98af`).
Require the two checkpoint and prior TEACH-0021 archive hashes to
match the published compact reports. Process all items, no clean
correctness filtering before inference.

Capture each native read's complete attention vector, valid-row
mask, target row from the public visible mapping and clean logits.
Run these finite `read.i.value` interventions:

1. `identity`: patch the native soft result with itself; logits
   must match clean within1e-5 on CPU.
2. `hard_first`: replace the first soft result with the model's
   own projected value of its **native first-attention argmax row**.
3. `hard_second`: replace only the second soft result with the
   model's projected value of its native second-attention argmax
   row, with the first read unchanged.
4. `hard_both`: use those two native argmax rows at both read-value
   sites; the second value is chosen from the clean run, *not*
   recomputed after the first patch, so this is a two-site
   intervention, not a new autonomous rollout.
5. `gold_first` and `gold_last`: exact correct visible-row values
   as sufficiency ceilings, to cross-check TEACH-0021.
6. `wrong_first`: deterministic first visible row whose right
   symbol differs from the correct first value.

Record exact answer, paired rescue/damage relative to clean, and
native first/second target-row attention argmax and mass. Report
normalized attention entropy `H/log n`, effective row count
`exp(H)` and top-1 mass by clean correctness. The primary
exploratory contrast is hard-first rescue **among all clean errors**
and among errors with a correct native first argmax; these
denominators must both be shown. An improvement from hard-first
would support mixture-induced second-address failure for this
model; hard-first failure with gold-first rescue would implicate
the selected row's value path, learned update or other state
geometry. Hard-second checks whether final output merely needs a
crisper selected row value. None alone proves a unique circuit.

## Validation and resource limits

Source-freeze a CPU runner, independent no-model all-row scorer and
sampled CPU full-logit checkpoint replay before inference. Archive
all 256 item rows including the native attention probabilities and
all condition predictions, with 2,064 logits for panel indices
0/64/127 in each seed/condition. The independent audit reconstructs
visible target and wrong rows, attention sums/masks, entropy and
all paired metrics, compares every clean prediction and both gold
interventions with the earlier TEACH-0021 archive, and verifies
source/checkpoint/manifest hashes. Replay full logits and native
attention on the six sampled items under 0.002 absolute/relative
tolerance. Enforce 20-minute CPU execution, 10-minute replay and
100-MiB new-artifact caps. Commit only compact audited results; keep
bulk rows ignored. This exposed-development result does not amend
TEACH-0014's incomplete confirmation or justify a Voynich
interpretation.
