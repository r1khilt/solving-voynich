# TEACH-0014: prospective parser-versus-router diagnosis

**Status:** diagnostic methods registered while the source-frozen v3 training
campaign is running and before any final neural prediction, checkpoint or
answer/parser decision was inspected. This document does not change that
campaign's metrics or labels. It specifies narrower interventions to explain
a possible raw-model failure; it does not create a decipherment claim.

The exact two-operand grammar is public: strip `EDGE`/`GAP` and pair ordinary
symbols. The answer-only raw model receives no row tensors, but its
overcomplete candidate set contains every adjacent-symbol pair, including
cross-row false edges. Its memory keys/values are projections of the
candidate left/right symbol embeddings, and a gate logit is added as
`logsigmoid(gate)` to each address score. Thus parser-gate leakage, content
addressing, recurrent state update and output readout are distinct failure
routes. See [TEACH-0014 design](TEACH-0014-design.md) and the source-based
[methods review](../research/latent-mechanisms-2026-09-21/METHODS_REVIEW.md).

## Entry and source isolation

Do not inspect or run these neural diagnostics until TEACH-0014's complete
artifact audit and sampled checkpoint-logit/gate replay pass. The public-row
oracle must meet the registered two-hop executor gate before assigning a
downstream raw bottleneck label. The shuffled-label null must pass its ceiling.
Use the frozen final suite only for the following **already registered
diagnostic comparisons**, and report all19 panels/two seeds. No tuning of a
mask constant, confidence threshold, arm, example subset or panel after
seeing results. For any subsequent mechanism search/confirmation, use fresh
TEACH-0015 discovery/confirmation families instead.

## Three parser interventions and matched controls

1. **Native**: evaluate the unchanged raw checkpoint. The saved final gate
   archive and independent parser audit supply exact-row, precision/recall
   and symbolic-solver scores; use the complete native answer archive.
2. **Gold candidate gate**: at the `edge_gate_logits` hook, assign `+20.0`
   to even-indexed valid adjacent-symbol candidates and `-20.0` to odd ones.
   This uses the public visible grammar **only at intervention time** and
   leaves the trained encoder, candidate features, K/V, reader and readout
   unchanged. It is an oracle injection, not learned parsing. The value20 is
   fixed before outcomes; `logsigmoid(-20)` strongly suppresses a false edge
   while `logsigmoid(+20)` is nearly zero. Check that the valid candidate
   count equals `2*rows-1` and that even candidates match serialized rows.
3. **False-only suppression**: preserve every native true-candidate gate
   logit, set false odd candidates to `-20.0`. This asks whether false edges
   hurt without directly improving positive-edge scores. Report the
   native-positive logit distribution; a negative true gate may still prevent
   rescue.
4. **Parity-reversed control**: apply `+20.0` to odd and `-20.0` to even
   candidates, identical magnitude and forward path to the gold injection.
5. **Count-matched random control**: for each episode, select exactly the
   true-row count from all valid candidates with a deterministic seed derived
   from its render ID; assign the same `+20/-20` values. Preserve the chosen
   indices in the artifact. Use several fixed seeds if the resource
   benchmark permits; do not cherry-pick the weakest random draw.

An optional **public-row bypass** constructs an oracle-row variant of the
same raw checkpoint with identical parameters and supplies rows parsed from
visible tokens. This removes false candidates entirely, but also changes the
model's parser/reader input distribution. It can corroborate a rescue; its
failure cannot rule out a parser bottleneck. A separately trained
`oracle_rows_workspace` checkpoint remains the executor positive control,
not a drop-in readout for raw-state coordinates.

For every condition, save full ordinary-answer predictions and sampled full
logits, exact render IDs, source checkpoint hash, intervention assignment,
condition count, elapsed time and MPS allocation. Numerical identity with
native logits under a native-gate replacement is mandatory. An independent
no-model scorer should recompute answer accuracy and all group metrics from
the same frozen manifest. A separate checkpoint replay should recompute the
sampled intervention logits, rejecting any mismatch.

## Address and state readouts

On all composed items, identify the *physical* true first edge `(n,k)` and
second edge `(k,a)` from the visible rows, never from the model's attention.
For each clean run and intervention, report attention mass and rank on these
two target candidate occurrences, total mass on cross-row candidates,
gate-logit margins, the first returned-value norm, `query.1` state norm,
and the final answer margin. Report marker-rich, fully marker-free,
long-distractor, alias and family-cross panels separately, in both seeds.
These are descriptive route measurements; attention mass by itself is not a
causal mechanism. Query.1 donor patches across changed G tables are reserved
for TEACH-0015's fresh assay and competence gate.

## Frozen interpretation order

- If oracle or null validity fails, withhold raw bottleneck labels.
- If native raw behavior fails, native parser/symbolic gates fail, and the
  gold candidate gate rescues at least40 percentage points on both
  factorial exact groups and marker-free boundary items in **both seeds**,
  report `PARSER-GATING BOTTLENECK SUPPORTED`. The false-only condition
  distinguishes excess false edges from weak true-edge scores; neither
  changes the primary threshold. Require gold to beat both reversed and
  count-matched random controls by at least35 points on those same measures.
- If native parser qualifies but native answers fail, report `ROUTING
  BOTTLENECK CANDIDATE`, then describe first/second target-row attention
  and state effects. A more specific first-address, state-update or
  second-address claim requires a separate finite intervention; descriptive
  mass cannot assign it.
- If native answers pass but the registered advantage criterion fails,
  report the original `RAW_BEHAVIOR_NOT_QUALIFIED` decision unchanged;
  diagnostic success does not override a strong control model.
- If the gold gate fails to rescue, report `NO PARSER-GATE RESCUE UNDER
  THIS INTERVENTION`; distribution shift, weak K/V or reader, and a
  genuinely non-discrete native code remain live alternatives. Do not
  equate this negative with proof that segmentation is correct.

This diagnosis is about a known synthetic grammar. Even a clean rescue only
licenses a better architectural test on known ciphers/unknown boundaries,
not a Voynich segmentation or historical interpretation.

`src/voynich/workspace/teacher14_diagnose.py` now implements the five
candidate-gate conditions and physical true-edge attention readouts without
loading any trained checkpoint. Random-weight CPU fixtures verify the native
identity, row/candidate indexing, deterministic count-matched random mask and
strong gold-mask suppression. It remains an **interface qualification**:
trained-checkpoint execution, finite resource estimate, complete result
archive and independent no-model decision audit must be separately frozen
and run after the TEACH-0014 primary artifact/replay gates.
