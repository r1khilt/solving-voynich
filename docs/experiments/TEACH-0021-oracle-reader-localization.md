# TEACH-0021: localize the public-row reader failure

**Status:** prospective, exploratory exposed-development assay, 2026-09-24.
TEACH-0014 stopped after 12/22 models; TEACH-0020 audited its first six
arms on exposed seed 74111. Both public-row oracle seeds failed the frozen
two-hop competence criteria. This assay uses only those two oracle
checkpoints and three already exposed development panels. It does not open
seed 84311 or qualify any downstream causal-intervention claim.

## Question and source basis

Does the model's two-hop error arise mostly from choosing the wrong visible
row, or does it persist after the correct row value enters the state-update
path? The public-row model receives exact visible rows, but uses a learned
key/query address, projected row value, nonlinear state update and tied
output embedding. The earlier TEACH-0004 typed-row reader reached perfect
row selection on its different supervised synthetic task. TEACH-0018
prospectively identifies reader/state/output failure as the branch if the
public-row arm fails. These comparisons motivate localization; they do not
establish that a particular circuit or architecture is correct. The
literature scope and Voynich-data mismatch are recorded in TEACH-0018 and
the local architecture review. No new literature claim is made here.

## Inputs, interventions and metrics

Use exactly the 128 episodes each from
`first_hop_confirm`, `direct_confirm` and
`composed_confirm_confirm` in the audited seed-74111 development manifest,
canonical SHA-256
`09930778461abea7618db8d31eb9aa82c610be0e05ca4d87493a7ff9834c98af`.
For each of two step-6000 `oracle_rows_workspace` checkpoints from the
interrupted v3 campaign, require its SHA-256 to equal the archived
TEACH-0020 report. Derive every target row by following the unique
left-to-right mapping in the publicly serialized visible rows, not by
reading `signal_paths`. Check that this gives the episode answer.

Capture native read-0 and read-1 attention and count exact target-row
argmax, including conditional counts on clean-correct and clean-wrong
episodes. Record exact answer counts for:

- `clean`: no intervention.
- `identity`: replace each active read result with its own captured
  vector; all logits and argmaxes must be identical within 1e-5 on CPU.
- `gold_first`: replace read-0 result with the model's own projected value
  of the correct visible first row.
- `gold_last`: replace the last active read result with the model's
  projected value of the correct last row.
- `gold_both`: replace both correct row values (one-hop panels coincide
  with `gold_first`).
- `wrong_both`: replace each active read result with the projected value
  of a deterministic different visible row whose right symbol differs
  from the correct value.

The intervention enters the existing `read.i.value` hook *after*
attention aggregation and *before* the model's own update, normalization
and output layers. It therefore tests whether a correct value suffices
through the remaining computation; it does not test whether the model
could have selected that row unaided. Report paired clean-error rescue,
clean-correct damage, and wrong-control accuracy, not only aggregate
accuracy. A native target-row attention hit is descriptive; only the
intervention supports a causal statement.

## Execution and interpretation

Run batched CPU inference under a 20-minute wall cap and 100-MiB new-result
cap. Archive per-episode visible ID, answer, native target-row/argmax
indices and all condition predictions. An independent no-model auditor
reconstructs all visible paths and metrics, compares clean predictions to
the TEACH-0020 archive, checks checkpoint/source/manifest hashes, and
refuses missing or duplicate rows. A separate sampled CPU full-logit
replay checks fixed panel indices 0/mid/last for every condition and both
seeds. Source-freeze before checkpoint inference; never rerun into the
same result namespace. The decision is exploratory: use error anatomy to
select a *new* reader architecture and fresh test, not to retrofit
TEACH-0014 confirmation thresholds.

If gold-last remains wrong frequently, output binding/update is defective
even with a correct final row value. If gold-last nearly always rescues but
gold-first does not, the earlier state-to-second-address path is implicated.
If gold-first rescues and native read-0 attention is poor, first addressing
is implicated. Mixed failures are possible; no aggregate alone proves an
exclusive mechanism. A perfect gold condition is a strong sufficiency
control, not evidence of manuscript semantics.
