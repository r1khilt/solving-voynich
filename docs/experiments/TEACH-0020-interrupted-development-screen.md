# TEACH-0020: interrupted-run development screen

**Status:** prospective exploratory evaluation, 2026-09-24. The frozen
TEACH-0014 v3 process disappeared after twelve completed arm/seed artifacts,
while its status file remained `running`. There is no complete primary
report. This screen uses **only the previously exposed seed-74111
development suite**, not the reserved seed-84311 final predictions. It
does not relabel or repair the interrupted primary campaign.

## Question and inputs

Do the six complete TEACH-0014 arm pairs show a clean gap between the
public-row oracle and the raw candidate parser/reader? Evaluate precisely
the first six frozen arms in original order, both seeds, on all nineteen
panels and 4,736 episodes of the already exposed development manifest
`outputs/TEACH-0016/teach14-74111.json`. Its canonical SHA-256 must be
`09930778461abea7618db8d31eb9aa82c610be0e05ca4d87493a7ff9834c98af`.
No checkpoint selection or optimization is performed here.

Before inference, verify each of the twelve final checkpoint files is
loadable and has exactly its registered arm, seed, step6,000, configuration
and parameter count; each corresponding loss archive has6,000 valid
step records. Compare answer-input/label hashes across the completed arms
with the oracle control. Check original benchmark and current model/task
source bytes against the archived `37288fe` source hashes. Record every
checkpoint and input hash in the screen archive.

## Evaluation and decision boundary

Run local MPS eval-mode batch32, archive every render ID/predicted symbol
for every completed arm/seed/panel, and full logits at panel indices0,
midpoint and last. Independently check the manifest, ordering, checkpoint
and source hashes, all prediction denominators and per-panel exact answers.
The original TEACH-0014 item/group and two-hop metrics may be shown on this
development suite for diagnosis, but **none is a registered confirmation
decision**. The screen should identify a likely failure site or a reason
to finish the campaign, not certify a causal mechanism. A separate CPU
sampled-logit replay is required before treating even these exploratory
predictions as checkpoint-derived.
Before any scores are read for architecture choice, replay full CPU logits
from every checkpoint at indices0/midpoint/last on four fixed panels:
`composed_confirm_confirm`, `factorial`, `boundary_groups`, and `long_ood`.
Require identical argmax and elementwise absolute/relative tolerance0.002;
a failed replay invalidates the exploratory score. The separate CPU replay
has a30-minute wall cap.

No trained inference on seed84311 occurs in this experiment. All later
frozen TEACH-0014, TEACH-0015 and TEACH-0016 admission gates remain closed
until an honestly labeled complete campaign and its independent replay
exist. Preserve the interrupted status and partial artifacts unchanged.

## Resources and failure

The original source-matched backward benchmark's slowest timed-step
median was0.2133s across the completed arms. A conservative forward
projection uses `1.5 × slowest median × 12 models × 148 batches + 300s`,
under900s; evaluation may have longer sequences, so this is only an
admission estimate. Enforce1h wall time,12GiB sampled MPS allocation and
1GiB new artifact caps. No paid service is used. Stop with a recorded
reason on any hash mismatch, absent checkpoint, nonfinite logits, resource
cap or failed replay. Keep bulk predictions ignored; commit only compact
score, audit, notebook and provenance.
