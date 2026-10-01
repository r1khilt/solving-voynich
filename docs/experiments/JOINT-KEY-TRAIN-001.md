# JOINT-KEY-TRAIN-001 — finite paired conditional-key learning

Prospective development training protocol, 2026-09-30. Question: does a
conditional whole-key inverse learn a better joint density and useful key
proposals on unseen dictionaries/selection-author text, and does larger capacity
help at matched exposures? This is not a blind reader qualification or a
manuscript experiment. Method/prior-work/math/duplication proof:
joint-key-training-design-2026-09-30.md and its linked reviews.

## Inputs, preparation and publication

Reuse LATIN-SOURCE-001 pinned Perseus revisioncc843833e101992ba54549005273cfe61a33504b,
manifestfcade8e816fac499284c103808be3b3e7c88e66711cbb6f426be089ec8ecc9c5. Original
notices/normalization/boundaries/64letter overlap and catalogue caveats remain.
The existing guarded loader opens only12training authors and Pliny; no Nepos,
Apuleius, earlier excluded authors or empirical cipher panel. Both architectures
use6,497,939training letters from>=512letter segments. Pliny original359,276letters
is filtered to>=224letter segments; record exact resulting source IDs/counts/hashes.

Publish full code/protocol/tests before **one** CPU preparation and **one** input
audit, each600wall/500CPU/4GiBsampledhost cap, exclusive start markers, no retry.
Persist compressed source texts/64validation episode metadata outsideGit, plus
compact input/source/window manifest and exact role/source/key replay receipt.
Then publish/verify this preparation checkpoint before one training campaign.
BASE_PATHS freezes all method/model/helper/source/auditor/test/protocol/lock and
original data/audit/resource records; INPUT_PATHS additionally binds the prepared
inputs, preparation and input-audit metadata before training. Source admission
precedes stage timers. Failure is retained; it cannot silently proceed to training.

## Generator and validation population

Each episode has two independently sampled source windows, lengths uniform over
64..224inclusive, then uniform eligible within-segment start for that length.
Draw23literal key rows independently/uniformly from the42one/two-glyph units over
ABCDEF, duplicates allowed. Reject dictionaries in the validation literal **or**
canonical complete-key sets before returning an episode, with at most16draws;
budget exhaustion fails. This conditioning is declared teacher assistance;
no fixed singleton count or bijection. Observed glyphs canonicalized by first
occurrence across the original pair. Source lengths, row presence, segmentation
and true key are supervision/provenance, never encoder inputs.

Network shape is batch4×4×448glyphs, PAD6: two original records duplicatedA,B,A,B.
No additional evidence or source exposure claimed for the repeated pair; the
architecture is invariant in real arithmetic and tested for predictions/gradients.
All records remain nonempty/suffix-padded; no truncation. Alphabet23is public.

Validation:64fixed independent Pliny episodes fromseed72221, same teacher family
and original variable-length policy. Exact literal/canonical collisions cause
preparation failure, not redraw. Both sets excluded from every training seed.
Source-role isolation inherits64letter overlap screening, not paraphrase/short
phrase independence. This validation is reused for checkpoint selection and is
not fresh qualification. Structured/null and fixed-count prior-shift recovery
need separately frozen future panels; neither is claimed tested by this fit.

## Four fits and finite exposure

Fixed sequential arms:small-72203,large-72203,small-72209,large-72209.
Configurations unchanged from KEY-PROPOSAL-SYSTEMS-001:control5,423,146and
large94,981,674parameters. Torch modelseed is the namedseed; separate NumPy
episode RNGseed+100003. Both capacities see exactly the same episode stream per
seed; verify rolling/full metadata digests and complete ledger replays. Same
seed does not imply identical parameters across differently shaped architectures.

20,000updates/fit, batch4:80,000generated dictionaries and1,840,000key-row targets
per fit, maximum80,000updates/320,000episodes/7,360,000row targets overall. Record
actual original source-letter exposure, distinct generated literal/canonical
dictionaries and rejected key draws. Do not infer uniqueness from chance alone.

Full23-row cross entropy; no unused-row target masks, no length/word-break aid.
AdamWdecay.01/foreachFalse/defaultbetasand epsilon, clipglobal norm1. Peaklr3e-4:
linear warmupsteps1..200, then cosine to3e-5at20,000. Float32MPS,CPUthreads2/BLAS1,
PYTORCH_ENABLE_MPS_FALLBACK=0; CPU MHAfastpath disabled for consistent references.
Every loss/preclipnorm finite; all gradients present/finite first/last step.
Norm checking eachstep covers aggregate nonfinite gradients. Do not stop early
for a favorable metric, restart failed arms, resume checkpoints or extend caps.

Save all snapshots at0,200,1000,4000,10000,20000; initial is eligible. Each records
all64full-row key log probabilities, free-runninggreedy full keys/logq, exactkey
count, overall and used-row diagnostics. Used-row truth mask is diagnosticonly.
Selectlowest validation meanNLLperrow, earlierstep on exacttie. No selection on
old cipher-panel answers, futureholdout or compute-rescued case.

Every snapshot: first16glyphs/all23decoder rows from the first validation episode
MPSvsCPUfloat64maxlogitdifference<=5e-5; MPS2vs4recorddifference<=5e-5 and CPUdouble
2vs4<=1e-10. These short checks are not full-input numerical certification.
Store float32checkpoint/config/seed/freeze/input digests plus compressed scores.
Record ledger/window/rawkey/canonicalkey/usedrows/cipherhash for every generated
episode, every optimizer loss/lr/norm/time/exposure/digest, and resource samples
after references/validation batches/checkpoints/every100updates. Progress snapshots
are actual work witnesses; never restart from an observation timeout.

## Resources and complete audit

Percontrol5400wall/4500CPUseconds, perlarge21600wall/18000CPUseconds. Wall alarms
begin after admission; CPU limits count absolute process CPU including setup,
while reported stage CPU excludes admission/imports.
outercap is per-armwall+120seconds. Continue all four terminal outcomes even if
one fails. Sampled Metaldriver<=16GiB/processpeakRSS<=8GiB; fixed shapes reduce
allocation variation, guards are sampled ratherthan physical total-RAM caps.
OneGPUprocess, twoCPUthreads; no overlapping model benchmark/training campaign.
Local paidAPI/cloud budget$0. Keep the machine awake only for the bounded parent
process if necessary. Parentexit0means inventory completion, not four successful fits.

Measured15step rates forecast8.85hours of updates before data/checkpoint/validation/
thermal overhead. Scheduled child envelope atmost15hours8minutes. Approximate
bulkcheckpoint retention4.82GBfor24snapshots plus ignored texts/ledgers/traces;
compact metadata onlyGit. Verify disk and live compute state before launch.

After terminalcampaign, oneCPU-only auditor3600wall/3000CPU/8GiBsampledhost,
twoCPUthreads. It retains complete failures/partial artifacts; successful arms
replay every sourcewindow/key/rejection/seed/disjointness/episode/optimizerdigest,
all checkpoints/config/rawstate finite values/parametercounts/artifactbindings,
every selection/timer/resource inventory. Initial AND selected checkpoints
recompute all64full-input CPUfloat64true-key andgreedy-key logq within.002nats.
Check greedy choices' CPUargmaxdeficit<=1e-4 to permit device-rounded near ties;
this is not bitwise CPU/MPS equality. Check every saved metric's arithmetic.
Other snapshots have finite/hash/accounting checks, not full density numerical replay.
Same-author alternate code path, not independent-agent approval. No repeated audit
or model inference after its failure/cap without an explicitly new protocol.

Total maximum preparation/audit/training scheduled envelope about16hours28minutes
plus admission/imports/parent overhead; actual cost/runtime reported. No open-ended
optimization or material spend. Check all outcomes and publish before new panels.

## Interpretation and decisions

Completed fit is an engineering outcome, not recovered cipherPASS. Development
density-learning criterion: selected full-row meanNLL improves from initialization
by>=.1nats/row in each seed/configuration; report failures ratherthan redefine.
Large-capacity preference requiresboth seeds largeNLL<=controlNLL-.05 and no
used-row greedy accuracy regression versus its matched control; neither gate
establishes recovery, calibration, source identity or causally meaningful features.
All initial/selected/failed outcomes and full denominators remain visible.

Subsequent use must generate whole key proposals, deduplicate/rebind unseen-glyph
orbits within a declared total budget, and apply unchanged exact global fitting.
Then evaluate all positive/null cases and eventually unused qualification. No
proposal count becomes prior mass; neural key probability is not a reader weight.
Mech-interp begins with actual competence and matched causal controls, not row-ID
probes/attention pictures. This experiment does not implement a Voynich translation.

```
PYTHONPATH=.:src .venv/bin/python scripts/run_joint_key_train001.py prepare --freeze <code-commit>
PYTHONPATH=.:src .venv/bin/python scripts/run_joint_key_train001.py audit-inputs --freeze <code-commit>

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONPATH=.:src \
PYTORCH_ENABLE_MPS_FALLBACK=0 caffeinate -i .venv/bin/python -u \
scripts/run_joint_key_train001.py campaign --freeze <prepared-commit>

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
PYTHONPATH=.:src .venv/bin/python scripts/audit_joint_key_train001.py
```
