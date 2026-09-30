# CONFIRM-002-DIAG-A: objective and discarded-restart diagnosis

Registered 2026-09-30 after the frozen fresh answer evaluation. **Exposed,
exploratory diagnosis**, not another qualification, repaired confirmation,
new key fit or new plaintext recovery. The original failure remains.

## Question and rationale

The fresh primary reader makes789/7,168 errors. Cases01/32 account723;
their true tuples are outside the fitted bank. Other14cases make66/6,272
errors, with each key's edit count equal to its true-key large-source decoder.
The first-stage source selects one winner among16restart trajectories before
higher-order refinement. Are bad dictionaries actually favored by the fitting
objective, or did search miss known better candidates? Did the discarded
restart endpoints already contain more useful starting basins?

Source review: [fresh qualification review](../research/blind-channel-fresh-qualification-review.md),
[original fresh registration](BLIND-CHANNEL-CONFIRM-002.md),
[prior source/key diagnosis](KEY-SOURCE-DIAG-001-results.md),
[Nuhn et al.2013](https://aclanthology.org/P13-1154.pdf) and
[Kambhatla et al.2018](https://aclanthology.org/D18-1102.pdf).
Their supplied-symbol searches motivate separating context-dependent scoring
from beam/restart coverage. Those published results do not establish our
variable-unit Latin recovery. No architecture or estimator changes here.

## Fixed inputs and actions

All16 original positives; no selected-case omission. Use the exact published
CONFIRM-002 fit panel, fit ciphertexts, answers, parents, banks and restart
traces, and audited fresh evaluation. Source versions/checksums, normalization,
geometric rho1/225, model code and library are inherited unchanged. The input
freeze must contain all original compact JSON manifests and new diagnostic
source/protocol/tests. The previous candidate/input hashes are rechecked.
Answers are already exposed by the original evaluation; nothing here is blind.

For each case:

1. Extract the terminal accepted dictionary of every scored order1 restart.
   Preserve unsupported starts, no-sweep states and accepted partial-sweep
   updates. Do not search its neighbors or select its best internal trace point.
2. Score each supported endpoint, the original selected order1 dictionary,
   order3 parent, expanded fit-selected dictionary and generating dictionary
   under the unchanged large exact source. Deduplicate equal dictionaries for
   computation while preserving each label. Maximum20labels/case,80native
   record scores/case,1,280total. These are fitting likelihoods, not new transfer
   readings or a recovered-key qualification.
3. Replay generating-key large likelihood independently with string-context
   backward inference on all64fit records. Check unchanged parent/best scores
   against saved banks. For the generating key under original order1/order3,
   compare production and separate manual backward likelihoods on all128
   case/source/fit records. Include literal model-code costs in each comparison.
4. Report learned-minus-generating costs, all restart endpoints and stop states,
   literal row matches, best endpoint under large fit objective, and its cost
   relative to original initialization and expanded selection. Finding a known
   better candidate disproves global optimality; not finding one does not prove
   it. More matching rows or a lower fitting cost is not a new reading result.

No source training/selection, changed key bank/weights, intervention, new key
optimization, transfer decoding or historical manuscript input. No synthetic
confirmation gate is changed. Results and failure records use a new namespace.

## Limits and verification

One local CPU process, numerical threads capped at1;900wall/750CPU seconds,
4GiB sampled host RSS, no paid/API/GPU use. Planning estimate1–5minutes; old
compressed restart traces can dominate parsing/memory. No retries, subset
completion or cap increases. Exclusive start/result writes retain failures.

Artificial tests check accepted versus unaccepted terminal updates, partial
sweeps, missing/duplicate/discontinuous restart traces, unsupported starts,
objective sign/code costs, nonfinite values and comparison tolerance. All
numerical replay tolerances1e-7nats; save maximum measured differences.
Publish source/protocol and the original failed evaluation before this one run.
Publish all16diagnostic outcomes with the original failure and complete resource
record. This can motivate a future multiple-basin fitting pipeline but cannot
establish that retaining endpoints will recover fresh unknown keys.
