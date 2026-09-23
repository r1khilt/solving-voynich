# How a raw binding circuit develops

Status: prospective analysis written while TEACH-0012 is running and before any final report or
confirmation score was read. It is conditional on behavioral eligibility and a separately
discovery-frozen final mechanism. It is not a result or experiment registration.

## Why loss curves are not enough

An apparent capability jump can have at least four causes:

1. a new reusable algorithm forms;
2. an existing weak algorithm becomes strong enough to cross an accuracy threshold;
3. a shortcut is replaced, suppressed or combined with another route;
4. the data distribution changes and the same circuit is merely evaluated under a new mixture.

[Wu, Geiger and Millière
(2025)](https://proceedings.mlr.press/v267/wu25j.html) report random, early-assignment heuristic
and systematic dereferencing phases in a serialized variable-binding Transformer. [Olsson et al.
(2022)](https://transformer-circuits.pub/2022/in-context-learning-and-induction-head/)
associate induction-head circuit formation with a phase change in in-context learning. [Nanda et
al. (2023)](https://arxiv.org/abs/2301.05217) show in modular addition that a seemingly abrupt
grokking transition can conceal gradual circuit formation followed by cleanup. These are direct
reasons to track mechanisms through training, but their phases are not templates that TEACH-0012
must reproduce.

Recent two-hop work also warns against equating a mature circuit with systematic transfer. [He et
al. (2026)](https://aclanthology.org/2026.findings-acl.1697/) report that generalization and a
reasoning path can dissociate under some data regimes. TEACH-0012 therefore needs behavioral,
geometric and causal trajectories measured separately.

## A major confound already built into TEACH-0012

The training distribution changes at steps 2,000 and 4,000:

- steps 0--1,999: first-hop/direct/copy; no composed task;
- steps 2,000--3,999: 40% composed with moderate render complexity;
- steps 4,000--7,999: 70% composed with harder distractors, marker dropout and gaps.

The stored 2,000 and 4,000 checkpoints are exactly at those boundaries: they are saved after the
last update of the preceding regime. This is scientifically useful because they provide clean
pre-transition states. It also forbids a strong claim that any subsequent circuit change is an
optimizer-intrinsic phase transition. Curriculum exposure is an alternative cause.

A future phase-transition claim would require a fresh constant-distribution replicate or several
curriculum schedules with registered checkpoints bracketing the same number of composed examples.
The current campaign can establish a developmental sequence under this curriculum.

## Available time points and their limits

The raw-deep checkpoints are 250, 500, 1,000, 2,000, 4,000 and 8,000 updates. Six samples cannot
locate a sharp transition or distinguish a sigmoid from a discontinuity. The analysis may label
regimes descriptively but must not estimate a critical update with false precision.

The looped model has only its final checkpoint. It supports final-mechanism comparison but no
developmental trajectory. If looped development becomes important, register a new run that saves
matched update and processed-token milestones.

Because all checkpoints lie on one continuous optimization trajectory, neuron/head coordinates
are directly inherited over time. This makes within-seed weight comparisons more meaningful than
cross-seed neuron matching. Function can still migrate between components, so native-coordinate
stability should never replace a causal assay.

## Freeze the final explanatory object first

Developmental analysis begins only after:

1. the final raw arm passes its registered behavioral gates;
2. a fresh mechanism study discovers a site/path on discovery families;
3. that study freezes semantic positions, layer/head/path masks, subspace rank, alignment method
   and all thresholds;
4. untouched final-checkpoint confirmation establishes necessity, sufficiency, recipient-specific
   transfer and numerical validity.

Then apply exactly that frozen explanatory object backward to every milestone. Searching a new
best head, layer or subspace at each checkpoint measures search flexibility, not circuit
development.

An additional exploratory analysis may ask what earlier checkpoints use instead, but it needs a
separate multiplicity-aware discovery split and cannot be presented as confirmation of the final
circuit's lineage.

## Fixed checkpoint panel

Create one fresh family-grouped panel, separate from TEACH-0012 confirmation and later mechanism
confirmation. It should contain:

- first-hop, direct, copy and composed tasks on identical logical graphs;
- easy marked and fully marker-free renderings;
- row-order rerenders and same-answer/different-layout pairs;
- 0, 2, 4 and 6 distractor chains;
- F-assignment and G-remapping factorial quartets;
- false paths sharing one endpoint with the signal path;
- recipient triples for reusable-key interchange;
- heuristic-conflict items where first/last/nearest/earliest-row rules disagree with the oracle.

All checkpoints see exactly the same panel. Report item and exact-group accuracy, legal-candidate
rate, target probability, entropy and shortcut destinations. Continuous target probability is
important before argmax competence, but it does not substitute for exact grouped behavior.

## Four synchronized trajectories

### 1. Behavioral trajectory

For every checkpoint and seed, record:

- clean task accuracy and target probability;
- every grouped counterfactual score;
- marker/order/distractor/length robustness;
- predictions of explicitly enumerated heuristics;
- error transitions for each item between adjacent checkpoints.

Classify items by transition pattern: always wrong, early correct then lost, late corrected,
unstable, or always correct. Cluster by logical family rather than treating rerenders as
independent samples.

### 2. Representational trajectory

Apply the frozen final key/binding/order/format contrasts and record:

- discovery-frozen probe or centroid readout on the fixed panel;
- confirmation CKA/RSA relative to the final checkpoint;
- Procrustes/principal-angle alignment using discovery examples only;
- subspace rank, participation ratio and explained covariance;
- action-aware Jacobian/Fisher distance and J-lens key-frame occupancy;
- format and physical-order nuisance components.

Decodability before causal effect means a variable is accessible but not yet shown to control the
answer. A late decline in decodability can reflect code rotation rather than information loss;
cross-check with discovery-fitted alignment and finite interventions.

### 3. Causal trajectory

Run the same frozen recipient-specific intervention at each checkpoint:

- clean base and donor competence;
- key sufficiency across three recipient G tables;
- reverse necessity and native-state rescue;
- fixed donor-answer injection;
- same-key rerender/distractor preservation;
- wrong-position, cyclic and matched-random controls;
- selected path/head versus complement;
- direct/copy specificity;
- finite effect versus local Jacobian prediction.

Use both the full panel and a checkpoint-specific both-clean-correct diagnostic. The full panel is
the only comparable primary denominator. A causal intervention cannot be declared absent solely
because a checkpoint cannot yet produce the correct clean answer; inspect target-probability
movement and the final-state answer-injection positive control.

### 4. Weight/circuit trajectory

For final selected components, measure:

- Q/K/V/O and MLP weight changes between adjacent milestones;
- singular spectra and effective rank of selected products;
- head-result norms and recipient-specific causal contributions;
- attention-source mass only as a descriptive companion;
- final-circuit restricted and excluded performance.

`restricted` means reconstructing or adding only the final selected causal writes/path on top of a
registered corrupted baseline. `excluded` means ablating the selected path while leaving the rest
of the checkpoint intact. Both require random path and equal-norm controls. A path that is
sufficient late but whose exclusion has little effect may coexist with redundant computation.

Do not interpret one weight norm as circuit strength. A product such as `W_OV`, its interaction
with source activations, and the finite downstream effect are more relevant than any factor alone.

## Progress measures

The final circuit supplies checkpoint-independent progress measures:

1. **recipient transfer:** exact three-recipient group rate;
2. **path sufficiency:** target-probability gain from selected writes;
3. **path necessity:** target-probability loss after selected-path ablation;
4. **specificity margin:** selected effect minus the largest matched control;
5. **format invariance:** effect retained across rerenderings;
6. **shortcut exclusion:** heuristic correctness after excluding the final circuit;
7. **restricted competence:** performance using only the selected path on the corrupted baseline;
8. **causal alignment:** cross-checkpoint transfer after the frozen discovery map.

Plot every progress measure with family bootstrap intervals and both seeds separately. A smooth
measure beneath an abrupt accuracy curve supports gradual circuit formation under this curriculum.
An abrupt causal measure between sparse checkpoints only bounds the change to an interval.

## Testing inheritance versus replacement

Three competing developmental stories should be distinguished.

### H-D1: amplification

The final circuit is weakly present early and gradually strengthens. Prediction: a final-frozen
path has increasing target-probability effect and stable source/recipient specificity even before
exact accuracy rises. Cross-checkpoint Procrustes alignment remains high.

### H-D2: repurposing

An early heuristic circuit later changes function. Prediction: the same components are causally
active early, but their effect follows physical order or early-row heuristics; later, effect
switches to logical recipient-specific transfer. Native component continuity is present while
effect signatures change.

### H-D3: route replacement or cleanup

An early route remains separately identifiable while a later systematic route forms and then
dominates. Prediction: excluding the final circuit leaves early checkpoint behavior unchanged but
damages the final checkpoint; excluding the early discovery route shows the reverse. For a period,
both routes may be sufficient or compete on heuristic-conflict items.

These stories are not determined by CKA or attention patterns. They require interventions on
matched inputs at each checkpoint.

## Cross-checkpoint causal transport

Use the final checkpoint as one donor only after fitting any coordinate map on discovery. For a
fixed logical episode, transplant an early key state into the final model and a final key state
into the early model at the corresponding semantic site.

Possible outcomes:

- bidirectional transfer: stable functional code across training;
- early-to-final only: early code is readable by the mature downstream circuit, but early readers
  cannot use the mature code or lack later computation;
- final-to-early only: unlikely but compatible with an early broad reader and later compressed
  code;
- neither direction despite similar RSA/CKA: geometric similarity without causal compatibility;
- transfer only after Procrustes: shared variable under a rotating basis;
- transfer after a high-capacity nonlinear map only: weak evidence, vulnerable to alignment
  overfit.

Controls include shuffled checkpoints, shuffled examples, random orthogonal maps, equal-rank CCA
maps and full late answer-state injection. Fit maps on discovery and evaluate only once on grouped
confirmation.

## Parameter interpolation is secondary

Adjacent checkpoints belong to one trajectory, so linear interpolation of weights can test for a
loss/behavior barrier. Evaluate several fixed interpolation coefficients on the frozen panel and
record behavior plus final-circuit effects. A smooth path suggests that sparse checkpoint sampling
hid gradual change; a barrier suggests functional reorganization.

Interpolation creates models the optimizer never visited and mixes all parameters at once. It
does not identify which update caused a circuit and cannot support a developmental claim by
itself. Componentwise parameter grafts are even more distribution-shifting and require identity,
random-component and normalization controls.

## Training-data attribution is a future experiment

TEACH-0012 can correlate circuit changes with curriculum exposure, but it cannot causally identify
which examples caused them. A future source-matched replicate could intervene on data:

- constant task mix versus staged curriculum;
- counterfactual factorial groups added or removed;
- marker-rich versus marker-free exposure;
- valid versus endpoint-sharing distractors;
- matched random examples with the same token frequencies and lengths.

Freeze total updates and examples. Ask whether removing one data family changes the final-frozen
causal progress measures, not merely validation loss. This would distinguish a mechanistic data
cause from temporal coincidence.

## Decision language

| Observation | Permitted conclusion |
|---|---|
| Final path effect rises smoothly before accuracy | Hidden gradual formation under this curriculum |
| Effect appears after step 2,000 or 4,000 | Change occurred after new curriculum exposure; cause unresolved |
| Early effect follows order, late effect follows logic | Repurposing or route replacement supported by interventions |
| Decoding precedes intervention effect | Variable accessible before demonstrated causal use |
| CKA rises but cross-checkpoint patches fail | Representational organization aligns; functional code does not |
| Restricted path improves while excluded path worsens | Selected circuit increasingly explains behavior |
| One seed changes earlier than another | Timing is seed-dependent; no universal update threshold |
| Sparse checkpoints bracket a jump | Transition interval located; discontinuity not established |

The strongest result would show, in both seeds, a final-confirmed recipient-specific circuit whose
effect can be traced backward from absent or heuristic use through formation to necessity, while
matched controls remain low. That would explain how one synthetic algorithm develops. It would
not imply that the same stages occur in natural language, other architectures or the Voynich
manuscript.
