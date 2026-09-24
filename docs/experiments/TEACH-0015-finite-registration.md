# TEACH-0015 full-state key-transfer assay: execution registration

**Status:** prospective, before TEACH-0014 final neural results or any
TEACH-0015 trained-model inference. This instantiates only the fixed full
`query.1` state-transfer stage of
`TEACH-0015-mechanism-preregistration.md`. The J-inspired geometry, sparse
token frame and neuron analyses remain separately gated on a passing finite
state assay and require their own source/selection freeze before confirmation.

The primary question is whether one unchanged state from donor `(F1,G0)`
causes the same frozen model to produce three different recipient-specific
outputs `G0(k1),G1(k1),G2(k1)` when inserted after the first read into
`(F0,G0),(F0,G1),(F0,G2)` respectively. The donor's fixed answer is a strong
wrong explanation for changed-G recipients. This is a synthetic relational
program with public labels; passing it does not decipher a manuscript.

**Prospective interpretation correction, before trained outcomes:** the G0/G1/G2
tables keep the same key-to-row-slot order under every common surface seed.
All1,024 surfaces in each frozen split place the `G_j(k1)` row in the same
physical slot across all three G tables. Therefore even a three-recipient
transfer could pass by carrying a G-row pointer rather than the symbol `k1`.
The registered full-state thresholds remain unchanged, but their strongest
permitted label is `PORTABLE-INTERMEDIATE-STATE-SUPPORTED`, meaning a state
that controls the tested recipients. They do **not** identify a symbol-key
code or Anthropic-style J-space. A separately preregistered cross-order assay
must break row-slot alignment before a key-specific interpretation is tested.

## Entry, population and fixed pairing

Require complete TEACH-0014 v3 no-model artifact and sampled checkpoint/gate
replay audits. Require the TEACH-0015 clean screen's no-model audit and sampled
CPU checkpoint replay. Screen all eligible arm/seed pairs, but admit a named
arm to finite confirmation only if **both its seeds** pass the screen's fresh
confirmation gate. Name answer-only, causal-interchange-supervised and
public-edge-supervised arms separately. If none enter, record
`NOT ENTERED: FRESH COMPETENCE` and do no finite checkpoint inference.

Use the frozen TEACH-0015 discovery and confirmation manifests, each128
groups×eight surfaces×three recipients=3,072 recipient attempts per arm/seed.
Surface order is distractor topology0/1, marker-rich/free, then physical row
order0/1. Every `(group,surface)` has the same donor F1/G0 reused across its
three recipients, with the native recipient G table otherwise unchanged.
There is no site, layer, item or surface search for this full-state assay.

Both wrong-key controls source the F1/G0 donor of another group at the **same
surface**. On the frozen ordered128-group manifest, a cyclic offset is valid
only when *every* target group has a different `key1` from the group at that
offset. The derangement offset is the smallest valid offset. For the second
wrong-key control, start at offset
`1 + SHA256(["TEACH-0015-wrong", split, ordered_group_ids]) mod 127`,
cycle through offsets1–127, and take the first valid offset distinct from the
derangement. Abort if fewer than two valid offsets exist. At each surface,
each offset maps every target group to exactly one donor and uses every donor
exactly once. Thus the controls are distinct, key-mismatched bijections with
matched surface and donor distributions. The independent auditor recomputes
both permutations from the manifest. No donor choice can use neural outputs.

## Interventions, denominators and decision

For every attempt archive native base/target/donor answers; identity patch;
unchanged donor `query.1` transfer; same-key opposite-marker donor; reverse
base-state patch into clean F1 target; donor `query.2` answer injection;
deterministic norm-matched random delta; norm-matched wrong-key donor; and
norm-matched cyclic derangement. Archive the actual 512D native and
replacement vectors and pairing IDs for every attempt. To enforce the2GiB
cap, each split/model/seed stores the eleven named vectors for all attempts in
one float32 `.npy` array of shape `(128,8,3,11,512)`, with ordered compressed
JSON metadata and SHA-256 checksums. This array preserves the float32 values
captured from the intervention calls; the independent auditor reconstructs
every row from the array. The full-logit sample
is the first, middle and final **surface** of each split, including all three
recipients and every condition. All other attempts retain predictions, vector
records and numerical identity/final-donor errors. Independent no-model
auditing checks all vectors, visible answers, pairings, conditions and grouped
denominators, with per-surface and per-group counts; separate CPU checkpoint
replay verifies every sampled full
intervention vector and logit. The clean native predictions must agree with
the already audited TEACH-0015 screen on the same render IDs.

**Prospective archive refinement:** the broader mechanism preregistration
requested full logits, probabilities and address masses for every item. This
finite implementation stores full logits only at the fixed sample surfaces to
remain within its2GiB cap. It does not archive per-item probabilities or
address masses, and therefore cannot support a claim about those quantities;
they need a separately registered diagnostic if desired. All primary finite
behavior criteria use exact predictions and unchanged state vectors and remain
the earlier frozen thresholds. This refinement was made before any trained
TEACH-0015 result or primary TEACH-0014 decision was inspected.

The fixed confirmation criteria retain the original preregistration:

- Transfer target accuracy≥75% over all3,072 recipients, **both seeds**.
- Exact three-recipient groups≥60% over1,024 `(group,surface)` triples.
- Changed-G non-injection≥90% over2,048 G1/G2 attempts: the transfer output
  must not simply equal the G0 donor's fixed answer.
- Reverse necessity≥70% among attempts whose clean base and clean target
  predictions are both correct. Report that eligibility count and reverse
  accuracy on the full denominator too; do not discard other items from the
  primary transfer denominator.
- Marked and fully marker-free transfer accuracy each≥70% over1,536 items.
- Correct donor transfer must beat **each** wrong-key, cyclic derangement and
  norm-matched random target-answer rate by≥35 percentage points on the full
  denominator, in both seeds.

The independent auditor applies these in conjunction after finite numerical
replay; a preliminary no-model result is only a candidate portable-state
label. Report all rates,
per-surface/group counts and4,000-resample logical-group bootstrap intervals
without adapting thresholds to the observations. Same-key, final-donor and
clean competence are controls; a positive final-donor answer injection is not
evidence of a portable intermediate. If clean competence fails on recheck,
the finite result is inconclusive. If necessary but not portable, use the
registered `NECESSARY, NOT PORTABLE` description only when its conditions are
actually shown; a failed transfer by itself is not proof that no key exists.

## Source, compute and artifact gate

The full runner, no-model audit, sampled CPU replay and all-surface random-
weight CPU smoke are implemented prospectively. The smoke verifies all eight
surface types and actual float32 archived vector reconstruction; a separate
three-surface random-weight CPU/MPS parity check found maximum absolute logit
error `5.41e-5`, maximum state-vector error `1.94e-4`, and no transfer-answer
differences in nine recipients. These are numerical checks, not trained-model
results. The full assay still requires a source-matched random-weight MPS
benchmark after a clean commit. Time representative complete surfaces
across marker/distractor/order variants with all controls and vector archiving.
The conservative projection is `1.75 × slowest surface-type median × 12,288
worst-case surfaces + 300s`, and must be under the2-hour wall cap; sampled MPS
allocation is capped at12GiB and finite artifacts at2GiB. An exploratory
eight-surface, uncommitted-code timing check during TEACH-0014 training showed
0.114–0.145s per batched surface, versus about0.252s for the equivalent
single-example path. This is only a feasibility observation, **not** the
source-matched admission benchmark. The full worst-case vector payload is
830,472,192B across twelve split/model/seed arrays before compressed metadata,
within the2GiB cap. All inference uses local MPS with no paid
API, and no bulk checkpoints/logits are committed to Git; compact audited
results, hashes, source versions and failures are. No trained intervention can
run before the primary and fresh-screen audits/replays pass, a named arm passes
both fresh-screen seeds, and the finite source-matched benchmark passes.
