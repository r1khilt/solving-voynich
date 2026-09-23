# Raw mechanism implementation-readiness audit

Status: blind source audit performed while TEACH-0012 is running. No confirmation prediction or
score was read. This document fixes engineering facts and foreseeable numerical/causal hazards;
it does not select a site, arm or result.

## What the current model already exposes

Both raw architectures use the repository's `ActivationContext`, which supports named activation
caches and shape-preserving interventions without persistent hooks.

The untied raw model exposes, for each zero-based block `l`:

- `blocks.l.resid_pre`, `attn_input`, `resid_mid`, `mlp_input`, `resid_post`;
- `blocks.l.attn.q_pre_norm`, `k_pre_norm`, `v`;
- `blocks.l.attn.q_normalized`, `k_normalized`, and post-RoPE `q`, `k`;
- `blocks.l.attn.scores`, `pattern`, `z`, per-head projected `result`, and summed `out`;
- `blocks.l.mlp.up`, `gate`, `act`, and `out`;
- global `embed`, `final_norm`, and `logits`.

The looped model exposes the same internal sites under
`passes.p.blocks.l.*`, plus `passes.p.input` for passes `p=0,1,2`. The four block modules are
shared across passes, but their activations are distinct. This is exactly what a pass-by-pass
iterative algorithm test needs.

`RawClassifier` returns the logits at the final `ANSWER` position while retaining full cached
tensors. The output classifier is tied to the input embedding table. A future J-lens can therefore
use the actual unembedding rows, but the tied row alone is not a J-lens: it must be composed with
the averaged downstream Jacobian from the chosen layer.

## Exact semantic positions can be reconstructed

Every serialized relation contributes exactly two ordinary-symbol tokens. `EDGE` and `GAP` are
below `SYMBOL_START`; the final task marker, query and `ANSWER` suffix are known. Within the body:

1. collect all token indices with ID at least `SYMBOL_START` before the three-token suffix;
2. group them in consecutive pairs;
3. verify each token pair equals the corresponding tuple in `episode.serialized_rows`;
4. map that row back to F, G or distractor membership and left/right role;
5. label the two occurrences of a signal key separately: F-right and G-left.

This avoids an ambiguous search by token identity because a key legitimately occurs twice. The
helper must assert the entire reconstruction and reject a malformed episode rather than guess.
The final query and answer positions are `length-2` and `length-1`.

Donor/base activation patches need matched batch width and serialization skeleton, or a position
index operation that modifies only validated semantic sites. Full-tensor patches between
differently padded batches are invalid even if the logical programs match.

## Causal ordering matters

The raw Transformer is causal. A relation token can affect later relation, query and answer
positions, but never an earlier position. Consequently:

- patching a late answer state can trivially inject an answer and is only a positive control;
- a query-position key state can influence the answer position in later blocks;
- a relation endpoint can be an earlier writer, but the relevant physical index changes with row
  order;
- dynamic routing may require an ordered path of writes rather than one permanent bottleneck;
- in the looped model, pass number functions like recurrent time even though sequence position is
  unchanged.

Site discovery should report an effect tensor indexed by architecture depth/pass, semantic
position and intervention family. It should not flatten physical indices across examples.

## Numerics must be qualified before science

When no cache is requested, attention uses fused scaled-dot-product attention. When any cache or
intervention is active, it explicitly computes scores, softmax, value aggregation and per-head
output projection. These paths are algebraically equivalent but can differ in floating-point
operation order. Earlier TEACH-0008--0011 studies showed that errors around `1e-6` can invalidate
an otherwise sensible assay.

Before site discovery on each trained checkpoint:

1. compare ordinary and `cache_names=['*']` logits on a fixed discovery-only numerical panel;
2. compare identity interventions at every site family with the cached path;
3. verify that per-head `result.sum(head)` equals attention `out`;
4. verify `resid_pre + attn.out = resid_mid` and the explicit MLP residual equals `resid_post`;
5. verify padding invariance and batch-versus-single-example logits;
6. check both float32-native and float64 analysis projections where source summation amplifies
   roundoff;
7. freeze a tolerance from the operation being checked, not from the desired behavioral result.

If the explicit path changes any argmax prediction, discovery is blocked until the numerical path
is repaired and independently tested. Preserve rejected outputs and label any rerun non-blind if
behavioral aggregates were exposed.

## Checkpoint facts

The campaign saves final checkpoints for every arm and deep-model milestones at 250, 500, 1,000,
2,000 and 4,000 updates. The final deep checkpoint is the ordinary 8,000-step checkpoint. Each
archive includes model state, frozen config, replicate, arm, update and parameter count.

The prospective developmental analysis should load these weights into fresh CPU or MPS model
instances and verify state-dict coverage plus archive SHA-256. It must apply one discovery-frozen
assay across milestones. Selecting a different best layer or subspace at each milestone would
confound representational development with repeated search.

Only deep milestones exist. A claim about looped algorithm development would require a new
prospectively registered training run with looped milestones; the current final looped checkpoints
support final-pass circuit analysis but not a developmental curve.

## Bounded discovery design

For each eligible arm and seed, use a fresh discovery suite with paired counterfactual skeletons.
Cache only the following first-pass screen to bound storage:

- every `resid_post` layer/pass at query and answer positions;
- every `resid_post` layer/pass at queried-F right and matched-G left/right positions;
- the corresponding `resid_mid` sites for effects that qualify at `resid_post`.

At width512, caching six semantic positions over12 depth steps for512 examples is about72 MiB in
float32 per model before metadata. Full all-position/all-component caching is unnecessary during
the first screen. Stream batches and retain compact effect rows rather than dense tensors unless
the later subspace stage needs them.

Discovery effects should be computed for F-value, G-value, pairing, format and distractor donors.
Select a site by the minimum recipient-specific effect across both seeds and hard strata. If no
single site qualifies, advance to a registered path screen instead of relaxing the single-site
threshold.

## Recipient-specific interchange implementation

For each group, construct a donor with `F1(n)=k1` and base programs with `F0(n)=k0` under three
recipient G tables. Use the same serialized row positions, marker pattern, gaps, distractor count
and total length. The donor G0 activation is reused across recipient bases.

At a full residual site, a position-selective intervention should clone the base tensor and replace
only `value[batch, semantic_position, :]`. At a head-result site, replace or add only the frozen
head subset in `value[batch, position, head, :]` or
`value[batch, position, head, d_model]`, according to the site's actual tensor contract. Every
intervention asserts shape, dtype, device, batch IDs and semantic labels.

Archive for every row:

- logical group, recipient and render identifiers;
- base/donor tokens and semantic-position maps;
- expected base, recipient-key and fixed-donor answers;
- clean predictions and probabilities;
- condition, site, layer/pass, position role and component mask;
- finite-patch prediction/probability and linearized prediction if applicable;
- patch norm, base/donor norm and numerical reconstruction errors.

The primary score uses all frozen items. Both-clean-correct is a separately reported diagnostic,
never an adaptive filter.

## Q/K/V and source decomposition details

The cached post-RoPE `q` and `k` have shape `[batch, head, time, head_dim]`; `v` has
`[batch,time,head,head_dim]`. Position code is already mixed into Q/K, so transplanting post-RoPE
vectors across physical positions also transplants position phase. To distinguish content from
position:

- edit `q_pre_norm`/`k_pre_norm` and let native normalization/RoPE recompute for the recipient
  position;
- compare with edits to post-RoPE Q/K as an explicitly position-entangled condition;
- use same-position skeleton pairs for the primary factorial;
- include moved-row counterfactuals as the position test.

The per-head `result` tensor is already split through the output projection and additive across
heads. Source contribution analysis can compute `pattern[..., source] * v[source]`, apply the
corresponding output-projection head slice, and verify its sum against `result`. Both removal of
the old source and addition of the new source may be required, as TEACH-0011 demonstrated in the
parsed circuit.

Q/K/V hybrids need the complete eight-cell factorial. “Q-only” or “V-only” must refer to which
donor components are substituted while all other components remain base, with endpoints verified
against all-base and all-donor reconstruction.

## Jacobian and J-lens implementation constraints

A literal dense Jacobian at width512 for all source/target positions, layers and2,064 logits is
unnecessary. Use vector-Jacobian products for selected legal key/object logit contrasts and
Jacobian-vector products for discovery-selected activation directions. Batch over examples and
average only after preserving task and recipient strata.

Estimate separate lenses for first-hop, direct and composed contexts. Fit them on discovery
prompts, and freeze layer, sparsity, regularization and token frame before confirmation. Because
the output embedding is tied, include controls using:

- the raw unembedding/logit lens;
- averaged-Jacobian J-lens;
- label-shuffled token rows;
- norm/eigenspectrum-matched Gaussian frames;
- PCA and random subspaces of the same rank.

Readout alone is not support. The J-lens coordinate swap must be executed as a finite activation
edit and must produce each recipient's `G_j(k1)` rather than a fixed donor token.

## Independent-audit boundary

The mechanism runner may use model code, but the auditor should not import its suite construction,
semantic mapping, scoring or decision functions. The auditor independently reconstructs logical
oracles and physical positions from archived tokens/rows; checks suite, source, checkpoint and row
hashes; recomputes every condition score and selection rule; and verifies that confirmation did not
influence site/subspace/head choice.

The auditor cannot prove that a checkpoint produced an archived neural value without rerunning
inference. Either rerun a deterministic stratified subset or state this limitation explicitly.

## Ready versus blocked

The current architecture is already instrumentable down to residual, Q/K/V, attention pattern,
head result and MLP sites. No model rewrite is needed for a raw causal study. Work remains blocked
only on scientific eligibility and registration:

1. complete and independently audit TEACH-0012 behavior;
2. choose eligible arm(s) by the frozen behavioral rule;
3. freeze fresh discovery/confirmation groups and causal thresholds;
4. implement semantic mapping and numerical qualification tests;
5. commit source before any trained-checkpoint mechanism screen.

This ordering preserves the ability to say that a later circuit was predicted and confirmed,
rather than reverse-engineered from the confirmation set.
