# TEACH-0012: raw serialized variable binding at serious scale

**Preregistration before any TEACH-0012 benchmark, training, checkpoint access or model
evaluation.** Generator structure and randomly initialized CPU forwards may be tested before
the source freeze. No trained TEACH-0012 outcome exists at registration.

## Question and restricted claim

Can a standard causal Transformer learn exact two-hop binding from a raw serialized token
stream when relation order changes, valid distractor chains are present, symbol roles are
episode-local, and row-boundary cues can move or disappear? If so, does depth or shared-weight
iteration matter?

This is the first bridge after TEACH-0004--0011 causally mapped a two-hop circuit in an
explicitly parsed, typed, four-row instrument. A pass establishes synthetic raw-sequence
competence only. It does not identify a Voynich token, language, cipher, word boundary or
historical mechanism.

## Prior work and design rationale

[Wu, Geiger and Milliere (ICML 2025)](https://proceedings.mlr.press/v267/wu25j.html)
trained a roughly 38M-parameter Transformer on serialized symbolic programs with variable
assignment chains and distractors. They report a progression from random behavior, through an
early-line shortcut, to systematic binding, and use causal interventions to identify dynamic
routing through the residual stream. Their programs still provide strong punctuation and
fixed training structure, and their finding does not predict success here. It motivates our
model scale, stored training milestones, raw-token input and explicit shortcut panels.

[Arora et al.'s MQAR work](https://arxiv.org/abs/2312.04927) shows that simple associative
recall tests can become misleadingly easy when query count, locations, vocabulary or sequence
length are small. It motivates a 2,048-symbol pool, multiple query counterfactuals, shuffled
locations and longer/distractor panels. MQAR is one-hop recall and does not demonstrate parsing
or composition. [Yao et al. (EMNLP 2025)](https://aclanthology.org/2025.emnlp-main.490/)
report that implicit multi-hop learning grows sharply in data and depth and that curriculum
helps but does not eliminate that burden. This motivates 12 layers, about 512,000 examples per
arm, and a frozen one-hop-to-composed curriculum; it does not establish our thresholds.

[CRiT-QA](https://aclanthology.org/2026.lrec-1.410/) and
[Bhuiya et al.](https://aclanthology.org/2024.emnlp-main.147/) show in natural-language
benchmarks that plausible distractor chains expose multi-hop shortcuts. Their tasks and models
differ; our valid independent chains and counterfactual groups are a synthetic adaptation.
The later mechanistic study will compare TEACH-0005--0011's reusable key-as-query mechanism
against binding-ID and dynamic-routing hypotheses motivated by
[Feng and Steinhardt](https://arxiv.org/abs/2310.17191) and
[Davies et al.](https://arxiv.org/abs/2307.03637). No such mechanism is assumed here.

Voynich-specific neural generators, predictors, embeddings and saliency analyses are already
catalogued in `docs/research/PRIOR_WORK.md`. The novel target is therefore not “train a
Transformer on Voynich.” It is to qualify a raw synthetic instrument under causal-quality
counterfactuals before any manuscript transfer.

## Frozen raw program

There is one shared pool of 2,048 opaque symbols. A symbol can be a name, key or object in
different episodes; no permanent name/key/object token classes exist. Each logical episode
contains four signal chains

`name -> key -> object`

represented as four F edges and four G edges, plus zero to four independent two-edge distractor
chains during training. Every left side is unique, so the rows define a function and the oracle
answer is identifiable. Tasks are explicit: first-hop, direct, composed and copy. The composed
answer is `G(F(query))`.

All 8--16 rows are globally shuffled. Each row is rendered as `EDGE left right`,
`left EDGE right`, `left right EDGE`, or--according to the frozen curriculum--bare `left right`.
Zero to two `GAP` tokens occur between rows. There are no F/G section markers, role embeddings,
row positions or type-specific symbol inventories. A task marker, query and final `ANSWER`
token remain supplied. Maximum context is 128 tokens; generation rejects overflow rather than
truncating.

F and G family partitions are SHA-256 hashes of their logical endpoint sets before distractors
or serialization. Buckets 0--7 train, 8 development and 9 confirmation. Training uses only
train/train families. Confirmation includes every crossed train/confirm cell. Surface variants
of one logical program cannot change its family split.

## Arms and scale

All learned widths are 512 and all symbol outputs use the same 2,064-token vocabulary, with
input/output embedding tying in raw arms.

| Arm | Frozen architecture | Parameters | Role |
| --- | --- | ---: | --- |
| `parsed_memory` | row-boundary oracle; two learned attention reads over all signal+distractor rows | 3,160,576 | task learnability and optimization upper bound |
| `raw_shallow` | 4-layer causal Transformer, 8 heads, FF 1,536, QK norm | 14,693,376 | depth control |
| `raw_looped` | same four learned blocks applied three times with pass embeddings | 14,694,912 | shared-weight 12-block-compute control |
| `raw_deep` | 12-layer causal Transformer, 8 heads, FF 1,536, QK norm | 41,965,568 | **primary raw arm** |
| `raw_null` | shallow architecture; composed labels sampled from visible signal objects | 14,693,376 | leakage/null control |

The primary is about 189 times the 222,215-parameter TEACH-0004 two-read model and is close to
the scale of the direct ICML variable-binding precedent. `parsed_memory` is intentionally not
parameter matched: it is an upper bound that receives row boundaries but no correct-row,
equality, intermediate-key or pointer labels. `raw_looped` approximately matches the primary's
block applications, not its parameters. `raw_shallow` matches its input and objective, not its
compute. Measured step time and total wall time will be reported; none is an equal-FLOP causal
comparison.

## Frozen training

Two fixed initialization seeds `(72121, 72131)`, stream seeds `(72221, 72231)` and evaluation
seed `72311`. Each arm uses exactly 8,000 AdamW updates, batch 64, learning rate `3e-4`, weight
decay `.01`, gradient clip `1.0`, dropout zero and the final checkpoint. No validation selection,
early stopping, extra seed or adaptive capacity increase.

- Steps 0--1,999: first-hop/direct/copy `45/45/10%`; 0--1 distractor chains; prefix markers;
  marker dropout zero; at most one gap.
- Steps 2,000--3,999: first-hop/direct/composed/copy `25/25/40/10%`; 0--2 distractor chains;
  three row styles; marker dropout `.10`; at most one gap.
- Steps 4,000--7,999: `10/15/70/5%`; 1--4 distractor chains; three styles; marker dropout `.25`;
  at most two gaps.

This presents 512,000 generated episodes per seed/arm. Repeated logical families and atomic
symbols are possible. Save primary-arm milestones at updates 250, 500, 1,000, 2,000 and 4,000,
plus every final checkpoint; milestones diagnose learning phases and cannot select the final
model.

## Frozen confirmation suite

Default group count is 128. The suite contains 4,096 items:

- 128 composed items in each train/train, confirm/train, train/confirm and confirm/confirm cell;
- 128 confirmation first-hop, direct and copy items;
- 128 four-query F groups and 128 four-query G groups, querying every relevant left side;
- 128 factorial quartets crossing an F assignment swap with an independently remapped G table,
  with four distinct answers;
- 128 four-render order groups with the same logical graph;
- 128 four-member distractor groups containing 0, 1, 2 and 4 irrelevant chains;
- 128 four-render boundary groups with marker dropout 0, .25, .50 and 1.0;
- 128 composed items with six distractor chains, beyond the training maximum.

Report per seed and arm: exact item accuracy, Wilson intervals, exact grouped accuracy, visible
answer-candidate membership, prediction changes, wrong-answer destinations, accuracy by length,
distractor count, marker dropout, answer-row location and serialized query distance. Grouped
variants share a logical core and are not independent items.

## Frozen decisions

`RAW-SEQUENCE-BINDING-QUALIFIED` requires both `raw_deep` seeds to satisfy all clauses:

- first-hop and direct confirmation accuracy at least 95%, and copy at least 98%;
- every composed crossed cell at least 90%;
- F-query and G-query four-item exact groups at least 85%;
- factorial quartet exactness at least 75%;
- order and distractor four-item exactness at least 80%;
- boundary four-item exactness at least 70%, with the fully marker-free slice at least 80%;
- six-distractor length-extrapolation accuracy at least 80%.

The positive control is healthy only if both `parsed_memory` seeds achieve at least 95% on every
ungrouped competence cell, at least 90% on both query-group panels and at least 85% factorial
exactness. If it fails, the raw campaign is `INCOMPLETE` as an optimization test. The null is
valid only if both `raw_null` seeds remain at or below 35% confirm/confirm composition and 10%
factorial exactness. A higher result triggers a leakage/audit investigation and invalidates the
campaign.

Separate labels, fixed before training:

- `DEPTH-HELPFUL` if `raw_deep` exceeds `raw_shallow` by at least 10 points on both
  confirm/confirm composition and factorial exactness in both seeds.
- `LOOPING-COMPETITIVE` if `raw_looped` itself passes every primary raw gate and stays within
  five points of `raw_deep` on confirm/confirm and factorial exactness in both seeds.

Failure of either comparative label does not alter the primary absolute verdict. High item
accuracy with failed query or factorial groups is a shortcut result, not binding. Boundary,
order, distractor and length failures are reported separately rather than averaged away.

## Mechanistic boundary

No site, head or subspace is selected in this behavioral campaign. If and only if raw competence
qualifies, TEACH-0013 will preregister a new discovery/confirmation suite and screen raw
layer-position sites without assuming that TEACH-0007's parsed cut-2 query slot survives. It
will test at least four hypotheses: reusable key-as-query, binding-ID similarity, ordering IDs,
and dynamic routing without one static state. Cross-G donor patches must follow each recipient's
`Gj(donor_key)` rather than a fixed donor answer, with full-state, complement, random, cyclic,
wrong-position, necessity and rescue controls. A probe, cosine cluster or attention map alone
will not count.

## Resource, provenance and stop rules

Commit registration, generator, models, trainer, auditor and tests before accelerator use. A
source guard rejects dirty registered paths. Benchmark 24 steps per arm (4 warmup, 20 timed),
record exact parameters and repeated-forward/finite-gradient checks, and require a conservative
1.5x projected campaign at or below six hours. Hard campaign ceilings: six hours wall time,
24 GiB sampled MPS allocation and 4 GiB ignored checkpoints/artifacts. A resource or numerical
stop is `INCOMPLETE`, with no automatic restart or partial-arm scientific verdict. CPU smoke
tests do not expose suite outcomes. No download, network service, paid API, manuscript training
or manuscript final-test score.

Retain source/config/suite/checkpoint/prediction SHA-256 hashes, all 8,000 losses, milestone
metadata, per-item rows, exact logical and render IDs, partitions, row order, distractor count,
marker dropout, lengths, answers and predictions. An independent CPU auditor must regenerate
the suite/oracles and decisions without importing the generator, trainer or model and must pass
before interpretation.
