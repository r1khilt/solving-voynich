# After raw Transformers: a bounded architecture ladder for serialized binding

**Prospective research memo, 2026-09-23.** This memo was written while TEACH-0012 was
running. Its registration and source were inspected, but no TEACH-0012 prediction, confirmation
score, trained checkpoint, report, or live output informed the proposals below. The program is
conditional on TEACH-0012 failing or passing only some of its registered raw-sequence gates. It
is not a preregistration, experimental result, model-selection license, or Voynich reading.

## 1. What the next experiment should diagnose

TEACH-0012 already makes a serious standard-architecture test. It supplies a raw causal stream
with a shared 2,048-symbol pool, shuffled relations, variable edge-marker placement and dropout,
valid distractor chains, and length extrapolation. Its primary model is a 41.97M-parameter,
12-layer Transformer; its shared four-layer-by-three-pass control separates some effects of
iteration from untied depth. Repeating this campaign with a somewhat wider or deeper Transformer
would add cost without identifying the missing operation.

A failure can arise at at least four different interfaces:

1. **Segmentation:** the model does not reliably infer which serialized occurrences form an
   ordered edge when markers move or disappear.
2. **Storage and addressing:** relations are parsed locally but cannot be stored and retrieved by
   episode-specific content.
3. **Execution:** first-hop information is available but is not iteratively transformed into the
   second-hop answer.
4. **Objective:** final-answer supervision does not reward a reusable intermediate or a stable
   latent graph strongly enough.

Those hypotheses predict different failure slices. Boundary failure with good marked-row
performance prioritizes latent parsing. Strong first-hop/direct accuracy with poor factorial
composition prioritizes iterative execution. Deterioration with distractor count or distance,
despite clean two-hop accuracy, prioritizes memory/addressing. Near-chance behavior everywhere,
including the shallow and looped arms, justifies testing both parsing and memory but does not
justify interpreting activations from an incompetent model.

The program below changes one major inductive bias at a time. Every positive arm has an ablation
that preserves most parameters, updates, data and supervision while removing the proposed useful
operation. The existing 4,096-item TEACH-0012 suite remains exposed after that campaign and must
not be reused as fresh confirmation. Each successor needs a new logical-family namespace and
separate development and confirmation families.

## 2. Source basis and strict transfer limits

### Writable and content-addressed memory

[Neural Turing Machines](https://arxiv.org/abs/1410.5401) couple a controller to differentiable
external memory and demonstrate copying, sorting and associative recall. The
[Differentiable Neural Computer](https://doi.org/10.1038/nature20101) adds allocation, temporal
links and multiple read/write heads for structured-data tasks. [End-to-End Memory
Networks](https://arxiv.org/abs/1503.08895) perform recurrent attention hops over memory, while
[Key-Value Memory Networks](https://aclanthology.org/D16-1147/) separate addressing content from
returned content. These results motivate explicit read/write and key/value factorization. They do
not show that a model can infer uncertain token boundaries, and a memory cell populated with
oracle rows would simply reintroduce TEACH-0004's parser.

### Recurrent and adaptive computation

[Universal Transformers](https://arxiv.org/abs/1807.03819) apply a shared self-attentive
transition recurrently and optionally halt by position. [PonderNet](https://arxiv.org/abs/2107.05407)
learns a distribution over computation steps with a cost regularizer. The
[Recurrent Memory Transformer](https://arxiv.org/abs/2207.06881) uses explicit recurrent memory
tokens across segments. These papers motivate shared iterative state and adaptive compute, but
TEACH-0012 already contains a fixed three-pass loop. A successor must differ through persistent
state, halting, or read/write operations rather than merely repeating blocks again.

### Sets, slots and inferred graphs

[Set Transformer](https://proceedings.mlr.press/v97/lee19d.html) builds permutation-aware set
processing with attention. [Slot Attention](https://arxiv.org/abs/2006.15055) competitively maps
perceptual features into exchangeable slots. [Neural Relational
Inference](https://proceedings.mlr.press/v80/kipf18a.html) learns a latent interaction graph and a
graph-network decoder, and the graph-network framework of [Battaglia et
al.](https://arxiv.org/abs/1806.01261) makes relational structure an explicit inductive bias.
These models were developed for sets, objects, or dynamical systems, not ambiguous symbolic
serializations. Slot identity is permutation-indeterminate, and a model given gold edge groups is
a parsed positive control rather than raw relation discovery.

### Neural algorithm execution

[Pointer Networks](https://proceedings.neurips.cc/paper/2015/hash/29921001f2f04bd3baee84a12e98098f-Abstract.html)
select positions from variable-size inputs. [Neural Algorithmic
Reasoning](https://arxiv.org/abs/2105.02761) argues for learned processors aligned with classical
algorithm structure, and the [CLRS benchmark](https://proceedings.mlr.press/v162/velickovic22a.html)
tests encode-process-decode models and out-of-distribution execution. This motivates a shared
message-passing processor and explicit position pointers. The target here is only a two-hop lookup;
passing it would not imply general algorithm induction.

### Latent dynamics and action-conditioned objectives

[Recurrent Entity Networks](https://openreview.net/forum?id=rJTKKKqeg) maintain gated entity
memories as a text-defined world changes. [MuZero](https://www.nature.com/articles/s41586-020-03051-4)
learns recurrent latent dynamics sufficient for predicting decision-relevant quantities, and
[Dreamer](https://dreamrl.github.io/) learns and rolls forward compact latent states. These are
agent/control settings. TEACH-style binding has no environment, reward uncertainty or need for
tree search. Their relevant idea is narrower: train an action-conditioned latent transition to
preserve what future counterfactual queries require. Reinforcement learning, value search and
imagined-policy optimization would be unjustified additions.

### Decipherment and linguistic limits

Classical computational decipherment assumes a specified plaintext language family and cipher
channel, as in [Knight et al.](https://aclanthology.org/P06-2065/). The Voynich evidence does not
supply TEACH-0012's graph, task marker, query, actions, answer labels or independently remappable
recipient tables. [Reddy and Knight](https://aclanthology.org/W11-1511/) document how little is
securely known about the manuscript's text, while [Lindemann and
Bowern](https://arxiv.org/abs/2010.14697) show why transcription system, glyph composition,
scribal hand and Currier partition matter to even descriptive entropy comparisons. A synthetic
architecture pass can validate a method for discovering and testing causal variables; it cannot
identify a manuscript language or make a translation more probable without an explicit historical
generator and held-out manuscript predictions.

## 3. Shared successor protocol

Every campaign below should regenerate the logical program before surface rendering and use a
fresh namespace. Keep TEACH-0012's shared symbol pool, tasks, family hashing and grouped
counterfactual philosophy, but add two prospective axes:

- **Hop length:** train on one and two hops; confirm separately on two, three and four hops.
- **Surface ambiguity:** cross edge-marker dropout with 0--8 distractor chains, row styles,
  repeated harmless symbol occurrences and 128/192-token lengths.

Confirmation should include at least 128 independent logical groups per panel. Continue to require
exact query groups, F/G factorials, render-order groups, distractor groups and marker-free groups.
Add an **alias control** in which the same logical symbol occurs in two irrelevant rows, so nearest
surface occurrence is not a valid strategy, and a **false-path control** containing a distractor
chain that shares exactly one endpoint with the queried chain.

All arms use the same logical episodes at every update, final checkpoints, AdamW settings and
legal output classes. Match example exposure exactly. Report parameters, timed updates, total
block/message-passing applications and peak memory separately; parameter matching and compute
matching answer different questions. Require two independent seeds for any support label. A
24-step source-matched local MPS benchmark must project the complete campaign within its wall-time
cap before scientific training begins.

The reusable behavioral gate is intentionally at least as strict as TEACH-0012:

- first-hop/direct confirmation at least 95% and copy at least 98%;
- every two-hop crossed-family cell at least 90%;
- query-group exactness at least 85% and factorial exactness at least 75%;
- order and distractor group exactness at least 80%;
- boundary group exactness at least 70% and fully marker-free items at least 80%;
- long/8-distractor items at least 80%;
- three-hop confirmation at least 75% and four-hop confirmation at least 60% for an
  `ALGORITHMIC-EXTRAPOLATION` label.

High item accuracy with failed grouped counterfactuals remains shortcut behavior. A positive
control failure makes an architecture comparison incomplete. A shuffled-label or shuffled-group
null exceeding its prespecified leakage ceiling invalidates the campaign.

## 4. Campaign A: external differentiable memory

### A1. Raw write/read memory

Use a small causal token encoder only to produce occurrence features. A learned controller scans
the serialization once and writes into 16 differentiable memory cells. Each cell stores separate
address and content vectors. The controller has two read heads and one write head. After scanning,
three recurrent read steps receive the query: first retrieve an edge whose address matches the
current query, update the query state with its content, then repeat. No row boundary, row type,
correct position, equality flag or intermediate-key target is supplied.

Concrete local configuration:

- width 384, eight heads, two causal encoder blocks, FF 1,024;
- 16 memory cells of width 384, content-based cosine addressing with learned strength;
- GRU controller, one write and two read heads, three shared retrieval steps;
- tied 2,064-token input/output embeddings;
- approximately **6--9M parameters** and roughly **3--5 Transformer-block equivalents** per
  episode, depending on the controller implementation.

The memory size is deliberately larger than the maximum relation count, but not proportional to
the vocabulary. Test 8, 16 and 24 cells only on a development split fixed before confirmation;
freeze 16 unless the positive control demonstrably cannot allocate all rows.

### A2. Decisive controls

1. `memory_content`: full content-addressed writes and reads.
2. `memory_no_write`: identical encoder/controller, but memory is the fixed mean of token states.
3. `memory_one_read`: one retrieval step, same controller parameters and training exposure.
4. `memory_random_address`: trainable content but independently permuted address logits each
   episode; this controls extra capacity and recurrence.
5. `oracle_row_memory`: receives gold serialized row spans but still learns addressing and values;
   this is an optimization/upper-bound control, never evidence of raw parsing.

`EXTERNAL-MEMORY-HELPFUL` requires both full-memory seeds to pass every shared two-hop gate and
beat both `no_write` and `one_read` by at least 15 points on factorial exactness and long-distractor
accuracy. Its allocation entropy must not collapse to one cell, and changing a nonqueried row must
preserve at least 95% of answers. A success only against `random_address` does not isolate writing.

### A3. Confounds and intervention

Soft memory can smear all rows across all cells; readable rows or sparse weights do not prove a
discrete table. After behavioral qualification, reuse cross-recipient interchange: patch only the
post-first-read state or selected memory content from a donor with a different intermediate and
require outputs to follow each unchanged recipient table. Also patch allocation/address weights
without content, content without address, a norm-matched random state and the full post-second-read
answer state. This distinguishes reusable lookup from answer injection and global-memory copying.

Estimated bounded campaign: five arms × two seeds × 6,000 updates, batch 64. At this scale plan
**1.5--3 hours** and a **4 GiB** sampled MPS cap, but let the source-matched benchmark decide.

## 5. Campaign B: adaptive iterative state, not another fixed loop

### B1. Stateful universal processor

Encode tokens once with two causal blocks. Append four persistent memory tokens and apply one
shared two-block transition repeatedly. The transition sees token states, persistent memory and a
step embedding; token states may be updated, but the persistent cells are the designated recurrent
state. Train with 2--6 transition steps sampled uniformly. At evaluation run exactly 2, 3, 4, 6
and 8 steps. A Ponder-style halting head is a separate arm, not silently folded into the main
model.

Concrete configuration: width 384, eight heads, FF 1,024, two input blocks plus two shared
transition blocks, four memory tokens; approximately **8--10M parameters**. Six transitions cost
about 14 block applications, comparable to TEACH-0012's deep/looped compute while using many fewer
parameters.

### B2. Arms and decisions

1. `stateful_fixed`: persistent cells, randomly sampled 2--6 training iterations.
2. `stateful_adaptive`: same model plus a Ponder-style halting distribution and prespecified
   compute penalty.
3. `stateless_shared`: zero the persistent cells between iterations while retaining repeated
   shared blocks and step embeddings.
4. `untied_compute`: six untied two-block transitions at approximately matched inference compute.
5. `fixed_three_pass`: same persistent model trained and evaluated only at three transitions.

`STATEFUL-ITERATION-HELPFUL` requires `stateful_fixed` to pass the shared gates and beat
`stateless_shared` by at least 15 points on factorial and three-hop accuracy. It must show a
nondecreasing 2/3/4/6-step development curve within five points of monotonicity; gains appearing
only at one exact step are format-specific rather than stable execution. `ADAPTIVE-COMPUTE-USEFUL`
additionally requires adaptive halting to match fixed-six behavior within five points while using
at least 20% fewer mean transitions, and to allocate more transitions to three/four-hop or
8-distractor examples than to clean one-hop items in both seeds.

The halting head can learn length rather than reasoning difficulty. Cross length and hop count so
equal-length examples require different hops and equal-hop examples have different lengths.
Persistent tokens can become an unconstrained answer cache; donor first-transition patches must
adapt across recipient G tables, while final-transition patches should behave as answer injection.

Estimated bounded campaign: five arms × two seeds × 6,000 updates. Depending on sampled
transition count, plan **2--4 hours**, **6 GiB** sampled MPS, and a four-hour hard stop.

## 6. Campaign C: latent row parsing, set structure and graph execution

This branch is justified when marked/canonical rows work but marker-free, order or false-path
controls fail. It must not receive gold rows in its primary arms.

### C1. Exchangeable relation slots

A two-block causal encoder yields occurrence features. Twelve exchangeable slots attend
competitively to token occurrences for three iterations. Each slot emits two pointer distributions
over input positions, interpreted as ordered left/right endpoints, plus an existence probability.
The resulting candidate edges are processed as a set by two Set Transformer blocks. The query
state attends to this set twice to produce the answer.

Use width 384, 12 slots, three slot iterations and four-head pointer distributions. This is about
**9--12M parameters** and **6--9 block equivalents**. Hungarian matching to gold rows is allowed
only for diagnostics in the answer-only arm; slot numbering is exchangeable.

### C2. Soft graph processor

Turn slot endpoint distributions into a soft directed multigraph over unique episode symbols.
Initialize nodes from all occurrences of the same token ID, then run a shared message-passing
processor for 1--4 steps. Read the queried node after each step. Pointer Networks motivate
position selection; graph-network and CLRS work motivate encode-process-decode execution. The
primary graph is inferred, not supplied.

Arms:

1. `slot_set_answer`: slots plus two query reads, final-answer loss only.
2. `slot_graph_answer`: the same slots followed by shared graph message passing.
3. `slot_graph_endpoint_aux`: add endpoint-set supervision after permutation-invariant Hungarian
   matching, coefficient frozen before training.
4. `slot_graph_shuffled_aux`: identical auxiliary weight with endpoint targets permuted across
   episodes.
5. `oracle_graph`: gold endpoints with the same graph processor, as an execution upper bound.
6. `token_graph_dense`: fully connected occurrence graph without slots, matched processor.

`LATENT-PARSER-SUPPORTED` requires `slot_graph_answer` to pass all two-hop surface gates and beat
`token_graph_dense` by at least 15 points on marker-free and false-path groups. It must recover at
least 85% of gold edge sets under permutation-invariant matching on untouched diagnostics, but
edge recovery alone is insufficient. `PARSE-SUPERVISION-HELPFUL` requires the endpoint auxiliary
arm to improve marker-free and three-hop panels by at least 10 points over both answer-only and
shuffled-aux arms. Report this as supervision dependence, not unsupervised discovery.

`GRAPH-EXECUTION-SUPPORTED` additionally requires the same trained shared processor to improve or
remain within five points when run from two to four message-passing steps, and to meet the
three/four-hop gates. Patch the first-step queried-node state across recipient graphs and require
recipient-specific downstream answers. Supplying gold endpoints is never allowed in this causal
claim.

The main confounds are slot collapse, duplicate edge slots, pointer reliance on adjacency, and an
oracle induced by endpoint supervision. Track slot utilization, duplicate rate, endpoint distance,
marker dependence and performance when complete rows are permuted. A set architecture should be
invariant to complete-row order but not to swapping only right-hand sides.

Estimated bounded campaign: use one frozen development seed for the six-arm parser screen, then
train only the best answer-only parser, its decisive ablation, shuffled null and oracle graph with
two fresh seeds. Cap the combined program at **4 hours** and **8 GiB**; do not promote a development
winner without fresh-family replication.

## 7. Campaign D: neural algorithmic execution with hint discipline

The slot/graph architecture can still learn an opaque graph classifier. A stricter neural
algorithmic experiment gives the processor optional execution hints while keeping raw parsing
separate.

Use the inferred graph from Campaign C and a shared message-passing processor. At step (t), the
processor produces a distribution over the current symbol node and a halt probability. Gold
synthetic execution states are known: query node at step 0, intermediate node after one lookup,
answer node after two. Compare:

1. final-answer supervision only;
2. node-pointer hints at every step;
3. the same hint weight with within-episode cyclically wrong nodes;
4. teacher-forced gold state during training but free rollout at confirmation;
5. oracle graph plus answer-only processor.

`ALGORITHMIC-HINT-HELPFUL` requires the hinted model to beat both answer-only and wrong-hint arms
by at least 15 points on three/four-hop exactness in both seeds, while free rollout remains within
10 points of teacher-forced rollout. If only teacher forcing works, exposure bias remains and no
autonomous algorithm was learned. A pointer trace is supervised evidence by construction; causal
interchange and length extrapolation establish whether it controls the answer.

This campaign should be small: width 256 or 384, **5--10M parameters**, four shared processing
steps, four arms × two seeds × 5,000 updates, approximately **1--2 hours** after a benchmark.

## 8. Campaign E: action-conditioned latent world state

This branch targets an objective failure: a competent parser or memory exists, but final-answer
training does not stabilize a reusable intermediate. It uses grouped synthetic supervision while
avoiding direct key labels.

### E1. State and action formulation

Let an encoder map the raw serialization to latent memory (z_0). A shared transition

\[
z_{t+1}=T(z_t,a_t,M)
\]

receives an action embedding for `LOOKUP(query)` at (t=0), then `LOOKUP(previous)` at later
steps. An answer head reads (z_t). The transition runs twice for two-hop examples and more times
for longer-hop confirmation.

The key additional training unit is a **counterfactual recipient group**: several episodes share
the same F table and query but independently remap G. Their post-first-transition state should
agree after a learned affine alignment, while their second-transition answer must differ according
to each recipient G. Conversely, episodes with different F(query) but the same surface answer
should have distinguishable first-transition states. This supervises functional equivalence and
distinction without naming the intermediate key.

### E2. Losses and controls

Use four prespecified terms:

- final legal-answer cross entropy;
- one-hop answer loss at the appropriate transition;
- post-first-state consistency for same-F/query, changed-G groups;
- counterfactual future loss: the same post-first state must predict correct outcomes under three
  independently remapped recipient G tables.

Compare `world_counterfactual`, an identical answer-only recurrent model,
`world_shuffled_group` with consistency groups permuted, and `world_no_bottleneck` whose first
state can directly retain all token states. Use a 64- or 128-dimensional first-transition
bottleneck; the full latent memory remains larger.

`COUNTERFACTUAL-STATE-HELPFUL` requires the full objective to pass all behavioral gates and beat
both answer-only and shuffled-group controls by at least 15 points on factorial and three-hop
accuracy. On untouched interchange groups, a donor first-state must produce at least 80% exact
recipient-specific groups, at least 40 points above shuffled donors, while a matched orthogonal
complement preserves at least 90% of base answers. The full loss may induce a useful state without
making it minimal or unique; report bottleneck rank curves and random-subspace controls.

No reward, policy, tree search or reinforcement learning is warranted. MuZero/Dreamer motivate
decision-relevant latent transitions only; this remains supervised synthetic system identification.
Estimated campaign: four arms × two seeds × 6,000 updates, width 384, approximately **8--15M
parameters**, **2--3 hours**, **6 GiB** sampled MPS.

## 9. Mixtures only after single-bias attribution

The strongest eventual system is likely a hybrid: latent relation slots write a content-addressed
memory, and a shared graph/recurrent processor executes reads. Building that first would make a
positive result uninterpretable. Run a prospective 2×2 factorial only after one parser and one
executor independently qualify:

| Parser | Executor | Question |
| --- | --- | --- |
| token encoder | dense answer head | raw baseline |
| selected latent parser | dense answer head | parsing alone |
| token encoder | selected memory/processor | execution alone |
| selected latent parser | selected memory/processor | joint and interaction effect |

Match widths and final heads, freeze the selected components before generating new confirmation
families, and report the factorial interaction rather than crediting the whole mixture. Require the
hybrid to improve by at least 10 points over the better single component on both marker-free and
three-hop exactness in both seeds. A negative interaction is informative: the parser's slot basis
may not align with the executor's address space.

Do not add a mixture-of-experts router in this stage. Surface-conditioned experts could partition
marker styles and hide rather than solve invariance. If experts are later tested, balance every
logical program across styles and require each expert/routing intervention to transfer across
styles.

## 10. Conditional priority order

1. **First/direct strong; composed/factorial weak:** Campaign B, then D. The parser can expose
   relations, but iterative execution is missing.
2. **Marked rows strong; marker-free/order/false-path weak:** Campaign C. Relation induction is the
   bottleneck.
3. **Clean two-hop strong; distance/distractors/long OOD weak:** Campaign A. Test explicit storage
   and content addressing.
4. **Looped improves substantially but misses exact gates:** Campaign B with persistent state and
   adaptive iteration, not another fixed-loop scale-up.
5. **All raw arms near chance while parsed memory passes:** run A's oracle-row memory and C's oracle
   graph as positive controls; then test latent parser and raw memory separately. Do not launch a
   hybrid until one side qualifies.
6. **Behavior competent but donor intermediate is unstable across recipients:** Campaign E. The
   issue is functional-state supervision, not raw answer capacity.

At most one campaign should own the accelerator at a time. Freeze each registration, tests,
auditor, benchmark and stop rules before training. Failed arms remain part of the record; do not
replace seeds or tune on confirmation. Each campaign should fit within four local hours, eight GiB
sampled allocation and four GiB ignored artifacts unless a new resource estimate is explicitly
registered.

## 11. What would actually advance the project

A useful endpoint is not merely a higher synthetic score. The program advances only if a model:

1. parses raw ambiguous relations without gold rows;
2. executes them across unseen logical families, longer chains and false paths;
3. exposes a causally reusable intermediate that transfers across recipient tables and surface
   renderings;
4. reproduces in two seeds and survives matched shuffled/random/complement controls; and
5. identifies which architectural bias supplied the gain through a decisive ablation.

Even that endpoint only qualifies an experimental instrument. Before manuscript use, repeat the
method on corpus-sized known natural-language, cipher and nonsemantic generators with uncertain
segmentation, freeze a generator-discrimination or decoding rule, and test on held-out physical
folios partitioned by section, Currier language, scribal hand and transcription. Synthetic graph
success supplies neither a plaintext language model nor a historical cipher channel.

### Source and inference ledger

Primary paper pages or author/venue copies linked above were reviewed on 2026-09-23. The papers
support the stated architecture families and their original task settings. The arm definitions,
parameter ranges, local resource estimates, conditional priority order and decision gates are this
project's prospective proposals. They are not results reported by those papers. Parameter totals
are planning estimates to be replaced by exact implementation counts; time and memory ranges are
extrapolations that require a source-matched benchmark.
