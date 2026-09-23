# Token and causal geometry beyond raw cosine

Status: prospective methods memo written while TEACH-0012 confirmation remains sealed. This is
not an experiment registration, result or Voynich interpretation.

## The useful part of the token-similarity idea

The user's suggestion is good if “similarity” means a tested relation between *contextual
computations*. It is weak if it means making one cosine-similarity heatmap of the embedding
table and naming nearby tokens as semantically related.

TEACH-0012 deliberately draws all ordinary symbols from one shared vocabulary and changes their
logical roles across episodes. A vocabulary row therefore has no fixed meaning such as “name,”
“key” or “object.” The relevant objects are occurrences: the state of token `x` when it is the
left side of the queried F edge, the same token when it is an unrelated G value, and so on. A
strong analysis asks whether occurrence states have the same counterfactual effect, participate
in the same relation, or can substitute for one another across contexts.

That distinction matters even in ordinary language. [Ethayarajh
(2019)](https://aclanthology.org/D19-1006/) found that contextual word representations are
anisotropic and become increasingly context-specific in upper layers. A single static vector
captures little of their variation. More generally, [Steck, Ekanadham and Kallus
(2024)](https://arxiv.org/abs/2403.05440) show analytically that cosine similarities of learned
embeddings can be arbitrary under degrees of freedom that leave the learned model's function
unchanged. Cosine is a coordinate statistic, not automatically a semantic fact.

## Three different questions that should not be collapsed

### 1. Are two individual states geometrically close?

For centered activation vectors `h_i,h_j`, ordinary cosine is

`cos(h_i,h_j) = <h_i,h_j> / (||h_i|| ||h_j||)`.

It ignores vector norm and depends on the coordinate system. It can be useful inside one fixed
layer and one fixed model when the layer's downstream operations actually use dot products in
that coordinate system. It is not automatically comparable across layers, seeds or models.

Centering removes a shared mean direction. Whitening by a discovery-set covariance `C` produces
`z = C_lambda^(-1/2)(h-mu)`, where shrinkage `lambda` is fixed before confirmation. Cosine or
Euclidean distance in `z` space is a covariance-normalized descriptive statistic. It can reveal
relations hidden by anisotropy, but whitening is itself a fitted coordinate choice and cannot
establish causal use.

### 2. Do two populations have the same representational organization?

Given activation matrices `X` and `Y` for matched examples, this is a layer/space comparison,
not a token-pair question. [Kornblith et al.
(2019)](https://proceedings.mlr.press/v97/kornblith19a.html) motivate centered kernel alignment
(CKA) and explicitly analyze which transformations a similarity index should ignore. Linear CKA
is invariant to orthogonal transformations and isotropic scaling, which makes it useful for
comparing seeds or checkpoints whose individual neurons are not aligned.

[SVCCA](https://proceedings.neurips.cc/paper/2017/hash/dc6a7e655d7e5840e66733e9ee67cc69-Abstract.html)
and [PWCCA](https://arxiv.org/abs/1806.05759) compare correlated subspaces after learned linear
alignment. They can be useful diagnostics of training dynamics, but sufficiently flexible
alignment can make unrelated high-dimensional spaces appear similar on small samples. Fit every
alignment on discovery, report its rank and free parameters, and test it on confirmation.

Representational similarity analysis (RSA) instead compares pairwise dissimilarity matrices.
Its virtue is that hypotheses can predict a geometry directly: same intermediate key, same
logical edge, same physical row, same surface format or same output. These competing model RDMs
can be crossed factorially and tested on held-out groups. RSA still measures association rather
than computational use.

### 3. Can two states do the same job?

This is the mechanistic question. Define a standardized bank of downstream contexts `c` and a
counterfactual effect signature

`e(h) = [ Delta log p(y_c | do(h)) ]_c`,

where each context changes the recipient G table, task, row order, boundary format or distractor
set while preserving the variable being tested. Two states are functionally similar when their
effect signatures match on unseen contexts. The strongest discrete version is interchange:
substituting one state for the other produces the recipient-specific oracle answer in every
context.

This is invariant to many harmless internal reparameterizations because it is defined by model
behavior. It also distinguishes a reusable key from a fixed answer: a key state yields different
`G_j(k)` under different recipients, whereas an answer state repeats one object.

## A behavior-aware local metric

Coordinate distance can be replaced with the model's own local sensitivity. Let `J_c` be the
Jacobian from an activation at a fixed layer/semantic position to downstream logits in context
`c`. Let `F_c` be a positive-semidefinite output weighting, such as the categorical Fisher matrix
or a frozen weighting over legal answer logits. Define

`G = E_c[J_c^T F_c J_c]`.

For a small activation difference `delta`,

`d_G^2(h,h+delta) = delta^T G delta`

approximates the expected downstream distributional change. Directions in the null space of `G`
are locally invisible to the chosen downstream contexts; directions with high eigenvalues are
high-leverage. This gives a principled alternative to treating every residual-stream coordinate
equally.

The metric is local and task-distribution dependent. A small `d_G` does not imply global
equivalence, and first-order predictions can fail for the finite interventions needed for causal
claims. Therefore use it to rank or summarize discovery candidates, then execute finite patches
on frozen confirmation data.

## What the J-space paper changes

[Gurnee et al. (2026)](https://transformer-circuits.pub/2026/workspace/index.html) do not define
J-space as ordinary token-embedding geometry. Their Jacobian lens averages the map from a layer's
residual stream to present and future final residual states across many contexts, then composes
that map with the model's unembedding. A J-lens vector is therefore indexed by a vocabulary token
but derived from its average potential effect on future verbal output.

The paper further treats these vectors as an overcomplete, non-orthogonal frame. J-space is a
union of sparse nonnegative cones, not one simple linear subspace. Its functional evidence comes
from swaps, ablations, flexible reuse across downstream tasks and broad read/write connectivity,
not from large cosine alone. The authors also state that the lens is approximate and restricted
by token verbalizability.

TEACH-0012 offers an unusually controlled miniature analogue because its intermediate keys are
sometimes explicit outputs in the `first` task but silent intermediates in `composed`. Define
task-conditioned average Jacobians at each layer and semantic position:

- `J_first`: effect on future key logits under first-hop requests;
- `J_direct`: effect on future object logits under direct-G requests;
- `J_composed`: effect on future object logits under composed requests.

Then ask whether a silent intermediate in a composed episode activates the same action-ready
key frame that the model uses when it must output that key directly. The decisive test is not a
J-lens label. Swap key-frame coordinates from `k0` to `k1` inside a composed run and require the
result to become the current recipient's `G(k1)` across several independently remapped G tables.
If it repeats the donor object's token, the intervention edited an answer route instead.

Even a positive result would support only a synthetic, output-addressable workspace-like
interface. It would not justify a consciousness claim or imply that a Voynich latent has a
verbal English gloss.

## Prospective experiment G1: static vocabulary sanity check

This is a diagnostic, never the primary result.

1. Extract input embedding rows and tied/untied output rows separately.
2. Report norms, mean pairwise cosine, eigenspectrum, effective rank and hubness.
3. Compare raw cosine, mean-centered cosine, shrinkage-whitened cosine and model-native bilinear
   scores where the architecture actually uses them.
4. Test whether nearest-neighbor identities are stable across the two seeds using overlap and
   rank correlation.
5. Compare against random initialization and frequency-matched permutations.
6. Regress geometry against token training frequency, output eligibility and positional exposure.

Because ordinary symbols change roles by episode, stable role clusters are not expected. A
cluster explained by output frequency or rare-token norm is an optimization artifact, not a
learned vocabulary ontology. Failure to find static structure is compatible with rich contextual
computation.

## Prospective experiment G2: factorial contextual geometry

Generate fresh discovery and confirmation episodes in which the same symbol occurrences are
crossed over:

- logical role: F-left, F-right, G-left, G-right, distractor and query;
- relation identity: same edge versus different edge;
- content identity: same token versus different token;
- physical position and distance;
- prefix/infix/suffix or marker-free rendering;
- task: first, direct and composed;
- correct recipient and distractor topology.

At every checkpoint, layer and semantic position, save contextual states without selecting on
confirmation accuracy. Build discovery RDM hypotheses for content, role, relation membership,
physical order, format and expected answer. Fit a regularized variance-partition model with
grouped cross-validation by logical family. Confirmation reports unique variance and family
bootstrap intervals.

A relation geometry must survive new contents, positions and formats. A content geometry must
follow the same symbol or intermediate across roles. An order geometry that disappears under
rerendering is a serialization shortcut. No RDM correlation earns a mechanism label without G4.

## Prospective experiment G3: cross-seed and developmental alignment

Use matched logical episodes but never reuse confirmation to fit maps.

For each pair of seeds or checkpoints:

1. compare centered Gram matrices with linear CKA;
2. compare hypothesis RDMs with RSA;
3. fit orthogonal Procrustes on discovery and report confirmation residuals and principal angles;
4. fit SVCCA/PWCCA only with fixed rank and shrinkage, and include shuffled-example controls;
5. test whether a discovery-fitted map transfers an activation causally in G4.

CKA can show that two layers arrange examples similarly even if neurons rotate. Procrustes can
provide an explicit orthogonal map for intervention. CCA can diagnose shared low-dimensional
information. Only transferred behavior shows that the aligned coordinates are functionally
compatible.

Developmental plots use the already frozen TEACH-0012 checkpoints. Freeze the final discovery
variables and alignment before applying them backward through training. Do not rediscover a new
subspace at every checkpoint. Track when content, role, order and causal-effect geometries appear,
and whether a shortcut geometry persists after systematic behavior emerges.

## Prospective experiment G4: causal equivalence matrix

Construct 128 fresh logical groups per split. Each group contains two F assignments and at least
three recipient G tables, with rendering and distractors independently crossed. For every
discovery-selected layer/position/subspace, measure:

- **identity:** replace a state with itself and reproduce logits;
- **same-variable nuisance:** transfer the same key across format/order/distractor changes;
- **different-key sufficiency:** transfer `k1` into a `k0` base and require `G_j(k1)`;
- **necessity:** restore `k0` in a `k1` computation;
- **relation re-pairing:** swap binding components while preserving the content multiset;
- **order control:** move rows while preserving logical edges;
- **random and cyclic controls:** matched norm/rank/search multiplicity;
- **complement:** remove or preserve the proposed subspace;
- **direct/copy specificity:** ensure the patch has the proposed scope;
- **finite versus linearized effect:** compare the Jacobian prediction with actual patch behavior.

Summarize each state by its confirmation effect signature. Cluster or embed these signatures only
after preserving their labeled contexts; distance in effect space is the substantive notion of
functional similarity. Require cluster stability under family bootstrap and both seeds.

## Prospective experiment G5: sparse J-lens analogue

Compute average Jacobians on discovery contexts only, separately by layer and task. Build token
frames from legal key/object output rows after the average Jacobian, and solve a regularized
nonnegative sparse coding problem. Freeze sparsity, regularization and layer ranges before
confirmation.

Primary comparisons:

- averaged Jacobian lens versus raw logit lens;
- current-token Jacobian versus present-and-future Jacobian;
- real token frame versus label-shuffled and Gaussian frames;
- sparse frame reconstruction versus equally ranked PCA and random subspaces;
- key-frame swap versus full-state and output-answer positive controls;
- shared swap across first-hop and composed tasks;
- ablation of top key coordinates versus equal-energy off-frame ablation.

Support requires all of the following in both seeds: reliable discovery-to-confirmation readout;
recipient-specific finite causal swaps; necessity and sufficiency; advantage over matched frames;
reuse across tasks or recipients; and low effect on unrelated copy behavior. High token rank or
reconstruction variance alone is descriptive.

## Implications for linguistics and decipherment

Language structure exists at several levels that need different units and controls:

- **graphic/phonotactic:** glyph or character neighborhoods and local constraints;
- **morphological:** reusable stems/affixes and paradigmatic substitutions;
- **syntactic:** contextual roles and dependency relations;
- **semantic/pragmatic:** reference and task-dependent consequences.

One token cosine matrix conflates them. For a manuscript with disputed transcription and word
boundaries, the problem is worse: a token may be a glyph, glyph group, scribal variant, word,
abbreviation or cipher unit. Geometry can suggest equivalence classes or contextual roles, but
cannot choose among those ontologies without controlled alternatives.

Before applying any learned geometry to Voynich, calibrate it on known systems with hidden labels:
natural languages, substitution and homophonic ciphers, nomenclators, synthetic generated
languages, copied pseudo-text and nonlinguistic structured controls. Freeze the metric and
selection procedure, then require recovery of held-out relations and rejection of structureless
or misleading controls. Compare multiple tokenizations and transcription variants.

On Voynich itself, the defensible output would be a prediction such as: occurrences in class A
substitute for class B in a specific held-out positional context, or a discovered relation predicts
unseen line/folio structure. It would not be a translation produced by nearest English embedding.

## Decision table

| Observation | Supported interpretation | Excluded leap |
|---|---|---|
| Static embedding neighbors repeat across seeds | Stable optimization geometry | Fixed token semantics |
| Centered/whitened role clusters generalize | Contextual role is linearly organized | Model uses it causally |
| CKA/RSA aligns layers across seeds | Similar example geometry | Same algorithm |
| Procrustes map transfers behavior | Shared causal geometry under that map | Unique coordinate basis |
| J-lens reads the intermediate key | Key is output-addressable | Key mediates composition |
| Key-frame swap yields each recipient `G_j(k1)` | Reusable intermediate is causally sufficient | Complete mediation |
| Necessity, sufficiency and rescue all pass | Strong causal abstraction at tested site/path | Voynich meaning |
| Geometry predicts known-system held-out relations | Calibrated structural instrument | Decipherment by itself |

The core principle is simple: use geometry to propose equivalences, and counterfactual behavior to
decide whether the model actually treats them as equivalent.
