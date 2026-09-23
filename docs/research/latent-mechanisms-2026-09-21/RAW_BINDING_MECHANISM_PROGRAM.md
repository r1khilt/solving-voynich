# Raw binding mechanism program: from behavior to a transportable causal algorithm

**Prospective theory and experiment-design memo, 2026-09-23.** No TEACH-0012
confirmation score, trained activation, checkpoint or prediction informed this memo. The
TEACH-0012 campaign was running while it was written. This document proposes conditional
follow-up work; it is not a preregistration, experiment result or Voynich claim.

## The question changes when the parser disappears

TEACH-0005--0011 mapped a complete algorithm in a parsed-row Transformer. A stable name address
selected the matching F row; F-row value content wrote a low-rank distributed key; that key
changed the next lookup's query; and all four downstream heads switched from the old G row to
the new G row. This was unusually clean because the architecture had six permanent slots and
each row was already compressed into one vector.

TEACH-0012 removes those gifts. A single relation can occupy two or three tokens, can appear
anywhere in the prefix, can lack its edge marker, and can be surrounded by syntactically valid
unrelated chains. The same symbol can serve as a name, key or object in different episodes.
A successful model must therefore solve at least four problems:

1. infer which adjacent tokens form a directed relation;
2. distinguish the queried chain from unrelated chains;
3. transport an episode-specific intermediate across token positions and layers;
4. use that intermediate to select an episode-specific answer.

The old “query slot contains a key” hypothesis is now only one member of a larger mechanism
family. Raw sequence competence could arise from a stable content vector, relational tags,
iterated pointer routing, a superposition of format-specific heuristics, or mixtures of these.
The follow-up must compare those accounts with counterfactual effects, not name whichever
activation is easiest to decode.

## Direct precedent and its unresolved gap

[Wu, Geiger and Milliere](https://arxiv.org/html/2505.20896) trained a 37.8M-parameter,
12-layer, eight-head, width-512 Transformer to dereference serialized variable assignments.
Their model passed through three phases: chance-like number prediction, early-line heuristics,
and a later systematic binding mechanism. Counterfactual residual and head patches traced value
information through right-hand-side token positions and finally into the query/output region.
They also found that the systematic circuit built on rather than erased earlier heuristics.

That study is the strongest direct precedent, but it leaves exactly the issues relevant here.
Its training programs used fixed line syntax, one-character variables, explicit equality and
newline tokens, and a fixed number of lines. Its counterfactual changes replaced a root value,
which traces answer content but does not by itself distinguish a reusable intermediate from an
answer already destined for one fixed context. TEACH-0012 randomizes row positions and boundary
cues, uses a much larger shared symbol vocabulary and independently remappable second-hop
contexts. A mechanistic follow-up can therefore ask whether a raw model forms a representation
that transfers across recipient tables and surface renderings.

Other primary work motivates competing hypotheses. [Feng and
Steinhardt](https://arxiv.org/abs/2310.17191) report binding-ID vectors that associate entities
and attributes in sufficiently large pretrained models. [Dai et
al.](https://arxiv.org/abs/2409.05448) report a low-rank ordering-ID subspace with causal binding
effects. [Davies et al.](https://arxiv.org/abs/2307.03637) localize shared arithmetic variable
retrieval using causal desiderata. These are large pretrained models and template tasks, not
predictions for TEACH-0012. They justify a multi-hypothesis test rather than assuming the parsed
key circuit will recur.

## Four mechanism hypotheses

Let a logical graph contain signal edges `n -> k` and `k -> o`. Let `p(x)` be the serialized
position of symbol occurrence `x`; a symbol can occur at more than one position. Let
`h_l(p; X)` be the residual state at layer cut `l`, position `p`, for input `X`.

### H1: content-key state

The model computes a representation of `k=F(n)` and transports it to a late query or answer
position. A donor state from an input with intermediate `k1`, patched into a base input whose
recipient G table is unchanged, should make the model produce the recipient-specific `G(k1)`.
The same donor content should work across row order, distractors and boundary renderings.

This is the closest raw analogue of TEACH-0007. It predicts a low-dimensional content geometry
whose causal meaning is preserved after alignment across seeds or formats. It does not predict
that individual neurons have stable names.

### H2: binding-ID matching

The model may represent content and relation membership separately. Matching endpoints receive
similar episode-specific binding IDs, and attention retrieves values by comparing those IDs.
Here, patching content alone may fail unless the associated ID is patched too. Swapping only
the binding-ID component between two edges should re-pair existing contents without inserting
a new key or object.

H2 predicts a relational similarity structure: within an episode, tokens belonging to the same
logical binding should share a causal component even when their symbol embeddings and physical
positions differ. Cross-episode raw cosine is not enough because arbitrary rotations and
nuisance variance can change it without changing the function.

### H3: ordering or position ID

The model may attach learned row-order or occurrence-order tags and use those tags to bind
fields. This can appear relational on ordinary examples while failing when physical order and
logical pairing are independently permuted. A true ordering-ID edit should swap bindings when
order is swapped, but its effect should track physical occurrence rather than the logical
edge under adversarial re-rendering.

TEACH-0012's four-render order groups weaken this shortcut behaviorally. Mechanistic tests must
go further by holding the token multiset and physical positions fixed while changing which
adjacent tokens form rows, and conversely by preserving rows while moving them.

### H4: dynamic routing without a stable bottleneck

Success may arise from sequential attention transfers along the two signal edges. No single
state needs to contain a linearly decodable full key or parse. Causal head/path patches can
still reveal a route even if every fixed-state probe is weak. Wu et al.'s layer-by-layer
right-hand-side transport makes this hypothesis particularly credible.

H4 predicts that a value perturbation advances through a sequence of layer-position edges,
with later hops depending on earlier writes. Patching one isolated late vector may be
insufficient, while patching the correct ordered set of head results succeeds. This is not a
failure of mechanistic explanation; it is a different computational ontology.

The hypotheses are not mutually exclusive. A model can use position heuristics for easy cases,
dynamic routing for hard cases and a late compact content state for output. The design must
measure mixtures rather than force a single global label.

## Stage 0: behavioral eligibility before mechanism claims

Mechanistic work begins only if the preregistered raw behavioral arm qualifies. It should also
stratify the frozen confirmation set before site discovery:

- correct answer row early versus late in serialization;
- zero, one, two, four and six distractor chains;
- marker-rich versus marker-free renderings;
- short versus long query-to-relevant-row distance;
- all four F/G family split cells;
- examples where nearest-row, first-row, last-row and most-recent-occurrence heuristics disagree
  with the oracle.

These strata must be defined from generator metadata, never from activation appearance. A
model that passes aggregate items but fails the anti-heuristic stratum has not earned a general
binding-circuit interpretation. A model can still support a narrower heuristic-circuit study,
clearly labeled as such.

## Stage 1: discovery map with paired counterfactuals

Create fresh logical families after TEACH-0012. Split them prospectively into discovery and
confirmation, preserving groups. For every base program construct multiple paired donors while
matching length, row count, delimiter pattern, physical row positions and distractor topology:

1. **F-value donor:** change only the queried F edge from `n->k0` to `n->k1`, preserving the G
   table. Expected answer changes from `G(k0)` to `G(k1)`.
2. **G-value donor:** preserve `F(n)=k0` and remap only `G(k0)`.
3. **pairing donor:** preserve the complete symbol multiset and row positions but swap the two
   values attached to selected left sides.
4. **format donor:** preserve the logical graph and query while changing row order, edge-marker
   style, marker dropout and gap pattern.
5. **distractor donor:** preserve the complete signal computation while independently replacing
   irrelevant chains.

At every cut and serialized position, patch the donor residual into the base and score exact
counterfactual answers. Use target probability change only as a secondary continuous metric.
Discovery selects sites separately for F content, G content, pairing and format invariance.
Selection must maximize the minimum effect across both model seeds and hard behavioral strata,
not the aggregate average of one seed.

Dynamic physical positions require semantic position labels derived from the known generator:
queried-F left, queried-F right, matched-G left, matched-G right, other signal endpoints,
distractor endpoints, task, query and answer. This metadata is an analysis oracle; it is never
fed to the raw model.

Confirmation then freezes the earliest reliable site or smallest reliable path for each
variable. If no single site qualifies, the result is “distributed/dynamic under this assay,”
not an excuse to report the visually strongest heatmap.

## Stage 2: recipient-specific interchange intervention

The decisive reusable-key test generalizes TEACH-0005. Choose paired first tables `F0,F1`
with `F0(n)=k0` and `F1(n)=k1`. Construct three recipient tables `G0,G1,G2` in which
`Gj(k1)` differs across `j`, while maintaining identical rendering skeletons. Compute one donor
state under `(F1,G0,n)` and transplant it into three base runs `(F0,Gj,n)`.

A reusable key mechanism predicts three different answers: `Gj(k1)`. Fixed `G0(k1)` across
recipients is answer injection. Failure to change outputs despite clean competence rejects
sufficiency at the selected site. Necessity patches must also restore `k0` behavior in an
`F1` run; sufficiency without necessity is compatible with a parallel route.

Primary controls:

- identity patch and native/manual recomputation;
- complete late donor state as answer-injection positive control;
- input-level signal-edge replacement as task positive control;
- same-key/different-format and same-key/different-distractor patches;
- cyclic other-group, wrong-position and norm-matched random patches;
- both patch directions;
- direct and copy tasks;
- full sample score and separately reported both-clean-correct score.

The old intervention worked at a permanent query slot. In the raw model, candidate sites can
be the query occurrence, final ANSWER occurrence or a dynamically located relation endpoint.
Recipient transfer should be tested at all three semantic roles on discovery and frozen before
confirmation.

## Stage 3: separate content from binding ID and order

Suppose a discovery state causally transfers the donor key. Fit the following representations
on discovery only:

- key-content centroids or supervised contrasts;
- relation-pair contrasts with content and position crossed factorially;
- physical-order contrasts with logical pairing held fixed;
- delimiter/format contrasts with graph held fixed.

Use balanced factorial data so each content occurs in every selected row position and binding
role. Simple linear classifiers otherwise confound content with location. Apply label-shuffle
and selectivity controls following the logic of control tasks for probes.

Then use causal decomposition on untouched confirmation examples. For an orthogonal basis of
candidate subspaces, patch:

1. content component alone;
2. binding component alone;
3. order component alone;
4. content plus binding;
5. orthogonal complement;
6. matched random subspaces.

H1 predicts content-alone recipient transfer. H2 predicts binding edits that re-pair preserved
contents and content+binding synergy. H3 predicts order-component effects that follow physical
positions under logical re-pairing. H4 predicts that no static subspace achieves the effect but
an ordered path patch does.

This analysis must treat the representation's basis as a gauge. Report subspace rank,
participation ratio, principal angles and cross-seed alignment. Native neuron masks should be
compared with random rotations, as TEACH-0006 already showed that a compact 11-dimensional
causal code can require all 128 native coordinates. “Neuron 73 is the key neuron” is not stable
unless its causal concentration survives function-preserving rotations or has an independently
fixed basis.

## Stage 4: dynamic circuit decomposition

For a causal state or path, decompose attention in the model's actual coordinate system:

- Q/K/V factorial interventions at dynamically labeled token positions;
- per-head output/result patches rather than attention weights alone;
- source-token contribution decomposition through output projection;
- path patches that isolate earlier writers from later readers;
- head subsets selected on discovery using minimum-both-seed effects;
- complement, MLP-only and full-residual controls;
- exact numerical reconstruction below a frozen tolerance.

The target factorization is not assumed. Possible outcomes include:

- query Q changes to address a matching row K, whose V carries the next content;
- relation IDs are written to K/Q while contents remain in V;
- one head moves a pointer/ID and another head retrieves content;
- MLPs construct a normalized relation code between attention stages;
- different surface formats use genuinely different circuits.

Format-specific circuit claims require cross-format transport. Discover a circuit on marker-rich
programs, freeze it, and test marker-free programs; then reverse. A shared algorithm should
retain causal effect after position alignment. If behavior is invariant but circuits differ,
the correct conclusion is redundant format-specific implementation, not one universal parser.

## Stage 5: developmental analysis

TEACH-0012 stores deep checkpoints at 250, 500, 1,000, 2,000, 4,000 and 8,000 updates. Apply
the final frozen discovery/confirmation assay to every checkpoint; do not rediscover a new site
at each time point. Track:

- legal-output probability;
- heuristic-stratum accuracy;
- exact query and factorial groups;
- recipient-specific interchange accuracy;
- selected path/head effects;
- content, binding and order subspace ranks;
- off-path heuristic patches.

This distinguishes smooth strengthening from phase transitions. It can test whether an early
position heuristic persists after systematic binding appears, as in Wu et al., or whether the
strong nuisance randomization suppresses it. Checkpoints are observational time points; changes
between them do not identify an optimizer-level causal event.

## Stage 6: looped versus untied algorithm

If both raw-deep and raw-looped models qualify, compare them as different computational
implementations. The looped model repeats four shared blocks three times, so pass index is a
natural causal coordinate. A two-hop algorithm might align with passes more cleanly than with
untied layer depth.

Fit no cross-model map on confirmation. On discovery, align corresponding semantic states with
orthogonal Procrustes or CCA-like methods, then test confirmation patches across:

- untied layers 0--3 versus loop pass 0;
- untied layers 4--7 versus loop pass 1;
- untied layers 8--11 versus loop pass 2;
- shuffled-pass and shuffled-example controls.

Successful causal transfer would show shared functional geometry despite different parameters.
Correlation or high canonical similarity without transferred behavior is weaker evidence.

## Subspace-patching cautions

[Distributed Alignment Search](https://arxiv.org/abs/2303.02536) provides a framework for
aligning high-level causal variables with distributed neural representations. But optimized
subspaces are powerful enough to create misleading interventions. [Makelov et
al.](https://arxiv.org/abs/2311.17030) show an interpretability illusion in which subspace
activation patching can appear causal through effects unrelated to the hypothesized variable.

Accordingly, any learned rotation or subspace must satisfy:

- discovery/confirmation separation;
- shuffled-label and random-subspace baselines;
- output specificity across factorial recipient contexts;
- necessity, sufficiency and rescue;
- complement preservation;
- multiple seeds;
- causal cross-format or cross-model transport;
- explicit rank/parameter count and search multiplicity;
- no selection on confirmation.

A subspace that changes answers but cannot follow recipient-specific `G(k)` is not a reusable
key. A subspace that decodes a key but fails intervention is an accessible correlate. A patch
that works only after a large learned rotation and fails shuffled controls may be an optimized
control direction rather than the model's algorithm.

## What a strong result would mean

The strongest plausible outcome is not “we found a key neuron.” It is:

1. exact raw behavior survives family, query, assignment, row-order, distractor, boundary and
   length counterfactuals in both seeds;
2. a discovery-frozen intermediate or path transfers `F(n)` across new recipient G tables;
3. necessity, sufficiency and rescue agree;
4. the effect survives changes in serialization while format nuisance components do not;
5. Q/K/V and source decomposition identify how relation addressing and content transport work;
6. a functionally corresponding code or route causally aligns across seeds and perhaps looped
   and untied models;
7. checkpoint analysis shows how that circuit emerges and which shortcuts remain.

That would demonstrate a learned, transportable causal abstraction in a raw synthetic sequence
model. It would justify using the architecture as a calibrated instrument on several known
linguistic, cipher and nonsemantic generators. It still would not make a manuscript activation
“a word,” “a plant,” “a cipher key” or “meaning.” Those names require external anchors and a
frozen decoder that predicts held-out physical material.

## What negative outcomes teach us

- **Behavior fails, parsed control passes:** serious scale and depth still did not recover this
  raw binding problem under the frozen data/objective; strengthen the data/task formulation or
  change inductive bias prospectively, not the thresholds.
- **Behavior passes only marker-rich cases:** the model learned a delimiter-dependent parser.
- **Behavior passes but no single mediator does:** search for dynamic path mediation; do not
  report the strongest probe as a key.
- **Recipient patches retain one donor answer:** answer transport, not reusable key transfer.
- **Binding edits track physical order under re-rendering:** ordering shortcut.
- **One seed or one format has a clean circuit:** not replicated or not format-general.
- **Looped succeeds while shallow fails:** shared iterative computation is sufficient evidence
  for this task, not proof that natural language or Voynich requires recurrence.
- **Deep succeeds while looped fails:** untied layer specialization or parameter capacity matters
  under this recipe; it does not isolate which without a new comparison.

Every one of these is more informative than another embedding plot because each narrows which
computations the model can actually perform and transport.
