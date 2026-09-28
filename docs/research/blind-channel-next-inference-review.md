# What a bounded channel-search failure would, and would not, establish

2026-09-27. **Results-blind mathematical and optimization review.** Read the model,
search, existing mathematical audits, and
[BLIND-CHANNEL-DEV-001 registration](../experiments/BLIND-CHANNEL-DEV-001.md).
No pilot source-selection output, generated ciphertext, fitted channel, gold
mapping, target text, or recovery result was opened. This document changes no
registration or result. The isolated bijective adapter described at the end was
implemented after this review; it changes none of the frozen pilot's search
files. Other interventions below are proposals for a separately bounded
development follow-up, not completed experiments.

The first pilot can test the implemented search under its stated budget. It
cannot establish that unknown-unit recovery is impossible. Several distinct
failures can produce poor decoding: an objective preferring a wrong explanation,
insufficient search within an adequate objective, a difficult path between
different unit dictionaries, an excluded model, or genuine information loss.
The following distinction is especially important:

**For family A, a perfect gold-channel decoder is automatic and does not validate
the source prior.** A known deterministic bijection has exactly one compatible
plaintext. With the registered positive source, that plaintext is decoded
regardless of whether its language-model score beats the scores of incorrect
keys. Family B's gold-channel decoder is informative about segmentation under
the given channel, but it still does not establish that the objective prefers
that channel over every wrong channel.

## Primary-source checks and the limits of their relevance

The existing [blind-channel design](blind-channel-recovery-design.md) supplies
the broader normalized-transducer, coding, identifiability, and Voynich review.
For this optimization question, additionally opened these primary PDFs:

| Source and reading depth | Relevant evidence | What is not established here |
| --- | --- | --- |
| [Nuhn, Schamper & Ney 2013, *Beam Search for Solving Substitution Ciphers*](https://aclanthology.org/P13-1154.pdf), definitions, search description and §6 | Separates search quality from language-model quality: higher-order scoring with bounded search can recover better than exact optimization of a weaker score. Also studies beam width and model order together. | Supplied substitution units and constraints differ from unknown variable emissions. Their scores and computational results do not transfer automatically to Latin or this transducer. |
| [Berg-Kirkpatrick & Klein 2013, *Decipherment with a Million Random Restarts*](https://aclanthology.org/D13-1087.pdf), §§2–3 | Demonstrates strong dependence of EM decipherment on initialization and restart budget for particular homophonic ciphers; compares found objective values with gold-initialized fits without claiming a global certificate. | Their HMM has supplied symbol units, a trigram prior, 200-update starts, tuned emission smoothing, and strongly weighted same-author language material. This neither authorizes an unbounded restart campaign nor predicts how many starts our problem needs. |
| [Chiang et al. 2010, *Bayesian Inference for Finite-State Transducers*](https://aclanthology.org/N10-1068.pdf), reviewed estimation/model sections alongside the prior project review | Normalized transition inventories and latent path counts support parameter estimation and segmentation models. | Learning weights within an inventory is different from constructing a new inventory. Our EM/local proposals are not their Bayesian algorithm or an exact posterior sampler. |

The algebra and proposed interventions below are deductions for this repository's
model. They are not novel cryptanalytic claims attributed to these papers.

## Separate four questions before interpreting recovery

Let `J(theta) = C(theta) - log2 p(Y_fit | source, theta)` under the same declared
coding context. Let `theta_gold` be the evaluator's generating channel and
`theta_found` the blind search result. Evaluator-only measurements can diagnose:

| Observation | What it supports | What it does not support |
| --- | --- | --- |
| Gold channel has poor B decoding | The supplied source/decoder cannot resolve the given channel's ambiguity on these records; decoding decision rule or intrinsic ambiguity may contribute. | An assertion that no better source or decision rule can decode it. |
| Gold channel decodes well and `J(gold) < J(found)` | A concrete better candidate was missed by this run, if gold satisfies the same grammar, pool and coding bounds. | Global optimality or identifiability of gold. |
| A wrong, poor-decoding channel has `J(found) < J(gold)` | This finite-sample objective prefers that wrong explanation to the literal generating description. | Proof that all optimal channels decode badly, or that ciphertext is intrinsically ambiguous. |
| Distinct decoders have equal probabilities for every possible ciphertext | Observational nonidentifiability for those exact model/source pairs. | Equality merely because a finite probe set or one fit sample scores similarly. |
| Same poor solution repeatedly wins restarts | Stability under the tested proposals and budget. | Global optimality, correct plaintext, or an identifiability theorem. |

For the third row, first inspect model bits and data bits separately, channel
equivalences, and source misspecification. A compact representation can have
different syntax from the generating description. Compare plaintext behavior,
not only literal key entries or arbitrary state names. Scores from different
source priors require a fixed source catalogue/context or explicitly labeled
conditional sensitivity analyses; do not silently compare uncharged choices.

Gold inspection belongs to a separate evaluator. Any subsequent design choice
made after that inspection is exposed development, requiring fresh cases for
confirmation. This review does not request an extra unregistered pilot scoring
pass or a change to the running protocol.

## An exact, cheap diagnostic for the bijective A subfamily

Assume one state, exactly one single-glyph emission per source letter, and a
bijection. Write `pi(g)` for the source letter assigned to glyph `g`. For a
collection of records, precompute:

- `b_g`: number of nonempty records beginning with glyph `g`;
- `C_gh`: count of adjacent ordered glyph pair `(g,h)`, never crossing records;
- `N`: total glyph count; `R`: number of records, including empty records.

For the registered order-one source, let `p0(a)` be its start probability and
`P(b|a)` its transition probability. The exact natural-log marginal is

```text
F(pi) = sum_g b_g log p0(pi(g))
      + sum_g,h C_gh log P(pi(h) | pi(g))
      + N log(1-rho) + R log rho.
```

There is only one latent path, so this is the full normalized likelihood, not a
Viterbi approximation. All such bijective channels have the same literal model
code length under the current context: the row counts, lengths, destinations,
weight-vector dimensions and number of glyph fields agree. Their model-code
**contents** differ, but their code **lengths** do not. Maximizing `F` therefore
exactly minimizes `J` inside this subfamily. Rare or absent glyphs remain in the
declared permutation; their zero counts contribute no direct evidence.

For a swap of mappings at glyphs `u` and `v`, only start terms for `u,v` and
bigram terms with either endpoint in `{u,v}` change. The exact delta is

```text
Delta F = sum_(g in {u,v}) b_g [log p0(pi'(g)) - log p0(pi(g))]
        + sum_((g,h): g in {u,v} OR h in {u,v})
          C_gh [log P(pi'(h)|pi'(g)) - log P(pi(h)|pi(g))].
```

Count each directed pair in this union once, especially `(u,u),(u,v),(v,u),(v,v)`.
One swap costs `O(K)` table operations after counting; a complete `K choose 2`
neighborhood costs `O(K^3)`, independent of corpus length. With `K=23`, one sweep
checks all **253** distinct swaps. Repeat best-improving or fixed-order improving
sweeps until a whole sweep fails, under an explicit sweep/start budget. This
certifies only a pair-swap local optimum of the restricted family.

Useful subsequent extensions are all 3-cycles (two directions for each triple,
**3,542** at K=23), bounded multi-swap perturbations followed by complete sweeps,
or a beam over partial permutations with every finished key exactly rescored.
These may cross pair-swap barriers; none guarantees a global solution. An
integer-programming/A* certificate would require valid bounds and an explicit
resource limit, not simply an exact objective evaluator.

The present generic proposal distribution spends half its attempts on immediately
inapplicable move types while a one-state incumbent has one alternative per row:
retarget, initial-weight transfer, row-weight transfer, and removal. Its two row
swap types coincide in one state and together receive one quarter of attempts.
If an unchanged incumbent received 1,000 such draws, about 250 swap attempts
would cover only about `253*(1-(252/253)^250)`, approximately 159 distinct pairs
on average. Incumbent changes make this an illustration, not the actual run's
coverage. A 1,000-proposal trace does not itself establish a complete local
neighborhood search.

**Required controls for a fast implementation:** on tiny positive sources compare
every permutation and every swap delta with full engine scores; include empty
records, directed asymmetric counts, self-pairs, absent glyphs, and record resets.
Use exhaustive small-alphabet permutations to check local/global distinctions.
For real development instances, independently rescore every final candidate with
the original engine. Report proposals, distinct evaluated keys, complete sweeps,
and actual CPU/wall cost separately. A faster subfamily is a disclosed diagnostic
arm and a possible ciphertext-only warm start for general search, not an oracle
gift about an unknown historical cipher.

### If the selected source is order zero

Inside the same bijective subfamily,

```text
F(pi) = sum_g n_g log p(pi(g)) + constant.
```

The rearrangement inequality gives a global maximum by pairing sorted observed
glyph counts with sorted source probabilities; ties can leave multiple maxima.
The first frequency initializer already implements that assignment when alphabet
sizes match. Further permutation search cannot improve this objective. If that
assignment is wrong, the source/order-zero objective is insufficient on these
data; it is not evidence of a hard permutation optimization problem. More
flexible nonbijective/variable-unit channels could still obtain a different score,
but that is a different subfamily and does not repair the order-zero key evidence.

Consequently the first source diagnostic is the **selected source order and
gold-key score neighborhood**, not A's oracle character error. Even for order one,
the correct key can have a wrong neighbor with a better score. Such a concrete
neighbor falsifies a claim that more exact search of this unchanged objective
must restore gold.

## Why B is more than a larger permutation problem

Family B starts with an unknown literal dictionary and ambiguous concatenations.
The current initializer emits one glyph per source character. With six glyphs
and 23 source letters, many source letters initially collide on the same glyph.
The gold generating family instead assigns 17 letters length-two emissions.
The initializer thus begins in a qualitatively different parsing regime.

Single-row changes can face three barriers:

1. Removing a singleton can destroy every parse of some record before other
   rows learn replacements. Exact rejection is correct, but a sequence of
   individually valid intermediate models need not be easy to find.
2. Adding a second alternative pays extra literal-string/weight code immediately
   and initially allocates it substantial mass. A useful collection of long
   units may be unattractive one unit at a time.
3. One EM update may not settle weights after support changes; EM cannot introduce
   an absent unit and cannot revive an exactly zero alternative. Exact EM improves
   continuous-weight likelihood, not necessarily the quantized two-part score.

These are optimization barriers, not proof that the true dictionary is best or
recoverable. Gold A/B channels have deterministic weights representable on
denominator32; grid resolution does not exclude their weights. The grid can
nevertheless impede intermediate soft explanations through its minimum positive
mass of 1/32 and the three-alternative bound.

For B's six-glyph/length-two setting there are only **42** possible nonempty units
(six singletons plus 36 pairs). The current observed-substring pool may omit an
unseen true pair, but its 128-unit cap does not truncate the full 42-unit universe.
A future proposal pool containing all 42 strings would remove this particular
inventory exclusion without revealing which 23 are used. It must still pay every
retained literal emission in the same model code. This removes a small search
restriction, not the combinatorial row-assignment and segmentation problems.

### Coordinated unit changes

Propose a complete k-row edit before evaluating it, with a bounded choice such as
`k in {2,3,4}`. Possible proposals replace units for several source letters,
exchange a set of singleton assignments for multi-glyph units, or combine births
and deletions while staying within each row's limit. Normalize each changed row,
quantize, and compute the exact marginal/code once for the completed proposal.
Do not reject because a hypothetical intermediate row edit would have failed.

Proposal ordering can use repeated substrings, source frequencies, or posterior
counts, but must include a seeded exploration component. Frequency alone confuses
true units with strings crossing two units. Compare k-row proposals against an
equal-cost single-row arm, record their number of changed rows and exact scoring
cost, and keep every rejection. Random k-row edits without informed ordering may
mostly fail; their benefit is a testable proposal, not assumed.

A more targeted **continuous likelihood derivative** can rank an absent arc
before a discrete proposal. For one state, add a new emission string `v` to source
row `a` with mass epsilon, scaling its old row by `1-epsilon`. For one supported
record, unnormalized forward/backward arrays `alpha,beta` give

```text
d log p(y) / d epsilon at zero
  = [sum_(i,h: y[i:i+len(v)] == v)
       alpha_i(h) (1-rho) p(a|h) beta_(i+len(v))(a)] / p(y)
    - E[number of source-a emissions | y].
```

The start context is included and the backward tail includes stopping. Sum this
quantity across independent records. This is a first-order insertion potential,
not an expected count of the absent zero-probability arc. Its positive term sums
paths using one new arc; the negative term accounts for scaling all old row
emissions. It ignores the discrete model-code jump, grid projection and interactions
between multiple new arcs. Use it only to order proposals, then evaluate actual
grid models. Verify with finite differences and exhaustive tiny path sums before
relying on it; it is not implemented here.

### Dense auxiliary support followed by explicit sparsification

One alternative is to initialize positive emission probabilities over all 42
units per source letter, run bounded EM to move mass jointly, then propose sparse
channels by retaining at most three alternatives per row. Use several explicit
initializations to avoid a symmetric stationary point. Pruning and quantization
are separate discontinuous operations: preserve parse support, evaluate several
bounded projections if needed, and rescore their actual two-part code.

Such a dense model **does not fit the current coding context**: 42 alternatives
exceed both `max_alternatives=3` and a denominator32 positive grid. It can only be
an explicitly labeled floating-weight *proposal workspace*, whose score is not
eligible as a final model under that context. Alternatively enlarge the context
before a fresh comparison and recode every candidate/baseline consistently.
Never grant dense support free code bits or report its likelihood as the old
model's evidence. Auxiliary compute counts toward the search budget.

Start with ordinary unsmoothed EM to preserve the existing likelihood statement.
If later using floors, pseudo-counts, a sparse prior or annealing, specify the
changed update/objective and tune it on separate development; do not carry over
ordinary-EM monotonicity automatically. Pruning can delete necessary rare units;
retain that failure and test rare-unit transfer explicitly. Compare dense-then-
sparse against equally budgeted direct sparse search and coordinated proposals.

## Interventions that isolate the missing capability

The following factorial distinctions are more informative than only blind versus
gold. All assistance must be generated in evaluator-owned files and explicitly
labeled; none should be silently added to the primary arm.

| Intervention | What is disclosed or changed | Main discrimination |
| --- | --- | --- |
| Restricted exact-neighborhood A | Bijective singleton family, same source, no mapping | Random neighborhood coverage versus discrete local optimum/objective quality. |
| True B dictionary, mappings hidden | Unlabeled complete unit set; boundaries remain hidden | Dictionary discovery versus joint key/segmentation search. This still contains ambiguous segmentation. |
| True unit boundaries | Each record's emission cuts, source labels hidden | Converts distinct deterministic units into substitution symbols; counts-based key search can isolate mapping difficulty. |
| Gold channel, hidden boundaries | Exact rows/weights/mappings | Decoder ambiguity under a fixed channel. A is trivial; B is informative. |
| Gold channel under uniform/order0/order1 priors | Channel fixed; source varied as a labeled diagnostic | Whether B's parsing actually benefits from contextual source information. |
| True source length only | Number of source steps per record, no units or mappings | Source-length/segmentation ambiguity. This is an extra oracle, not a free use of the registered mean. |
| Gold key with fixed 2/4/8-swap perturbations | Explicitly gold-derived starting points | Local basin size and objective drift from good keys; never a blind recovery success. |
| Ciphertext-only dictionary/permutation warm starts | Independently computed starts using fit ciphertext and fixed priors | Initialization quality, with cumulative compute charged. |
| Small teacher-consistent controls | New samples from the exact declared source/stop law | Separates some model misspecification from search trouble in a correctly specified toy family; finite-data ambiguity remains. |

Known boundaries are a stronger gift than a known dictionary. They identify each
realized unit occurrence; an unlabeled dictionary alone does not. Conversely,
hidden rare units need explicit treatment even in the boundary arm. Compare
like-for-like code contexts and budget actual inference operations, not just
the number of proposals.

For source upgrades, first use source-only held-out prediction to select a small
finite-order candidate and then test gold-key neighborhood rankings on exposed
development as a diagnostic. A better average source loss need not improve all
rare discriminating contexts. Stronger order-two/three sources increase the
reachable context lattice for B; measure that cost before expansion. The current
order-zero/one limit is a deliberately weak source constraint. A sparse exact DP
may visit relatively few longer contexts when each observed offset matches few
source letters, while an aliased initializer can activate many contexts. Neither
case supplies a runtime guarantee before measurement. A small factorial of
**source order/capacity × optimizer**, with source hyperparameters selected only
on P2 and comparable inference budgets, separates source improvement from better
optimization more cleanly than a generic larger architecture. Do not merge neural
hidden vectors as if they were exact finite contexts. For A,
unique decoding permits direct higher-order or neural candidate scoring, but a
neural prior in ambiguous B cannot simply replace the finite-state source while
retaining the current exact marginal claim. Neural models are initially safer
as proposal rankers followed by the existing exact scorer, with that limitation
explicitly reported.

The registered fixed 224-character excerpts also differ from the model's geometric
length law. At rho=1/225, shortening a path from 224 to 112 source characters changes
the geometric continuation factor by only `112*log2(225/224)`, about **0.72 bits**.
The mean is useful but does not enforce a narrow length. This is not a normalization
bug: the full source/channel still defines a proper law. It is a reason to inspect
decoded lengths and include a separately labeled true-length oracle before
blaming all variable-unit failure on mapping search. A narrower proper length
model needs source-only justification, explicit assistance, and a new frozen
comparison rather than an after-result correction.

## Faster exact inference and warm starts need explicit contracts

Before larger models, cache matches of each unit at every observed offset;
source probabilities and record boundaries are fixed across proposals. Candidate
changes alter transition weights/assignments, not the observed matching relation.
For order-one sources, sum incoming source-context mass once per state/source
letter and then distribute through matching emission alternatives. Dense batched
linear algebra or compiled loops can reduce repeated Python work while retaining
the same finite sum. Benchmark real reachable lattices: more states can be cheap
on an artificially deterministic path and expensive under broad ambiguity.

Variable-length transitions combine messages from different offsets. Naively
normalizing each offset separately and adding those arrays loses their relative
scales; use verified log-space operations or explicitly tracked scale factors.
All accelerations need exhaustive small-path and original-engine comparisons,
including posterior counts, impossible records, and extreme lengths. Batching
multiple restart/candidate scores is computational parallelism, not independent
scientific replication.

A future warm-start interface should serialize a schema such as:

```json
{
  "schema_version": 1,
  "role": "blind_fit_only | dictionary_oracle | boundaries_oracle | gold_perturbed",
  "source_sha256": "...",
  "coding_context_sha256": "...",
  "fit_ciphertext_sha256": "...",
  "parent_artifact_sha256": "... or null",
  "recipe": "fixed algorithm name and version",
  "seed": 0,
  "assistance": {
    "true_units": false,
    "true_boundaries": false,
    "true_mapping": false,
    "true_lengths": false,
    "target_plaintext": false
  },
  "prior_compute_seconds": 0,
  "channel": {}
}
```

The role value is a single enum choice, and a real `channel` must pass all model,
grid, context and observed-support validations. Reject assistance-bearing starts
in a blind arm. Input hashes and explicit false flags are provenance assertions,
not proof against an unrecorded leak; the builder/solver separation still matters.
Charge previous fit time when continuing a checkpoint. Reusing an exposed
same-key result is continuation, not a fresh independent restart.

A good one-state blind model can seed a two-state search by cloning identical
self-loop emission rows and splitting initial mass across the clones. Summing
over clones leaves the ciphertext and plaintext law unchanged, though the
two-state model code costs more. This is a checkable initialization, not new
evidence for statefulness. Subsequent asymmetry proposals must earn their cost;
record both the equivalent start and the actual improvement. The present API
does not accept warm starts; adding this is a proposed future change.

## What would support an information-loss conclusion

An exact source symmetry illustrates genuine ambiguity. If a nonidentity source
permutation `sigma` preserves both `p0(a)` and every `P(b|a)`, permuting all channel
letter assignments by that symmetry leaves the ciphertext law unchanged while
changing plaintext labels. The project's uniform-source identity/swapped-channel
fixture is a special case. State relabeling is different: it can preserve both
ciphertext and decoded plaintext and should not count as a wrong key.

For small rational models, the composed finite weighted automata can be compared
by an exact equivalence procedure, as discussed in the existing design's
[Kiefer 2020 reference](https://arxiv.org/pdf/2009.01217). Source/channel arithmetic
and expansion details must actually satisfy that method's assumptions. Finite
probe agreement, close floating scores, or similar repeated decodes are weaker
observations. Near-equal rare-letter alternatives can reflect weak finite-sample
evidence without exact all-string nonidentifiability.

Finally, stronger search can find more persuasive false explanations for
nonlanguage input. Keep shuffled and stronger prospective structured negatives,
the exact ambiguity fixture, and independent transfer checks alongside every
proposed improvement. A result with lower `J` but worse literal recovery is useful
failure evidence, not permission to rename fluency as decipherment.

The most informative first follow-up is therefore: establish complete A swap
neighborhoods and objective rankings; distinguish B dictionary and boundary gifts;
then test coordinated or dense-to-sparse proposals under the same final score.
Escalate model capacity only after those comparisons say what is missing. A short
random local search is a legitimate bounded engineering attempt, not a theorem
about the recoverability of the manuscript or of the declared channel family.

## Isolated adapter implemented after this review

`src/voynich/bijective_channel_search.py` now implements the narrow sufficient-
statistics reduction above without changing the frozen general-search code or
reading pilot artifacts. Public helpers expose record statistics, normalized
mapping likelihood, swap deltas, and conversion of a glyph-to-source mapping
into an ordinary `Channel`. `search_bijective_channel` uses seeded frequency/random
starts and complete best-improving pair sweeps under explicit restart, sweep and
cooperative wall-time limits. The original engine checks the chosen channel's
full score once before return; the actual model-code length remains charged.
The search prepares the source log tables once and then uses O(K) deltas. The
convenience public delta helper prepares those tables per call, so its complete
per-call cost includes O(K²) preparation for an order-one source.

Each sweep records its parent mapping and every evaluated pair/score. A partial
sweep retains the best completed key but cannot establish local optimality. A
certificate concerns the exact parent mapping of a complete sweep with no gain
above the declared tolerance. Because a sub-tolerance neighbor can still be the
best scored key, the result explicitly distinguishes the count of certified
parents from `best_is_certified_pair_local_optimum` for the returned mapping.
All certificates are restricted to pair swaps, this source, this ciphertext,
this candidate subfamily and the numerical tolerance. Zero-probability starts
are recorded and skipped; no supported start yields no result. Preprocessing,
one swap evaluation and the final whole-corpus engine check can overrun the
cooperative deadline.

`tests/test_bijective_channel_search.py` has 21 passing checks including exhaustive
small-key and neighbor comparisons with the full engine, directed/self-pair and
record-start accounting, empty/absent glyph cases, code-cost invariance, all 253
neighbors in a 23-letter sweep, order-zero optimal assignment, trace replay,
seed reproducibility, zero-probability handling and deadline rollback. A positive
five-letter cyclic source supplies a strict pair-local maximum worse than another
permutation, explicitly demonstrating that local optimality is not global. This
is hand-specified mathematical test material, not an empirical recovery result.
The separate [adapter audit](bijective-channel-search-audit.md) records independent
checks. The adapter is available for a separately frozen follow-up only after the
original pilot has been evaluated and closed; it does not retroactively strengthen
that pilot's search budget or outcomes.

One artificial throughput check reused the explicitly specified 23-letter,
four-record/896-glyph cyclic fixture from
[the structural-search timing recipe](finite-state-structure-search.md).
For all 253 swaps of the identity-labeled mapping, public-helper scores took
0.01164 seconds versus 0.74104 seconds for separate full-engine corpus scores;
maximum log-likelihood difference was 2.73e-12. This approximately 64-fold ratio
includes repeated source-table preparation in the public helper. It is one local
timing observation on hand-specified data, not a promise about end-to-end search,
a benchmark key-recovery result, or a measurement on the pilot ciphertexts.
