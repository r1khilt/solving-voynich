# Local repair can be exactly trapped; changing the support is a different intervention

Own constructive proof and proposal, separate from the frozen
[fresh-key recovery comparison](../experiments/WINDOW-READING-RECOVERY-001.md).
No new corpus, neural fit or manuscript result. The comparison keeps its original
allocation and local kernels despite this independent theoretical investigation.

## An explicit counterexample at the original alphabet shape

Use 23 source rows, six observed glyphs, and the original 42-unit pool. Bind every row
to a different two-glyph unit; take any 23 of the 36 pairs. In each of two records,
emit every bound row once in the same order. Each observed record has 46 glyphs, and
every row has an occurrence in the other record.

For every valid width1/2 observed window, all 23 bindings are used outside its interior.
Therefore none is released. Every existing emission has width2, so a valid window must
be an aligned whole pair. There is exactly one row bound to that pair. A two-letter
replacement would require singleton-bound or free rows; neither exists. Thus every valid
conditional family contains only the current interpretation. Other windows bisect an
emission and are identity by construction. The whole local kernel is the identity at
this state, whatever positive language model or proposal policy is used.

There are 182 observed windows: 46 valid singleton families and 136 invalid identities.
Another complete literal reading exists: use six singleton-bound rows to emit the same
observations, with 92 source letters across the two records instead of 46. Both states
have positive target under a positive source model. A constant artificial source with
six common rows and 17 extremely rare rows makes the singleton reading strictly higher
target than the locked reading. This removes the explanation that the trapped state
must already be optimal; it makes no assertion about the original Latin model's scores.

[Two deterministic proof fixtures](../../tests/test_window_support_obstruction.py)
check all 182 windows against the separately implemented full row-tuple enumerator,
literal replacement identity, positive full targets and the strict inequality. They
also check all 10,626 ordered distinct-row global pair triplets: every rewrite is
identity. A merge needs singleton source rows; a split needs two compatible singleton
or unused rows, which are absent. These are artificial mathematical unit checks, not
a newly registered empirical recovery run or unseen data qualification.

Adding global row-label transpositions can permute labels but preserves the condition
that all rows bind distinct pairs and are used in both records. The resulting set of
all such permutations is still closed under label, window and distinct-triplet pair
updates. The old regrowth mixture has a positive root proposal and is different: this
counterexample does **not** refute its previously derived ideal full-support argument.
It also does not establish that the active fresh-key trajectories enter this closed set.

## What follows, and what does not

Exact detailed balance and target invariance do not imply convergence from an arbitrary
starting reading. A larger neural network selecting from the same conditional families
cannot create an absent edge. Likewise changing acceptance temperature cannot help at
an identity-only family: the proposal support is unchanged. A better language expert
can alter preferences, but cannot repair that missing transition either.

The general distinction is standard. [Tierney1994, section2.4 / printed page1710](https://www.ma.imperial.ac.uk/~das01/MyWeb/SCBI/Papers/Tierney94.pdf)
distinguishes invariant but nonirreducible kernels and discusses mixtures/cycles; a
mixture with an irreducible component inherits that property. This selected page was
visually read from the original scanned PDF, not inferred from its abstract. Our
23-row counterexample and its consequences above are our deduction.

This does not prove local repair is useless at every initialization, that original
source probabilities are adequate, or that root proposals mix quickly. The active
comparison still answers whether conditional local repair helps in its fresh-key
development setting. No result-dependent change to that comparison is made.

## Proposal: temper the channel constraint, while keeping the exact cold target

Related primary methods exist. [Tavares et al.2019, sections4.1–4.3](https://arxiv.org/html/1901.05437v1)
broaden a predicate's support and use replica exchange; their continuous equality
conditions can require an approximate positive minimum temperature. Our proposed finite
discrete construction has a nonempty exact hard-constraint layer. We do not import their
soft Boolean implementation or claim its empirical results transfer to decipherment.
[Gront, Kolinski and Hansmann2005](https://pmc.ncbi.nlm.nih.gov/articles/PMC1473033/)
temper constraint strength at fixed temperature in protein simulations. That is evidence
that varying constraint strength is an established sampling design, not evidence of
our cipher solver's performance.

Own proposed state z consists of:

- A partition of each observed record into width1/2 spans.
- One source-row label per span.
- A single unit binding for every visited source row; unused bindings stay marginalized.

Temporarily allow a span's observed glyphs to disagree with its row's bound unit. Let
D(z) be the sum of integer Levenshtein distances between these two strings, per span.
The source string is the span-label sequence, so its continuation/EOS law remains
explicit. Let B(z)=Q(source(z)) U^(-m(z)), with original source and visited-row prior.
Counting measure is over the stated partitions/labels/visited bindings, with no hidden
extra alignment multiplicity. Define

    T_epsilon(z) = B(z) epsilon^D(z),   0 < epsilon <= 1
    T_0(z)       = B(z) 1[D(z)=0].

D=0 requires exact literal equality, including width agreement. Each hard state therefore
corresponds to exactly one original complete reading, not several alignments. On that
layer the original cold target is unchanged. Finite observed records give a finite state
space, and every soft state has positive mass when the source coefficients are positive.

This is an auxiliary search measure, not permission to accept mismatching ciphertext as
a translation. Only exact D=0 cold readings may be returned. A nonzero epsilon alone
would change the decoding model and is not silently used as the final target.

For symmetric replica exchange between finite rational epsilons, the target ratio is

    (epsilon_i / epsilon_j)^(D(y)-D(x)).

The B factors and normalizers cancel because the base measure is identical. This is a
positive rational ratio, not an irrational heat-bath over a tempered source model.
At the hard layer, a proposed incoming D>0 state has zero acceptance. If both states
have D=0, exchange ratio is one. This cancellation assumes the same B and counting
measure; adding source temperatures would introduce their additional target factors.

Local proposals in the soft layer could relabel a span, rebind a visited row, or split/
merge adjacent observed spans, with new visited bindings sampled explicitly. They must
account for forward/reverse birth/death counts and state-dependent choices. Merely
calling these actions symmetric is not enough. The new constraint_reading.py now implements
validated states, targets and deterministic proposal components; a production stochastic
sampler, action RNG controller and original-source admission remain unimplemented.
[Prospective finite qualification](../experiments/CONSTRAINT-READING-THEORY-001.md) follows
the artificial preparation; its result is not assumed here.

An ideal connectivity proof is available for those positive-support actions: relabel all
spans to row0, deleting bindings as their last use disappears; rebind row0 to a fixed
singleton; split every pair span into singleton spans. Every intermediate has finite D
and positive soft mass. All states can reach this common state, and each move can be
reversed with positive probability if new bindings have full unit support. This argument
requires the specified action support; it does not establish finite-budget hitting time,
efficient return to D=0, or correctness of an unimplemented transition controller.

The hard-layer obstacle becomes a measurable soft-to-hard return problem. A relaxed
replica that almost never reconstructs D=0 can consume compute without helping. Sampling
the first apparently valid warm endpoint after adaptive stopping is also not justified
as an exact posterior draw. Fixed-time stationary conditioning and finite hitting-time
selection are different laws.

## Connection to world models, diffusion and interpretation

A repair world model could propose joint revisions across occurrences while predicting
future constraint residuals. Discrete denoising could operate on the extended partition/
label/binding state, with an explicit exact verifier and actual proposal probabilities.
These are proposed architectures, not fitted diffusion or physical world models.

[CREPE's ICLR2026 proceedings abstract](https://proceedings.iclr.cc/paper_files/paper/2026/hash/a17a13f0dce74fcd1a8bd78e537e582b-Abstract-Conference.html)
describes diffusion inference control through replica exchange. Only its abstract was
reviewed here; the linked paper fetch failed. It is a relevant broader connection,
not a verified discrete-cipher algorithm or a basis for adopting its theoretical claims.

Mechanistic analysis should test whether a learned policy represents consequences of
changing repeated bindings, including recipient sites in other records. It should not
mistake supplied dictionary embeddings or an external hard mask for learned reasoning.
A policy restricted to singleton families has no behavioral freedom to reveal through
activation patching: successful decoding needs action support before neuronal analysis
can explain how it uses that support. The exact graph obstruction is not neural mechinterp.

Next admission would first implement the soft state and independently enumerate a tiny
finite joint measure, hard-state bijection, all local reverse probabilities, exchange
flux, component connectivity and zero-constraint return controls. Only after qualification
and measured original-source costs should a fresh sealed recovery experiment or learned
policy be registered. Preserve current failed runs, frozen source and actual budgets.
