# After fragment failure: retain compatible source/key states

2026-10-01. Post-outcome reasoning and next design, **not implemented or run**.
SOURCE-FRAGMENT-001 retains its original negative criteria. This note proposes
a finite-state route for a future separately frozen solver, not an architecture
claim or a learned mechanism.

## What the evidence changes

Literal twelve-letter retrieval excluded the true fragment at every selected
offset. Across the full exposed texts, library coverage12=4/1064,6=859/1112,
4=1122/1128. Simply increasing retrieval breadth or neuron count does not repair
that particular missing support. Short fragments have coverage but impose
fewer rows; the coupling study already showed locally harmful correct changes.
Joining consistent short hypotheses without an immediate score gain, or
using the source's full smoothed support, remains a meaningful next question.
The three greedy rounds did not test it: every round kept the same incumbent.

These are four generated Markov keys, not historical text or independent
replications. No claim that every valid source fragment must be literal Latin
corpus text. The neural training campaign uses actual Latin corpus windows;
this analysis does not explain its weak inverse validation by distribution
substitution. Source quality, neural competence and inference coverage remain
separate questions.

## Own source-weight decomposition

At a history h, let N_d be its retained suffix count at depth d, zero when
absent or unavailable. With tau64, alpha_d=N_d/(N_d+64). The actual source
recursion is q_d=alpha_d times its empirical conditional distribution plus
(1−alpha_d)q_(d−1). Therefore the **mixture coefficient** on depth≥k is
`1 − product_(d=k..12)(1−alpha_d)`. This does not require reading the next
target letter or calling the scorer. It is not the fraction of a target's
probability supplied by those depths: each component distribution differs.

The exposed1152 histories yield mean weights≥4=.8747958026,
≥8=.0586365302,≥12=.0003857404. Counts/history support and these coefficients
are in the reproducible AFTER-outcome record, not registered success metrics.
Deep counts can be small and heavily smoothed. A source called order12 can
generate novel fragments through its lower-order components. It retains
long-context predictive information, but a literal12gram library is not its
support and is a poor substitute for its transition law in this setting.

## A stronger exact state abstraction before heuristic pruning

The generic `source_prefix_inverse.py` accepts a full-history provider and
cannot safely merge histories using only their suffix. In our fixed statistical
source, the validated dense automaton **is** a sufficient source state.
Exploit that specific assumption in a new engine instead of silently changing
the generic API or the frozen experiments.

For two records, a state can contain both observed glyph offsets, both source
automaton states, and all23 partial key rows (each −1 or a unit index).
Choose the next unfinished record deterministically from its observed glyph
progress, with a fixed tie rule. For every source row, a previously bound unit
must match literally; an unbound row branches over the two observed lengths,
with once-per-row factor1/42. Each emitted letter has source-transition factor
and `(1−rho)`. Add rho when its record ends; non-erasing units forbid further
letters at a finished record. Reset its irrelevant future context to a common
sentinel. No true source length or supplied boundary enters the state.

Different source histories arriving at **identical** offsets, contexts and
partial keys have identical future possibilities and weights. Their joint
prefix masses may be added by log-sum-exp. Maintain a separate maximum-path
backpointer if a representative reading is needed; a maximum is not a sum.
Do not merge merely because two keys have similar embeddings, row counts,
canonical names, literal support or a shared suffix of unspecified length.
No similar exact merging claim applies to a neural language model unless its
entire sufficient future state is genuinely identical.

Process states in topological **total consumed-glyph** order, because every
edge increases it by1or2. Arrival at the same offsets can have different
numbers of source-letter actions. Pruning or expanding a state before all its
incoming contributions arrive would invalidate a purported exact merge.
For a frozen glyph-based scheduler every completed reading/key has one action
ordering, avoiding duplicate schedule mass. Tests must compare schedule
independence and full-key/string enumeration on exhaustive tiny alphabets.

With all states retained, summed terminal masses yield the joint evidence:
each source history is weighted once and unused key rows are integrated out.
Terminal partial-key cylinders can overlap as sets of full keys while their
source-history events are disjoint. Thus a terminal state's mass is **not** a
full-key posterior or a marginal plaintext MAP score. Complete candidate keys
by a declared rule, then use the unchanged full-key native likelihood/reader.
Account for duplication rather than calling a visit count a Bayesian prior.

## Where ambition requires controls

Compile compact states and explicitly bounded frontiers before a larger beam.
The fragment engine scanned1.964billion nodes in≈38seconds, but its stack was
small: that does not predict hash-table frontier memory or state-merging speed.
Measure throughput/RSS on exact tiny and moderate cases before registering
100k–1M-frontier proposals or extrapolating hours of search. Width/resource caps
must report lost mass/states and cannot certify completeness or MAP. Preserve
the full positive source support in expansion; a finite beam can still lose
truth. Distinguish evidence lower sums from normalized finite-bank bias.

Compare merging with equivalent unmerged work, preserved hypothesis diversity,
future guidance and complete-readout recovery. Keep tiny positive/null and
independent enumeration controls. A guide or learned action policy may later
rank states, but must not add a false sampler law or discard inconvenient
counterfactuals. Existing once-binding, geometric-stop and future-guidance
derivations remain relevant. No new GPU training before the original campaign's
completion audit; no additional experiment is authorized as an open-ended loop.
This is a source/key inference program that can eventually supply trustworthy
mechanistic tasks, not a license to infer meaning from arbitrary latent geometry.
