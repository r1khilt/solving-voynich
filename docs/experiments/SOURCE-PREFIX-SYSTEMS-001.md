# SOURCE-PREFIX-SYSTEMS-001: cipher-only growing-key inference core

2026-10-01, engineering qualification. This implements the source-prefix
direction after the supplied-source observability oracle. It does not train a
new model, score an empirical cipher panel, validate plaintext recovery or infer
a historical channel. The original95M training campaign remains frozen/live.

## Method and prior review

PRIOR_WORK.md preserves prior Voynich substitution/anagram applications and
their flexible-source artifacts. A consistent reading under supplied Latin and
the six-glyph channel family is not an identification of Voynich's language.
[Nuhn, Schamper and Ney2013](https://aclanthology.org/P13-1154.pdf), §§3–4,
defines partial mappings, beam extensions, score estimation and pruning under
single-token substitution/homophonic assumptions. We retain its general
search-with-consistency principle, not its bijection or bounded-homophone rules.
[Meister, Vieira and Cotterell2020](https://aclanthology.org/2020.tacl-1.51.pdf),
§§3–4, discusses monotonic scoring, stopping and the effect of heuristic pruning.
Its beam equivalence theorem is not claimed for this different growing-key
queue. Existing word-equation, collapsed-key and proper-loss memos supply the
project's counting derivation; the residual-mass bound below is a project claim
checked against independent rational full-key/source enumeration.

## What the engine knows

Only unsegmented ciphertext integer records, declared source/glyph alphabets,
a fixed normalized source callback and a declared geometric stop probability
are supplied. No gold source, key, target source length, unit boundary or
used-row mask enters search. The callback receives only a hypothesized current
source prefix and resets at each record. It defines a source prior independent
of ciphertext; a ciphertext-conditioned proposal is not silently a source prior.

A state contains complete previous source records, current source prefix and
cipher offset, and one shared partial dictionary. A letter action checks an
assigned row's emitted unit, or binds a new row to the next one/two observed
glyphs. Duplicate assignments are allowed. Row values never change within a
branch. An exhausted record closes with EOS and resets source history; later
records reuse the same dictionary. Unseen rows stay unassigned and integrate
out of the iid uniform key prior.

Each letter contributes log(1−rho)+logp(letter|prefix); a newly bound row
contributes−log(g+g²), and each EOS contributes logrho. Complete leaf mass is
p(source-pair)/(g+g²)^M for its particular used dictionary. Different compatible
used dictionaries for the same reading must be summed. Visited assignments
give a lower mass unless all search is exhaustive. Global canonical naming,
when explicitly requested and validated, multiplies by g!/(g−m)!; absolute
evidence and all residual bounds use the same frame. The systems workloads use
literal glyph labels, avoiding an undeclared orbit factor.

## Incomplete search and bounds

The finite queue retains at most4096active states for large workloads, with
dual heaps and lazy-ID compaction. Each evicted subtree's probability upper
bound is accumulated; expansion/terminal caps preserve the remaining frontier.
No unsafe history/key/cipher-state merge is performed. Only identical complete
reset source prefixes share callback probabilities, with at most one cache
entry per source query/expanded node.

If L glyphs remain in a record, any completion emits n source letters with
ceil(L/2)≤n≤L. The normalized geometric law gives interval mass
`(1−rho)^ceil(L/2) * [1−(1−rho)^(L−ceil(L/2)+1)]`.
Multiply the prefix's prior mass by these interval masses over remaining
records; source probabilities and remaining key constraints can only reduce
that sum. At L=0 the factor is rho, not1. Critically, do not use a single
length's rho factor as a bound on a sum of multiple possible lengths.

Found complete mass is a lower bound. Frontier+evicted subtree bounds upper
bound unvisited mass; adding found mass bounds total evidence. A reading's
visited lower mass exceeding the strongest other visited mass plus all
unresolved mass gives sufficient separation under the declared model. The
1e-9numerical margin and scalar tests are **not interval arithmetic**, a
historical truth guarantee or a claimed rigorous floating-point certificate.
Ties remain unseparated. No early success criterion hides other outputs.

Queue priority is the subtree bound plus a fixed progress bonus times consumed
cipher glyphs. The two registered bonuses are0and4nats/glyph. This is an
exploration heuristic only: all stored source/key masses remain unchanged.
Heuristic pruning can change coverage/answers and is not covered by a generic
beam-optimality theorem. Both modes retain discarded-mass bounds.

## Fixed workloads, inputs and gates

One benchmark contains288tiny calls: all36pairs over the six binary observations
of lengths1or2, bonuses0/4, and (frontier,expansion) configurations
(10000,10000),(1,3),(3,20),(3,60). The source has two rows and a fully specified
rational first-order law; rho=1/4, six possible units/row. A separate full-key
and source-string enumerator calculates exact rational reading/evidence masses.
Check exhaustive support/scores, every incomplete evidence interval and every
reported reading separation against it. Unsupported masses serialize as null,
while unassigned key rows retain their distinct null meaning on replay.

Then eight full-source workloads: four generated fixtures ×bonuses0/4. Reuse
the unchanged selected LATIN-SOURCE-COMPACT-001 large source, alphabet23/order12,
archive SHA9769415ac460c56778a461af7646635abf5906fa726484ba84d98d4f401815f6,
its manifest-selected tau and dense adapter. No retraining/new corpus/holdout.
Fixtures in fixed order: repetitive64,repetitive224,Markov64,Markov224, with
NumPyseeds73141..73144 and iid42-way row keys. Repetitive fixtures repeat the
most probable root letter in both records; duplicates provide no extra evidence.
Markov fixtures sample source letters from reset source transitions. Their
source lengths are forced for resource measurement, not drawn from the decoder's
geometric prior and not supplied to it. They are artificial, not manuscript or
independent recovery qualification cases.

Each full-source call uses5000expanded nodes,4096active states,512terminals,
rho=1/225. Report all eight counts, completions, caps/pruning, unresolved mass,
source calls, array bytes and elapsed/CPU/RSS. A zero-completion workload is
reported, not retried or relabeled successful decoding. Literal re-encoding and
independent scalar source-state/whole-prior replay verify every returned leaf
and reading aggregation≤1e-9; raw source arrays hash identically before/after.
Engineering PASS means the fixed workload/transport/resource/numerical checks
complete, not that all workloads produce readings or recover their generation
answers. No gold-reading accuracy or semantic claim is made.

## Execution and audit

Publish source/tests/protocol and verify exactremote before one invocation.
One600stagewall/500absoluteCPU-second run,2GiBsampled host,BLAS1, no GPU/neural
inference/training or paid API; estimated1–5minutes but capped at10minutes.
One same-budget auditor replays288tiny+8full-source calls, inputRNG/source hashes,
bound/leaf/accounting arithmetic and alternate scalar scoring. It is the same
large-search algorithm and author, not independent scientific confirmation.
No retry/cap extension. Exclusive start markers and ignored full traces/fixtures;
compact results tracked. The main four-fit training completion audit is separate
and remains pending. Fresh realistic recovery/null comparisons follow only after
this systems result is published; none is silently queued by this registration.

```text
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONPATH=.:src \
.venv/bin/python -u scripts/benchmark_source_prefix_systems001.py --freeze <commit>
```
