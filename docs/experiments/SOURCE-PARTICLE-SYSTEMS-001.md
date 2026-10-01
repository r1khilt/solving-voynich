# Shared-key particles: systems qualification and synthetic diagnostics

2026-10-01. Exploratory CPU experiment after the source-prefix systems failure.
This implements the probability law derived in
`docs/research/source-key-particles-2026-10-01.md`. It is not a Voynich reading,
fresh historical recovery qualification or a change to the live95M campaign.

## Prior work and hypothesis

The linked memo reviews Doucet/Johansen's SMC weighting/resampling tutorial,
Loula et al.'s controlled-LM conditioning and Hauer et al.'s cipher MCTS,
with direct primary links and differences from our duplicate-allowing
one/two-glyph family. PRIOR_WORK.md's Voynich/anagram and flexible-source
limitations remain. Correct conditioning under an assumed Latin source does
not identify the manuscript's source or channel.

Hypothesis: interleaving shared-key constraints and maintaining a population
may avoid the zero-completion behavior observed in the earlier small prefix
budget. This is an exploratory comparison, not a promised improvement. No
post-observation retuning of source, particle counts, schedule or diagnostics.

## What the solver receives

Only literal unsegmented ciphertext records, declared alphabet sizes, the
unchanged source probability/goto arrays, geometric rho=1/225, population size,
seed and schedule. No gold source, key, source length, boundary or used-row mask.
All source records reset at state0. Row priors are iid uniform across42units,
including duplicate row assignments. Unused row priors integrate out.

Every state has partial source-context states, ciphertext offsets, closed flags
and one shared dictionary. A deterministic schedule chooses either first-open
record or least fraction of observed ciphertext consumed (integer cross-products,
index ties). It does not branch on orderings or multiply-count a leaf.

First-binding action weight=(1−rho)p(row|context)/42; matching assigned-row
weight=(1−rho)p; EOS=rho; already-completed absorption=1. Each letter consumes
one/two observed glyphs. Fixed horizon=sumcipherglyphs+recordcount covers every
surviving path. Never re-charge EOS, drop early completions or refill extinction.

Each tick computes local total G for every parent, multiplies mean G into an
evidence estimate, multinomial-resamples parents proportional to G, and samples
their normalized actions. All RNG draws and ancestor/action arrays can replay.
The exact-arithmetic normalizer identity does not make a finite posterior or
log normalizer unbiased, prove coverage, or certify floating-point bounds.

Ancestry uses4-byte parent/2-byte record/2-byte source-action entries. −1 action
means EOS,−2 absorption. Group only by an actual terminal ancestor before
reconstructing; context/key similarity is not enough to merge source histories.
Scalar literal emission and full source/stop/first-binding score replay check
every distinct terminal ancestor. Equal final ancestor groups retain equal
keys/scores. Complete source/key duplicates merge particle counts, never count
a literal prior mass twice. Final particle frequencies are approximations;
visited unique-leaf prior mass is a separate finite-bank lower mass.

## Fixed workload and inputs

Use the selected frozen LATIN-SOURCE-COMPACT-001 large model, manifest-selected
tau/order12/23letters, counts archive19,566,187bytes SHA256
9769415ac460c56778a461af7646635abf5906fa726484ba84d98d4f401815f6.
No refit, new corpus, reserved author or old cipher panel is opened.

Four new artificial fixtures, in order: repetitive64,repetitive224,
Markov64,Markov224, NumPyseeds74401..74404, iid42-way dictionaries. Draw key
first, then two reset source strings. Repetitive strings are identical copies
of the most probable root letter; they provide no independent source evidence.
Markov strings are independently sampled from the trained source. Their lengths
are forced for resource measurement and **not** passed to the solver. They do
not represent the decoder's unconditional geometric-length population.

Exactly32calls:4fixtures ×schedules(sequential,balanced) ×populations(512,4096)
×inference seeds(74411,74413), in that product order. Source cases and seeds are
paired across schedules/populations; they are not32independent data cases.
Maximum theoretical particle actions42,614,784 under the one/two-unit length
envelope. No learned guidance/twist, rejuvenation, posterior decoding adjustment
or secondary architecture is trained or enabled.

Write an immutable population and prediction bank before scoring its generation
answer. Decision rule: most frequent source-record pair by particle multiplicity,
lexicographic ties. For its key diagnostic, most frequent leaf key within that
reading, lexicographic ties. Report exact records, edit errors/true letter count,
matches over gold-used rows, absent completed decisions, finite evidence
estimates, terminal ancestors/initial ancestors/distinct keys/readings and
time/memory/work. Extinction uses empty predictions and zero used-row matches,
remaining in every diagnostic denominator. No full-key accuracy claim for
unobserved rows. No confidence interval or null-acceptance claim from this grid.
Edit distance encodes each source-row identity as one distinct Unicode codepoint;
it preserves unit-cost source-symbol edits, not glyph-length or byte edits.

Engineering PASS requires all32calls/transport and literal/scalar/ancestry
checks, with extinctions retained. It does not require gold accuracy. Report
synthetic diagnostic failures prominently. Realistic held-out Latin, nulls,
restart qualification and manuscript unitization remain separate requirements.

## Tests, resources and single execution

Before launch, independent full-key/source enumeration on all36tinycipherpairs
checks the production action law under both schedules, including local
normalization/importance correction and failures. Tests also cover incompatible
shared records, exact one-reading evidence, early absorption, source resets,
zero-mass categorical boundaries, allocation caps, replay/no source mutation
and the actual32-cell CLI/audit transport with small artificial inputs.

Publish code/tests/protocol and verify exactremote before one invocation.
One600stagewall/500absoluteCPU-second/2GiBsampled host run; same-budget one
auditor afterward. BLAS1, zero GPU/neural inference/training/paid API/newholdout.
Estimated1–5minutes per stage, strict10minute wall cap; no retry or extension.
128MiB ancestry allocation per call and256MiB conservative observation/particle
work-array envelope (excluding the caller-owned source);1GiB sampled total
ignored bulk output guard.
These are planning limits, not measured performance. Keep large populations
and traces ignored; compact32summaries/input/result/audit manifests tracked.

The auditor hashes all stored populations/predictions, regenerates fixture RNG,
replays all32populations exactly, checks every array/dtype/shape/trace, repeats
scalar/literal/ancestry checks and diagnostic arithmetic. Same author/algorithm,
not independent scientific confirmation. Keep all22trainingfreeze paths and
the previous prefix registration unchanged. No second GPU workload is launched.

```text
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONPATH=.:src \
.venv/bin/python -u scripts/run_source_particle_systems001.py --freeze <commit>
```
