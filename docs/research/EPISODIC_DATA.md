# Episodic synthetic data contract

Implementation for EXP-0012, schema version 1. This is a small, controlled process-inference benchmark. It contains no manuscript text, plaintext, translations, semantic anchors, or claim of historical applicability. The experiment registration determines scientific seeds, splits, counts, gates, and resource limits; the seeds in unit tests are engineering checks only.

## What is inherited and what is adapted

The immediate design inputs are [NEXT_DESIGN.md](deep-review-2026-09-21/NEXT_DESIGN.md) and [THEORY.md](deep-review-2026-09-21/THEORY.md), read together with the current project memory, notebook, and research protocol before implementation. Their source audit records reading depth and limitations. Relevant source-based reasons are:

- Edge-emitting forward filtering and joint probabilities implement the equations in THEORY, not a new inference theorem. Computational-mechanics and predictive-state work motivate evaluating whole futures and consistent updates; predictive states need not equal an author's internal states. Reviewed primary sources: [Shalizi and Crutchfield](https://arxiv.org/pdf/cond-mat/9907176v2), [Shalizi and Shalizi](https://arxiv.org/pdf/cs/0406011v1), and [Littman, Sutton and Singh](https://proceedings.neurips.cc/paper/2001/file/1e4d36177d71bbb3558e43af9577d70e-Paper.pdf).
- Independently sampled generator parameters are a limited sequence-task adaptation of prior-data fitted task training, following the rationale in THEORY T12. The [PFN paper](https://arxiv.org/pdf/2112.10510v7) does not establish that this four-symbol simulator prior transfers to the Voynich manuscript. No frontier-scale result is extrapolated into an expected success here.
- Prefix-only first-occurrence canonicalization and a NEW event implement the already proposed representation in NEXT_DESIGN. Removing arbitrary symbol names is different from learning homophony, omissions, segmentation, or meanings.
- The repository's completed EXP-0008 used a small fixed collection of keys and found failures on unfamiliar keys. EXP-0011's Finnish latent recovery also failed. Fresh parameters, whole-family holdouts, and explicit observational-equivalence controls address specific limitations; they are not evidence that those limitations are resolved.

These implementation choices and toy family definitions are local adaptations for falsifiable tests. They are not claimed as original research or replications of the cited papers. The implementation block used the repository's reviewed sources; it did not independently reread or verify every original paper.

## Task representation and families

`make_task(family, seed, alphabet=4)` returns a JSON-serializable dictionary containing `schema_version`, `family`, `seed`, `alphabet`, `edge`, `prior`, `params`, and `task_id`. Exactly four symbols are currently supported. `edge[a,i,j] = P(emission=a,next state=j | current state=i)`. Every source-state row sums to one across symbols and next states. Priors are stationary, including in the periodic families, so a fixed episode boundary does not reveal the latent phase.

Every task samples a uniformly random symbol permutation plus continuous probabilities. New seeds therefore alter process parameters as well as arbitrary symbol names. SHA-256 `task_id` hashes a canonical JSON representation of schema, family, alphabet, edge, prior, and parameters. The seed is retained for reproduction; task identity depends on the generated parameters. Scientific split construction must allocate disjoint underlying tasks before sampling streams. Multiple samples from one task are correlated at the task level and do not become independent tasks by receiving different stream seeds.

| Family | Intended allocation | Hidden representation | Continuous nuisance parameters |
| --- | --- | --- | --- |
| `cycle` | Training / fresh-parameter within-family evaluation | Four noisy-symbol states, moving to the next state or dwelling | Four progress probabilities, emission noise, background probabilities |
| `branch` | Training / within-family evaluation | Four states: 0 forks to 1 or 2; either joins 3; 3 returns to 0; each can dwell | Branch probability, four dwell probabilities, emission noise, background probabilities |
| `pair_parity` | Training / within-family evaluation | Six states: ready parity mode 0/1, or first bit remembered under either mode | Tag bias, noisy parity probability, parity-mode switching probability |
| `iid` | Mandatory null | One state; unequal symbol probabilities | Dirichlet symbol probabilities |
| `rr_xor` | Whole-family holdout | Seven representational states: ready, first bit remembered, or both first bits remembered; third signal is their noisy XOR | Tag bias and parity noise |
| `switching` | Whole-family holdout | Two slowly switching hidden regimes with overlapping symbol distributions | Two switch probabilities, two tag biases, signal noise |

Before the random renaming, parity and switching symbols encode a binary signal and an independent binary tag as `2*signal + tag`. The tag is a nuisance variable that ensures four-symbol support while allowing a clear binary dependency. Its task-level bias is observable from enough data and must be preserved by any claimed selective causal intervention.

Pair-parity state indices are `ready(c)=c` and `remember(c,x)=2+2*c+x`. From either ready state the first signal bit is fair; the tag law is also the same. Thus ready states 0 and 1 have *exactly the same immediate distribution*. Their two-symbol joint distributions differ because the second signal is `x XOR c`, with task-specific noise. After that second emission the mode can switch. This supplies a genuine joint-future counterexample to one-step-only state tests.

The seven-state RRXOR representation deliberately stores both bits even though only their XOR matters for the third emission. It is consequently nonminimal: second-bit-memory states with the same XOR have the same observable future law. Hidden labels and representation state counts are diagnostic metadata, never unique-recovery targets. A separate exact `equivalent_state_split` null is available for every family.

Variable copy-lag is deferred from this first edge-process interface. A variable-history copy process needs a separate explicit history/oracle contract or an exponentially larger finite-state representation. It must not be silently called a small edge model.

## Sampling and oracles

- `sample_task(task,n,length,seed)` returns independent streams of shape `[n,length]`, `int64`. It uses one shared transition CDF and vectorized draws over streams.
- `sample_tasks(tasks,length,seed)` returns one stream per supplied task, shape `[len(tasks),length]`. It pads different state counts and vectorizes draws over tasks at each time step. Padding states are unreachable from a valid prior.
- `oracle_belief(task,prefix)` returns a row belief over the state **after** consuming the complete prefix. Filtering normalizes after each observed symbol. An impossible prefix raises an error instead of inventing a posterior.
- `oracle_next(task,prefix)` predicts the next symbol.
- `oracle_joint(task,prefix,horizon)` enumerates normalized probabilities in lexicographic continuation order. For example, at horizon two the order is `00,01,02,03,10,...,33`. Horizon zero is `[1]`; allocations above one million continuations are refused. Scientific use is expected at horizons at most four.
- `rename_task(task,permutation)` maps each raw symbol `a` to `permutation[a]` without changing hidden transitions. It composes the stored emission permutation and recomputes the hash.
- `equivalent_state_split(task,state=0,weight=0.5)` duplicates a state. All incoming and initial mass splits by `weight`; the copies have the same aggregate outgoing laws. Every finite observable continuation distribution is unchanged. Therefore correctly discovering two different physical hidden machines is impossible from these observations alone.

Task dictionaries contain oracle truth and must stay separate from learner inputs. The batch sampler uses this truth to generate text; it does not append family IDs, seeds, parameters, state labels, state counts, or posterior beliefs to the text.

## Prefix-only canonicalization

`canonicalize(tokens,alphabet)` takes integer `[B,T]` raw IDs. It returns:

| Output | Shape | Meaning at position t |
| --- | --- | --- |
| `canonical` | `[B,T]` | First-occurrence rank of the current symbol |
| `counts` | `[B,T]` | Number of distinct symbols seen **through** the current symbol |
| `inverse` | `[B,T,A]` | Rank-to-raw-ID mapping after the current symbol; unused ranks are `-1` |

For raw `2,2,0,2,1`, canonical ranks are `0,0,1,0,2`; counts are `1,1,2,2,3`. The inverse map at the second position is `[2,-1,-1,-1]`, even though the complete sequence later contains 0 and 1. Each row starts with an empty mapping. No full-sequence alphabet statistics are used.

`canonical_targets(tokens,alphabet)` returns `[B,T-1]`: an already seen next symbol uses its assigned rank; an unseen next symbol is the event `NEW=A`. Full-sequence canonical ranks are permitted for target construction only because the comparison to the count at the prediction position converts all newly appearing symbols to NEW. The input snapshot remains independent of that target.

`canonical_raw_probs(logits,counts,inverse)` is a NumPy reference for the neural output conversion. The output head has `A+1` events: ranks `0..A-1` and NEW. Unassigned rank events are masked before softmax, as is NEW once all symbols have appeared. Valid rank mass returns to its corresponding raw ID. NEW mass is divided uniformly over unseen raw IDs. An empty prefix therefore assigns equal mass to all symbols. This is a known-fixed-alphabet convention, not a claim of identifying an exact previously unseen symbol.

## Engineering validation

The dedicated tests independently enumerate hidden paths and visible futures; compare joint marginals and sequential conditionals; check empirical sampled frequencies against exact oracles; test stationary priors, deterministic regeneration, changing nuisance parameters, JSON serialization, and valid probability rows; exhaustively test the equivalent-state null on all prefixes through length two; and check all 24 alphabet permutations.

Canonical tests alter every unseen suffix, compare truncation to full-sequence prefix snapshots, reconstruct raw IDs from inverse maps, and check NEW masking for zero, partial, and complete observed alphabets. Invalid probabilities, impossible prefixes, malformed inverse maps, and excessive joint enumeration fail explicitly. These checks validate the implementation. They do not measure neural transfer or establish a decipherment.
