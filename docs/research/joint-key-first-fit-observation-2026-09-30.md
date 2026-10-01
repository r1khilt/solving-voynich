# First completed joint-key fit: density learning is weak key recovery

2026-09-30, partial campaign observation. The first5,423,146-parameter control
finished its fixed20,000 updates,80,000 generated dictionaries and23,037,420
original source letters in1,813.793 stage-wall seconds,376.406 stage-CPU seconds.
This is one of four scheduled fits; the95M first-seed fit is running. The single
full-campaign replay auditor has not yet run. No early stop, retry, extra model
inference, holdout use or alteration of the frozen training criteria.

The selector chooses step20000 with Pliny development NLL3.40778293 nats per
row versus3.90086854 initially. Free-running used-row accuracy is49/1248=3.9263%,
versus26/1248 initially and73/1248 at step4000. Complete greedy keys:0/64 at
every saved checkpoint. The engineering runner's PASS denotes completion and
numerical/resource checks; it does not qualify key recovery. The complete
two-seed capacity comparison remains pending.

An additional **retrospective arithmetic diagnostic**, using only the stored
final64 true-key log probabilities, makes the sampling issue concrete:

| Quantity under the saved canonical neural proposal law | Value |
| --- | ---: |
| Mean probability of the literal true complete dictionary | 5.6480e-34 |
| Largest such probability among64 cases | 5.7889e-33 |
| Geometric mean | 9.1291e-35 |
| Uniform42-choice/23-row complete dictionary probability | 4.6266e-38 |

For N independent draws for each case, linearity of expectation gives
`E[exact literal truth hits] = N * sum_i q(K_true_i | C_i)`.
Thus100,000 hypothetical draws **per case** (6.4million total) have expected
literal-truth hits3.6147e-27 under the saved law. No draws were performed.
The union bound also limits the probability of at least one such literal hit
by that expectation. CPUdouble completion replay is pending; this diagnostic
uses stored MPS scores and does not replace that numerical audit.

This calculation concerns **one full named dictionary** per case. An unused
source-letter row can vary harmlessly; alternative dictionaries can decode the
same records. We have not summed that equivalence-class mass. The calculation
does **not** bound successful reading, native-search support, unseen-glyph orbit
union proposals or a future repaired dictionary. It does show why improved
proper loss alone is insufficient reason to expect exact full-key samples.
Greedy outputs also need not be the most probable complete autoregressive key.

For the next stage, preserve the frozen four-fit comparison and full audit.
Then test whole proposals plus unchanged exact fitting and compare every case
against existing search, with fresh qualification still required. Before
latent-neuron interpretation, separately test whether ciphertext order supplies
causal information beyond symbol counts and canonicalization priors, and
whether that information improves actual reading. A cipher-blind density
baseline and order-preserving versus order-destroying controls are candidate
follow-ups, not registered or executed experiments. No mechanism or historical
language has been established by this observation.

[Archival receipt](../../results/JOINT-KEY-TRAIN-001/launch-observation-003.json)
binds the closed first-arm files and then-closed first large-arm snapshots;
[arithmetic diagnostic](../../results/JOINT-KEY-TRAIN-001/small-72203-probability-diagnostic.json)
binds its exact input scores. Neither is the final campaign auditor.
