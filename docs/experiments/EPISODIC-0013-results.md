# EPISODIC-0013 results — Joint-future interventions and temporal consistency

Completed 2026-09-21. **Neither recurrent family establishes the registered primary claim.** Learned GRU interventions transfer part of a donor's future predictions and beat the registered controls, but change the recipient's immediate answer too much and operate on an insufficiently accurate teacher. The signed recurrence fails additional selectivity, temporal-consistency and untrained-control requirements.

This is the episodic study originally registered as EXP-0013, distinct from the null-removal/CTC experiment with that number. Frozen JSON retains the original identifier under `results/episodic-20260921/`. See the [registration](EPISODIC-0013.md), [primary gate summary](../../results/episodic-20260921/EXP-0013/summary.json), and [complete audit inventory](../../results/episodic-20260921/EXP-0013/campaign_summary.json). Scientific source `c9c95a0cfef9bdfc542136e4d38482e7d37c59ea` remained unchanged through every phase; interruption/recovery and full resource accounting are reported in [EPISODIC-0012 results](EPISODIC-0012-results.md) and the [resource and restart record](../../results/episodic-20260921/provenance/resource_and_restart_summary.json).

## Frozen question and measurements

Could a small fixed subspace transplant a recurrent model's three-symbol future distribution, preserve its immediate output when donor and recipient agree there, and continue updating coherently after another symbol? The target was the frozen model's donor distribution. No hidden-state labels, oracle distributions or manuscript data were used to fit the explanatory basis.

Both recurrent architectures, all three seeds, were audited at rank 4 (primary) and rank 16 (secondary). Each of the 12 audits included an independently initialized same-architecture control; its initialization seed was shared across the two ranks for a given backbone seed. Fresh pair-parity pools contained 8 fit, 4 development and 8 confirmation keys, each with 24 prefixes of length 64. Parameter seeds, exact task descriptions and sampler provenance are in the [task manifest](../../results/episodic-20260921/EXP-0013/task_manifest.json). Keys and exact prefixes were disjoint across pools.

Donors came from the same key. Strict matches shared the final symbol and had teacher immediate-distribution TV≤0.05; among them, the fixed pairing rule chose the largest later-future discrepancy. A learned orthogonal basis and shuffled-target counterpart each received 120 optimization steps, with development checkpoint selection. PCA, output-Jacobian and three displacement-matched random controls were retained. Future-Fisher and delayed-Fisher directions were secondary diagnostics fitted only on the fit pool.

The main gap is KL(donor future || patched future) minus its first-symbol KL. Lower means better transplantation of conditional future predictions. “Commuting” compares patch-then-update with update-then-patch, averaging symmetric KL over possible next symbols using donor probabilities. It tests whether the intervention continues to behave coherently, rather than only changing one answer.

## Primary results

All table values use equal-key means on the immediate-matched confirmation subset. The primary claim required **all three seeds** of a family to qualify, pass the numerical checks, have sufficient matched evidence, reduce the future gap by ≥20% and ≥0.005 bits, keep immediate TV≤0.02 and commuting KL≤0.02, beat every registered control, and establish trained-model specificity.

| Model / seed | Teacher oracle H3 KL | Future gap: unchanged → learned | Reduction | Immediate TV | Commuting KL | Matched pairs / keys |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| GRU 121 | 0.542294 | 0.068764 → 0.026435 | 61.56% | 0.045231 | 0.008641 | 65 / 8 |
| GRU 122 | 0.535754 | 0.053959 → 0.023872 | 55.76% | 0.042599 | 0.008591 | 65 / 8 |
| GRU 123 | 0.535560 | 0.053297 → 0.017559 | 67.05% | 0.043331 | 0.005755 | 61 / 8 |
| Signed 121 | 0.685590 | 0.140864 → 0.155886 | −10.66% | 0.048736 | 0.028520 | 64 / 8 |
| Signed 122 | 0.684536 | 0.152835 → 0.142382 | 6.84% | 0.041337 | 0.028797 | 45 / 8 |
| Signed 123 | 0.700004 | 0.176928 → 0.141721 | 19.90% | 0.053536 | 0.024177 | 47 / 7 |

**Teacher qualification fails for all six backbones:** oracle H3 KL exceeds the registered 0.30-bit ceiling. This measurement compares each teacher against the known process after the same prefix; it includes uncertainty about the task's parameters, not only failure to represent its state. It is an average qualification test, not a guarantee for individual donor pairs.

GRU rank 4 passes gap reduction and commuting requirements and beats all six registered controls after averaging seeds within keys and bootstrapping eight common keys. Learned-minus-output-Jacobian gap is −0.030490 bits, 95% interval [−0.042337, −0.020904]; learned-minus-shuffled is −0.022494 [−0.028384, −0.017225]. Nevertheless, every seed violates immediate preservation. Its untrained baseline gaps are only 0.006682–0.008147 bits, below the 0.01 qualifying floor. Specificity is therefore **inconclusive**, not positive evidence for learning.

Signed rank 4 misses the 20% reduction requirement in every seed, including seed 123's 19.90%. It also fails immediate preservation and commuting. It beats the three random controls but does not establish superiority to shuffled targets, PCA or the output-Jacobian span. The shuffled comparison has mean +0.011581 bits and interval [−0.000444, 0.027802]. These control comparisons use seven keys present in all seeds. Qualifying untrained models reduce their own future gaps by 92.47–95.48%, exceeding the trained reductions; trained specificity fails. Untrained teachers select their own matched pairs, so raw gap magnitudes are not directly comparable.

## Secondary findings: preserving one answer is not enough

These methods/ranks cannot rescue the primary claim. Their most informative pattern is the separation between immediate preservation, future transplantation and temporal consistency.

| Secondary intervention | Future-gap reduction across seeds | Immediate TV | Commuting KL |
| --- | ---: | ---: | ---: |
| GRU delayed-Fisher, rank 4 | 53.09–63.87% | 0.000357–0.000633 | 0.014436–0.023823 |
| Signed delayed-Fisher, rank 4 | 53.56–67.60% | 0.001930–0.003078 | 0.041555–0.081786 |
| GRU learned, rank 16 | 84.84–90.00% | 0.040195–0.041224 | 0.005790–0.007543 |
| Signed future-Fisher, rank 16 | 89.56–92.53% | 0.038577–0.045035 | 0.009316–0.011540 |

Delayed rank-4 directions preserve the immediate answer much better while moving substantial future behavior. However, all signed seeds fail temporal consistency, and one GRU seed exceeds the commuting ceiling. Larger/Fisher subspaces can transfer more behavior while still changing the immediate answer too much. Every underlying teacher remains unqualified. These are useful distinctions between intervention properties, not recovered latent variables.

Coordinate-update accuracy alone is also insufficient. Signed learned rank 16 achieves mean coordinate MSE only 0.001064 times the fit-mean baseline, yet reconstructing the state with a fixed complementary component incurs 0.187610 bits of current joint KL and 0.205141 bits after its affine update. A coordinate system can be predictable while omitting important behavior. Conversely, a valid nonlinear or rotating representation could fail this fixed-subspace/affine test.

## Validation, limitations and next state

Independent read-only recomputation checked every method's per-key counts and means in all 24 trained/untrained reports and reproduced all 12 primary control-bootstrap intervals. Full donor reproduction has exactly zero joint KL. Every report records unchanged and restored weights/buffers; identity-state and identity-prediction errors are exactly zero. Maximum normalization error is 2.384×10⁻⁷ and basis orthogonality error 3.576×10⁻⁷. Signed state-encoding replay differs by at most 4.992×10⁻⁷ across batch sizes; it is not exactly zero. These checks support functioning intervention plumbing, not the scientific claim. Causal analysis took 183.01 seconds within the completed analysis phase.

The output-Jacobian control samples the first 16 fit contexts, all from one key under this pool ordering. Delayed-Fisher excludes only that sampled rank-matched span, not every direction capable of changing immediate outputs. Confirmation pairs are deliberately selected for teacher disagreement; they are not representative random context pairs or oracle-defined counterfactuals. Known task grouping is supplied externally, hybrid states may lie off the natural state manifold, and immediate-output preservation is not an independent semantic nuisance test. Only eight confirmation keys and three shared-data model seeds limit inference.

The next method should first obtain qualified predictive teachers on fresh pair-parity tasks, then test an intervention objective that jointly addresses immediate preservation and temporal consistency. Reusing these observed secondary results as confirmatory evidence would require new final keys and frozen rules. All pools here are exposed. No manuscript glyph assignments, historical mechanism or translation follows from this study.
