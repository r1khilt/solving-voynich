# Episodic rule recovery implementation

Implemented and completed in an isolated worktree on branch `codex/episodic-rule-recovery`, preserving concurrent work in the original checkout. All scientific phases used frozen source `c9c95a0cfef9bdfc542136e4d38482e7d37c59ea`. The public aliases EPISODIC-0012/0013 distinguish these studies from the separate latent-recovery series; historical configuration/output identifiers remain unchanged.

- [Data and canonicalization](EPISODIC_DATA.md): six process families, fresh parameter tasks, exact oracles and observational-equivalence controls.
- [Models](EPISODIC_MODELS.md): existing transformer reuse, fast GRU, inspectable signed product recurrence, canonical probabilities and explicit edge-HMM fitting.
- [Causal battery](EPISODIC_CAUSAL.md): exact joint futures, multiple alternative subspaces, finite temporal interventions, independent update fits and failure witnesses.
- [EPISODIC-0012 registration](../experiments/EPISODIC-0012.md) and [results](../experiments/EPISODIC-0012-results.md): 21 neural runs with fixed versus fresh tasks and equal maximum training budgets, plus bounded explicit probabilistic models. Development-selected endpoints received different exposure.
- [EPISODIC-0013 registration](../experiments/EPISODIC-0013.md) and [results](../experiments/EPISODIC-0013-results.md): new-key causal tests for six recurrent backbones, two ranks and independently initialized controls. Neither primary causal claim was established.

Execution: `scripts/run_episodic_campaign.py` reuses the tested process-group supervisor. Scientific outputs/checkpoints stay under ignored `outputs/EXP-0012/`; completed compact archives are under `results/episodic-20260921/`. A host interruption required replaying two preserved partial runs without changing source/settings; provenance and monitoring limits are retained in that archive. No manuscript inputs or paid API were used. Synthetic success is not historical decipherment.

The initial model sizes span399,882–10,625,664parameters. Hardware timing and explicit caps are in EXP-0012; memory usage is measured separately from compute throughput. Added compute buys153,600fresh tasks, a larger capacity comparison, independent seeds and stronger causal controls, rather than a target RAM occupancy.
