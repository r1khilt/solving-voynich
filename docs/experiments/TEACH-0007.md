# TEACH-0007: discovering an emergent causal key site in the dense-row transformer

**Preregistration before generating the new suite or loading dense checkpoints.** TEACH-0004's dense-row controls achieved perfect fresh-family behavior without an explicit two-read state. TEACH-0005/0006 identified a causal distributed key code at the memory model's architecture-defined `first` site. This study asks whether a comparable variable can be discovered inside the dense transformer without assuming its layer or slot in advance.

## Fresh split and discovery screen

Generate 256 new cross-G groups from seed `67111`; first 128 discovery, last 128 confirmation. Load only the two TEACH-0004 `dense_row` checkpoints. Reproduce native logits by manually applying the interface, each of four frozen `TransformerEncoderLayer`s and final norm. Exact manual/native error must be below `1e-6` in each seed.

At cuts 0–4 (parsed six-slot input and after each layer), patch one donor slot at a time into the base: F rows 0/1, G rows 2/3, marker 4 or query 5. The donor state is computed once under `G0` and reused under recipient `G0,G1,G2`. Score recipient-specific donor-key answers, exact three-G groups and fixed-donor-answer avoidance. Screen all 30 cut×slot sites on discovery in both seeds.

Choose one global site without confirmation access. Find the earliest cut containing a slot whose **minimum across seeds** discovery exact-group accuracy is at least 65% and cross-G fixed-answer avoidance at least 90%. Within that cut choose the slot with highest minimum group accuracy, then highest minimum item accuracy, then lowest slot index. If no site qualifies, freeze the maximum-minimum group site with the same tie rules and mark site discovery unqualified. This screen is causal but exposed; only confirmation supports a final claim.

## Confirmation interventions and controls

At the frozen site on 128 confirmation groups:

- selected single-slot donor patch;
- norm-matched Gaussian slot state, seed `67211`;
- same slot from a same-key/different-distractor donor;
- neighboring wrong slot at the same cut, choosing the lowest adjacent slot index;
- both donor F-row embeddings at cut 0, a supplied-relation positive control;
- all six donor slots at the selected cut, an answer-injection/broad-state control;
- clean base/donor and exact manual/native recomputation.

The same-key control replaces the irrelevant name with the lowest-token alternative whose F family remains held out when one exists, otherwise the lowest available alternative; its own F partition does not enter the success denominator. This deterministic fallback avoids silently dropping confirmation groups when the finite hash split has no alternative held-out distractor.

Fit a key-centroid subspace using only discovery selected-slot states, exactly as in TEACH-0006: float64 centered SVD, rank at most 11. On confirmation patch only the projected selected-slot donor change or its orthogonal complement; compare 32 deterministic same-rank random subspaces, seed `67311`. Report rank1/2/4/8/full curves, nearest-centroid key accuracy, native-neuron leverage and 16 coordinate rotations descriptively.

## Frozen decisions

`EMERGENT-KEY-SITE-SUPPORTED` requires discovery qualification and, in both seeds: clean base/donor and cut-0 two-F-row positive controls≥95%; selected-site donor item accuracy≥80%, exact group accuracy≥65% and cross-G non-injection≥90%; selected-site donor advantage over norm-random≥40 points; same-key patch preserves base≥90%; and manual/native error<`1e-6`.

`DENSE-CAUSAL-SUBSPACE-SUPPORTED` additionally requires in both seeds: selected-slot discovery centroid rank≤11; confirmation key-span donor items≥75% and exact groups≥60%; orthogonal complement preserves base≥90% and donor≤10%; key-span donor advantage over the 95th percentile random subspace≥40 points; and confirmation nearest-centroid key accuracy≥90%. Failure does not exclude multi-slot, nonlinear or later answer representations.

The selected site is not presumed to be unique. Row parsing is still supplied, and cut-0 F-row replacement is a broad input intervention, not a discovered latent. Compact rows, discovery screen, source/checkpoint/suite hashes and controls are retained; independent audit regenerates the suite and decisions without rerunning inference. CPU cap 600 seconds. No training, paid API, download, manuscript data or final-test scoring. A pass is synthetic mechanistic evidence, not Voynich semantics.
