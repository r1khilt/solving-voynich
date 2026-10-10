# WINDOW-READING-RECOVERY-001

Prospective fresh-key recovery comparison, not manuscript decipherment. Question: does
exact source-weighted conditional replacement recover hidden source text better than
uniform replacement with exact cold MH, when both can revise the same local intervals?
Prerequisites: [finite laws](WINDOW-READING-THEORY-001-results.md) and
[original-source cost/full audit](WINDOW-READING-ADMIT-001-results.md) PASS; the prior
[fresh replica recovery failure](TEMPERED-READING-RECOVERY-001-results.md) stays preserved.

Method rationale follows the primary Ravi/Knight2011 type-versus-point/boundary sampling
review in [occurrence repair](../research/occurrence-repair-2026-10-10.md), the
[conditional-family derivation](../research/window-repair-2026-10-10.md), and linked prior
Voynich/contextual key-Gibbs reviews. Their models and supervision differ from ours;
neither historical applicability nor efficient global mixing is assumed. Small intervals
can change source equality, local bindings and segmentation; distant coordinated changes
can remain inaccessible in the allocated budget. No new model architecture is fitted.

Freeze every PATHS entry in scripts/run_window_reading_recovery001.py, commit/push and
verify exact origin before ONE producer and ONE full auditor. Exclusive output files;
no retry, resume, replacement seed/case, cap extension, extra steps or tuning after results.

Original source unchanged: 6,497,939 Latin training letters / 12 authors, order12,
23 source rows / 42 one/two-glyph units, six observed glyphs, stop1/225, grid32. Use already
exposed Pliny development windows, two records of 64–224 source letters per key. Generate
16 fresh positive keys with seed96401, excluding original manifest forbidden raw/canonical
keys, previous64 validation keys, the previous fresh16 keys, and new duplicates. This is
fresh-key development, not new-author/language/overlap isolation or disjointness from every
past training key. Checksum-bound ignored allocation records all provenance and rejections.
First8 positives receive paired shuffled and IID controls plus one declared ambiguity null
using seed96411: 33 cases total. Controls remain in all traces; no semantic classification.

Both arms use the same cipher-only source-greedy initialization and declared singleton
fallback on death. Seeds96421/96429, each cell seed=seed_base+case. Both cold arms take
256 attempts, 132 cells / 33,792 attempts. Step i uses a new PCG64 stream initialized by
SeedSequence((cell_seed,i)); select uniformly among ALL observed width1/2 windows.
Thus window ranks are paired across arms despite different replacements; invalid endpoints
are identity without refill. This deterministic stream schedule is finite algorithm design,
not a guarantee about a PRNG's ideal Markov transition law.

At a valid window, BOTH arms enumerate/evaluate/save every compatible local family target
with qualified exact ratios. Conditional arm draws the complete normalized source-target
family; uniform arm chooses uniformly then exact MH with ratioTnew/Told, including an
ordinary identity proposal and its acceptance draw. Changed retained readings get full
source scoring and forced reference replay. No best family member or rejected candidate
can enter final selection. Both arms remain cold; no regrowth, label transport, replica
exchange, warm conditional draw or learned proposal is added.

The uniform arm's full-family evaluation is deliberately unnecessary for its draw; it
controls available evaluations for the distribution comparison. This is not its natural
optimized runtime baseline. Match attempts and paired observed windows, NOT exact CPU,
candidate count or state-dependent work; report all three actual costs. No equal-compute
advantage follows. Complete per-step candidates, rational ratios, selected member, exact
acceptance/categorical blocks, PCG before/after and retained paths stay in ignored archives.

Select the earliest exact target maximum among initial and actually retained cold states,
using integer comparisons. Seal ALL 132 selected outputs before first known-answer metric.
Unchanged competence thresholds: ≥90% used bindings, ≤10% source edits, ≥50% exact records
and ≥50% complete used keys. Success additionally requires conditional edits strictly lower
and used matches strictly higher than BOTH uniform and initialization at BOTH seeds. Report
all gates including failures, not only average gains. Null metrics are not language labels.
Post-seal Gold-v-selected target and label-orbit/inventory diagnoses are predetermined;
they cannot guide the search or prove global optimality/identifiability.

Producer hard7200 wall / 7000 absolute CPU seconds, 2GiB RSS / 4GiB ignored output / CPU1 / $0.
Audit hard10800 wall / 10400 absolute CPU seconds / 2GiB RSS / CPU1 / $0. Measured prior
0.019829 CPU seconds/valid table suggests ~11 minutes raw producer window work and roughly
3.3× for full-target audit; evolving states/family sizes differ, so planning10–60 /30–150
minutes, hard caps rather than promised duration. No pilot with new keys is run.

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m scripts.run_window_reading_recovery001 --freeze REVISION
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m scripts.audit_window_reading_recovery001 --freeze REVISION

Complete auditor separately enumerates row tuples/reconstructs all bindings, fully scores
every candidate, replays categorical lattice and MH integer quantile decisions, every PCG
state and retained reference path, allocation exclusion, all cold-state exact selection,
integer-DP edits/used-key metrics, costs/caps/gates and frozen hashes. Qualified source and
reference backend, initialization, allocation and Gold-orbit helpers are shared. Same
researcher, no independent expert or neural mechanism claim.

Artificial preparation covers full-family trace replay/ratio and selection corruption,
fresh-key exclusion, complete temporary controller/seal/full-audit and exclusive outputs.
Only artificial sources/records enter those fixtures. Any preparation error is recorded;
it does not authorize repeating a frozen scientific producer or auditor.
