# TEMPERED-READING-RECOVERY-001

Prospective exploratory fresh-key inverse comparison; not a historical decipherment test.
Prerequisites: closed finite replica qualification and original-source admission/full replay.
[Method review/derivation](../research/tempered-reading-implementation-2026-10-10.md) and
[prior full-key tempering failure](TEMPERED-RECOVERY-001-results.md) remain binding context.
No old training/chain resumed, original source/policy changed, paid API, neural fit, new
author or manuscript holdout. The existing Latin/channel hypothesis is still unverified.

Freeze exactly PATHS in `scripts/run_tempered_reading_recovery001.py`, commit/push/verify
exact origin before ONE producer and ONE auditor. Exclusive namespaces; no retry/resume,
extra seeds, adaptive stopping, temperature tuning or revised success criteria after outputs.

Generation seed96201:16uniform23-row dictionaries over42one/two-glyph units, two independently
drawn within-segment Pliny development windows per dictionary, lengths64–224 as in the frozen
EpisodeSampler. Canonicalize glyphs jointly across the two records. Forbid every raw and
canonical key listed in prior SOURCE-ACTION-TRAIN-002 inputs, its earlier forbidden sets,
and every new key already allocated here. Rejection budget16draws per key unchanged.
This is fresh relative to those frozen validation keys; no claim of disjointness from every
historical neural training episode or a fresh language/author. Development text can overlap
previous windows. Source roles/checksums/segmentation come from the original audited inputs.

Controls seed96211: for first8positives, one within-record shuffle and one IID glyph panel
at matched observed lengths, plus one deliberately nonidentifiable two-record all-zero
panel;33cases total. SourceGold/windows/dictionary metadata are generation provenance,
never parameters of the inference routine. Every case/arm/seed retained, including nulls.
Controls test behavior on nonsemantic/ambiguous inputs; no automatic semantic classifier
or Bayes factor is being proposed and a literal null reading is not called a translation.

Common initialization per observed case: frozen source_increment greedy. If it truly dies,
use the declared always-complete singleton reading (glyph g maps to row g) instead; report
fallback incidence. No refills or choice against Gold. Seeds96221/96229, same seed+i per
case in both arms for paired comparisons, not independent arm replication.

Cold arm: degree1,1024sweeps. Replica arm: degrees(1,2,4,16),256sweeps. Both use the same
fixed half-regrowth/quarter-pair/quarter-label local mixture and exactly1024local attempts
per case. Replica arm additionally attempts384adjacent exchanges per case. All132cells
therefore135,168local proposals and25,344exchanges. Match is proposal count, not exact
CPU or wall time; source lengths, power arithmetic and serialization can differ. Record
actual cost, failure/self-loop counts and acceptance/transport per arm.

Prediction in BOTH arms: highest original cold T among initial and all recorded after-sweep
cold states, exact integer cross-products, earliest tie. No warm candidate is selected unless
it has actually been exchanged into the cold slot. Store every trace/candidate/reference
path/RNG/radical fingerprint. Seal ALL selected cold outputs and input allocation before
the first known-answer endpoint metric or Gold-score/orbit diagnostic.

Primary success requires, for BOTH seeds, replica-arm competence under the unchanged
reading_endpoint_metrics thresholds (≥90%used bindings,≤10%normalized edits,≥50%exact
records,≥50%complete-used keys) AND strictly better edit total and used-binding matches
than BOTH cold-only and initial. Report cold competence separately. Do not assert these
gates as execution requirements: a completed scientifically valid run may fail every gate.
Failed endpoints receive full-length edit penalties/zero exact/bindings; unused rows never
inflate accuracy. No filtering of difficult cases, best seed, longest/hardest window, null
or failed proposal. Tables/case details remain visible for negative outcomes.

Prespecified post-seal diagnosis: score the generating reading under the unchanged cold
source/prior and compare exact target to selected states; independently known reachable
better score diagnoses a missed available point, not global optimality. Compute label-orbit/
unit-inventory ceilings against Gold; this diagnoses chosen states, not all structural search.
Gold never tunes an operator, selected reading, temperature or stopping rule.

ONE producer:

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m scripts.run_tempered_reading_recovery001 --freeze REVISION

Hard7200wall/7000absoluteCPU seconds/2GiBRSS/4GiBprivate/CPU1/$0. Planning20–75minutes
from actual admission .013563CPU/local attempt, crude135168×=.1833e4seconds≈30.6min;
longer trajectories and source/selection/serialization can change this substantially.

ONE full auditor:

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m scripts.audit_tempered_reading_recovery001 --freeze REVISION

Hard10800wall/10400CPU/2GiBRSS/CPU1/$0, checks private input≤4GiB. Planning30–100minutes
from102.366CPU/6208admission attempts≈2228.8seconds≈37.1min before extra selection/metrics.
All limits include source/generation/reader/scoring; spare budget does not extend allocation.

Audit fresh allocation/provenance and all fixed cells, full saved trace via a copied,
allocation-generalized alternate admission replay (new selectors/maps/targets/suffix ratios/
root quantiles/swaps/RNG). Original fixed auditor source stays unchanged. Independently
rescore every distinct cold reading and verify earliest exact-target selection, linear literal
state validation, separate integer edit DP/used binding/exact record/full-key metrics, seal
binding, resources and gates. Legacy quantized suffix policy/source-greedy initialization,
summary helper and post-seal oracle/orbit helper are shared; not independent expert review.
No chain extension, posterior convergence certificate, neural-circuit interpretation,
semantic null discrimination or historical decipherment claim from any synthetic result.
