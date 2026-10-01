# DICTIONARY-SMC-001 — whole-key observation-conditioned stability

2026-10-01. Exploratory EXPOSED calibration. No recovery or historical qualification. New namespace; [prior calibration](SHARED-KEY-GUIDE-001-results.md) remains closed and failed. [Method/literature](../research/dictionary-smc-2026-10-01.md).

## Fixed inputs and assumptions

Reuse EXACT twelve queries from the closed SHARED-KEY-GUIDE-001 result, original source counts/arrays, four previously exposed Markov synthetic cases and original archived prefix states. All source/fixture/pruning/build/transitive paths153remain frozen. New runner PATHS explicitly add12files,165total. Admission binds parent result and complete audit hashes and reconstructs all query definitions. No original search or neural source is rerun. Known-answer correct-prefix/altered-binding diagnostics remain labeled; altered states are hypothetical, not necessarily reachable.

Target ONLY the previous conditional shared-dictionary IID surrogate: q=(original root probabilities+1e−8)/(1+23e−8), actual one-state source with all goto states0, g6/U42/rho1/225. Unknown dictionary rows independent uniform, assigned rows conditioned; same key shared across two suffixes. Supplied original contexts are intentionally unused by this explicit IID control. Intermediate prefixes allow cuts inside emission pairs, and final exact closed likelihood adds EOS. This is not the true contextual future H, a future bound or support certificate. Fixed population initialized from prior; no truth, source length, source reading, neural fit or gold row proposal is used.

## Design fixed before opening outcomes

12queries ×4originalseeds81401..81404 ×3arms=144calls. Every call128particles, balanced relative prefix stride4, multinomial resampling EVERYstage, including final. Same conditional initial bank for all arms/seeds as appropriate. Arms:

- no_move: zero mutations after resampling; control for conditioning/resampling without key repair.
- row: two symmetric single-free-row MH proposals per particle per stage.
- joint: two symmetric MH mixture proposals per particle per stage,3/4single,3/16two-row,1/16all-free rows, uniform new codes including self.

Uniform prior cancels; proposals are symmetric; weight THEN resample THEN move. Changes affect every occurrence automatically. Extinction is a recorded outcome, not repaired or retried. No mode/seed/sample/stride tuning after outcomes. Whole-key irreducibility of the joint kernel is finite formal reachability, not a practical mixing guarantee. no_move is lower cost; row/joint have equal declared proposal/table counts only if they complete identical stages, not necessarily equal CPU. Completed final keys are finite approximate posterior samples, not exact all-key posterior mass or decipherments.

Per-arm calibration gate: ALL48calls positive; every query's four-seed log-evidence spread≤2nats; every nonextinct stage's maximum normalized incremental weight≤.5; correct-prefix remaining estimate exceeds altered-binding estimate in≥12of16case/seed pairs. A positive prefix and zero altered counts preference; zero prefix never counts. ALL clauses required, unchanged after results. Incremental SMC weights are not old independent-prior contribution shares; resampled equal weights do not prove stable/accurate posterior. Report evidence estimates/extinction, stage ESS/weight maxima, distinct full keys, acceptance/actual changes, work/runtime and all failures. Distinct full keys may differ in unused mappings; they are not distinct readings or effective independent samples. Seed stability alone cannot establish unbiased accuracy.

## Resource bounds and execution

There are1593totalstages over the12query definitions; maximum218percall. At most5,709,312fixed-key evaluations across144calls (4×128×1593×7), lower on extinction. Per-call max_tables is128×stages×(1or3); conservative row-operation cap128×3×stages×23×sum(remaining glyph lengths). No cache/scoring shortcut or silent early fit stop. Bridge128MiB owned work/scratch and particle32MiB envelope, global2GiB sampled process peak RSS/128MiB ignored compressed bulk. Source loader overhead included in process peak. CPU1/BLAS1,0GPU/newtraining/paid/cloud/finalholdout. Expected3–20min per stage, conservative extrapolation from vectorized prior guide plus thousands of prefix DPs and Python proposal loops; not measured full-scale runtime. ONErun hard1800wall/1600absoluteCPU seconds; ONEfullaudit same. No retries/resume/extensions/retuning. Failure receipts preserved.

Registration commit/push/exact remote verification BEFORE first actual query call:

```
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.:src .venv/bin/python -u scripts/run_dictionary_smc001.py --freeze <registration-sha>
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.:src .venv/bin/python -u scripts/audit_dictionary_smc001.py
```

Compact per-call metrics/hash manifests/results/audit are tracked; full final banks, scores and trajectories compressed under ignored outputs/DICTIONARY-SMC-001. Single full audit regenerates ALL144stochastic trajectories/RNG/keys/stage weights/proposals/accepted moves/table counts and verifies EVERYcomputed likelihood with alternate outgoing-prefix DP, including mid-unit cuts and final EOS. Comparison≤1e−7nats with exact zero agreement. Same author/RNG-core replay plus alternate source calculation, not independent full-solver review; exact invariant-kernel/transport references provide separate finite-law evidence. Verify all165frozen files, source identities, all artifacts, inventory, version, resource/decision aggregation. A successful execution/audit is not a passed inference gate.

Preparation:22focused tests include full144call tiny transport and full outgoing audit; exact theory receipt as above. Full-tree checks required before freezing. No actual panel call yet at registration preparation. Actual Voynich UNSOLVED; long-term goal ACTIVE.
