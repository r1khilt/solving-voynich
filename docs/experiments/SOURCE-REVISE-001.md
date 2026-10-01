# SOURCE-REVISE-001: whole-key revision from particle candidates

Exploratory development registration, 2026-10-01, BEFORE one invocation.
Question: does revising early shared-key assignments improve the same exact
reading decision on exposed synthetic candidates? Rationale, primary-source
review and probability law: `docs/research/source-key-revision-2026-10-01.md`.
No fresh recovery qualification, manuscript result or posterior claim.

## Inputs and isolation

Reuse the FOUR exposed SOURCE-GUIDE-001 Markov fixtures, source lengths
64,64,224,224, generation seeds75501..75504; two independent source-reset
records and a shared23-row iid42 key per fixture, duplicates allowed.
Parent population seeds75511/75513, constant/iid guidance yield exactly16
ordered warm-start banks. Parent result/audit/16compact cells are frozen;
all parent fixture and prediction archives are byte/hash bound.

Unchanged trained Latin order-12/23-row source from
results/LATIN-SOURCE-COMPACT-001/large.json: count archive19,566,187bytes,
SHA2569769415ac460c56778a461af7646635abf5906fa726484ba84d98d4f401815f6.
Immutable dense probability/transition arrays hashed before/after.
Native build/admission from NATIVE-SUFFIX-001/GLOBAL-SEARCH-SYSTEMS-001;
no rebuild or scorer tuning. The frozen cpp/library manifests bind compiler,
flags, ABI and exact bytes. No old cipher panels, reserved authors, final
Voynich holdout, new corpus/download or neural checkpoint accessed.

Fit receives only ciphertext, completed warm key, fixed source/native scorer,
registered config and resource observer. Gold source/key/length/boundaries
stay outside fit arguments and are used only AFTER immutable fit/prediction
serialization. This is a process isolation guard, not an external blinded
researcher or new unseen evaluation.

## Fixed search and predictions

Select highest original literal leaf; lexical partial-key/source tie. Retain
bound rows, fill unseen rows with NumPy default_rng seed
75701+101*case+(particle_seed-75511),23 iid pool draws. Both paired guidance
arms use the same filling vector but may bind different rows.

Native score: -23log42 + sum_records log(sum_ALL_source_paths
rho*(1-rho)^length*p_source(path)); rho=1/225. No MDL key-length penalty.
Whole-key search seed75811+101*case+(particle_seed-75511), same across paired
guidance arms. Eight fixed temperatures1,2,4,8,16,32,64,128; symmetric
replace/swap/block weights2:1:1, block sizes2/3/6, exchange every4iterations.
One warm start scored once and replicated eight times. Retain all unique
scores and all proposal/exchange events. Maximum2048iterations,
8192unique keys and120search wall seconds PERcell; stop at first bound.
Finite caps are valid finite search outcomes, not exhaustive enumeration,
stationarity or global optimality. No restarts/resume/extension/retuning.

Every native record has exact graph limits2,000,000nodes/8,000,000edges.
Warm/final/supplied-gold fixed-key Viterbi uses record_kbest(count=1),
500,000nodes/2,000,000edges/50,000prefix expansions; no approximate answer
on cap. Literal forward emission, independent scalar source score and
k-best/native marginal agree within1e-7. Errors raise and stop the stage.
Both before/after predictions use the same best-key/fixed-key-Viterbi rule;
no direct comparison to the parent's different modal reading decision.

## Metrics and fixed exploratory signal

Report all16cells: warm/revised/generating-key-reader edit errors and exact
records; matched gold-used rows; best score gain; gold full-key score minus
selected; full/used gold mapping bank membership; all scores/events/cache
statistics/stop reasons/resources. Scores are not textual or semantic truth.
Supplied generating key is a post-fit diagnostic, never a search seed.

Primary exploratory signal SUPPORTED only if all16cells finish, revised
aggregate edits are at least25% below warm aggregate edits, and at least12
cells have strictly fewer edits. Otherwise NOT_SUPPORTED; failure/cap of
native/Viterbi stage remains an execution failure, never a fake zero score.
Four keys repeated across seeds/guidance are not16 independent datasets.
No significance or fresh null/positive qualification is claimed.

## Bounded resources and verification

ONE run and ONE completion audit, each hard2400wall/2200absolute CPU seconds,
4GiB sampled peak host RSS. Bulk run output≤512MiB; ignored archives retain
all scored keys/events and immutable predictions. BLAS/environment threads1,
one CPU process, zero GPU/new training/paid API use; cost$0. Existing source
model prediction occurs. Sixteen120second searches allow1920seconds plus
480seconds source loading/decoding/serialization; count bound≤131072keys,
≤262144record search scores plus diagnostic/decoder scores. Prior artificial
native benchmark2034keys/10.059search seconds suggests minutes, but exposed
longer-key lattice cost is unknown; estimated5–20min/stage, hard40min.
The audit replays each actual retained key, not the theoretical maximum.

Auditor independently reconstructs Python-random mixture draws, MH/exchange
uniforms/acceptance, event order/first-seen indices, counts and final replicas.
It replays EVERY stored key with native exact scoring and checks warm/final
full-record scalar DP traversal, predictions, source/parent/archive hashes,
post-fit gold diagnostics and aggregate signal clauses. This is same-author
auditing with alternate scalar traversal, not an independent global search or
agent review. Tests cover actual16cell transport on a tiny mock source.

Before launch: relevant tests/lint/diff, original training/guide freezes
unchanged, notebook/memory, coherent commit+push+exact remote verification.

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
    VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONPATH=.:src
    .venv/bin/python -u scripts/run_source_revise001.py --freeze <registration>

Same environment, once after terminal successful run:

    .venv/bin/python -u scripts/audit_source_revise001.py

Log terminal status, failures and actual work/cost. Preserve existing original
training campaign and its single all-four-fits completion audit. No retry of
a failed empirical stage under this registration.
