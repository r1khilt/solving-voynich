# SOURCE-GUIDE-001: future-guided shared-key inference

Registered exploratory method comparison, 2026-10-01, BEFORE its one invocation.
Motivation/primary-source reading and complete probability derivation:
`docs/research/source-guidance-2026-10-01.md`. Previous DIAG-A is exposed prior
evidence; no old population or success threshold is reinterpreted.

## Inputs and fixed assumptions

Reuse the unchanged selected Latin order-12/23-row source and count archive from
LATIN-SOURCE-COMPACT-001/large.json. Its manifest binds 19,566,187 bytes, SHA256
9769415ac460c56778a461af7646635abf5906fa726484ba84d98d4f401815f6.
Source probability/transition arrays are hashed before/after and never mutated.
Raw arrays and populations remain ignored; compact manifests/results tracked.

Exactly FOUR new fixtures: source lengths (64,64,224,224), seeds75501..75504.
Draw the entire 23-row key first, iid over all42 single/double units from six
glyphs; duplicates allowed. Then draw two independent reset Markov passages of
the fixed length from the existing source. Emit literal ciphertext using the
shared key. No manually chosen Latin sentence or ciphertext-selected example.
These are new artificial cases, not reserved authors/final pages. Forced source
lengths are a design limitation, not a decoder input; geometric rho=1/225 stays
the target. Corpus/source-family/latent assumptions remain restricted.

## Fixed comparison

16 ordered calls: fixture, particle seed (75511,75513), guidance (none,iid).
512 particles each, deterministic balanced record scheduling; fixed horizon
sum(cipher lengths)+2. Both arms use identical new log/Gumbel-max transport.
The iid guide reads the source root row, smoothed1e-8, and all remaining C;
unbound units are fresh each occurrence, all segmentation paths summed.
Total log guide floor -2000, complete h=1. Corrections and initial h0 retain
the original target in exact arithmetic. No guide-strength selection or fitting.
Source contexts, once-only key priors, EOS and literal leaf scoring unchanged.
Extinction retained; never refill/restart or retry failed cells.

Archive complete population and prediction bank before gold diagnostics.
Reading choice: most final-particle frequency, lexical tie; key modal within
chosen reading, lexical tie. Scalar/literal replay of every terminal ancestor
checks the original contextual source/length/key score within1e-9.
Report all cells: exact records, edit errors, gold-used-row matches, genealogy,
best visited leaf, gold-used-leaf/reading membership, known gold leaf versus
best leaf and Zhat, and runtime/table counts. The known leaf lower-bounds
evidence; it is NOT proven globally optimal. No complete reading-marginal
count or posterior coverage certification is performed in this comparison.

Primary exploratory signal is SUPPORTED ONLY IF all16 populations complete,
guided total edit errors are at least10% lower than matched local total AND
guided best visited literal leaf is better in at least6/8 paired cases/seeds.
Otherwise NOT_SUPPORTED. This is a prioritization signal, not recovery/null
qualification. Also report exacts and all failures regardless of outcome.
Four artificial keys and paired seeds are not16 independent corpora; no
asymptotic significance claims or extrapolation to Voynich.

## Budget, freeze and audit

One BLAS1 CPU invocation,1800wall/1600absoluteCPU seconds,2GiB sampled host,
512MiB sampled bulk,128MiB ancestry and256MiB conservative candidate-work
envelope per call. Guide:1024 cache entries plus<=64 build tables within64MiB,
max200,000 total table builds per call. Expected5–15minutes per full stage,
uncalibrated engineering estimate; HARD30minutes. No extension/performance
stop/retuning. Zero GPU, neural model calls/training, paid spend/new download/
new historical holdout. Actual statistical source predictions occur.

Publish source/tests/protocol and verify exact remote freeze BEFORE:

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONPATH=.:src \
    .venv/bin/python -u scripts/run_source_guide001.py --freeze COMMIT

Then exactly one auditor, same environment/budget:

    .venv/bin/python -u scripts/audit_source_guide001.py

Auditor replays all16 source/fixture/RNG/arrays/ancestry/literal/scalar/diagnostic
and comparison/hash/accounting outcomes. Same author/algorithm for full-size
replay; independent alternate algorithm only in tiny law checks. It is not
independent historical validation. Existing JOINT-KEY-TRAIN-001 four-fit GPU
campaign and frozen paths continue unchanged. No extra GPU model/resume.
Python3.12.13/NumPy2.5.3 environment; actual resource/versions saved.
