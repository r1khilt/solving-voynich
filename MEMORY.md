# Project memory

Last updated: 2026-09-20 (America/Los_Angeles).

## Current state

- Phase: **active research and implementation**. The user explicitly authorized prior-work/methods research, corpus acquisition, architecture design, and code; the earlier preparation-only restriction is superseded.
- Repository: `/Users/rikhil/coding/solving-voynich`.
- Configured remote: `https://github.com/r1khilt/solving-voynich`; branch at setup: `main`.
- The user describes the repository as private; remote visibility has not been independently verified.
- Initial local checkout had no commits or project files. The project now has a documented corpus pipeline and interpretable-model implementation.
- Setup checkpoint `e07c276` was successfully pushed to `origin/main`; see notebook entry NB-0002 for validation and environment limitations.
- Targeted prior-work and architecture reviews are complete, with primary-source links and limitations. This is not an exhaustive literature survey or a replication of prior results.
- Official ZL3b transcription acquired and prepared: 226 modeling pages, frozen section-aware physical-group split with 177/24/25 pages, train-only 112-entry vocabulary. Raw/derived text stays Git-ignored; provenance/manifests tracked. See DATA.md.
- Dense reference and controlled variants implemented with native causal hooks, training/resume, baselines and explicit test-evaluation gating. Current validation: 237 tests passed plus 23 subtests; lint and documentation checks passed; live MPS identity/restoration controls passed.
- EXP-0001 completed on clean source revision `4fc9019`: 1,814,208-parameter reference trained 200 CPU steps, validation 1.960671 bits/token vs five-gram 2.087802 on identical targets. Single seed, no architecture ranking or decipherment claim. Test split unscored. Results and local-checkpoint digests tracked in `results/EXP-0001/`; checkpoints themselves stay in ignored `outputs/`.

## Durable intent and preferences

- Pursue actual decipherment by identifying the underlying encoding/generative process.
- Take ambitious explanations seriously, but demand falsification, independent checks, and out-of-sample performance.
- The charter is flexible. Abandon or revise its proposals if evidence favors another account.
- Investigate structured null/filler material and non-one-to-one or state-dependent mappings without assuming either exists.
- Consider synthetic ground-truth tasks, small interpretable models, and causal analysis. Keep frontier models in the researcher role where useful.
- The user is particularly enthusiastic about mechanistic interpretability and argues that routine use would disfavor cumbersome codebook lookup. Treat reader learnability as a conditional modeling preference, not established evidence of a specific encoding; execution was authorized after that discussion (NB-0004).
- Do not assume spaces delimit plaintext words; consider multiple representational levels.
- Keep an updated Markdown notebook. Commit and push progress regularly; routine Git checkpoints have standing user authorization.
- Ask for concrete resources or assistance when needed. Do not repeatedly ask for permission already granted.

## Resources reported by the user

Reported on 2026-09-20; not verified account balances:

- Approximately $1,000 in GPT-6 Astra credits available.
- Approximately $20,000 in GPT-6 API credits expected in about one week (roughly 2026-09-27); availability must be checked before reliance.
- User is willing to help with downloads and additional resources.
- No API credentials or paid-run spending schedule established; local compute and EXP-0001 limits are recorded below.

Local implementation inventory: arm64 Mac, 64 GiB memory, 18 logical CPUs; user identifies it as M5 Pro. MPS is available outside the sandbox and tensor execution verified. A bounded reference-size synthetic benchmark measured 0.01947 s/step on MPS versus 0.06590 on CPU; use approved outside-sandbox runs for GPU access. Python 3.12.13 / PyTorch 2.14.0 / NumPy 2.5.3 installed in ignored `.venv`, dependency resolution tracked in `uv.lock`. No paid research API used.

The user's reference to an AI model solving Navier–Stokes is motivation, not verified evidence in this project. No conclusion about that claim has been drawn and no independent verification has been performed.

## Reading map

- Original intent: `docs/RESEARCH_CHARTER.md`.
- Complete work record: `NOTEBOOK.md`.
- Evidence/source index: `docs/knowledge/INDEX.md`.
- Hypothesis IDs and tests: `docs/research/HYPOTHESES.md`.
- Research standards: `docs/research/PROTOCOL.md`.
- Deferred candidates: `docs/research/BACKLOG.md`.
- Literature and architecture: `docs/research/PRIOR_WORK.md`, `docs/research/ARCHITECTURE_REVIEW.md`, `docs/research/ARCHITECTURE.md`.
- Reproduction commands: `docs/RUNBOOK.md`; completed experiment registrations/results: `docs/experiments/EXP-0001*` through `EXP-0004*`; visual overview: `results/research-round-2026-09-20/overview.png`.

## Latest experiments and next state

- **EXP-0002 completed:** 18 Voynich runs/34,300 updates on MPS; compact 430,720-parameter model mean validation 1.839734 bits, MTP 1.837866. Registered near-tie rule selects compact. All 18 trained on clean `f79e51b`. Larger-model minima usually around 800–1,200 updates under this schedule. Results: `docs/experiments/EXP-0002-results.md`.
- **EXP-0003 completed:** same 192 validation targets/10 leaf groups across three seeds. Truncating to 16 previous units costs mean 0.154093 bits; shuffling distant 112 while preserving last 16 costs only 0.001685 on average, with inconsistent seed effects. Could reflect distant symbol mixture/style; not proven. Seed-42 L1H3 has largest zero-ablation damage, but no function assigned. Identity/full residual controls exact. Results: `docs/experiments/EXP-0003-results.md`.
- **EXP-0004 completed:** six independent synthetic models/6,000 updates. Known four-state, two-homophone, 35% ambiguous-filler generator plus IID independent-label control. Supervised readouts fitted after text-only LM training recover structured hidden state 86.25% (Bayesian reference 86.61%) and roles 81.58% (reference 82.73%). IID state 25.39%/chance25%, roles equal majority guessing. **Probe-direction steering failed**: donor KL 1.437 unchanged vs1.431 state patch vs1.431 norm-matched random; full residual0.047. Do not conflate readable information with a discovered causal circuit or unsupervised decoding. Clean source `a84e46d`. Results: `docs/experiments/EXP-0004-results.md`.
- All 24 new training runs used local MPS, 40,300 updates, about 11.93 minutes summed measured training time; analysis/startup additional. No paid research APIs. Manuscript final test remains unscored; synthetic test is now exposed and must not be reused as a fresh holdout for adaptive causal-method tuning.
- Next useful branches are registered matched distant-content/order controls and harder/fresh synthetic causal-state and label-free recovery benchmarks. See `docs/research/BACKLOG.md`. Review methods before each attempt, keep finite budgets, preserve negative results, commit/push coherent checkpoints. Historical filler/homophony/state hypotheses remain unresolved.
