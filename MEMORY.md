# Project memory

Last updated: 2026-09-21.

## Current state

- Phase: **active research and implementation**. The user explicitly authorized prior-work/methods research, corpus acquisition, architecture design, and code; the earlier preparation-only restriction is superseded.
- Latest checkpoint: **CAMPAIGN-0001 completed**, all three tracks and 25 new models; no job remains running. Read `docs/CURRENT_STATUS.md` and `docs/experiments/CAMPAIGN-0001-results.md` before new work. Historical hypotheses remain unresolved.
- Repository: `/Users/rikhil/coding/solving-voynich`.
- Configured remote: `https://github.com/r1khilt/solving-voynich`; branch at setup: `main`.
- The user describes the repository as private; remote visibility has not been independently verified.
- Initial local checkout had no commits or project files. The project now has a documented corpus pipeline and interpretable-model implementation.
- Setup checkpoint `e07c276` was successfully pushed to `origin/main`; see notebook entry NB-0002 for validation and environment limitations.
- Targeted prior-work and architecture reviews are complete, with primary-source links and limitations. This is not an exhaustive literature survey or a replication of prior results.
- Official ZL3b transcription acquired and prepared: 226 modeling pages, frozen section-aware physical-group split with 177/24/25 pages, train-only 112-entry vocabulary. Raw/derived text stays Git-ignored; provenance/manifests tracked. See DATA.md.
- Dense reference and controlled variants implemented with native causal hooks, training/resume, baselines and explicit test-evaluation gating. Current validation: 295 tests passed plus 23 subtests; lint and whitespace checks passed; earlier live MPS identity/restoration and cached-forward controls passed.
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
- Explain progress in plain English before reporting technical metrics. The user explicitly says the terminology and progress are hard to understand. Separate real-manuscript findings from synthetic accuracies; define vocabulary/tokens, baselines and bits when relevant. See `docs/PROGRESS_EXPLAINED.md`.

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
- Reproduction commands: `docs/RUNBOOK.md`; completed experiment registrations/results: `docs/experiments/EXP-0001*` through `EXP-0010*`; latest visual overview: `results/CAMPAIGN-0001/overview.png`.

## Latest experiments and next state

- **EXP-0002 completed:** 18 Voynich runs/34,300 updates on MPS; compact 430,720-parameter model mean validation 1.839734 bits, MTP 1.837866. Registered near-tie rule selects compact. All 18 trained on clean `f79e51b`. Larger-model minima usually around 800–1,200 updates under this schedule. Results: `docs/experiments/EXP-0002-results.md`.
- **EXP-0003 completed:** same 192 validation targets/10 leaf groups across three seeds. Truncating to 16 previous units costs mean 0.154093 bits; shuffling distant 112 while preserving last 16 costs only 0.001685 on average, with inconsistent seed effects. **This order hint is superseded by EXP-0005's larger sample below.** Seed-42 L1H3 has largest zero-ablation damage, but no function assigned. Identity/full residual controls exact. Results: `docs/experiments/EXP-0003-results.md`.
- **EXP-0004 completed:** six independent synthetic models/6,000 updates. Known four-state, two-homophone, 35% ambiguous-filler generator plus IID independent-label control. Supervised readouts fitted after text-only LM training recover structured hidden state 86.25% (Bayesian reference 86.61%) and roles 81.58% (reference 82.73%). IID state 25.39%/chance25%, roles equal majority guessing. **Probe-direction steering failed**: donor KL 1.437 unchanged vs1.431 state patch vs1.431 norm-matched random; full residual0.047. Do not conflate readable information with a discovered causal circuit or unsupervised decoding. Clean source `a84e46d`. Results: `docs/experiments/EXP-0004-results.md`.
- All 24 new training runs used local MPS, 40,300 updates, about 11.93 minutes summed measured training time; analysis/startup additional. No paid research APIs. Manuscript final test remains unscored; synthetic test is now exposed and must not be reused as a fresh holdout for adaptive causal-method tuning.
- Subsequent context and causal controls are completed below. Review methods before each new attempt, keep finite budgets, preserve negative results, commit/push coherent checkpoints. Historical filler/homophony/state hypotheses remain unresolved.

## Latest continuation completed

User requests further theories/experiments and authorizes local RAM/GPU. No paid API or cloud work used; all three bounded experiments completed and no training job remains running.

- **EXP-0005**, clean `933bade`: 768 validation targets; distant shuffling consistently hurts by mean 0.037202 bits, correcting the earlier 192-target hint. Random cross-vs-same illustration-category gap 0.172691 falls to 0.128584 after approximate histogram matching, but histogram imbalance remains. Fixed histogram/category n-gram controls help and still trail the transformer. Results: `docs/experiments/EXP-0005-results.md`.
- **EXP-0006**, clean `2a36727`: fresh 7,168-context synthetic pools; 14 fits / 5,600 alignment updates in 42.17 seconds, peak process RSS 1.32 GB. Frozen backbone verified. Late learned steering achieves 97.07% desired next-category agreement and 0.034799 oracle KL bits, but simple output-weight span achieves 0.036336. Early true supervision fails against shuffled supervision. Strong supervised output control does not recover a unique algorithm. No key/family transfer or label-free inference. Results: `docs/experiments/EXP-0006-results.md`.
- **EXP-0007**, clean `389a12c`: same 768 manuscript validation targets; matched character-minus-group loss difference only 0.005090 bits, all three descriptive leaf-bootstrap intervals include zero. Both matched conditions preserve separators/local suffix and change 43.14% of remote positions. Block-scale effects are not monotonic. Results: `docs/experiments/EXP-0007-results.md`.
- New synthetic evaluation pools are now exposed; adaptive successors need fresh pools. Manuscript final test remains unscored. No Voynich filler assignments, coding rules or translation established.
- Candidates proposed after EXP-0007 were subsequently partly tested in EXP-0008/0009/0010 below. Further line/recency isolation, explicit transitions and broader family/key transfer remain open; see `docs/research/BACKLOG.md`. Hypotheses are not findings or scheduled background work.
- Explanation/compute planning, 2026-09-21 UTC: live read-only hardware check confirms Apple M5 Pro, 18 CPU cores, 20 GPU cores, 64 GB memory, MPS available and Metal recommended working set 51.8400 GiB (not free memory). User wants useful experiments that exercise the machine. Recommended first larger direction is blind synthetic mechanism recovery, preceded by a scaling benchmark and bounded by a proposed 8-hour campaign. This is a proposed time budget, not a runtime estimate or launched job. Detailed guide: `docs/PROGRESS_EXPLAINED.md`.


## Latest completed parallel campaign

- User authorized all three tracks in parallel with delegated agents. All scientific runs used clean, pushed `72f6328`, with unchanged source through completion. Three MPS jobs launched together under finite time/memory caps; all completed in **54.62 minutes elapsed**, with no resource-stop or numerical-control failure. Maximum sampled summed process RSS was 1.5703 GiB, not total GPU/physical memory. No paid research API or manuscript final-test scoring.
- **EXP-0008:** four synthetic models /16,000updates,160 blind partitions,320 frozen files. Larger models improve familiar-key prediction and some blind groupings, but no unfamiliar cycle/branch key passes the stronger criterion. Every IID/RRXOR partition selectsK1. Copy control mainly identifies the next copy source; later decisions fail. Output-probability clustering generally matches or beats residual clustering. All32 held-out/training key pairs are observably distinct. Results: `docs/experiments/EXP-0008-results.md`.
- **EXP-0009:** seven frozen-model causal maps. All three trained synthetic models pass future horizons2/4; untrained and all three manuscript models fail. Successful changes transplant broad earlier-component outputs, not a state-only variable or transition rule. Numerical identity/restoration controls exactly0. Results: `docs/experiments/EXP-0009-results.md`.
- **EXP-0010:**21models /25,200updates /88,088,063 sampled manuscript targets. Identical target exposure and initial weights verified per matched seed/capacity. None of four longer-context comparisons passes. Main2048 loses0.034861bits on average vsmain256, although its own extra context helps. Boundary merge costs0.035553bits overall; glyph-only evidence varies. Full2048subset has only4pages/2leaves. Results: `docs/experiments/EXP-0010-results.md`.
- All adaptive synthetic final pools are now exposed. New recovery attempts need fresh keys/families/contexts and frozen rules. Candidate next directions: key-invariant inference, explicit probabilistic transitions, joint-future prediction and more selective causal changes. Source reviews in `docs/research/PREDICTIVE_RULES.md` and `BELIEF_NET_REVIEW.md` were added after registration and did not change this campaign. No subsequent run is scheduled.
- Compact results and scientific overview are in `results/CAMPAIGN-0001/` and `results/EXP-0008/0009/0010`; raw data/checkpoints remain ignored under their recorded local paths. Final checks:295tests plus23subtests; archival hash/exposure/control audits pass. Notebook NB-0020 records the completed checkpoint.
