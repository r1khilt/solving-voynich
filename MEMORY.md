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
- Dense reference and controlled variants implemented with native causal hooks, training/resume, baselines and explicit test-evaluation gating. Pre-pilot validation: 222 tests passed plus 23 subtests, lint and documentation checks passed.
- EXP-0001 completed on clean source revision `4fc9019`: 1,814,208-parameter reference trained 200 CPU steps, validation 1.960671 bits/token vs five-gram 2.087802 on identical targets. Single seed, no architecture ranking or decipherment claim. Test split unscored. Results and local-checkpoint digests tracked in `results/EXP-0001/`; checkpoints themselves stay in ignored `outputs/`.

## Durable intent and preferences

- Pursue actual decipherment by identifying the underlying encoding/generative process.
- Take ambitious explanations seriously, but demand falsification, independent checks, and out-of-sample performance.
- The charter is flexible. Abandon or revise its proposals if evidence favors another account.
- Investigate structured null/filler material and non-one-to-one or state-dependent mappings without assuming either exists.
- Consider synthetic ground-truth tasks, small interpretable models, and causal analysis. Keep frontier models in the researcher role where useful.
- The user is particularly enthusiastic about mechanistic interpretability and argues that routine use would disfavor cumbersome codebook lookup. Treat reader learnability as a conditional modeling preference, not established evidence of a specific encoding or authorization to begin experiments (NB-0004).
- Do not assume spaces delimit plaintext words; consider multiple representational levels.
- Keep an updated Markdown notebook. Commit and push progress regularly; routine Git checkpoints have standing user authorization.
- Ask for concrete resources or assistance when needed. Do not repeatedly ask for permission already granted.

## Resources reported by the user

Reported on 2026-09-20; not verified account balances:

- Approximately $1,000 in GPT-6 Astra credits available.
- Approximately $20,000 in GPT-6 API credits expected in about one week (roughly 2026-09-27); availability must be checked before reliance.
- User is willing to help with downloads and additional resources.
- No API credentials or paid-run spending schedule established; local compute and EXP-0001 limits are recorded below.

Local implementation inventory: arm64 Mac, 64 GiB memory, 18 logical CPUs; sandboxed PyTorch reports no available accelerator. Python 3.12.13 / PyTorch 2.14.0 / NumPy 2.5.3 installed in ignored `.venv`, dependency resolution tracked in `uv.lock`. No paid research API used. EXP-0001 bounds the initial local CPU pilots.

The user's reference to an AI model solving Navier–Stokes is motivation, not verified evidence in this project. No conclusion about that claim has been drawn and no independent verification has been performed.

## Reading map

- Original intent: `docs/RESEARCH_CHARTER.md`.
- Complete work record: `NOTEBOOK.md`.
- Evidence/source index: `docs/knowledge/INDEX.md`.
- Hypothesis IDs and tests: `docs/research/HYPOTHESES.md`.
- Research standards: `docs/research/PROTOCOL.md`.
- Deferred candidates: `docs/research/BACKLOG.md`.
- Literature and architecture: `docs/research/PRIOR_WORK.md`, `docs/research/ARCHITECTURE_REVIEW.md`, `docs/research/ARCHITECTURE.md`.
- Reproduction commands: `docs/RUNBOOK.md`; active pilot: `docs/experiments/EXP-0001.md`.

## Next state

The requested targeted review, data acquisition, architecture and runnable pipeline are complete, with bounded EXP-0001 validation. Next research candidates: a separately registered multi-seed ablation comparison, synthetic known-generator calibration, and prespecified causal behaviors. Continue to review both Voynich-specific precedents and the underlying methods before attempts. No full sweep or semantic/cipher-recovery experiment has run.
