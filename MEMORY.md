# Project memory

Last updated: 2026-09-20 (America/Los_Angeles).

## Current state

- Phase: **preparation only**. The user explicitly said not to start research tasks yet.
- Repository: `/Users/rikhil/coding/solving-voynich`.
- Configured remote: `https://github.com/r1khilt/solving-voynich`; branch at setup: `main`.
- The user describes the repository as private; remote visibility has not been independently verified.
- Initial local checkout had no commits or project files. This checkpoint establishes documentation only.
- Setup checkpoint `e07c276` was successfully pushed to `origin/main`; see notebook entry NB-0002 for validation and environment limitations.
- Limited source checks supported the feasibility discussions in NB-0003 and NB-0004; the source register includes those references. No systematic literature review, corpus, trained model, experiment result, or decipherment claim exists in the project yet.

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
- No API credentials, compute inventory, spending schedule, or per-experiment budget has been established.

The user's reference to an AI model solving Navier–Stokes is motivation, not verified evidence in this project. No conclusion about that claim has been drawn and no independent verification has been performed.

## Reading map

- Original intent: `docs/RESEARCH_CHARTER.md`.
- Complete work record: `NOTEBOOK.md`.
- Evidence/source index: `docs/knowledge/INDEX.md`.
- Hypothesis IDs and tests: `docs/research/HYPOTHESES.md`.
- Research standards: `docs/research/PROTOCOL.md`.
- Deferred candidates: `docs/research/BACKLOG.md`.

## Next state

Wait for the user's instruction to start research. The backlog is a menu of possibilities, not permission to execute it. NB-0005 records a proposed first modeling sequence: prepare transcription and held-out splits, train a small standard model from scratch on Voynich-only training text, evaluate, then interpret. No architecture is implemented or empirically ranked, and the discussion has not started research execution.
