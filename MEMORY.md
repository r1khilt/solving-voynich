# Project memory

Updated 2026-09-21 after the episodic campaign completed. Detailed history belongs in `NOTEBOOK.md`.

## Current phase and evidence

- **Active research is authorized.** Continue bounded experiments, source review, data acquisition, implementation and mechanistic interpretation without repeated approval. Actual decipherment is the objective. No Voynich word, language, filler assignment or historical encoding rule has been established.
- **Latest completed campaign:** EPISODIC-0012/0013, isolated branch `codex/episodic-rule-recovery`, scientific source `c9c95a0cfef9bdfc542136e4d38482e7d37c59ea`. All 21 neural runs, 12 explicit fits and 12 causal audits with untrained counterparts completed. Read `docs/experiments/EPISODIC-0012-results.md` and `EPISODIC-0013-results.md`.
- **Findings:** fresh tasks improved raw-symbol prediction over 32 fixed tasks; canonical fixed training nearly matched fresh. Larger capacity brought a familiar-family gain below the registered practical threshold. Neither primary causal claim passed: all recurrent teachers failed oracle qualification, and learned GRU interventions changed immediate outputs too much. Secondary delayed interventions require fresh testing.
- **Execution caveat:** a host interruption killed supervision. Recovery preserved 13 completed runs and two partial attempts, then replayed the partial runs with identical manifests/initialization before confirmation. Elapsed time from original launch through analysis was 93.55 minutes. Monitoring has a gap; exact discarded work is unknown. No paid API or manuscript input. Provenance: `results/episodic-20260921/provenance/`.
- **Naming/isolation:** episodic studies were originally EXP-0012/0013 at c9. Publication aliases and result namespace prevent collision with the separate null-removal/CTC series. Historical JSON/config/output paths retain original IDs. Preserve concurrent main-branch work in `/Users/rikhil/coding/solving-voynich`.
- Campaign workers completed; no successor is scheduled. Its final pools are exposed. This work did not score the manuscript final test. Adaptive successors require fresh final tasks and frozen rules.

## Durable decisions

- Be ambitious about hypotheses and strict about evidence. The charter is a guide, not fact; keep `docs/RESEARCH_CHARTER.md` unchanged.
- Before attempting a method, review prior Voynich applications and relevant ML/cryptography/decipherment literature. Deep review: 93 records /92 works, with reading depth and limitations, under `docs/research/deep-review-2026-09-21/`. Search with `scripts/research_catalog.py`.
- Use known-answer synthetic positive controls, structured nonsemantic/null controls, unseen keys/families, restart stability, complexity penalties and held-out physical folios where appropriate. Prediction, readable activations, causal transplantation and decipherment are different claims.
- Do not assume spaces are plaintext word boundaries, EVA codepoints are verified glyphs, or filler exists. Retain alternative units and copy/mutate explanations. Routine-book usability is a user hypothesis, not proof against codebooks.
- Explain results in plain English; the user does not find ML vocabulary self-explanatory. `docs/PROGRESS_EXPLAINED.md` defines terms.
- Update notebook after substantive blocks. Preserve source/config/data hashes, seeds, splits, failures and costs. Commit/push coherent checkpoints and verify the remote; do not force-push. Corpora, arrays, weights, credentials and environment files stay ignored. Local bulk artifacts are not backed up by Git.
- Preserve concurrent changes; inspect branch/worktree status before editing.

## Earlier evidence and data

- Official ZL3b: 226 modeling pages; frozen physical-group split 177/24/25; train-only 112-entry vocabulary. Provenance: `docs/research/DATA.md`.
- EXP-0002 selected compact manuscript prediction; it beat five-gram on development data. EXP-0005 found distant-shuffle harm; EXP-0007's group-vs-character effect was uncertain. No semantics follow.
- CAMPAIGN-0001: blind unfamiliar-key recovery failed; broad synthetic causal tests passed while manuscript tests failed; longer context did not win registered comparisons. `docs/experiments/CAMPAIGN-0001-results.md`.
- EXP-0011 Finnish null-removal holdout failed after one registered fix; no resulting Voynich deletion run. Separate latent-recovery work proceeds on main; check its state live.
- Bowern/Lindemann's linguistic interpretation is contested, not a constraint. D'Imperio is a historical survey. PDF/source qualifications: notebook NB-0021.

## Resources and next steps

- Verified local hardware: Apple M5 Pro, 18 CPU /20 GPU cores, 64 GB memory; MPS available outside the sandbox. Existing environment `/Users/rikhil/coding/solving-voynich/.venv`: Python 3.12.13, PyTorch 2.14.0, NumPy 2.5.3; lockfile tracked.
- Episodic completed work: 25,200 updates /309,657,600 scored training targets, including repeated exposure and excluding discarded partial attempts. Sampled summed RSS peaked at 1.82 GiB; maximum reported Metal driver allocation 4.14 GiB. These overlap and are not total physical-memory peaks.
- User reported about $1,000 in Astra credits and $20,000 in API credits expected around 2026-09-27. Unverified planning context, not balances or authorization to exhaust them. No paid research API credentials/spending schedule established. Estimate spend before paid runs; ask for concrete missing resources.
- User's AI/Navier–Stokes claim remains unverified motivation.
- Complete publication/integration of this campaign and external-audit fixes. Then register fresh tests separating parameter diversity from symbol renaming and obtain qualified parity teachers before stronger causal claims. Do not scale simply to fill RAM.
- Start with `AGENTS.md`, `docs/CURRENT_STATUS.md`, latest notebook and `docs/research/PROTOCOL.md`. Reproduction: `docs/RUNBOOK.md`. Implementation: `docs/research/EPISODIC_IMPLEMENTATION.md`. Hypotheses/backlog: `docs/research/HYPOTHESES.md` and `BACKLOG.md`.
