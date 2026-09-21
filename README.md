# Solving Voynich

A long-term research project aimed at deciphering the Voynich Manuscript by reverse-engineering the process that produced its text. Modern AI is a research tool; fluent generation, attractive translations, and statistical similarity are not themselves decipherment.

**Current phase: active research and implementation.** The user has authorized a prior-work and methods review, manuscript acquisition, and an interpretable Voynich-only modeling pipeline. Design choices, validation, and completed runs are documented in the notebook; no decipherment claim is made.

## Start here

- [AGENTS.md](AGENTS.md): instructions for agents working in this repository.
- [Project memory](MEMORY.md): compact continuity record and current constraints.
- [Research notebook](NOTEBOOK.md): chronological record of work, decisions, validation, and unresolved issues. Update this whenever substantive work happens.
- [Original research charter](docs/RESEARCH_CHARTER.md): the user's text, preserved verbatim; a guide rather than a fixed plan.
- [Knowledge index](docs/knowledge/INDEX.md): evidence labels, source records, and what we do and do not know.
- [Hypothesis register](docs/research/HYPOTHESES.md): competing explanations and proposed falsification tests.
- [Research protocol](docs/research/PROTOCOL.md): reproducibility, evaluation, and decipherment standards.
- [Deferred work](docs/research/BACKLOG.md): possible next steps, not an active task queue.

## Working practice

Use versioned Markdown while the knowledge base is small. Add searchable source records and experiment manifests as material arrives; adopt retrieval infrastructure only when volume warrants it. Preserve raw evidence and keep interpretations traceable to it.

Commit coherent checkpoints and push them to the configured GitHub remote. Never commit credentials, private keys, or uncontrolled bulk datasets/model artifacts. Every research checkpoint should include its notebook update and enough detail to reproduce or inspect the result.

## Research and implementation

- [Prior work](docs/research/PRIOR_WORK.md): existing Voynich GRU/GPT experiments and related cipher/decipherment research.
- [Architecture review](docs/research/ARCHITECTURE_REVIEW.md): modern DeepSeek/Qwen/Moonshot ideas, evidence at small scale, and interpretability tradeoffs.
- [Implemented model](docs/research/ARCHITECTURE.md): a roughly 1.8M-parameter reference transformer, smaller/attention-only controls, and isolated MTP, QK-normalization and gating variants.
- [Corpus documentation](docs/research/DATA.md): the official ZL3b transcription, normalized EVA units, uncertainties, frozen physical-group splits and provenance.
- [Runbook](docs/RUNBOOK.md): acquisition, training, evaluation, checkpoint resumption, ablation plans and causal interventions.
- [Registered validation pilot](docs/experiments/EXP-0001.md): limits and acceptance criteria established before real-data training.

```sh
uv sync --extra dev --locked
.venv/bin/python scripts/download_data.py
.venv/bin/python -m voynich.data
.venv/bin/python -m pytest -q
.venv/bin/python -m voynich.train --config configs/smoke.json --run-dir outputs/my-smoke
```

The pipeline needs no paid APIs or pretrained weights. Training evaluates validation pages; final test scoring requires a separate explicit command. Raw text, derived corpora, and checkpoints stay local and are reproducible from tracked code/manifests. Architecture configs are candidates, not a completed ranking or evidence of decipherment.

## Initial measured result

[EXP-0001](docs/experiments/EXP-0001-results.md) completed: the 1,814,208-parameter reference reached **1.961 validation bits/token** after 200 CPU steps, versus **2.088** for a matched five-gram baseline. This is a single-seed engineering pilot; the test split remains unscored. All 222 implementation tests and 23 subtests passed before the run. Multi-seed architecture comparisons and actual mechanism recovery remain future experiments.
