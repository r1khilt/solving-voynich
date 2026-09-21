# Solving Voynich

A long-term research project aimed at deciphering the Voynich Manuscript by reverse-engineering the process that produced its text. Modern AI is a research tool; fluent generation, attractive translations, and statistical similarity are not themselves decipherment.

**Current phase: active research and implementation.** The user has authorized continued bounded experiments, corpus research, model training and mechanistic interpretation. Design choices, validation, and completed runs are documented in the notebook; no decipherment claim is made.

## Start here

- [Current status and agent handoff](docs/CURRENT_STATUS.md): active work, live monitoring paths and continuation instructions.
- [Three-track campaign](docs/experiments/CAMPAIGN-0001.md): registered blind recovery, causal mapping and longer-context experiments running under finite local resource limits.
- [Progress in plain English](docs/PROGRESS_EXPLAINED.md): what the models actually do, the terminology, which results concern artificial data, and proposed useful larger experiments.
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

The pipeline needs no paid APIs or pretrained weights. Training evaluates validation pages; final test scoring requires a separate explicit command. Raw text, derived corpora, and checkpoints stay local and are reproducible from tracked code/manifests. Architecture comparisons and interpretation experiments are documented below; none establishes decipherment.

## Current measured results

- [EXP-0002: 18-model comparison](docs/experiments/EXP-0002-results.md): compact model **1.839734 validation bits/unit**, MTP **1.837866**, five-gram **2.087802**. Registered near-tie rule selects the compact 430,720-parameter model. More elaborate architectures made small differences under this schedule.
- [EXP-0003: initial context and causal head tests](docs/experiments/EXP-0003-results.md): removing distant context hurts; the small sample's inconsistent shuffling effect is superseded by EXP-0005. A candidate influential head was localized in one seed, with no function or semantic assignment.
- [EXP-0004: synthetic calibration](docs/experiments/EXP-0004-results.md): diagnostic readouts recover hidden state at **86.25%** and signal/filler at **81.58%** in a known toy generator. An IID control stays near chance/majority guessing. Probe-direction causal steering **failed** to outperform its matched random control meaningfully.
- [EXP-0005: content, frequency and category controls](docs/experiments/EXP-0005-results.md): larger 768-target sample finds consistent distant-shuffle damage, mean **+0.037202 bits/unit**. Approximate frequency matching reduces the cross-category replacement penalty, but does not remove all confounding.
- [EXP-0006: learned causal interventions](docs/experiments/EXP-0006-results.md): supervised late interventions achieve **97.07%** agreement with the desired next category in the toy cipher. A simple output-weight control nearly matches this; early interventions fail the shuffled-supervision control. No unique algorithm recovered.
- [EXP-0007: group forms versus ordering](docs/experiments/EXP-0007-results.md): the matched character-versus-group contrast is only **+0.005090 bits/unit**; all three descriptive leaf-bootstrap intervals include zero. Boundary/recency effects remain an untested explanation.
- [Latest visual overview](results/research-round-2-2026-09-20/overview.png), [first-round overview](results/research-round-2026-09-20/overview.png), [research notebook](NOTEBOOK.md), and [initial 200-update pilot](docs/experiments/EXP-0001-results.md).

Earlier synthetic probes and intervention fitting use known-generator supervision after text-only language-model training; this is not unsupervised decipherment. EXP-0008 separately registers blind fitting before truth diagnostics. Manuscript hypotheses remain unresolved and its final test set remains unscored. Current implementation checks: **295 tests and 23 subtests passed**; live MPS intervention and cached-forward controls also passed. All research runs used local compute, with no paid API calls.
