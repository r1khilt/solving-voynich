# Repository instructions

## Scope and continuity

1. Read `MEMORY.md`, the latest entries in `NOTEBOOK.md`, and the relevant knowledge/research documents before substantive work.
2. The initial user instruction is **preparation only; do not begin research tasks yet**. Setup is authorized. Do not acquire corpora, conduct a literature review, run experiments, train models, or call paid research APIs until the user asks to begin. Once that happens, update the phase in `MEMORY.md` and `README.md` and record the transition in `NOTEBOOK.md`.
3. The long-term objective is actual decipherment, not merely Voynich-like generation or another descriptive statistical report. The charter is a source of hypotheses, not mandatory methodology or established truth.
4. Use the user's latest instructions to resolve scope. Be ambitious about questions and strict about evidence.

## Research records

- Update `NOTEBOOK.md` after each substantive work block and before committing. Record the question, actions, exact inputs, observations, interpretation, validation, limitations, and next state as applicable. Record negative results and failures.
- Keep `MEMORY.md` concise: current phase, durable decisions, resource constraints, blockers, and links. Update it when these change. Do not use it as a duplicate experiment log.
- Keep the original `docs/RESEARCH_CHARTER.md` unchanged. Put proposed revisions and decisions elsewhere.
- Clearly separate sourced observations, user suggestions, working hypotheses, and model-generated speculation. Never invent citations, experiment results, data availability, or external accomplishments.
- Preserve source versions, checksums, preprocessing, manuscript identifiers, seeds, splits, software versions, commands, and costs when available. Keep raw and derived data distinguishable.
- Follow `docs/research/PROTOCOL.md` for experiment design and claims. Do not retrofit success criteria to observed results without labeling the work exploratory.

## Git and resource use

- The user explicitly authorizes routine commits and pushes to this repository without repeated confirmation. Commit and push coherent completed checkpoints, including notebook updates; avoid one commit per trivial edit.
- Inspect repository status and remotes first. Preserve unrelated user changes. Never force-push or rewrite shared history merely for tidiness.
- Verify pushes and report failures honestly. A local commit is not a remote backup. Follow environment-required approval/escalation flows when a sandbox blocks an authorized action.
- Keep secrets, tokens, local environment files, large downloaded corpora, checkpoints, and generated bulk outputs out of Git. Track provenance/manifests and compact results instead. Verify rights and repository size before adding third-party assets.
- Resource figures in `MEMORY.md` are user-reported planning context, not live account balances or authorization to exhaust them. Estimate material spend before paid runs; log actual usage when available. Do not launch open-ended paid loops.
- The user welcomes requests for downloads, access, compute, or other assistance. Ask for concrete missing resources when needed; do not infer credentials or paid-service access from a stated budget.

## Completion

Validate the changed artifacts, update the notebook and any affected project memory, commit and push the checkpoint, then report what actually happened and what remains unresolved. Do not claim a decipherment based on plausibility, prediction quality, or cherry-picked translations.
