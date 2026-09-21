# Research notebook

This is the chronological record of the project. Append dated entries; preserve previous results and explain corrections explicitly. Use local dates with a timezone and UTC timestamps when execution timing matters. Link detailed source/experiment records instead of duplicating large outputs.

## Entry template

```text
## YYYY-MM-DD — Short title [NB-NNNN]
Phase / question:
Inputs and provenance:
Actions and artifacts:
Observations / results:
Interpretation and limitations:
Validation (commands, checks, or independent replication):
Decisions / next state:
Git checkpoint / remote status:
Resource use (if applicable):
```

## 2026-09-20 — Preparation and charter intake [NB-0001]

- **Time:** preparation started around 20:34 PDT (2026-09-21 03:34 UTC).
- **Phase / question:** establish a durable research workspace without starting research tasks.
- **Inputs:** the user's message and attached `Pasted text.txt`, copied verbatim into `docs/RESEARCH_CHARTER.md`. Original attachment: `/Users/rikhil/.codex/attachments/975d1ba2-4794-4308-9206-2f54a80f2515/Pasted text.txt`.
- **Repository observation:** local checkout contained only `.git`; `git status --short --branch` reported no commits on `main`. `origin` was configured as `https://github.com/r1khilt/solving-voynich`. Its tracking reference was reported as gone; no remote-state conclusion was inferred from that local status.
- **Actions:** read the full charter; established a README, repository agent instructions, project memory, this notebook, a knowledge index, a hypothesis register, a research protocol, and a deferred backlog. Added Git exclusions for secrets and bulk generated/downloaded material.
- **Decisions:** use plain Markdown as the initial knowledge system; preserve the charter; distinguish hypotheses from observations; record work and failed attempts; commit and push coherent checkpoints under standing user authorization.
- **Interpretation:** the charter suggests promising lines of inquiry but supplies no verified experimental evidence. Predictive similarity or similar learned mechanisms would not by themselves identify the historical generator. Distinguishing indistinguishable signal/filler processes may require assumptions or additional evidence.
- **Limitations:** no external literature or manuscript sources reviewed, no datasets downloaded, no models trained, no experiments run, no paid research APIs called. The reported credits and external AI achievement remain unverified user context.
- **Validation:** the charter is byte-for-byte identical to the attachment (SHA-256 `ba74631f91464449097cd230bc7045cd58876c90b56be1292f723c185d407604`). An initial generic trailing-whitespace check flagged the charter's original Markdown hard-break spaces; these were preserved deliberately and exempted in `.gitattributes`. Authored documents are checked separately. Local Markdown links and the preparation-only state are checked before committing.
- **Git / environment:** the first remote check failed because the sandbox could not resolve GitHub. The approved network retry succeeded; `git ls-remote --heads origin` returned no heads. The initial documentation commit and push are pending; completion will be recorded in the next entry.
- **Next state:** preparation only; wait for the user's instruction to start research.
