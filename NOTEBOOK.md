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

## 2026-09-20 — Setup validated and published [NB-0002]

- **Phase:** preparation only; no research execution.
- **Validation results:** all 9 Markdown files inspected by the validation script; all 11 relative Markdown links resolved; authored files passed whitespace checks; the original charter remained byte-for-byte unchanged. Preparation-only wording is present in both README and project memory. `git diff --cached --check` passed before the initial commit.
- **Git checkpoint:** `e07c276` (`Initialize Voynich research knowledge base and notebook`) created the 11 setup files. `git push -u origin main` succeeded and established `main` tracking `origin/main` on the configured GitHub repository.
- **Environment limitation:** the sandbox initially blocked `.git/index.lock`; an approved escalation allowed the commit. Network access and push also used the environment's approval flow. These requirements are separate from the user's standing authorization for routine Git work.
- **Follow-up:** this entry records the successful initial publication and is included in a documentation-only follow-up checkpoint. No corpus downloads, dependency installations, training runs, or paid research API calls were performed.
- **Next state:** wait for the user's instruction to begin research; maintain these records at future substantive checkpoints.

## 2026-09-20 — Initial feasibility assessment [NB-0003]

- **Question:** the user asked for the assistant's own assessment of whether decipherment is possible.
- **Scope:** discussion and a limited factual source check; the research-execution phase remains inactive. Consulted Yale's manuscript overview, a Yale interview with Claire Bowern, and the bibliographic record/introduction of Reddy and Knight (2011). Source records: SRC-0003 through SRC-0005. No systematic literature review or experiments performed.
- **Source observation:** Yale continues to describe the text as undeciphered. Bowern emphasizes how many assumptions theories require while distinguishing structural knowledge from semantic understanding. These are source reports, not independent replication.
- **Assistant assessment, not an experimental finding:** full decipherment is possible but cannot responsibly be called likely from present evidence. A constrained, consistent encoding with surviving redundancy would offer a better prospect than arbitrary private conventions or text without recoverable semantic content. More capable models cannot uniquely recover information absent from the evidence.
- **Method assessment:** synthetic ground truth combined with causal model analysis is the most compelling proposal to investigate, without claiming novelty or demonstrated transfer. A learned predictor can reveal useful structure without recovering the historical generator or meanings. Unrestricted context-dependent mappings and filler exceptions would make a proposed decoding unfalsifiable.
- **Evidence that would increase confidence:** a compact frozen mechanism or decoder making independently checkable predictions on excluded manuscript material; progressively stronger evidence would be required for semantic claims. No numerical success probability assigned.
- **Validation / checkpoint:** source links and attribution scope recorded; preparation-only status preserved; documentation link and whitespace checks accompany this checkpoint. Commit and push outcome is verified in the session's tool record.
- **Next state:** continue discussion or await instruction to begin; no direction selected for execution and no resources spent on training or paid research API calls.

## 2026-09-20 — Reader usability and mechanistic interpretation [NB-0004]

- **User position:** cumbersome private codebook lookup seems unlikely for an everyday reference; the user remains particularly optimistic about mechanistic interpretability.
- **Limited source check:** SRC-0006 reports Davis's assessment of wear consistent with repeated use. This supports the user's premise in qualified form; it does not establish daily-use consensus or a particular encoding.
- **Assistant clarification:** the arbitrary private-codebook example was a limiting case for identifiability, not a favored historical explanation. Frequent use would disfavor continual expensive lookup, conditional on that use being established. It would not exclude learned private conventions, memorized vocabulary, or shared abbreviations.
- **Working methodological inference:** reader learnability can help prioritize compact, reusable candidate rules. Ease for a trained reader does not guarantee recoverability for an outsider. This is a modeling preference, not a new observed manuscript property.
- **Potential interpretability question:** whether different surface forms converge on a common internal representation that causally supports predictions across contexts. Such a representation would still need synthetic calibration, competing generator controls, and independent validation before being interpreted as a historical or semantic unit.
- **Validation / checkpoint:** preparation-only state retained, no experiment executed; documentation links and whitespace checked before the discussion checkpoint. Git publication verified in the session tool record.
- **Next state:** continue discussion; mechanistic interpretability is a user preference, not yet an authorized execution task.
