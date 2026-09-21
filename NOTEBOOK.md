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

## 2026-09-20 — Proposed first modeling sequence [NB-0005]

- **Question:** whether the first step toward mechanistic interpretation is to build a model architecture and train it exclusively on essentially the whole manuscript.
- **Proposed sequence, not an executed experiment:** establish a traceable transcription and preserve layout/uncertainty; define development and test pages or blocks before fitting; initialize a small conventional decoder-only transformer from random weights; train next-unit prediction on the training portion of Voynich alone; check generalization against simple baselines; then apply causal interpretation and synthetic calibration.
- **Architecture clarification:** a custom architecture is not a prerequisite. Start with an architecture that is straightforward to inspect, and choose size after corpus inspection. Character/transcription-unit modeling is a candidate starting point; do not silently equate transcription characters with historical glyphs or assume spaces are plaintext word boundaries.
- **Evaluation clarification:** exclusive Voynich training does not require exposing every manuscript page during the first run. Reserve validation material for choices and an untouched test set for final evaluation, checking repeated material and overlapping windows for leakage. A later full-corpus exploratory model is possible, but its training pages can no longer serve as independent evidence of generalization for that model.
- **Modeling purpose:** establish a reproducible predictor with inspectable weights and activations. Prediction quality and mechanistic findings do not alone establish semantics or the historical generator. Save intermediate checkpoints where useful and compare multiple seeds once a minimal pipeline works.
- **Status / validation:** discussion only; no corpus acquired, architecture implemented, framework selected, training started, or API call made for research. Documentation whitespace checked; checkpoint publication verified in the session tool record.

## 2026-09-20 — Research and implementation authorized [NB-0006]

- **User instruction:** investigate earlier Voynich neural models and related cipher/decipherment methods; consider recent architecture advances including official DeepSeek work; design and implement a strong, causally inspectable model and obtain manuscript text. The user asks for background research on both the specific problem and the method before future attempts.
- **Phase change:** active research and implementation; preparation-only restriction superseded. Local tests and bounded validation pilots support the requested runnable implementation. No open-ended paid training authorized or launched.
- **Parallel assignments:** prior Voynich/neural-decipherment literature; recent architecture/interpretability literature; authoritative transcription acquisition and data pipeline. Root integrates research, implements model/training/interventions, validates, and maintains records.
- **Scientific constraint:** architectural quality must be measured on a tiny corpus with held-out data. Frontier-scale results motivate controlled options, not automatic claims of superiority. Preserve a directly inspectable reference model.
- **Status:** in progress; results, exact artifact paths, validation commands and publication checkpoints will be added as work completes.

## 2026-09-20 — Literature-informed implementation checkpoint [NB-0007]

- **Background reviewed before implementation:** original Voynich GRU and character/token GPT reports, including a 2025 five-fold saliency study; supervised/unpaired neural decipherment, recurrence encoding, ALICE, and generative controls; current official DeepSeek/Qwen/Moonshot architecture work and causal-patching methods. Detailed attributed findings and access limits are in `PRIOR_WORK.md` and `ARCHITECTURE_REVIEW.md`. These reports do not establish any decipherment.
- **Design:** small dense pre-RMSNorm/RoPE/SwiGLU decoder, full independently accessible heads, additive residuals, untied readout, and individual optional QK normalization, scalar head gating, and direct +2 prediction. Smaller and attention-only controls are included. Frontier-scale MoE/compressed attention/multiple residual streams are deferred for lack of an established need at corpus scale.
- **Implementation:** PyTorch model and hook API, train-only tokenizer/data interface, page-contained horizon targets, masked uncertainties, validation selection, resumable model/optimizer/RNG snapshots, matched n-gram/copy baselines, explicit final-test gate, and causal intervention CLI with identity/shuffled/reverse controls. Bounded ablation launcher defaults to plan-only output.
- **Resources:** arm64 machine, 64 GiB RAM, 18 logical CPUs; PyTorch's sandbox probe reports no available accelerator. Created Python 3.12.13 environment and locked PyTorch 2.14.0/NumPy 2.5.3 and validation dependencies. Environment creation/installation required approved sandbox escalation. No paid API calls or remote training.
- **Data acquisition:** downloaded official 411,671-byte ZL3b transcription, source digest in EXP-0001. Maintainer CC0 statement documented by the data agent. Sandbox DNS failure was followed by an approved successful download. Parser corrections addressed unreadable/empty alternative notation and a metadata-tag collision before preparation passed. Raw/derived text remains ignored.
- **Split design correction before training:** the first uncommitted random split omitted whole sections from validation. Replacing it with metadata-stratified physical-group assignments before any model result; preserve the rare-section limitation rather than breaking leaf groups. This is a preregistration/design revision, not an outcome-selected split.
- **Implementation defects found and fixed:** validation of malformed model settings; best-state preservation and initial baseline during resume; interpretation input guards. Independent tests cover causal-prefix invariance, forward-path parity, head decomposition, target masking, no test-data use in training, checkpoint exact resumption on CPU, and intervention controls.
- **Next checkpoint:** finish data stratification, run full checks, publish source before bounded EXP-0001 pilots; append exact measured outcomes. No architecture-efficacy or semantics claim from unit tests.
- **Pre-pilot completion:** split v2 finalized (177/24/25 pages, 112 vocabulary entries), unchanged-source rebuild reproduced all derived hashes, source cache checksum verified. Full `.venv/bin/python -m pytest -q` passed **222 tests and 23 subtests**; Ruff, Git whitespace, and local Markdown-link checks passed. Final audit also added registered-corpus hash enforcement, layout-aware patch-report validation, and checkpoint/code provenance to evaluation reports. This source checkpoint is committed and pushed before real-data pilots; its exact revision is recorded by each subsequent run manifest.

## 2026-09-20 — EXP-0001 completed and measurements retained [NB-0008]

- **Source and provenance:** code/research checkpoint `4fc9019` pushed before pilot. Both run manifests confirm that revision with a clean tree, exact source/data/tokenizer checksums, locked environment and effective configs. Commands and full interpretation limits: `docs/experiments/EXP-0001-results.md`.
- **Runs:** 40-step 96,576-parameter smoke and 200-step 1,814,208-parameter reference; both CPU/float32 with four threads, started concurrently. Training-only text, random initialization. Reference saw 358,387 eligible sampled targets; best step 200. Reported elapsed times approximately 0.456 s and 14.528 s, respectively; not a hardware benchmark.
- **Observed validation result:** reference **1.960671 bits/token**, five-gram **2.087802**, copy/unigram **3.377559**, unigram **3.997425**, all on identical 25,917 eligible targets across 24 validation pages at 256-unit context. One seed and a short run; does not rank architecture variants or establish semantic knowledge.
- **Intervention plumbing:** identity patches left logits unchanged (maximum difference 0); replacing the final residual exactly recovered the clean prediction. Illustrative head patch had a small effect and negative normalized recovery; no function assigned. Saved clean/corrupted contexts, reversed/shuffled controls and checkpoint digests. This is a forward-computation check, not circuit discovery.
- **Artifacts:** compact manifests, summary/history, baseline comparison and intervention reports copied to `results/EXP-0001/`; weight snapshots retained locally under ignored `outputs/EXP-0001-*`. Regeneration commands and SHA-256/size records tracked. No raw/derived bulk corpus or weights pushed to Git.
- **Other validation:** bounded ablation launcher printed its 15-run plan without executing it. No test-model loss computed, no paid training/API calls, no full ablation sweep, no synthetic cipher solver or SAE experiment.
- **Interpretation / next state:** infrastructure is functional and the initial reference outperformed registered simple baselines on validation. Further architecture ranking requires a registered multi-seed comparison; historical hypotheses require separate synthetic calibration and causal tests. The requested research/design/data/code deliverable is complete; final result documentation checkpoint follows.

## 2026-09-20 — Continued experimentation authorized [NB-0009]

- **User scope:** proceed with ambitious experiments, theories, hypotheses, and mechanistic interpretation. User reports M5 Pro with 64 GiB RAM. This authorizes bounded local research beyond the initial implementation pilot.
- **Hardware correction:** sandboxed MPS availability was false; approved outside-sandbox PyTorch probe reports MPS available and successfully executes a GPU tensor operation. The earlier statement described sandbox visibility, not absence of usable GPU hardware.
- **Before attempts:** reviewed existing Voynich/cipher and architecture records, plus primary induction-head/state-tracking literature for planned causal/synthetic follow-ups. Register EXP-0002 for six architectures × three seeds, up to 2,000 steps each, using unchanged data/validation policy and no test scoring.
- **Next:** benchmark CPU/MPS on bounded artificial inputs, select a common backend, commit registration/source, run the comparison while implementing separately registered causal/synthetic tests. No paid APIs or remote compute.
- **Hardware benchmark:** 20 measured synthetic optimizer steps after five warmups, batch 8/context 256. CPU averaged 0.06590 s/step; MPS 0.01947 s/step (~3.38× faster). Cached/ordinary maximum logit differences were below 8e-7 on both. Select MPS for all EXP-0002 runs; hardware report tracked. Added requested intermediate checkpoints at 100/500/1000/2000, with targeted retention/validation tests.
