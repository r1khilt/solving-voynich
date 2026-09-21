# Project memory

Last updated: 2026-09-21 (EXP-0021 FAIL: rate-free EXP-0016 decode rejected; EXP-0018 fixed-rate FAIL retained).

## Current state

- **Communication-system workbench implemented; qualification pending:** The user explicitly authorized implementation after the research dossier. `src/voynich/communication/` now contains joint masked inference, executable world/language/channel models, compiler-guided search, learned action dynamics, causal controls, relational grounding and Bayesian evidence selection. Read `docs/COMMUNICATION_SYSTEM.md`; registration `docs/experiments/WMD-0001.md`. Default joint model: 4,944,924 parameters, at most 2,000 updates/1,200 seconds; action model: at most 1,000 updates/180 seconds. No paid compute or manuscript final-test access. Initial full-checkout validation: 667 tests and 23 subtests passed, one accelerator test skipped in sandbox. No scientific training score yet for this new system.
- Phase: **active research and implementation**. The user explicitly authorized prior-work/methods research, corpus acquisition, architecture design, and code; the earlier preparation-only restriction is superseded.
- **World models / diffusion / linguistics review completed:** approximately 35,000 words, 11 documents, 98 annotated source records with explicit reading depths. This specific user request was **research and ideation only**; no implementation, training or manuscript scoring was performed for it. Leading proposal: infer a document-producing system using external multi-system training, globally revisable structured inference, explicit rule execution, independent grounding and causal extraction. Five ambitious research directions remain unimplemented hypotheses. Start at `docs/research/world-models-diffusion-2026-09-21/README.md`; notebook NB-0032-WMD. This does not change the scope of separate experimental tasks.
- **Latent recovery track:** EXP-0011–0013 FAIL (threshold decode). **EXP-0014 PASS:** frozen EXP-0013 weights + exact-count; Finnish recon 0.218 > frozen matched-random 0.2043 (confirms peeked ~0.218). **EXP-0014b FAIL** on copy_mutate without retrain (recon 0.185). **EXP-0017 PASS:** typed program search selected `exact_count_neural`. **EXP-0016 PASS** (1,063 varieties / 270 family holdouts; macro recon 0.3155 > 0.1918) and **EXP-0016b copy_mutate PASS**. Prior ZL3b validation label-free under 0.30 null was negative (other session). This block did **not** rescore ZL3b. Synthetic PASS ≠ decipherment.
- **R3 / HYP-005:** **EXP-0019 FAIL** (`iid_control_failed`): iid matched Zipf+adjacent under frozen floors (2/4 separate; need ≥3); search never forced ≥3/4 optimizable surface match. Held-out not interpreted. HYP-005 unresolved. Optional R3 redesign (new id, harder iid falsifier) not this stretch.
- **R4 / NEXT_DESIGN P2:** **EXP-0020 FAIL** (`joint_not_better`): validity + equivalent-generator controls passed; structured fresh-key Δ_joint=0.0028<0.05 (copy_lag TF win cancelled cycle/parity edge wins). Ranked next: **R5**.
- **EXP-0018 FAIL (durable):** Frozen EXP-0016 checkpoint, exact-count rate grid on ZL3b validation with seed-4018 select/confirm split. SELECT chose r=0.70 (only candidate with neural Δ vs matched-random > 0). CONFIRM: mean_gain −0.152 > mean_random −0.196, but fraction_beats_random 0.459 ≯ 0.50. **Fixed-rate exact-count transfer from the multilingual deletion model is rejected on this validation split.** Not a decipherment; ZL3b test unscored.
- **EXP-0021 FAIL (durable):** Same frozen EXP-0016 checkpoint and same seed-4018 confirm pages; rate-free keep iff P(signal)≥0.5 (not tuned on Voynich). Confirm implied null rates all 0 (std 0); mean bits-gain 0 (= delete-nothing); fraction_beats_random 0. **Absolute keep-scores saturate near 1 on ZL3b validation, so the threshold deletes nothing.** **Neither fixed-rate (EXP-0018) nor this rate-free decode of EXP-0016 establishes a Voynich null layer.** Not a decipherment; ZL3b test unscored. See NOTEBOOK NB-0038b.
- Earlier research checkpoint: **deep research/design review completed**, 93 source records /92 distinct works and resources, with reading depth and a searchable catalog. Read `docs/research/deep-review-2026-09-21/README.md` and `NEXT_DESIGN.md`. CAMPAIGN-0001 remains the latest completed neural campaign; historical hypotheses remain unresolved.
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
- **PDF assessment (NB-0021, 2026-09-21):** Bowern & Lindemann (*Annu. Rev. Linguist.* 2021) is a useful linguistics survey; their “encoded natural language (not hoax)” conclusion is contested author theory, not a project constraint. Do not drop structured nonsemantic / copy-mutate controls. D’Imperio (1978) is a historical NSA survey; the user’s OCR scan is not a clean primary source. Neither PDF establishes a language ID or decipherment. Details: `NOTEBOOK.md` NB-0021; local extracts under ignored `outputs/pdf-assessment-2026-09-21/`.
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

**Navier–Stokes (sourced 2026-09-21):** OpenAI published a claimed finite-time blowup for 3D Navier–Stokes **with a designed smooth force** (zero initial velocity; bounded energy; velocity \(L^\infty\) blowup), aimed at Clay alternatives **C/D**, plus a Lean artifact. This is **not** unforced global regularity (A/B), **not** a PINN/FNO theorem, and **not** a CMI prize award (CMI 11 Sep 2026: “apparently been settled”; rules require qualifying publication, ≥2 years, community acceptance). This repo did not compile the Lean. Memo: `docs/research/ai-hard-problems-2026-09-21/README.md`. Transferable lesson: generate + **exact verifier**, not EVA-like generation. Highest-EV scientific next remains synthetic exact-count/program-search gates (EXP-0013/0016 track) plus typed decoder search—not a PDE-net or LLM translator.

## Reading map

- Original intent: `docs/RESEARCH_CHARTER.md`.
- Complete work record: `NOTEBOOK.md`.
- Evidence/source index: `docs/knowledge/INDEX.md`.
- Hypothesis IDs and tests: `docs/research/HYPOTHESES.md`.
- Research standards: `docs/research/PROTOCOL.md`.
- Deferred candidates: `docs/research/BACKLOG.md`.
- Literature and architecture: `docs/research/PRIOR_WORK.md`, `docs/research/ARCHITECTURE_REVIEW.md`, `docs/research/ARCHITECTURE.md`.
- Expanded review: `docs/research/deep-review-2026-09-21/README.md`; four topic reviews and source ledgers, `CATALOG.md`, `SOURCE_AUDIT.md`, `NEXT_DESIGN.md`. Search with `python3 scripts/research_catalog.py --query <term>`; validate with `--check`. Candidate direction: fresh-key episodic inference, explicit edge-emitting beliefs, joint continuations and selective causal update tests. These are proposed methods, not findings or a launched campaign.
- World/action models, diffusion, separate linguistic foundations, and mechanistic identification: `docs/research/world-models-diffusion-2026-09-21/README.md`; the integrated proposal is `SYNTHESIS.md`. Research only; 98 source records are not 98 unique or fully read papers.
- AI on hard problems (NS, Erdős, formal math) and Voynich analogues: `docs/research/ai-hard-problems-2026-09-21/README.md`.
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
