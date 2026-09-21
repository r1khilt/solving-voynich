# WMD-0001 — Implemented system and measured limitations

Completed 2026-09-21 under the [frozen registration](WMD-0001.md). The integrated software runs and its invariant checks pass. The learned joint solver does **not** yet recover executable world explanations in this evaluation: all accepted candidates came from compiler search under the nonsemantic copy family, and training reduced their document coverage. No accepted procedure or taxonomy explanation was recovered. This is a negative recovery result alongside positive reconstruction and separately supervised action-learning results. There was no preregistered scientific pass threshold to declare passed.

## Execution and provenance

Training, free-running evaluation, the causal report and the fixed demonstration used unchanged communication-module bytes from `eeba1468d01e438a5c9009283327abe156f86f22`. Other repository tasks committed unrelated work concurrently; source-byte manifests, rather than the moving repository HEAD, establish the scientific code version. The later direct-API final-test guard is post-run hardening and did not produce these scores.

The joint Transformer has 4,944,924 parameters: six layers, width 256, eight heads, dropout zero, observation capacity 128 and alphabet capacity 24. It trained for 2,000 AdamW updates on 32,000 generated episodes, seed 41021, in **453.117 seconds** on local MPS. It completed within its 1,200-second limit. Fixed masked validation loss fell from **1.1525028224 to 0.7238970943**; the best checkpoint is step 2,000. All 20 validation checks used the same 64 examples and 2,993 masked targets. This is a development set used for checkpoint selection, not a fresh confirmatory evaluation.

The action model has 105,284 parameters. It completed 1,000 updates on 64,000 labeled examples, seed 41022, in **39.998 seconds**, within its 180-second limit. Runs overlapped in wall time; their durations should not be added and represented as total elapsed time. Python 3.12.13, Torch 2.14.0 and device details are recorded in the manifests. No paid service or remote compute was used.

Checkpoints remain local under ignored `outputs/WMD-0001/`. Their sizes and SHA-256 values are in [the artifact manifest](../../results/WMD-0001/artifact-manifest.json). The approximately 1.3 MB tracked result bundle includes every evaluation document, all negative controls and compressed candidate audit records, not model weights or manuscript text.

## Joint recovery: frozen comparison

Each condition uses the same 64 validation worlds with the held-out grammatical combination `(VOS, prefix)`, four denoising proposals, 12 denoising steps and a compiler cap of 20,000 expansions with beam 32. Zero-anchor and two-anchor conditions use the same underlying worlds; two anchors are **synthetic oracle correspondences**, not discovered historical evidence. Candidates are ranked without gold. The first candidate is scored even when invalid, and invalid selections are marked abstained.

“Trained” and “untrained” below both include the identical compiler mechanism and search budget. The neural network supplies proposal distributions and search preferences. The two simple baselines have no compiler. Scores are fractions; higher is better except where explicitly stated otherwise.

| Anchors | Method | Unanchored non-null key accuracy | Keep F1 | Boundary F1 | Decoded similarity | Selected explanation valid |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 0 | Trained + compiler | 0.1142 | 0.8808 | 0.6435 | 0.2796 | 19/64 = 0.2969 |
| 0 | Untrained + compiler | 0.1365 | 0.9286 | 0.7009 | 0.2025 | 25/64 = 0.3906 |
| 0 | Random key | 0.0487 | 0.6272 | 0.3743 | 0.2300 | 0/64 |
| 0 | All null | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0/64 |
| 2 | Trained + compiler | 0.1966 | 0.8964 | 0.6591 | 0.3985 | 20/64 = 0.3125 |
| 2 | Untrained + compiler | 0.1576 | 0.9329 | 0.7305 | 0.3277 | 27/64 = 0.4219 |
| 2 | Random key | 0.0627 | 0.6904 | 0.4175 | 0.3098 | 0/64 |
| 2 | All null except anchors | 0.0000 | 0.3131 | 0.2735 | 0.1867 | 0/64 |

Decoded similarity is one minus normalized Levenshtein distance, not translation accuracy. Exact lexical/key scores depend on anonymous-entity naming conventions and do not establish identifiability. Including true null keys increases trained key accuracy to 0.1559/0.2107, but also gives the all-null baseline 0.2280/0.2643; that metric alone rewards deletion and is inadequate.

The trained solver improves decoded similarity by +0.0771 without anchors and +0.0708 with two anchors. It loses 0.0938 and 0.1094 in selected-valid fraction, respectively, and performs worse on both keep and boundary F1. Non-null key accuracy also worsens without anchors. These mixed results do not support a general recovery claim.

Post-run descriptive paired bootstrap intervals use 10,000 resamples of the 64 worlds, seed 41023. The 95% intervals for similarity differences are [0.0277, 0.1214] and [0.0316, 0.1075]; those for selected-valid differences are [-0.1719, -0.0312] and [-0.1875, -0.0469]. This uncertainty calculation was not preregistered, does not account for checkpoint selection or training-seed variation, and does not make the development result confirmatory.

The audit identifies the source of apparent successes:

- **Zero of 512 trained denoising proposals passed the candidate verifier** across the two anchor conditions. The untrained denoiser also produced zero valid proposals out of 512.
- Every valid candidate came from compiler-guided proposals. Trained candidate pools contained 202/320 and 184/336 valid compiler proposals; untrained pools contained 165/380 and 176/405. A larger valid proportion within returned candidates does not compensate for fewer documents with any valid candidate.
- Consistency projection changed 237/256 and 227/256 trained denoising candidates, versus 256/256 in each untrained condition. Repairs are recorded, rather than represented as unmodified neural predictions.
- Every world involved beam pruning. No evaluation exhausted the global expansion cap. Failure is not proof that no compatible explanation exists; bounded beam search is incomplete.
- The untrained evaluations were repeated once to retain full inference metadata; their document scores and aggregate metrics reproduced exactly. No favorable repeat was selected.

Independent review checked all 2,465 candidate records, their rank-zero identities, position coverage, global mappings, edit-score arithmetic and search budgets. It confirmed all 727 valid candidates were compiler-origin. As an additional post-run prevalence check, the observations contain 2,085 signal positions out of 2,422 positions: always keeping everything would obtain F1 0.9252, above both trained conditions. High keep F1 alone therefore does not establish null recovery. This analytical reference is not a newly tuned solver.

**All 727 accepted candidates have status `nonsemantic_surface_only`.** They satisfy the declared copy-family surface rules; none provides an executable procedure or compatible taxonomy witness. Consequently the table's validity column measures surface-valid nonsemantic explanations in this run, not recovered world semantics. The implementation can execute procedure/taxonomy witnesses in independent fixtures, but the scored solver did not recover any. Action-model accuracy below concerns separately supplied true states and actions and cannot fill this inference gap.

The fixed CLI demonstration at validation index zero abstained, with four invalid denoising proposals and no complete compiler proposal. The optional action prior correctly reported `not_applicable` for these candidates. It was not given a better-looking replacement example. Independent unit fixtures verify scoring of executable procedure candidates without changing their rank.

## Action model and causal checks

On the fixed action audit, exact valid next-state accuracy improved from 0 to 1.0 and balanced action-validity accuracy from 0.5381 to 1.0. Full-transition accuracy improved from 0.1934 to 1.0. However, training covered **all 1,152 possible state/action/type combinations in this generator**. All 745 unique evaluation combinations were seen; the audit has 2,048 examples, split evenly between legal and illegal actions. Unseen-input metrics are correctly null. This establishes learning within the finite domain, not new-action or new-world generalization.

The denoiser causal report uses the trained checkpoint, fixed validation indices zero and one, seed 41021, on CPU. Identity intervention changes output probabilities by exactly zero; full final-hidden-state donor replacement reproduces donor probabilities exactly. Targeting the 16 mutable key positions changes their mean total variation by 0.03208 versus 0.01085 for the norm-matched random control. Other positions remain unchanged in this final-readout intervention. **All argmax decisions remain unchanged.** These are controlled computational effects from one pair and one seed, not a recovered algorithm or historical semantic circuit. Multi-seed causal stability remains untested.

Analytic CLI fixtures separately check supplied evidence updates (posterior 0.9/0.1), query information gain (0.3680642072 and zero nats), ambiguous graph alignment (two ties), and labeled transition extraction (coverage 0.5 with unseen transitions explicit). These fixtures establish software behavior; their labels and likelihoods are supplied by construction.

## Manuscript boundary and corrections

The format adapter successfully exported the registered **training** split: 177 pages, 1,532 windows, 108 literal observed units, four explicit uncertainty markers and 75 rare-glyph escapes. A pre-training audit corrected an initial mistake that treated all private-use glyphs as uncertainty. The earlier local manifest remains retained as an incorrect intermediate artifact; only `manuscript-format-export-v2` is authoritative. A regression test distinguishes rare glyph escapes from uncertainty markers.

No manuscript was used in either training run or recovery evaluation. The 24-symbol checkpoint cannot consume the 108-unit manuscript inventory; it refuses incompatible capacity. The adapter hashes corpus files to verify their preparation manifest, including final-test file bytes, but it does not parse or score final-test records. No final synthetic holdout was scored. Unit fixtures use separate tiny seed domains to test split handling; they are not benchmark results.

After scientific runs, independent review found that the CLI rejected final-test inference while the direct Python `infer` API did not. The backend now also refuses by default and requires an explicit boolean `allow_test=True` for a separately frozen evaluation. The guard is regression-tested using a validation observation relabeled as test, without opening a final holdout. This hardening changes the post-run source hash; the archived scientific manifests keep the original hashes.

Final whole-checkout validation returned **682 tests and 23 subtests passed**, with one accelerator-only test skipped inside the sandbox. That test separately passed on local MPS. Targeted lint, whitespace checks, archive/checkpoint hashes, finite JSON, compressed-audit restoration, source provenance and document links passed. The original research charter is unchanged. Independent CPU replay also reproduced all action-checkpoint final metrics and the evaluation input checksum.

## Reproduction and next state

The exact training configuration and seed policies are in the registration and manifests. Commands actually used:

```sh
.venv/bin/python -m voynich.communication train --config configs/communication-local.json --run-dir outputs/WMD-0001/joint --device mps
.venv/bin/python -m voynich.communication train-actions --run-dir outputs/WMD-0001/actions --device mps --steps 1000 --max-seconds 180 --seed 41022
.venv/bin/python scripts/wmd0001_qualification.py --checkpoint outputs/WMD-0001/joint/best.pt --output outputs/WMD-0001/qualification --device mps
.venv/bin/python -m voynich.communication intervene --checkpoint outputs/WMD-0001/joint/best.pt --output outputs/WMD-0001/interventions.json --seed 41021 --index 0 --device cpu
.venv/bin/python -m voynich.communication generate --output outputs/WMD-0001/demo-observations --count 8 --anchors 2 --seed 41021
.venv/bin/python -m voynich.communication infer --checkpoint outputs/WMD-0001/joint/best.pt --action-checkpoint outputs/WMD-0001/actions/model.pt --observation outputs/WMD-0001/demo-observations/observations.jsonl --index 0 --output outputs/WMD-0001/demo-inference.json --candidates 4 --steps 12 --seed 41021 --device cpu
.venv/bin/python scripts/wmd0001_summarize.py --run-dir outputs/WMD-0001 --output results/WMD-0001
```

Use fresh output directories when rerunning. The qualification harness refuses source/checkpoint mismatches. To replay this old checkpoint after later source changes, expose `eeba146`'s `src` from an isolated checkout through `PYTHONPATH`, while invoking the retained harness and local checkpoint by absolute paths. New training should use the current source and generate a new manifest. Exact same-platform replay was checked for the untrained evaluation and CPU training resume; cross-version or cross-device bitwise equality is not promised.

The implemented workbench covers the computational core of the five research directions. It is not yet a multilingual foundation model, visual perception system, general VLA, stateful historical cipher solver or Voynich decoder. The full ambition requires independently annotated evidence and broader linguistic/world data, together with stronger joint search and honest null controls. A successor should address the observed gap between masked reconstruction and complete valid explanations under a new registration. No historical hypothesis was confirmed, and no WMD-0001 job remains running.
