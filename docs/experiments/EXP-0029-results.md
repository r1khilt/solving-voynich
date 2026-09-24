# EXP-0029: matched locus association does not pass held-out controls

**Registered decision: `no_support_under_registered_controls`.** A train-leaf co-occurrence feature improved the matched-word log loss slightly, but the gain was below the predeclared size floor, its physical-leaf interval included zero, and it was below the 95th percentile of page-bag-preserving shuffled-training models. This result concerns a deliberately restricted masked-word task; it is **not** evidence that Voynich has no semantics or that published whole-manuscript co-occurrence patterns are invalid.

## Execution and input scope

- Registration and source-frozen runner: `docs/experiments/EXP-0029.md`, commit `bcf2e37`. Command: `PYTHONPATH=.:src .venv/bin/python -m voynich.locus_association`.
- Same pinned ZL3b derived train/validation SHA-256 as EXP-0027/28; no final-test text accessed. Fit65 physical training leaves, fit paired readout on16 other training leaves, refit association counts on all81 training leaves, then score validation.
- Calibration:796 eligible masked-word/matched-negative pairs. Validation:1,008 pairs from24 pages but **9 physical leaves**: the tenth, f73, has no `P0` running-text loci in this transcription. Pair exclusions in validation:34 short loci,250 repeated targets,652 rare targets,1,070 without a fully matched same-page negative. Each selected target occurs once in its locus; the negative is on the same page but another `P0` locus, with matched length, count ratio, edge character and nearest edit distance to context.
- CPU wall8.70s, no paid API/download. Pre-run full repository checks:1,052 passed,8 skipped,23 subtests; changed-file Ruff and diff hygiene passed. The complete per-pair archive is ignored locally at `results/EXP-0029/rows.json` (SHA-256 `79d79c782efebca5be63d1ed1bbc52f21a9cd650c30ded0793cb686e06246afd`); the tracked compact `results/EXP-0029/results.json` (SHA-256 `1de833e771e3f91999ad5d73444c71ef7aba9569885a25b71b04e97d92c3532a`) retains the calibration, all100 null gains, selection counts, observed metrics and per-leaf sums. The complete archive is reproducible by rerunning the source-frozen command above and then `.venv/bin/python scripts/exp0029_compact.py` in a clean output directory.
- The independent auditor's **first** invocation failed its own assumption that all10 validation leaves must have `P0` pairs. f73r/f73v consist only of `Lz`/`Cc` loci. The auditor was corrected to compare represented leaves with the actual `P0` leaf set; no runner, data, score, gate or result changed. The corrected independent audit passed, reconstructing every pair from the validation source, every exclusion count, score aggregate, bootstrap interval and decision. This is an audit fix, not an experimental retry.

## Observed predictive readout

| Same-page matched-word predictor | Bits per pair (lower better) | Accuracy |
| --- | ---: | ---: |
| Strong static spelling/frequency and context-form features | 1.00318 | 50.99% |
| Same features + training-locus association | **0.99688** | 51.98% |

Association saves **0.00630 bits/pair**, below the registered≥0.020 threshold. Paired physical-leaf bootstrap95% interval is **[−0.00704, 0.01417]**. The 95th percentile of100 shuffled-training null gains is **0.00736**, above the real0.00630; their mean is0.00138. Accuracy remains below the registered55% floor. Every primary gate fails except the direction of the point estimate. Two large leaves f111 and f76 contribute758/1,008 pairs, so token-level independence would grossly overstate evidence; the leaf bootstrap is the relevant uncertainty view. The trained association coefficient is positive (0.1354 after calibration RMS scaling), but that coefficient is not itself evidence of a reliable historical relation.

## Interpretation

This particular count-based, shrinkage locus-association model does not establish a relation that transfers across physical leaves once same-page vocabulary and basic orthographic/frequency cues are controlled. It also does not test all distributional representations: rare words, non-`P0` labels, more complex morphology, section-conditional rules, and larger latent models remain open. A cosine map or cluster name must not be promoted to a semantic reading without a held-out task and an external anchor. The next useful step toward actual decipherment is to connect a *frozen* textual relation to independent manuscript evidence, especially labeled imagery or historically constrained candidate readings, rather than continuing to optimize descriptive word networks on the same exposed validation leaves.
