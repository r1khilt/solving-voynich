# TEACH-0005 results: perfect cross-table key transfer, failed task-specificity gate

**Registered verdict: NOT SUPPORTED. Independent artifact audit: PASS.** The post-F-read state passed 13 of 14 preregistered clauses in both checkpoint seeds. It transferred the counterfactual key perfectly across every recipient G table, but it also redirected direct-key questions, violating the frozen requirement that direct lookup degrade by no more than five points. The composite claim of a reusable **and task-specific** intermediate therefore does not pass. The cross-G result is still strong causal evidence about what the state does in this synthetic model.

## What ran

The frozen CPU-only analysis used TEACH-0004's two successful two-read checkpoints without training, fitting a probe or selecting an intervention site. It generated 128 new groups from suite seed `65111`. Each group had a base `n -> k0`, donor `n -> k1`, and three independently remapped held-out G tables with six distinct output objects. The architecture-defined 128-dimensional `first` vector from the donor under `G0` was reused unchanged in all three base G computations. Total inference and artifact writing took 0.087 seconds. Exact unpatched restoration error was zero in both seeds.

## Registered results

All counts below are on the preregistered clean-correct denominator; all 128/128 groups were eligible in both seeds.

| Metric | Seed 0 | Seed 1 | Gate |
| --- | ---: | ---: | ---: |
| Donor-first recipient-specific items | 384/384 | 384/384 | ≥90% |
| Exact three-G donor-first groups | 128/128 | 128/128 | ≥80% |
| Cross-G avoidance of fixed donor answer | 256/256 | 256/256 | ≥90% |
| Norm-random donor-target items | 176/384 | 184/384 | donor advantage ≥40 points |
| Mean donor-target probability increase | 0.99937 | 0.99976 | ≥0.40 |
| Zero-first base accuracy | 233/384 | 177/384 | drop ≥30 points |
| Base-first rescue | 384/384 | 384/384 | ≥95% |
| Same-key/different-distractor patch | 384/384 | 384/384 | ≥90% |
| Donor-attention-only patch | 384/384 | 384/384 | ≥90% |
| Full donor-second fixed-answer control | 384/384 | 384/384 | ≥80% |
| Patched copy | 384/384 | 384/384 | loss ≤5 points |
| Patched direct | **61/384** | **90/384** | loss ≤5 points |

The primary discrimination was exact: a donor answer copied from `G0` must fail under `G1,G2`, because all six group outputs are distinct. Instead, the unchanged donor-first state produced `Gj(k1)` for all 768 recipient items across both seeds and avoided fixed-answer injection in all 512 changed-G cases. Patching the complete downstream donor `second` state did the opposite and preserved the fixed `G0(k1)` answer on all 768 items. This separates the two intervention sites behaviorally: `first` adapts through the recipient table; `second` carries the donor answer.

Zeroing `first` reduced base accuracy by 39.32 and 53.91 points. Restoring the original vector restored every answer exactly. A same-key state computed with a different distractor name/key preserved every base answer, and transplanting only donor attention weights while retaining base F values also preserved every answer. Thus the transported content, rather than attention weights alone or irrelevant row surface, determines the counterfactual key effect. Norm-matched random states remained near the two-candidate regime and trailed true donor states by 54.17 and 52.08 points.

## Why the registered verdict failed

The design treated direct-key lookup as unrelated behavior that a composed-path state patch should preserve. The architecture, however, feeds `first` into the G query for every non-copy task. The frozen result shows this was not a harmless implementation detail. When donor `first` was inserted into a direct question for `k0`, the prediction became the recipient's `G(k1)` on 323/384 items in seed 0 and 294/384 in seed 1; every remaining prediction was the original `G(k0)`, with no third destination. The internal key state and explicit direct query therefore compete for the same G read. That finding explains the failed specificity clause but cannot retroactively remove it, so the registered verdict remains `not_supported`.

This pattern is inconsistent with a state that merely stores one answer: its output changes correctly under three independent recipient mappings. It is compatible with a reusable key-like control state shared across composed and direct routes. It does not show that the state is the unique or minimal key representation, that every coordinate has stable meaning, or that the dense-row model uses a similar mechanism. A new study may prospectively test the shared-bottleneck explanation; this report labels the direct-destination count as exposed-data exploratory diagnosis.

## Independent verification and limits

`scripts/teacher0005_analyze.py` regenerated the full deterministic suite, recomputed all symbolic labels and eligibility, checked frozen source/checkpoint/row hashes, validated every token sequence, prediction and probability range, and independently reconstructed summaries and decision. `results/TEACH-0005/audit.json` reports `pass / not_supported`. It does not rerun neural inference.

This is a deliberately typed synthetic system. Its symbols, row boundaries, task markers, key role and counterfactual truth are supplied. The result validates a mechanistic method and a model-internal variable; it does not identify a Voynich key, word, language or reading.
