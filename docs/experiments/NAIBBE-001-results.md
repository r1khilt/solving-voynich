# NAIBBE-001 result — exact shared key and near-exact text under supplied grouping

**Registered competence gate PASS; independent audit PASS.** The source-frozen decoder recovered all23 plaintext-letter assignments from the first8,192 Naibbe ciphertext tokens. The same unchanged key decoded the next evaluated8,192 tokens with **4 edits in12,451 characters**, CER **0.03213%**. Both remaining errors are ambiguous two-character chunks. This exactly matches the known-key oracle. No Voynich text was deciphered.

The largest caveat is also the main next research question: the decoder received the **true grouping of codewords representing the same letter**, plus role grammar and alphabet. That removes most cipher discovery. Even the wrong-language English prior found the correct23-letter key. This is therefore a successful, relatively easy engineering control, not evidence of language identification or a solved verbose cipher.

## What was frozen

[Registration](NAIBBE-001.md), source, data manifest and six new tests were pushed and remote-verified at `93b9243da606405cbd58393b50ed504f46a65c5d` before any target key optimization. Fit used only a separate `fit_input.json` and independent Caesar language-model text. Each source prior had317,326 normalized characters. The original Pliny plaintext did not enter either model.

All three learned keys and56-cycle search traces were pushed and remote-verified at `6787b35762e6712ac5d0ff1b719dc4589b5fb727` before answer access and transfer decoding. The independent auditor was implemented/tested without results and remotely frozen at `5f5af5c4019a5c239bf2806980dcdf25a8447964` before evaluation. Evaluation ran once, then the independent auditor ran once; neither required a result-dependent repair.

Source search took6.819s (Latin joint),5.636s (Latin fixed parse),7.460s (English joint), **19.915s total**, all local CPU with no paid spend or model training. Latin joint reached its selected key at the first cycle after6 improving swaps,0.187s. Subsequent registered restarts did not improve it; global optimality is not proved. These timings describe this heavily assisted task; they predict neither blind homophone discovery nor manuscript difficulty.

## Held-out recovery

| Frozen decoder | Key assignments correct | Character edits /12,451 | CER | Exact chunks /8,192 | Correct ambiguous chunks /46 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Latin joint key/parse, primary |23/23|4|0.03213%|8,190|44|
| Latin fixed first anonymous parse |23/23|68|0.54614%|8,158|12|
| English joint key/parse |23/23|8|0.06425%|8,188|42|
| Latin known-key oracle |23/23 supplied|4|0.03213%|8,190|44|

All23 true source letters occur in both evaluated blocks; there are no zero-occurrence key types in this published sample. Weighted key recovery is100% in all three learned arms. Primary criteria pass: CER≤2%, within0.5percentage points of oracle, and weighted key≥99%. Fit Latin joint CER is6/12,483=0.04807%, with61/64 ambiguous chunks correct. Fit fixed parse has88 edits; English joint16. The development block was unused in fitting/selection; the post-evaluation integrity audit reads its prepared candidates/answers to validate data support, so it should not later be described as researcher-unexposed.

The joint parser reduces transfer edits from68 to4 with **the same recovered key**, isolating an actual benefit of source-conditioned latent parsing on this control. The two Latin joint errors also occur with the correct key supplied: more key optimization cannot remove them under the frozen4gram parser. A richer source model, card/table-sequence constraints or explicit uncertainty could be tested later on fresh data, but chasing four residual errors is lower priority than withdrawing the major grouping gift.

All learned keys equal the gold permutation, and every arm's fit/transfer gold-minus-learned objective is zero. Thus neither search failure nor a wrong-key source-objective preference explains this result. The English arm's success shows that exact key recovery here does not require a language-specific Latin prior; a lower source score for English is not a calibrated language-identification result.

An explicitly **post-hoc initializer diagnostic**, performed after this evaluation while002 ran, reconstructs the frozen frequency-rank starting key: Latin initially gets13/23 types and58.98% source-frequency-weighted assignments correct; English4/23 and24.76%. Both finish23/23 after sequence-based search. A separate Counter-based reconstruction agrees exactly. Thus this was not already solved by the initializer, although it remains heavily assisted and is not language identification. Values are in [the diagnostic record](../../results/NAIBBE-001/posthoc_initializer.json); this was not a registered gate or a new held-out experiment.

## Audit and limits

The separate auditor reconstructs codeword support by trying each glyph-string split, source probabilities from dictionary ngram counts, exact lattice best scores with its own forward DP, and Levenshtein distance with Wagner–Fischer rather than the evaluator's bit-vector implementation. It checks **all65,536 emitted chunks** across oracle/three arms and two splits, **all18,432 prepared token lattices**, each frozen key, source/input/output hashes, every metric and the decision. Maximum source-score difference is1.24×10⁻¹⁰; audit time3.571s. It does not independently rerun key optimization or establish historical encoding events/table choices.

- [Evaluation](../../results/NAIBBE-001/evaluation.json), SHA-256 `6c2abf0946ebee5c72033e92230424793458a7c5917742720bfcd95a8c5d945f`.
- [Audit](../../results/NAIBBE-001/audit.json), SHA-256 `cc2e711054a93ee7e880e6f6a2bd68788afdedb8c537eee2e1f9eb0351ddb9d6`.

Full project suite: **1,172 passed,8 skipped,23 subtests passed**,200.03s, using `PYTHONPATH=.:src`. New tests and changed-file Ruff/compilation pass. Full-tree Ruff still reports five pre-existing unused imports/variables in unchanged legacy files; it is not globally clean. The first full-suite command omitted the repository-root import path and failed collection; the corrected command above completed. No unrelated working-tree changes were included.

This is one published control with a supplied alphabet, language candidate, role grammar, observed ciphertext-token boundaries and true cross-table equivalence. Scores maximize legal source paths and omit the card-deck emission law; they are not marginal ciphertext evidence. The cipher construction is not established as Voynich's mechanism. No manuscript language, null layer, key, word or semantics follows.

## Next: withdraw grouping, preserve explicit predictions

The predeclared next rung removes known same-letter links across the six tables, while retaining a disclosed role grammar. The solver must infer which different codewords represent the same source letter while producing a readable message with one reusable key. This directly tackles a missing part of decipherment. New data/holdout decisions must be frozen before evaluating that harder model; the present transfer result cannot be relabeled as a fresh confirmation. Image-gap and mechanistic/neural branches should be reintroduced when they address a demonstrated ambiguity in this recovery task.
