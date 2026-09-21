# Review of the supplied research audit

Reviewed 2026-09-21 against repository checkpoint `3e956c8`, with fixes published as `aa502e9` on `codex/research-audit-fixes` and subsequently integrated into `codex/episodic-rule-recovery`. The review began while the separate campaign was running; that campaign has now completed using unchanged source `c9c95a0`. No historical result is reclassified by this review.

The audit identifies useful weaknesses, but mixes them with incorrect descriptions of the code and untested explanations. Its introductory praise and assertion that every factual claim was checked are not evidence. The attached network-error line also does not establish what its author actually verified.

## Findings and actions

| Audit claim | Assessment | Action |
| --- | --- | --- |
| Positional reconstruction gives excessive penalties after an insertion/deletion | Correct for interpreting partial recovery. It is still a defined positional score, not an implementation of edit similarity. | Preserve historical `recon_acc`; expose aligned reconstruction and exact-sequence recovery consistently, including deletion baselines. Future gates must register the intended metric before scoring. |
| Retained-text bits can reward aggressive deletion and re-ranking | Correct concern; incorrect description as entropy estimated only from the selected string. The bigram model was already fitted separately. | Add a fixed-representation diagnostic with normalized probabilities and count-matched deletion comparisons. Keep legacy scores explicitly named; low bits alone never certify recovery. |
| Fixed training chunks create boundary artifacts | Correct about restricted history and fixed positions; false that a boundary-adjacent next-symbol pair is never trained. Targets already extend one symbol past each input window. | Add reproducible, page-contained random offsets for training. Keep deterministic evaluation and old configurations available for exact reproduction. |
| EXP-0010 proved long history useless because padding diluted attention | False description of both conclusion and mechanism. Limited full-length page support is a real limitation already reported. | Document the distinction; do not remove padding masks, learnable position parameters that do not exist, or page isolation based on this explanation. |
| EVA subdivision creates a 0.25-bit ceiling | Token representation matters; the claimed ceiling and amount attributable to pen strokes are unmeasured. | Keep alternate reversible unitizations as a controlled experiment. Compare common raw-character units and equal targets; do not compare unlike bits/token directly. |
| Voynich causal tests failed because information is diffuse across four layers | Unsupported explanation; the manuscript models tested there had two layers, and the sweep included full residual interventions. | Preserve failure as failure of the tested intervention protocol. Do not assign a distributed mechanism without measuring it. |

## Why aligned scoring matters without turning old failures into wins

The positional score compares characters at identical indices. Dropping one character near the beginning can therefore spoil many later comparisons even when those later characters are correct. Edit distance instead finds an alignment and charges for the single deletion. Exact sequence equality answers a third, stricter question. These measures should be reported separately.

Checked directly in the existing implementation: deleting `c` from `abcdefghij` gives positional score **0.20**, aligned similarity **0.90**, and exact recovery **false**. This is a deterministic software example, not a manuscript result.

An aligned score can still look good when a method deletes little: a contaminated string often contains the whole answer as a subsequence. Keep delete-nothing, amount-matched random deletion, null precision/recall, and exact reconstruction alongside it. Repeated characters can also permit multiple equally good alignments; a good string score does not uniquely identify null positions.

The implemented [recovery diagnostics](RECOVERY_METRICS_AUDIT.md) add aligned/exact reconstruction and counts to ordinary and random-baseline reports. The new normalized fixed-rank proxy is explicitly opt-in and independently fitted; it freezes the original input's encoding and supplies count-matched and bucket-histogram-matched deletion controls. Because that original map uses whole-input frequencies and its description cost is uncharged, this is an **offline conditional diagnostic**, not raw-symbol likelihood or compression. The legacy scorer also has non-normalized fallback rows; its historical outputs remain labeled legacy rather than silently reinterpreted. Future experiments should register aligned/exact recovery measures and independent controls instead of reusing the offset-sensitive positional score as their sole reconstruction gate.

The audit quotes the original branch's null-aware EXP-0012. Its neural mask accuracy was about 0.740, while the registered requirement was the majority baseline 0.714 plus 0.05, about 0.764. Thus replacing its reconstruction score would not by itself make that run satisfy the entire registered rule. The existing negative result remains; any aligned reanalysis is exploratory and needs fresh confirmation before a new positive claim.

## Random-offset implementation and limits

`PageWindows.sample_random_windows` samples a page with probability proportional to its number of raw next-symbol positions, including uncertainty/unknown targets later excluded from loss, then chooses an inclusive uniform start among all full windows on that page. Short pages use their one available padded window. The page weights are independent of requested context length. Draws use the existing saved sampler RNG, so resuming training restores the same sequence of batches.

This removes permanently fixed training boundaries. It does **not** make target exposure uniform: interior targets appear in more possible full windows than edge targets, and short pages contribute fewer targets per sampled row. It does not cross folios, create more independent manuscript evidence, or guarantee better validation performance. Future comparisons must record the policy and match scored-target exposure where required.

Use `configs/small_random_offsets.json` or `--window-sampling random_offsets` with the ordinary training entry point. The new configuration is an implementation example, **not** a launched scientific experiment. Old configurations default to `fixed_windows`; changing a sampler on resume is rejected as a changed training configuration. Evaluation continues to score each eligible next-symbol target exactly once with the existing declared context policy.

Method precedents: [nanoGPT's training implementation](https://github.com/karpathy/nanoGPT/blob/master/train.py) samples random contiguous offsets; [Hugging Face's fixed-context perplexity guide](https://huggingface.co/docs/transformers/en/perplexity) explains how disjoint evaluation chunks reduce available history. Our adaptation preserves manuscript page boundaries. Sliding-window *evaluation* is a separate future comparison, not silently substituted for the archived metric.

## The parallel architecture campaign already addresses several proposals

The isolated episodic branch already implements prefix-only canonicalization, explicit edge-emitting state models, exact joint continuation scores, recurrent comparisons, and stricter intervention controls. Canonicalization removes arbitrary symbol names under a fixed known alphabet; it does not automatically solve homophony, changing alphabets, word boundaries, or unfamiliar process families. An HMM that predicts synthetic data well is a compact model of those observations, not evidence that a medieval reader used that mechanism.

**Identifier collision resolved:** concurrent branches independently used `EXP-0012` and `EXP-0013`. The original latent-recovery series retains those names. Episodic publications now use [EPISODIC-0012](../experiments/EPISODIC-0012-results.md) and [EPISODIC-0013](../experiments/EPISODIC-0013-results.md), with results under `results/episodic-20260921/`. Their original c9 JSON/config/output identifiers remain historical. Neither registration/result series was overwritten.

The current campaign is not changed in response to this audit. Any new control or metric that could affect a scientific decision needs its own declared analysis and appropriate fresh evidence.

## Checks on tokenization, padding, and cited conclusions

Reproducible local checks and input/source hashes are in [the original audit record](../../results/AUDIT-20260921/descriptive_checks.json) and [integrated-source verification](../../results/AUDIT-20260921/descriptive_checks_integrated.json), generated by `scripts/audit_external_review.py` from train and validation only. The latter also records the new recovery-metrics dependency and reproduces identical input hashes and measurements; the original is preserved. The actual 256-wide training partition contains **811 windows**, not the attached estimate of approximately 650.

The tokenizer explicitly uses normalized EVA codepoints. A direct train/validation-only check found 184,576/26,323 text codepoints, or 184,930/26,371 tokens including page BOS/EOS. These are transcription units, not identified manuscript glyphs. `c` is followed by `h` in 8,774 of 10,571 training occurrences (83.00%) and 1,308 of 1,568 validation occurrences (83.42%). This is strong local regularity, but not determinism or a measurement of how much model capacity it consumes. The 1.839734-bit result also belongs to the two-layer compact model, not the four-layer reference.

Of 177 training pages, 145 have fewer than 2,048 codepoints; the corresponding validation count is 20/24. But pages and training blocks are different denominators: the 2,048-wide training representation has 209 blocks. Overall padded-slot fraction is 56.84%, and 85/209 blocks (40.7%) have 50–75% padding. The paper/code conclusions do not support the external audit's claim that over 80% of training examples had that padding range. Padding costs compute; the implemented attention mask excludes padded keys. RoPE here is a deterministic rotation, not a table of separately trained position embeddings.

The claimed glyph-grain explanation also misses directly relevant prior work: [Lindemann and Bowern, version 2 (2021)](https://arxiv.org/abs/2010.14697v2) report that the compositional changes they tested did not remove Voynich's unusual conditional-entropy behavior. This does not settle every possible segmentation, but it rules out presenting regrouping as an already demonstrated solution.

The supplied table's blanket “verified true” labels should be replaced with claim-specific statements:

- [Montemurro and Zanette (2013)](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0066344) measure organized word distributions and interpret them as supporting linguistic content. Our inference: that evidence does not prove meanings or exclude every structured nonsemantic generator; the paper's semantic interpretation is not decoded ground truth.
- The [Timm and Schinner publisher record](https://www.tandfonline.com/doi/abs/10.1080/01611194.2021.1911875) confirms a 2021 review in *Cryptologia* 45(5), 434–438. Full publisher text was inaccessible in this check, so the attached audit's stronger claim that the review “proved” reproduction of all relevant tests is not verified here.
- [Gassiat, Cleynen and Robin](https://arxiv.org/abs/1306.4657) require restrictions including full-rank transitions and linearly independent emissions. Identifiability from the distribution of observation triples is not recovery from only three observed tokens, nor a general guarantee for arbitrary edge-emitting or neural models.
- [Yale's catalog](https://pre1600ms.beinecke.library.yale.edu/docs/pre1600.ms408.HTM) supports 102 surviving folios and a sextuple foldout. The checked passage does not date the disappearance of every missing leaf. Material composition compatible with historical production does not independently date the ink; preserve the distinction between parchment dating and writing date.

This was a targeted verification. No claim of exhaustive historical fact-checking, new language identification, semantic proof, or decipherment follows.

Validation: 357 tests and 23 subtests passed on the isolated audit branch; targeted metric tests passed 43/43 and targeted sampler/training tests passed 46/46. An independent sampler review found no blocking defect. Twelve legacy scoring/decision functions retain byte-identical source; additive diagnostics do not read or alter historical pass rules. No new scientific training was launched by these fixes.

After integration with the episodic implementation, **471 tests plus 23 subtests passed**, including MPS, and full source/test/script lint passed. Integration removed one previously unused local RNG assignment from the gated label-free helper; actual independently seeded random draws and decision rules are unchanged. Both original descriptive measurements and episodic archives were preserved.
