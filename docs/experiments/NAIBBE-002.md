# NAIBBE-002 — infer cross-table homophones instead of receiving their groups

**Pre-search registration, 2026-09-25.** An exploratory harder follow-up to [NAIBBE-001](NAIBBE-001-results.md), with a fresh later-block transfer evaluation. No new published-target search or held-out recovery score has run. The exposed first8,192 tokens remain the fit block. This is a step toward explicit decipherment, not a Voynich reading.

## Question, previous work and remaining assistance

NAIBBE-001 recovered all23 key entries and near-perfect plaintext, but true cross-table equivalence was supplied; English also recovered the key. The next question is whether source consistency can discover those equivalences across different observed codewords. Greshko's [published construction](https://github.com/greshko/naibbe-cipher) and [paper](https://doi.org/10.1080/01611194.2025.2566408) provide ground truth. Direct Voynich and search/segmentation prior-art review remains as recorded in NAIBBE-001: [Hauer/Kondrak2016](https://aclanthology.org/Q16-1006/), [Nuhn et al2014](https://aclanthology.org/D14-1184/), [Ravi/Knight2011](https://aclanthology.org/P11-1025/), [Berg-Kirkpatrick/Klein2013](https://aclanthology.org/D13-1087/), [Aldarrab/May2022](https://aclanthology.org/2022.emnlp-main.44/). None establishes this custom solver's performance or a Voynich mechanism.

Replace the single23-class key by **138 anonymous local classes**, one for each of23 within-table symbols in six tables. Each table has its own unknown23-letter permutation. Cross-table same-letter links are no longer supplied. Keep table membership, within-table unigram/prefix/suffix same-letter linkage, role grammar, token boundaries and23-letter alphabet supplied. These are still substantial oracle clues. Do not call this unrestricted discovery of the cipher family, number of symbols, segmentation or all homophones.

Identical observed glyph strings do not force anonymous classes to merge. In particular `dar` can be a unigram in either of two tables; the solver must retain both candidates without inheriting their true equality. Original table names/letters are absent from solver inputs, class/table IDs and within-table mappings are independently randomized. The public builder can reconstruct the answer, so isolation is auditable code/data separation rather than secrecy from the researcher. Search never imports the builders.

Before scoring, an independent ciphertext-only inventory found fit candidate paths rise from8,256 to8,394:202 tokens have two anonymous paths,7,990 have one.138 additional ambiguities arise from `dar`; exact last-three-class symbolic state count never exceeds4. The latent parse search therefore remains tractable. The difficulty added is138 coupled mapping entries, not a huge segmentation lattice. Three tables expose22 of23 classes in possible fit paths; the other three expose23.

## Data, splits and source model

Use the same immutable source tables, encrypted Pliny sample and exactly the same317,326-character Latin/English source LM texts as001. Builder seed6069202. Raw/derived files remain ignored; [manifest](../../data/manifests/naibbe002_data.json) records checksums and all preprocessing. Separate fit `[0,8192)` and new transfer `[18432,26624)` inputs. The old dev/transfer were exposed in001 and are excluded from this new transfer. New transfer plaintext is used mechanically for data-support validation, without printing/scoring it. It opens for recovery evaluation only after learned-key freeze. This is a different contiguous block of the same source publication, not independent historical validation.

Source prior, smoothing, alphabet normalization and latent Viterbi grammar are unchanged from001. No new neural model, plaintext corpus, dictionary, external API or target-trained LM is introduced. Alphabetic source1–4gram models use normalized fixed interpolation and additive.1 per character. As before, only legal source strings are scored; original card-deck probabilities and table multiplicities are not modeled. Best-path source scores are not ciphertext likelihoods or Bayes factors.

## Search and frozen controls

Within every table, initialize a bijection by its own class frequency rank against source-language unigram rank. A local move swaps two letter assignments within one table:6×253=1,518 possible moves. The primary search also offers253 **coordinated plaintext relabelings** that swap a letter pair in all six tables at once; this allows a consistent but globally wrong relabeling to change without temporarily corrupting cross-table consistency. This custom optimization proposal is a hypothesis, not a claimed reproduction of prior beam-search work.

Three arms: **Latin joint local+coordinated** (primary), **Latin local-only** (search-neighborhood control), **English joint local+coordinated** (wrong-language prior). All use seed920201,8 starts,7 cycles per start. Restart r applies2r random within-table swaps in each table to its frequency initial key. Each cycle alternates up to3 times between steepest improving key swaps and exact Viterbi; each climb permits at most300 accepted moves. Perturb the global incumbent by6×(2+cycle mod4) random within-table swaps between cycles. Fit objective alone selects the key. Every candidate preserves a23-letter permutation inside each table. Source/protocol and all chosen keys are separately remotely frozen before their respective evaluation stages.

Resource probe before target scoring: six independently encrypted tables on12,500 later Caesar characters, language prior on first250,000 Caesar characters,2 starts×2 cycles, took24.965s and recovered100% of occurring source characters,132/138 key types. Missing/rare symbols make full key-type recovery stricter than plaintext recovery. This only estimates throughput/competence; not the published target. Approximate56-cycle cost from that probe is6minutes per arm, with actual convergence variable. Cap20minutes search per arm, checked each climb move; a final lattice decode can finish. Three arms≤roughly1hour CPU, no paid spend, bulk outputs modest. No automatic continuation beyond those caps. No interrupted-run retry without preserving its status.

## Primary evaluation and identifiable metrics

Freeze all selected keys and fit traces, push and verify before new transfer or answer scoring. Report fit and transfer Levenshtein CER, exact token chunks, anonymous-path-ambiguous token correctness, all138 key entries and per-table counts. A table-draw trace is **not** available in the published aligned plaintext: where two codewords have the same glyph string and source letter, the original table draw is unidentifiable. Therefore do not score purported latent draw accuracy or invent true per-table emission counts.

For frequency-weighted key recovery count only ciphertext positions whose candidate paths all agree on the local class at that source-character position. Exclude unresolved class positions from this denominator and report their number. This differs from001's source-letter frequency weighting. Also report unweighted macro accuracy across all138 entries, errors and classes with zero uniquely assigned evidence. Per-table type accuracy can be established from published tables even without a draw trace. Joint parse accuracy concerns emitted plaintext chunks, not the unobserved generating table.

Known-key Latin Viterbi is the oracle. Primary competence **PASS** requires transfer CER≤.02, CER≤oracle+.005, macro key accuracy≥.95, unique-class-frequency-weighted key accuracy≥.99. All four must pass; controls cannot rescue failure. Gold-key objective comparison diagnoses search misses versus a source model preferring a wrong answer. Do not tune thresholds or optimize on transfer after observing results. One source publication gives no broad generalization guarantee; wrong-language behavior is a control, not language-ID certification.

Meaningful tests enumerate a small two-table key space and compare neighborhood/local optima, preserve all table bijections under local/global moves, replay joint returned scores, and test exclusion of unresolved table draws from frequency weights. Independent audit must reconstruct mapping compatibility, all plaintext/key metrics, legal lattice scores and gate decisions. No semantic/manuscript claim is permitted from this control.

```
PYTHONPATH=.:src .venv/bin/python scripts/build_naibbe002_data.py --preflight
PYTHONPATH=.:src .venv/bin/python scripts/run_naibbe002.py --arm latin_joint
PYTHONPATH=.:src .venv/bin/python scripts/run_naibbe002.py --arm latin_local
PYTHONPATH=.:src .venv/bin/python scripts/run_naibbe002.py --arm english_joint
# Freeze/push/verify keys, then:
PYTHONPATH=.:src .venv/bin/python scripts/evaluate_naibbe002.py --key-freeze-commit <verified-commit>
```

If this passes, withdraw within-table role linkages or table/permutation structure next, using an explicit new registration and retained source/control constraints. A failure instead prioritizes the measured search-versus-objective gap and identifiability. External historical controls and ultimately manuscript-wide consistent predictions are still required. Mechanistic/neural analysis should target demonstrated solver ambiguities, and direct-ink work remains deferred until it tests a concrete decoding prediction.
