# Creative hypothesis sprint

2026-10-12. User-directed change: prioritize unusual manuscript hypotheses and
actual discriminating experiments over further mathematical sampler qualification.
These are proposed experiments, not results or established explanations. No new
fit, ink acquisition or manuscript score has occurred in this work block.

## Borrow mechanisms, not method names

[Genie](https://arxiv.org/abs/2402.15391) learns latent action codes alongside
video dynamics without action labels. Its abstract was refreshed today; its
large video setting does not establish that adjacent manuscript tokens are
states of one process. Our earlier WORLD_MODELS_ACTION.md already proposed
writer inverse dynamics. The new requirement is an action that reuses across
unseen token families and predicts compositions, not a target-word code.

[Gaussian splatting](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)
represents a scene using spatial components with learned covariance and a
visibility-aware renderer. Its useful inspiration here is shared components,
rendering and multiple views. A flat manuscript does not require a 3D scene.
The analogy supplies a hypothesis, not evidence about the manuscript.

The prior [Voynich structural review](https://aclanthology.org/W11-1511/),
project EXP0027/0028 copy/edit experiments, failed PARAGRAPH-CARRY-001 transport,
and earlier world-model notes prevent presenting token similarity or generic
latent actions as new discoveries. Refreshed primary abstracts/project pages
are a targeted methods check, not an exhaustive survey.

## Six genuinely different bets

| Bet | Proposed hidden process | First observable prediction | What would defeat this version |
|---|---|---|---|
| Reusable writer actions | A small vocabulary of operations transforms a reusable root/template: change one component, extend a run, switch an ending, select a variant | Operators learned on one set of roots predict withheld root/operator combinations and two-step compositions | Gains disappear when whole roots are held out or reduce to nearest-word copying |
| Rendering before letters | Named glyphs are mixtures of shared visible components; context changes their rendering | A shared, bounded stroke/component renderer predicts withheld combinations and scribe variants better than whole-glyph templates | Components merely memorize crops or destroy distinctions needed to re-render withheld ink |
| Two messages braided together | A fixed interleaving rule mixes two sequences, such as item names and operations | Separating the predetermined streams improves prediction on new leaves, and the same rule survives both readings | Length/position-matched shuffled streams do equally well or the rule needs per-page tuning |
| Position as a cipher dial | The same source unit renders differently under a clock driven by token index or physical position | A fixed low-period dial transports across leaves and accounts for variants through one shared mapping | Only flexible page-specific clocks work or position-only/copy baselines explain the gain |
| Meaning with multiple renderings | Some roots are one latent item expressed through several scribe/context-dependent channels | Transformations inferred on common roots transfer to withheld roots; forward/backward transforms agree | Pairings require guessed meanings or free root-specific exceptions |
| Document as a procedure graph | Reading order follows diagram connections, recurring labels or referenced items rather than simple left-to-right adjacency | A graph built without target text predicts masked textual links better than ordinary/local reading order | Gains need text-derived graph edges or hand-selected botanical identifications |

For rendering and multiple-rendering bets, similar-looking ink is not necessarily
the same symbol or meaning. Keep the original distinctions and uncertain cases;
do not solve the initializer problem by arbitrary glyph collapse. GC and ZL
are two readings of the same ink, not independent semantic supervision.

## First two bets to execute

**Reusable writer actions:** first use TRAIN-only clean whole-EVA words, preserve
compound units, and infer a small shared edit vocabulary on development leaves.
Withhold complete root families as well as physical leaves. Compare one-edit
copying, static frequency and character prediction, building on EXP0027/0028
instead of redoing them under a world-model label. Test operator composition
on deliberately withheld combinations. Use keyed morphological, copy/mutate,
and topic-mixture synthetic documents to distinguish compositional reuse from
easy copying. A positive result constrains how forms are made; it does not name
an operation such as “boil” without independent evidence.

**Braided streams:** fix a narrow initial portfolio before scoring: ordinary
order, odd/even token streams, and period3 streams within a line; never move
across uncertainty or excluded-locus gaps. Score only positions with comparable
available history, include a pooled model with the same parameter budget,
position-only models and matched order-shuffled controls. Train on development
leaves and predict other development leaves in both readings. Recover known
interleaving in positive controls and retain failures on ordinary/topic/copy
controls. Report every rule, rather than selecting a visually attractive winner.
This targets a different process from the failed previous-terminal transport.

These two bets are first because existing data and code permit direct tests.
The ink-component experiment needs a trustworthy crop/alignment cohort; the
procedure graph needs image relations that are annotated without text answers.
Those prerequisites must remain visible, not be replaced with invented labels.

## Operating decision

No new sampler-proof campaign is the default. Spend local compute on different
process hypotheses, compare their predictions and retire failures. Each actual
run still gets a short registration with fixed inputs, splits, controls, seeds
and resource cap; this is bookkeeping for honest experiments, not a prerequisite
to proving a general theorem. Initial runs should use existing local resources,
no paid APIs and a concrete time limit. Larger learned action/rendering models
earn compute by recovering their own planted mechanism and transferring their
predictions, rather than by looking sophisticated.

The outstanding Borg published-answer check remains an unfinished capability
check. This creative pivot does not convert its unassessed candidate into a
historical decoding success. No Voynich plaintext or semantics are claimed.
