# BLIND-CHANNEL-DEV-001: discover units under a fixed computational budget

Registered2026-09-27 before source-selection scores, generated ciphertext,
fitting or recovery metrics. **Exposed engineering development, not sealed
confirmation.** This implements the first A/B portion of the
[blind-channel design](../research/blind-channel-recovery-design.md); no
stateful family-C examples, final authors or manuscript are used.

## Question and assumptions

Can exact normalized channel search discover a reusable encoding without a
true unit dictionary, a legal parse lattice, table membership or role labels?
If it fails, compare with the gold-channel decoder to separate source-model
adequacy from search failure. A supplied Latin source alphabet/model, complete
glyph inventory, record boundaries, computational bounds and approximate
source-record length remain assistance. Latin is the sole candidate; this is
not a language-identification test. Fixed-length source excerpts do not follow
the model's geometric stopping law exactly; this misspecification is explicit.

Prior-method review: the linked design reviews Chiang etal2010 and Ravi/Knight
2011 finite-state decipherment, Ristad/Yianilos normalized string channels,
Grünwald/Roos explicit MDL, Voynich precedents Reddy/Knight and Hauer/Kondrak,
and known identifiability limits. The [search implementation review](../research/finite-state-structure-search.md)
documents our local optimization rather than attributing an unknown-structure
recovery guarantee to those papers. NAIBBE001–003 supplied units/role grammar;
this pilot deliberately removes that largest remaining shortcut.

## Inputs, source model and splits

Use only the three pinned files in the
[development corpus manifest](../../data/manifests/blind_channel_development_corpora.json)
and its [extraction audit](../research/blind-channel-development-corpora.md).
Each author contributes its first50,000eligible letters. Estimate a positive
order0/1 character prior on Caesar P1: unigram counts+.5; order1 rows back off
to that unigram with total Dirichlet mass τ in{.25,1,4,16,64,256}.
Choose minimum bits/letter on Virgil P2, stable listed-order ties; include
order0 as the first candidate. Body boundaries reset context and are never
counted as adjacent characters. Refit the chosen configuration on P1+P2.
Virgil is verse and the other texts prose. Record every candidate score;
no cipher answer enters selection. Fix geometric stopping rho=1/225, giving
expected source length224, and source-start probabilities to the unigram.

Four positive tasks A/B×keys1/2 use disjoint Cicero blocks. Task index i=0..3
uses base=4000i; fit starts base+{0,256,512,768}, transfer starts
base+{2048,2304}, each224letters. The within-author transfer is exposed
development; it does not establish cross-author recovery. Two keys per family
are a small optimization check, not a confidence interval. Key seeds47001+101i;
shuffle seeds+1,000,000; search seeds48101+101i+null_indicator. Public seeds
make solver/answer separation procedural, not cryptographically secret.

Family A: a random one-to-one mapping from23source letters to23single glyphs.
Family B: six glyphs; unknown deterministic codebook contains all six singleton
strings and17random distinct two-glyph strings, randomly assigned to23letters.
Units are distinct but not prefix-free; ambiguous segmentations are real.
Each positive has a paired null independently shuffling the glyphs within
each fit and transfer record, preserving exact record lengths/unigrams.
These eight runs test only this weak matched null. The exact identical-output-
law/opposite-plaintext counterexample already independently proved in the
[math audit](../research/finite-state-channel-math-audit.md) remains a ninth
analytical risk control, not a new empirical task or a learned ambiguity
detector. No broad nonlanguage-rejection claim follows from four shuffles.

## Search, controls, outputs and resource limits

One fixed search per case: states(1,2), two restarts per state,1000proposals
per restart, one exact EM update for candidates with free probabilities,
120seconds cooperative deadline,128candidate-unit pool cap. Include every
declared singleton, then frequent within-record substrings up to length2.
First one-state restart uses source-only versus ciphertext unigram ranking;
remaining starts random. All eight moves and every acceptance/rejection are
recorded. State count stays fixed within a restart; incomplete searches report
which candidates actually ran. This is greedy bounded optimization, not
sampling or exhaustive search. Fixed coding context: denominator32,
maxstates2,maxemissionlength2,maxalternatives3, one source candidate,rho1/225.
Every completed candidate is evaluated with its actual quantized weights,
exact full-path marginal and actual model-code bits. Choose minimum total.

Compare gold-channel exact marginal/joint-Viterbi decoding on each positive,
with no key optimization; this identifies errors the fixed source prior and
decoder make even with the right channel. Decode transfer with unchanged
selected channels. Baseline is a proper independent-glyph/geometric model,
fit only on ciphertext: unigram+.5 rounded to positive counts summing256;
stop probability rounded to r/4096,r=1..4095 from geometric MLE. Its actual
code pays one family-selector bit,12stop bits and the positive-composition
weight rank. Channel models also pay one selector bit. Both share the given
glyph inventory; compare actual encoded probability laws, not an arbitrary
parameter penalty. Baseline is weak and does not prove natural-language status.

Report character edit rate, exact record recovery, decoded length, learned
versus oracle likelihood and errors, code/data bits, transfer likelihood
against the frozen baseline, supported/unsupported records, all search costs
and per-key differences. Unsupported prediction counts as deleting every
gold character; strict JSON uses null likelihood, not Infinity. Preserve full
predictions ignored; track compact results and hashes. A *diagnostic flag*,
fixed before scores, requires≥32bits fit two-part saving over iid baseline and
≥.05bits/glyph transfer likelihood gain. Count flags on positives/nulls and
positives with transfer CER≤10%. These engineering cutoffs are not significance
tests or final competence gates. No detector is qualified unless later fresh
controls validate it. A flattering likelihood with bad recovery fails the
purpose of this pilot.

Run at most two independent CPU workers,8GiB aggregate planning cap,120seconds
per case plus one atomic inference overrun. Eight fits nominally≤16minutes
combined arm wall time; entire pilot ceiling1CPU-hour, no paid compute/API.
Track process CPU seconds and peak RSS. Stop on inference/hash mismatch,
unexpected gold access or resource excess. Trace cap200MiB compressed per case;
all bulk ignored. Artificial896-glyph timing is0.003–0.080seconds per likelihood
pass and0.010–0.230seconds for expected counts; this is not a worst-case bound.

Freeze implementation, extraction and this protocol before generating source
scores/data. Then freeze selected source and data manifest before fitting.
Freeze selected models and independently written auditor before transfer/answer
evaluation. Run independent likelihood/max-path/edit/hash/decision audit,
update notebook and publish regardless of outcome. On failure, use the
oracle/baseline gap to choose the next intervention; do not relabel exposed
data as confirmation or silently extend the budget.
