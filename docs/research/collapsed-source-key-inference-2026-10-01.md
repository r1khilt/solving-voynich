# Integrate nuisance dictionary entries; search for a consistent reading

2026-10-01, project derivation after KEY-OBSERVABILITY-001. The new result is
conditional on **supplied** source text. This memo proposes a ciphertext-only
search direction; it does not implement or qualify that decoder. The existing
four-fit training schedule, proper key loss and prior failures remain unchanged.

## A different search variable

Predicting all23arbitrary dictionary rows asks a network to guess entries that
never affect the observation. The current full-key objective remains proper:
it can represent that uncertainty. But literal complete-key success is a poor
summary of reading quality when every development case has unused rows.

A more direct target is a candidate **pair of plaintext strings** X. Given X
and observed cipher C, the new oracle exactly counts the compatible assignments
to the M source rows actually used by X, leaving U=23−M rows unrestricted.
Let S(X,C) be that count, assuming exhaustive search. For literal glyph labels,
independent uniform42-way row priors and a deterministic channel,

`P(C|X) = S(X,C) * 42^U / 42^23 = S(X,C) / 42^M`.

Proof: each compatible complete dictionary has prior42^(-23), there are42^U
completions of each used dictionary, and the deterministic observation supplies
an indicator, not another language score. Unused entries integrate out exactly.
Neither the oracle nor the proposed inference may use the true key to guide
the search. X is a candidate, not a known answer, in the proposed application.

For globally first-occurrence **canonical** C using m observed glyphs, each
compatible canonical dictionary corresponds to `6!/(6−m)!` equally likely raw
relabelings, including the declared ordering of unused raw glyphs. Therefore

`P(C_canonical|X) = [6!/(6−m)!] * S(X,C_canonical) / 42^M`.

The relabeling factor is constant over X for a fixed C, so it can cancel in a
candidate-reading comparison. It cannot be omitted from absolute evidence,
cross-observation comparisons or a normalization claim. This calculation is
for the iid prior, not the existing training prior conditioned on excluding
finite validation key sets, an injective prior or another historical channel.

Under a declared normalized source prior p(X), candidate-reading posterior
mass is proportional to `p(X) * S(X,C) / 42^M` for fixed canonical C. This
retains the different key-count penalty and actual support multiplicity.
Choosing one compatible key and ignoring S is a different approximation.
Missing, partial or capped counts cannot be labeled exact marginal likelihood.

## Important checks before confusing this with decipherment

Source probabilities must include their declared length and record reset laws.
For the current teacher that means its finite segment/window/length sampling
law; an autoregressive language model with EOS or a geometric length law is a
different prior, not an exact reproduction of that teacher. Do not divide a
whole-sequence training/likelihood score by a candidate length and still claim
the original normalized posterior. The supplied language/alphabet/unit family
remain assumptions; plausible text under them is not historical identification.

Count compatibility for the **joint pair sharing one dictionary**, rather than
multiplying two independently marginalized single-record likelihoods. For a
two-row/two-glyph toy, known sources(a,a) with separate cipher records(x,y)
have zero shared-key support. Each record alone has positive support; the
independent-record product incorrectly admits the contradiction. More records
can help because they share constraints, not because duplication supplies data.

The oracle is cheap on the64supplied-source cases, but unknown X ranges over an
enormous language space. This derivation does not make exhaustive realistic
reading search cheap. Candidate proposals, a constrained beam or learned
actions could help; they require actual cost/recovery measurements and fresh
synthetic calibration. A language model can favor the wrong compatible source,
and the true source can remain outside a visited bank. Report both failures.
Do not turn a normalized posterior within a finite candidate bank into a
whole-space posterior/evidence or global optimality claim.

## Why a source-prefix action can be useful

A possible state combines a hypothesized source prefix, partial shared
dictionary, cipher offset and explicit record resets. A next-source-letter
action either checks an assigned unit, or binds a new row to the next one/two
cipher glyphs. Such transitions are exactly executable. A source model can
propose the letter; a verifier can reject inconsistent assignments immediately.
The second record must reuse first-record assignments. Unused rows are never
filled merely to claim a complete key.

This is a candidate architecture/search environment, not a trained action model
or a unique mechanistic interpretation. Finite beams can prune the correct
source, long-range choices can need block revisions, and target support can be
disconnected under local updates. Learned rewards do not replace literal
constraint checks or the declared source score. The existing complete-key and
native decoder baselines remain required controls. Broader unit inventories
and historical transcription uncertainty need separate qualification before
any manuscript use.

## Exact arithmetic performed, with explicit scope

In a three-glyph/three-row toy,12possible units per row, sourceab and canonical
cipher010, direct enumeration gives144raw complete dictionaries out of12³.
The oracle gives two used dictionaries, one unused row and relabeling factor6:
`6*2/12² = 144/12³ = 1/12`. The extra12-way unused choices cancel, rather than
requiring successful guesses.

A separate two-row/three-glyph exhaustive enumeration checks **all21canonical
outputs** generated by144raw dictionaries for sourceab. For each output, the
collapsed count times its own relabeling factor equals the directly counted
probability, and the21probabilities sum exactly to1. The shared-record
contradiction above is also checked. These use rational arithmetic and an
independent full-unit dictionary enumeration. They are mathematical fixtures,
not realistic-scale source recovery, posterior calibration or a Voynich reading.

[Manea and Schmid2019v2, §§1–3](https://arxiv.org/html/1906.06965v2) supplies
the uniform-substitution word-equation formulation. The iid counting and
canonical-orbit calculations here are project derivations and checks, not
claimed theorems from that survey. See the experiment protocol and
key-proposal-uncertainty/key-inverse-next-objectives memos for proper loss,
correlated rows, proposal laws and repair-support limits.
