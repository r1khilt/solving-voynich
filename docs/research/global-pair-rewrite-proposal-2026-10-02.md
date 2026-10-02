# A prospective global merge/split move across every record

Own derived candidate, not implemented, registered, qualified or empirically tested.
This supplements label revision by changing segmentation and repeated-letter structure.
The current all-pairs diagnostic, existing training and completed recovery remain unchanged.
Further primary-method/Voynich review and finite controls are required before attempting it.

## Why this could address a different obstruction

Labels preserve source lengths, unit boundaries and the partition of positions into equal
source letters. Regrowth can change these but often revises only a short suffix. A global
rewrite changes a repeated pattern everywhere, retaining literal encoding and a tractable
inverse. This is a potential inference operator under our existing restricted synthetic
channel, not evidence of a historical grammar, word meaning or manuscript ligature.

## A discrete partial involution with fixed triplet selection

Choose an ordered triplet (a,b,c) of DISTINCT source-row labels uniformly from all
R(R−1)(R−2) possibilities. Selection is state-independent and remains the same in reverse;
invalid triplets are self-loops. Do not normalize only over currently valid triplets without
deriving the resulting selection-probability correction.

Merge branch: a is unbound; b and c are each bound to a one-glyph code; at least one
adjacent b,c appears in the current source readings. Replace EVERY such pair in EVERY
record with a, bound to the concatenation of b's and c's two glyphs. Release b or c only
if it becomes unused. Other rows keep their assignments. Distinct b,c prevents overlapping
matches. Every replacement emits the same two observed glyphs; literal offsets at the end
stay fixed, but source length and segmentation change globally.

Split branch: a is bound to a two-glyph code and appears; b and c are either unbound or
bound to the appropriate first/second single glyph. The current readings contain NO b,c
adjacent pair anywhere. Replace EVERY a with b,c, bind b/c to those glyphs and release a.
Other assignments remain unchanged. The no-existing-pair condition is essential: otherwise
the reverse merge would collapse old pairs that this split did not introduce.

For an eligible merge, no old b,c pair remains afterward, because every one was replaced
by a distinct third label a. No new b,c adjacency can cross that inserted a. The inverse
split restores each replaced pair and the old bindings, including any released b/c rows.
For an eligible split, each newly created b,c is precisely the introduced pair; neighboring
old b/c letters cannot create a second match across its boundaries, since b≠c and the
unsplit state had no b,c pair. The reverse merge therefore restores the original reading
and visited-only key. Rejected conditions define identity behavior. This is the proposed
involution argument; exhaustive independent checks must catch any overlooked boundary case.

## Target correction and bookkeeping

Because triplet selection is uniform and branch direction is determined by the current
dictionary, the proposed MH ratio is the COMPLETE unchanged target T(new)/T(old).
It must include changed source lengths, EOS/continuation factors and visited-row prior
U^(-m), unlike a pure label swap. All transformed source contexts need rescoring. A
source-policy q ratio does not belong to this deterministic triplet transformation.
Accepted candidates would reconstruct reference policy counts for future regrowth only.

The action path must be rebuilt using the normalized-consumption record scheduler and
new unit lengths; simply editing the old interleaved action sequence would be wrong.
No old future binding may be forced into a reference prefix. The complete transformed
texts and key determine the literal path without Gold or target source lengths.

## What would falsify the proposal before large runs

Enumerate all complete readings of short multi-record binary observations with3/4 rows,
duplicate codes and unbound rows. For every ordered distinct triplet, independently apply
both branches and verify literal encoding, the full state involution, correct action
reconstruction, target ratio and transition flux/stationarity. Include repeated patterns,
record boundaries, unused-row release, pre-existing-pair rejection, asymmetric source
contexts, length changes, and deliberately omitted prior/EOS/selection corrections.

Qualify exact target arithmetic, cheap invalid-triplet checks and original-source cost
before a registered comparison. Its state-independent selection can waste most proposals;
a mathematically valid kernel may still be ineffective. Fixed mixtures with existing
regrowth and labels need explicit resource/behavior comparisons on fresh known-answer
keys and retained nulls. Nothing here licenses an open-ended chain or a decipherment claim.
