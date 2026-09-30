# Image uncertainty belongs inside decipherment, but supervision matters

2026-09-30. Primary-source review and a proposed next model, not an implemented
or trained system. The running fresh-key experiment remains unchanged. The new
[Borg visual resources](borg-glyph-resource-review-2026-09-30.md) make this branch
more concrete; they do not establish an unknown-key image decoder.

## What the recent evidence actually tests

[Kang et al., April 2026](https://arxiv.org/html/2604.23683v1) fine-tune TrOCR
with paired Copiale images and German plaintext: 1,269 training lines, 175
validation and 370 test lines. Generic handwriting pretraining precedes this
cipher-specific supervision. Reported test CER improves from46.10% to11.03%.
This supports visual transfer and supervised reading of a known solved system.
It does not test discovering a new key without target plaintext. The paper's
attention visualization is not a causal mechanism experiment. Its results
cannot be supplied as a Voynich competence certificate.

[Oliveros-Blanco et al., June 2026](https://arxiv.org/html/2606.27700v1) explicitly
train on the fixed Copiale substitution. Direct image decoding improves real
manuscript CER from43.0% to39.3% relative to their two-stage pipeline, while
remaining far from reliable reading. Their large synthetic corpus also uses
that known key. My assessment: the scale comparison changes corpus and
language alongside sample count, so it does not isolate a causal data-size
effect. Similarly, architecture and end-to-end training both change between
arms. Neither result demonstrates that more model capacity alone discovers
an unknown historical mapping. The April and June percentages concern
different protocols and should not be treated as a common leaderboard.

[Yin et al., 2018 preprint / ICDAR2019](https://arxiv.org/abs/1810.04297) already
study unsupervised segmentation, image clustering and cipher decoding with
pipelined and joint models. Thus joint image decipherment is prior art; the
novelty of a future implementation must not be asserted from adding a vision
encoder. This review used the abstract-level description only, without opening
potential Borg solution examples in the full paper. Exact historical baseline
reproduction still needs a solution-isolated method review.

## Proposed factorization

Keep three uncertainties separate: visible marks, the encoding rule, and the
source language. With fixed crop boundaries for the first qualified version,
write the generative model as

\[
 p(I,x,g\mid K,L,\phi)=p_L(x)\,p_K(g\mid x)
                 \prod_{j=1}^{|g|}p_\phi(I_j\mid g_j).
\]

Here `I` is the image evidence, `g` glyph identities, `x` source letters,
`K` the unknown encoding rule, and `L` the source prior. Integrating over `x,g`
preserves uncertain readings instead of committing to the recognizer's best
guess before key search. A finite source/channel and fixed observation lattice
permit exact dynamic programming in a bounded initial family. State counts and
approximation errors must be reported before broadening it. Unknown cropping,
insertions, deletions and overlapping ligatures require an additional normalized
segmentation/rendering model; they are not solved by the displayed factorization.

A discriminative recognizer supplies `q(g|I)`, not `p(I|g)`. Under a calibrated
fixed visual training distribution with positive class prior `pi(g)`, Bayes gives

\[
 p(I\mid g)=q(g\mid I)\,p(I)/\pi(g).
\]

For the *same fixed crop sequence and recognizer*, the product of `p(I_j)` is
constant when comparing keys, so observation weights may use `q/pi`. Using
`q` directly generally imports the recognition class prior again. This is an
elementary derivation, not a new theorem or a guarantee under domain shift.
If segmentation, crop counts or recognition models change, those constants
need not cancel. Such weights alone are not a normalized image code length and
must not be compared to the current glyph-iid baseline as if they were.

## Where a larger model could help

Use capacity for learning shape invariances across independently held-out
scribes and symbol systems, and for proposing keys across many newly generated
encodings. Do not supervise the visual encoder with target decoded words.
Randomize the key and glyph names separately per training episode; hold out
keys, source documents, fonts/scribes and channel families at evaluation.
An episode-level key is shared across all its lines, preventing each line from
inventing a different translation. A proposal network can accelerate exact
scoring without becoming the judge of its own fluent output.

For recognition similarity, compare held-out calibrated confusion distributions
and their effect on key recovery. Embedding cosine can be a diagnostic, but
its usefulness must survive style changes and predict actual confusions. A
shared visual prototype does not establish a shared plaintext letter.

Do not jointly reward unconstrained visual relabeling solely through language
fluency: the visual component could learn to reinterpret ink in the direction
the source model already prefers. Keep independently measured visual accuracy,
calibration and image likelihood or justified likelihood ratios in the test.

## Causal tests that would be worth the compute

1. **Known hidden keys, unfamiliar handwriting.** Compare hard transcription,
   soft evidence and oracle transcription with the same key-search budget.
   Report both symbol errors and plaintext errors. Improvement only in fluent
   output, without correct key transfer, fails the intended claim.
2. **Controlled ambiguous marks.** Replace one real crop with another verified
   glyph while holding page context fixed. Test whether the predicted change
   follows the frozen key. Compare style-only perturbations, unrelated crops
   and blank/noise evidence. Begin with known synthetic/historical answers;
   Voynich lacks the answer labels needed for this causal correctness test.
3. **Separate visual and key representations.** If a learned system qualifies,
   exchange a visual prototype or inferred episode-key state across matched
   examples. Measure intended transfers and unaffected records. Compare random,
   norm-matched and mismatched-key edits; decoded fluency and activation movement
   alone do not count as success.
4. **Unseen mechanisms and nonsemantic controls.** Include irregular homophones,
   nulls, changing states and structured gibberish, with explicit abstention and
   failed-case accounting. Do not rename poor performance on unseen families
   as evidence that the historical manuscript uses a different mechanism.

## Decision before new model training

First finish the already running fresh recovery panel. For each positive,
compare learned/oracle decoding, dictionary support floors and the same-objective
fit scores. A better oracle objective identifies a concrete missed search point;
a worse oracle objective with better reading exposes a score/recovery mismatch.
A positive dictionary-only edit floor proves exact reading is unavailable under
that learned key, even with a better source decoder. These are diagnostic
distinctions, not mutually exclusive causes or proofs of a global optimum.

Then complete the Borg image/transcription ledger and physical-leaf identities.
Only after those checks, register a bounded soft-observation control with real
uncertainty patterns and matched oracle/hard/null arms. Training a giant direct
image-to-plaintext network on a solved key would answer a different question
from the unknown-key competence we currently need. No new training or resource
allocation is authorized by this proposal itself; any run needs its concrete
budget and registration under the existing research authorization.
