# Condition on a small observed interval, then repair its entire interpretation

Own derivation, following the [single-occurrence proposal and primary decipherment review](occurrence-repair-2026-10-10.md).
It additionally targets wrong unit boundaries, which a single-occurrence reassignment preserves.
The active TEMPERED-READING-RECOVERY-001 controller and criteria remain unchanged.

## Conditional family, not an append or global-type edit

Choose uniformly from **all** width1/2 observed intervals across records. The number
A=sum_r(2|c_r|-1) depends only on the observed cipher, never the current reading.
If an interval endpoint cuts through an existing emission, do nothing. Otherwise its
interior currently contains one or two source letters. Keep every exterior source
position and binding fixed; release bindings whose last occurrence was in the interior.

The candidate family includes every compatible one-letter interpretation of a one-glyph
window, and every compatible one-letter/two-letter interpretation of a two-glyph window.
Rows used outside must retain their code. Interior-only rows may receive a new code.
For a repeated candidate row in a two-letter interpretation, both emitted glyphs must
agree. Candidate count is at most R+R² (552atR23). No candidate is selected using Gold,
source fluency, an approximate future guide or an externally supplied dictionary.

This permits local source equality changes, binding births/deaths/rebindings and unit
splits/merges, including repeated row pairs. Existing matching pairs elsewhere are
untouched. It avoids the involution restriction that a global split must not accidentally
merge preexisting pairs on reversal. Literal observations and exterior interpretations
remain exact. Source length may change by one; the source model must rescore consequences.

## Why uniform MH needs no proposal correction here

For a fixed observed interval, every family member has exactly the same exterior
interpretation. Releasing interior-only bindings recovers the same outside key. Thus
the compatible candidate family C is identical after any replacement. Old interpretation
is always a candidate. The component forward/reverse probabilities are both1/(A|C|).
Different intervals can lead to the same endpoint; keep those augmented components and
sum them, rather than treating distinct endpoints as equally likely proposals.

With targetT=Q(x)U^(-m), acceptance at degree k is

    min(1, [T(new)/T(old)]^(1/k)).

Q includes continuation^(source length), record stop factors and every conditional source
coefficient. The visited-row penalty remains inside the temperature power. A changed local
letter can change later source contexts; scoring only its immediate coefficient is wrong.
The implementation uses complete source rescoring and full accepted reference replay.
The reference action-path probability is unrelated to this conditional proposal law.

Selecting only currently valid endpoint intervals would change the anchor probability.
The count of such intervals can differ after a split/merge. That variant would require
its actual reverse correction; it is not silently substituted for uniform-all windows.
A learned window/replacement policy likewise needs its correctly marginalized reverse q.

For any positive component, raising detailed-balance flux to the integer k is injective.
With symmetric q and r=T(new)/T(old), exact rational verification reduces to

    T(old) q^k min(1,r) = T(new) q^k min(1,1/r).

This proves the intended root acceptance's component flux. Combined with normalized
selection and identity/rejection mass, finite-state invariance follows. An exact rational
full transition calculation additionally verifies cold stationarity. Finite acceptance
block/integer caps can abort a run; this is not stationarity of an absorbing failure state.

The additional window_conditionals.py implements a cold **heat-bath** over all T-weighted
conditional family members. It uses exact positive rational weights and a bounded raw64
integer-CDF interval sampler, including multiblock/abort tests. Source-ratio computation
cancels the common prefix and exterior records; the unchanged suffix cancels ONLY once
old and new source contexts agree. Until then, every suffix coefficient is rescored.
There is no assumed forgetting horizon or immediate-letter approximation. Continuation
and visited-row changes remain explicit. Retained choices receive full source/reference
replay. The candidate family can still require552scorings and repeated literal/prefix
validation; fewer coefficient evaluations is not yet measured runtime acceleration.

Uniform-all windows and identical conditional families give exact cold flux

    T(s) T(s') / [A sum_{z in C} T(z)],

which is symmetric. This is a cold conditional Gibbs update; a warm heat-bath would need
normalized irrational root weights and is **not** implemented. Warm replicas retain the
uniform-family MH proposal. No original-source cost or empirical recovery qualification
has yet been run for either new variant.

## Preparation versus experiment

New reading_window.py, eight artificial tests and a prospective finite checker are separate
from the running recovery method. Tests independently filter the complete reading space by
outside observed intervals, check every family/reverse map and exact cold stationarity;
seeded warm fixtures check literal/reference replay. One additional short preparation panel
checks the alternate finite checker and root-quantile replay before registration.

WINDOW-READING-THEORY-001 prospectively freezes eight constant/contextual binary panels
over3/4source rows, all complete states/windows/replacements, four exact powered flux laws,
cold MH and heat-bath stationary equations, every incremental full-target ratio,
wrong-prior/wrong-continuation/uncorrected-valid-anchor negative controls and256seeded
MH plus256heat-bath oracle-replayed steps per panel. A separate1024categorical decisions
and four forced multiblock/abort witnesses qualify the categorical law. Some tiny panel shapes were used in
preparation tests and are documented; this is a mathematical qualification, not unseen
empirical recovery. An original-source cost admission and a separately sealed new-key
comparison remain required. No competence, manuscript or neural mechanism claim follows.

This family still has limitations: a window bounded by two already committed emissions
cannot initially edit its interior if an endpoint bisects a unit. Uniform invalid anchors
are self loops. Small windows need not efficiently coordinate distant key changes, and a
good local score can remain a wrong reading. Additional global moves may be essential.
No practical mixing bound or general decipherment theorem is asserted.
