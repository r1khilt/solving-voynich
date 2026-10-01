# Source/key state lattice after literal-fragment failure

2026-10-01. Own derivation and new engineering implementation, initially
unmeasured at corpus scale. SOURCE-FRAGMENT-001's negative result remains
unchanged. This is a Markov-specific inference engine, not a neural mechanism
or a historical decipherment. See SOURCE-STATE-SYSTEMS-001 for the frozen run.

## Sources reviewed and what transfers

[Mohri 2002, Semiring Frameworks and Algorithms for Shortest-Distance Problems](https://cs.nyu.edu/~mohri/pub/jalc.pdf):
reviewed introduction/algebra definitions and selected §4 DAG/topological
passages through Corollary2/Figure3; not a complete audit of all proofs. Sums
of path products and maximum path products require different aggregation
operators. Topological processing handles all preceding arrivals in an acyclic
graph. This supplies standard algorithmic precedent, not a proof that our
chosen key/source abstraction is sufficient. That argument is given below.

[Nuhn, Schamper and Ney 2013, Beam Search for Solving Substitution Ciphers](https://aclanthology.org/P13-1154.pdf):
reviewed selected §4 search/pruning passages. Partial-key extension and finite
histogram beams are precedent for keeping competing assignments. Their
one-to-one/homophonic constraints and key-cardinality steps differ from our
duplicate-allowing one/two-glyph units and consumed-glyph DAG. Their results do
not establish completeness for our beams or recovery on historical Voynich.

[Berg-Kirkpatrick and Klein 2013, Decipherment with a Million Random Restarts](https://aclanthology.org/D13-1087.pdf):
rechecked selected §2–2.1 model/trigram-HMM/EM setup. A fixed language backbone
with separately inferred cipher parameters is relevant. Our deterministic
once-bound unit assignments, finite DAG and log-sum histories are different
from their stochastic emission fitting and GPU restart program. Their scale
does not justify an open-ended local budget.

Prior Voynich application was already reviewed in
[Hauer and Kondrak 2016](https://aclanthology.org/Q16-1006.pdf), selected artificial
vocabulary limitations and §5 Voynich assumptions, and fragment proposals in
[Hauer, Hayward and Kondrak 2014](https://aclanthology.org/C14-1218.pdf), selected
§§3–5. Neither establishes an accepted manuscript reading. The new method
removes our literal12gram library restriction; it does not validate Latin,
word boundaries, anagram assumptions or any historical encoding family.

## Exact state and probability law: own reasoning

For observed records C1,C2, a state is `(i1,i2,s1,s2,K)`: glyph offsets,
validated Markov source states, and all partial key rows. Unknown K[r]=−1;
otherwise K[r] indexes one of U=42 literal one/two-glyph units. Different rows
may share a unit. There is no ciphertext relabeling or supplied true boundary.

Choose the unfinished record with smaller consumed fraction i/len(C), using
integer cross products; ties go to the first record. The source state for a
finished record is reset to zero because it has no future emissions. A row r
with source probability p(s,r)>0 emits its fixed unit if it matches at i. If
unassigned, branch over the one/two observed glyphs and bind once with factor
1/U. Every letter has factor `(1−rho)p(s,r)`; add rho immediately when that
record reaches its observed end. Non-erasing units preclude another letter
after that end. The source-state transition is the original dense goto table.

Two histories with the identical state have the same possible next steps and
future weights. Prefix masses may therefore be added; keep a separate maximum
leaf mass for comparison. The fixed scheduler assigns a unique ordering to
each completed pair of source texts/key. Since each edge increases i1+i2 by
one or two, process total consumed glyphs topologically. Processing by number
of source letters would expand some merged states before all arrivals.

For a completed history with M used rows the collapsed key prior is U^(-M).
Unseen iid rows integrate to one. Different completed source histories are
disjoint events even when their partial-key cylinders overlap as sets of full
keys. Exhaustive terminal sums yield evidence; a terminal group is neither a
full-key posterior nor a plaintext MAP score. This implementation stores no
reading backpointer. Its optional reading comes from a declared full-key fill
and the unchanged exact fixed-key Viterbi reader.

The generic full-history provider is untouched: suffix equivalence is valid
here because this particular validated source automaton is sufficient. No
neural hidden-state equivalence or cosine-similarity merger is implied.

## Finite beams, bounds and controls

For g remaining glyphs, future letter count lies between ceil(g/2) and g. The
sum of geometric length probabilities over that interval is a conservative
upper bound after ignoring all literal/source/key constraints. Multiply these
bounds across unfinished records. Completed-record EOS is already in prefix
mass. Sum the bound over discarded and unexpanded prefix histories; their
disjoint history events avoid duplicate descendant accounting. Found terminal
mass is a lower sum. Floating arithmetic is not an interval certificate.

The unmerged control uses unique path IDs and retains the same original edge
law. A beam is applied only after every arrival to a glyph layer. The ordinary
ordering uses prefix mass times the geometric bound. Guided ordering first
retains 4×width by that bound, then uses the previous iid suffix heuristic:
root letter weights with1e−8 smoothing, bound rows fixed, unknown row units
resampled at every occurrence, backward one/two-glyph DP plus EOS, floor−2000.
That surrogate is not a valid shared-key future likelihood. Both pruning
stages lose mass explicitly; guide ranks never alter actual edge weights.

Whole-layer expansion/generation preflight caps and time stops retain the
entire unexpanded layer for unresolved bounds. An allocation or guide-work
cap is a failure. Pending-state cap excludes the popped incoming vector;
peak unpruned layer and sampled process RSS must also be reported. Terminal
output truncation is separate from already accumulated terminal evidence.

Independent tiny rational full-key/full-string enumeration validates summed
support/mass and maximum leaf mass, both schedules, zero-support controls,
different action-length arrivals, beams, guide-only ordering and truncation.
The separate Python state implementation is an additional implementation
check. Realistic full replay uses the same compiled engine/author and is not
independent scientific confirmation. The first corpus-scale benchmark fixes
both controls and a32× wider guided frontier before seeing its outcome.
No100k–1M-width extrapolation from the fragment matcher's small-stack speed.
