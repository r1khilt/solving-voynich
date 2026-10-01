# Source-state search: one exact synthetic recovery, three failures

2026-10-01. **All16 cells and the complete audit passed execution checks.**
The width4096 guided arm recovered both64-letter passages of case0 exactly and
all19 dictionary rows those passages used. The other three ciphers remain
wrong. This is a concrete synthetic recovery under a supplied source model,
not a Voynich reading or a general recovery qualification.

Registration59595647838d7c244a7854f36d088954e98e22fc was pushed and exact remote
main verified before ONE run87986terminal0. ONE complete audit59045terminal0.
No retry/resume/extension/retuning/new training/GPU/paid calls/holdout access.
Read [frozen protocol](SOURCE-STATE-SYSTEMS-001.md) and
[method/source review](../research/source-state-lattice-2026-10-01.md).

## What changed in plain English

The previous fragment method searched for memorized long strings. The actual
source can generate strings absent from that library, so that route missed the
correct text. The new method follows the source's full smoothed transition
law, tracks a single shared dictionary across both passages, and keeps partial
readings without demanding an immediate full-key score improvement. When two
histories have exactly the same future possibilities, it adds their weights.

Keeping4096 candidates per glyph layer instead of128 brought the correct
dictionary into the returned candidates for one cipher; the unchanged reader
then reproduced its exact source text. Search quality therefore mattered in
at least this case. Correct bookkeeping alone was insufficient: the same
architecture still failed three cases, and wider search made case3's reading
worse despite a higher visited likelihood. No guarantee that more width alone
will solve them.

## Every case and arm

Edits are insertions/deletions/substitutions against both known source passages
combined. Missing readings are shown as missing, never zero errors. Dictionary
matches count only rows used by the true passages; unused rows are uncertain.

| Case / true letters | Width128 unmerged edits | Width128 merged edits | Width128 guided edits | Width4096 guided edits | Wide used-row matches | Wide exact passages |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
|0 /128|113|113|103|**0**|**19/19**|**2/2**|
|1 /128|111|111|110|99|3/18|0/2|
|2 /448|missing|missing|333|313|6/21|0/2|
|3 /448|missing|392|353|376|1/20|0/2|

Both guided arms returned readings in all four cases. Narrow guided899 versus
wide788 edits/1152 source letters is a **12.347% descriptive error reduction**;
wide used-row matches29/78 versus narrow11/78. Wide returned true used mappings
in1of4 cases; none of the narrow arms did. These repeatedly exposed fixtures
are not independent new holdouts. No accuracy threshold was registered under
this systems ID, so do not relabel this as a passed recovery gate.

Unmerged reads2/4 and merged3/4: their224 and616 total edits cover different
denominators and are not aggregate wins over either guided arm. Merging did
produce a complete candidate on case3 where the unmerged beam died, but its
392 errors are still poor. At equal width, merging changes which weighted
histories survive; this is not an equal-count comparison of source paths.

## Work, mass and limits

| Arm | Expansions | Merged arrivals | Cells with terminal keys | Exhaustive cells |
| --- | ---: | ---: | ---: | ---: |
| unmerged |137582|0|2/4|0/4|
| merged |164122|121472|3/4|0/4|
| guided |244374|255215|4/4|0/4|
| wide_guided |8330819|10729737|4/4|0/4|

Across all16 cells:8876897 expanded states,75553077 generated edges,
11106424 merged arrivals,55546984 pruned states,10311330 guide-table builds,
22395 terminal key groups. These are not counts of independent full dictionaries
or individually scored full readings. The engine can aggregate many histories.
Peak pending250794/unpruned layer168375, below all declared allocation caps.

Every call stopped at frontier exhaustion, but **every search was incomplete
because of beam pruning**. No expansion/time/generated/guide/allocation cap
was hit. Returned terminals were truncated to512 for wide cases0/1/2; their
total groups9998/11004/615. Case3 has125. Gold-mapping diagnostics cover returned
keys only; absence is not proof of absence from every internal terminal.

| Wide case | Found terminal log mass | Evidence log upper |
| --- | ---: | ---: |
|0|−319.100661|−12.821751|
|1|−418.108782|−11.841109|
|2|−1430.278699|−10.675925|
|3|−1402.653040|−9.981900|

These gaps are enormous. The conservative unresolved bounds do not certify
posterior coverage, the best dictionary, the most probable plaintext or an
interval proof. Terminal masses are sums of history events with unused rows
integrated out, not full-key posteriors. One selected partial key is filled
once by the fixed cipher-only rule and read by the original exact fixed-key
reader. No answer key enters selection or search.

Wide calls summed120.174 measured wall seconds versus narrow guided2.213,
roughly54.3× more time for the descriptive gain; not an equal-compute win.
Total run123.792802wall/123.422629CPU seconds,894828544peakRSSbytes.
Audit123.514055wall/123.390103CPU seconds,894025728peakRSSbytes.
Ignored bulk273963bytes; all3600wall/3200CPU/2GiB/256MiB caps held. Preparation
estimate2–35minutes was conservative; actual≈2.06minutes per stage. This
measurement concerns this engine/frontier, not a neural-model speed claim.

## Validation and next question

Full state/edge/pruning/guide/cache/terminal/trace replay, all selections/fills/
likelihoods/literal readings/diagnostics/config/hash/source identities and
aggregate/resource checks passed.26 alternate full-record Python DP scores
agree with native within6.821210263296962e−13, versus1e−7 limit. Realistic
state replay is the same implementation/author, not an independent agent
review. Tiny independent rational full-key/source-string enumeration validates
the probability law, different-length arrivals, nulls and truncation. Before
registration27newtests passed; final full2640tests+23subtestsPASS/13skip158.71s;
scoped lint passed, five old unrelated full-tree findings remain.

Result6901bytes SHA256
abd62ad93dbed62b1a19ed7f30123df5e1bfcc4f9c968cc7fed874599dcf4fd6.
All compact manifests/cells and audit retained; bulk paths/hashes permit local
reproduction. Original training/guide/revision/coupling/fragment freezes stay
unchanged. Original large neural fit continues separately; all-four training
audit is still pending. No successful neural mechanism claim follows here.

The next useful question is **where the three correct hypotheses are lost**:
before adequate repeated-symbol evidence, through the iid future surrogate,
or through ranking/truncation after arrival. A separately registered known-answer
pruning diagnostic can discriminate those possibilities without changing this
run. Larger/diverse frontiers can then be compared using the measured costs.
See [post-outcome next-design note](../research/after-source-state-2026-10-01.md).
