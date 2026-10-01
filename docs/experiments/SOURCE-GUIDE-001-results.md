# Future guidance improves the search, but still does not recover the answer

Registered freeze `a424beaf229959210cbd4c5ad76fb1767275ed02` was pushed and
exactly verified on origin/main before one run76140 (exit0). One fixed
auditor11970 (exit0) replayed all16 populations, RNG/source/fixture hashes,
ancestry, literal/scalar scores, decisions and diagnostic arithmetic.
Original model/training/particle sources were not changed. No retries or tuning.

All three prespecified exploratory clauses pass: all16 cells complete, total
edit errors fall at least10%, and best visited literal leaf improves in at
least6/8 paired cells. Actual improvement is **34.02%**, with lower edit errors
and higher best-leaf mass in **all8 pairs**. This supports further investigation
of future guidance under the restricted synthetic model; it does not establish
recovery or historical decipherment.

| Source length / record | Seed | Local edits | Guided edits | Local used-row matches | Guided used-row matches | Gold used rows | Guided best-leaf gain (nats) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 64, case0 | 75511 | 180 | 108 | 0 | 2 | 19 | 89.31 |
| 64, case0 | 75513 | 165 | 125 | 1 | 0 | 19 | 15.24 |
| 64, case1 | 75511 | 148 | 104 | 1 | 3 | 18 | 112.71 |
| 64, case1 | 75513 | 159 | 82 | 0 | 6 | 18 | 75.41 |
| 224, case2 | 75511 | 561 | 345 | 0 | 4 | 21 | 339.77 |
| 224, case2 | 75513 | 588 | 347 | 3 | 6 | 21 | 377.62 |
| 224, case3 | 75511 | 520 | 362 | 1 | 5 | 20 | 304.94 |
| 224, case3 | 75513 | 492 | 383 | 1 | 2 | 20 | 15.93 |

Each cell contains two independent reset source records sharing one key. Each
arm has2,304 true source letters across the repeated-seed grid. Local total
2,813 edits versus guided1,856; edit ratios122.09% and80.56% are NOT accuracies
(insertions can make edit distance exceed true length). Used-row matches7/156
versus28/156 are repeated-case diagnostics, not156 independent dictionaries.
One pair's key matches get worse despite better plaintext edits.

## The remaining failure is large

Both arms select **0/16 exact passages**. None of the16 finite banks contains
the complete generating reading pair or its known used-key leaf. Gold's
literal leaf beats the best visited leaf in every cell: local245.94–1066.77
nats, guided160.63–781.75nats. Therefore guidance reaches better explanations
but still misses an available vastly higher-mass explanation. This does not
prove gold is globally best or that all reading marginals have that gap.

The realized log evidence estimates also remain below one known leaf's mass
(which lower-bounds evidence): local174.48–894.86nats, guided136.88–705.78nats.
Positive-guide correction is mathematically valid; finite particles still
seriously underestimate evidence on these outcomes. That is compatible with
unbiased evidence in exact arithmetic and rare large outcomes. Neither finite
posterior frequencies nor log evidence estimates are unbiased.

Final banks contain1–5 distinct used dictionaries locally and1–3 with guidance,
and3–22 local readings /1–4 guided readings. All16 have one initial ancestor,
but initial particles have identical states; that statistic alone does NOT
establish meaningful key/trajectory diversity loss or identify its cause.
No rejuvenation move revisits an already bound row. The guide uses iid letters
and fresh unknown units per occurrence; it cannot express contextual Latin
structure or the true shared-key completion likelihood. These are plausible
limitations, not separately established causal explanations.

## Computation and checks

512 particles per call in both arms. Local cells total17.310wall seconds;
guided cells48.033seconds, **2.77times the computation**, with734,001 total
surrogate table builds (61,459–129,221 per guided call, below200k cap).
This is a matched-population comparison, NOT an equal-compute win over larger
local populations or independent restarts. Four fresh artificial keys and two
paired seeds give limited evidence; no significance/generalization claim.

Whole run66.094wall/65.982CPU seconds/865,501,184peakRSSbytes;
audit63.492wall/63.423CPU/870,907,904RSSbytes.9,107,111ignored bulk bytes.
All registered30min/1600CPU/2GiB/512MiB bounds held. The prospective5–15minute
estimate was conservative; actual full stages each took about a minute. No
GPU/neural/training/paid/new downloads/historical holdout. Statistical source
prediction did occur. Source arrays unchanged. Full2589tests+23subtestsPASS,
13skips/151.94s;50focusedPASS; scoped changed Ruff/diff hygienePASS. Five old
unrelated full-tree lint findings remain. Full-size replay uses the same
algorithm/author; independent alternate algorithms only for tiny law checks.

Compact result5,678bytes SHA256
81e8afd531d7c3b4b8404f2199688db43e97de6d20e5081f9292b934ca826a36.
All population/fixture/prediction bindings and16 ordered compact cells are
stored in results/SOURCE-GUIDE-001; ignored bulk remains in outputs/.

Next justified questions are an equal-compute restart control, a guide that
retains source context/key sharing, or valid joint trajectory/key moves that
can revise early assignments. None is launched by this report. Original
training campaign proceeds with its frozen four-fit schedule. Voynich remains
unsolved; restricted synthetic progress does not identify its historical system.
