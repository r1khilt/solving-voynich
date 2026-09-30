# KEY-SOURCE-DIAG-001 results

**The improved sources reduce reading errors but do not repair the old learned keys.** This is an exposed fixed-key diagnostic, with zero new key fits and no changed qualification.

| Source | Unchanged learned-key edits /3584 | True-key edits /3584 | Learned supported /16 | True-key exact /16 |
| --- | ---: | ---: | ---: | ---: |
| Historical old order3 | 395 | 142 | 15 | 1 |
| statistical-small | 373 | 126 | 15 | 0 |
| statistical-large | 309 | 37 | 15 | 7 |
| neural-31103 | 288 | 39 | 15 | 4 |
| neural-31109 | 296 | 35 | 15 | 6 |

All new arms use the same eight old positive keys and passages, plus all eight old glyph-shuffle controls. The historical row is the previously frozen result, not a new measurement or a pure data-size control. New small/large statistical arms share the same estimator and selection policy. The neural sources were selected on Pliny before this diagnostic, but the diagnostic itself is adaptively chosen and not fresh.

## Exact statistical key preferences

Positive gaps below mean the true key is a known better fitting candidate. Negative gaps mean the unchanged wrong key has a better exact marginal-plus-code objective. Neither establishes the global optimum.

| Key | Small: learned minus true, bits | Large: learned minus true, bits | Large code-cost gap | Large data-cost gap |
| --- | ---: | ---: | ---: | ---: |
| 1 | -5.805944 | -0.746531 | -9 | 8.253469 |
| 2 | -6.373043 | -6.017230 | -6 | -0.017230 |
| 3 | -10.046010 | -9.014312 | -9 | -0.014312 |
| 4 | 136.877878 | 291.919783 | -9 | 300.919783 |
| 5 | -6.105334 | -6.008246 | -6 | -0.008246 |
| 6 | -6.239069 | -6.049845 | -6 | -0.049845 |
| 7 | -4.296420 | -15.557127 | -12 | -3.557127 |
| 8 | 53.770759 | 112.375720 | -15 | 127.375720 |

Keys4and8 now have known better true-key candidates under the large statistical source by291.920and112.376bits. No new search was run, so these keys are still not recovered. Key7still prefers the wrong key by15.557bits. Key6has a6-bit coding advantage for the shorter learned dictionary, with only0.050bits of additional data advantage; the resulting missing singleton still makes one transfer record unsupported.

A new source therefore does not erase uncertainty about unseen assignments. Forkey6, the large statistical true-key reading is perfect, while the unchanged learned key incurs224errors from one unsupported record. No source can create a parse outside the fixed dictionary support. This is a dictionary limitation, not a new reading failure.

## Neural estimates have a different meaning

The table below compares found joint key/text scores in bits, not marginal likelihoods. Positive favors the found true-key candidate. Every pair of completion-bound intervals overlaps, so none of these differences certifies ordering of the exact joint MAP objectives.

| Key | Neural A found true minus learned | Neural B found true minus learned |
| --- | ---: | ---: |
| 1 | -0.630797 | 1.427279 |
| 2 | -6.000051 | -5.999994 |
| 3 | -9.000000 | -6.503735 |
| 4 | 332.643190 | 334.046710 |
| 5 | -5.999999 | -6.000039 |
| 6 | -6.000005 | -5.404509 |
| 7 | 21.575750 | 35.927297 |
| 8 | 132.933694 | 139.326509 |

These candidate scores motivate further investigation, especially key7, but do not justify substituting a beam score for exact evidence or declaring a recovered key.

## Validation, scope and next action

All576readings completed. Statistical backward marginal/MAP/node and path checks agree within1.48e-12nats; neural incremental/full-forward paths within5.01e-5nats. Supported readings re-encode, Boolean support checks agree, and all384positive record edits match a separate full-grid calculation. Nulls have no true plaintext; their full scores and support remain in all16case reports per source. No null was dropped.

Elapsed438.419s, host CPU271.784s, peak RSS1182957568bytes. The run exceeded the rough six-minute estimate but stayed within its fixed900second cap. Zero paid API/cloud use; no restart or key alteration.

The next pipeline needs stronger search and explicit key uncertainty. The separately implemented [shared-key decoder](../research/shared-key-mixture-decoding-2026-09-30.md) has artificial checks only and has not been applied to these cases. It cannot fix a candidate bank that omits every useful key. A subsequent empirical bank policy and fresh recovery qualification must be frozen separately. The original confirmation FAIL and later supplied-key reader PASS retain their original meanings.
