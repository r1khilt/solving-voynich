# EXP-0003 results — Context matters; distant ordering remains unestablished

Completed 2026-09-20 PDT / 2026-09-21 UTC from clean analysis source `a84e46da3bb5508898acf4bac8df7ba51fea5a90`. [Registration](EXP-0003.md), [summary](../../results/EXP-0003/summary.json), [seed-42 full head interventions](../../results/EXP-0003/small-seed42.json).

## Paired context experiment

All three selected compact-model seeds used the same 192 fixed targets: eight per validation page, 24 pages, 10 physical-leaf groups. Their preceding 128 units contain no target or future text. Final manuscript test remains unscored. These sampled positions differ from the whole-validation denominator in EXP-0002; do not compare absolute loss across experiments as improvement.

| Available preceding context | Mean bits/unit across seeds | Damage against full context |
| --- | ---: | ---: |
| Full 128 units | 1.773438 | 0 |
| Last 64 | 1.826200 | +0.052763 |
| Last 16 | 1.927530 | +0.154093 |
| Last 4 | 2.048072 | +0.274635 |
| Full 128, distant 112 shuffled; last 16 unchanged | 1.775123 | +0.001685 |

Removing older context worsens average prediction for every seed. Shuffling its order has inconsistent effects: +0.009528, +0.041295, −0.045767 bits for seeds 42/43/44. In seed 44 shuffling actually improves this sampled score. Equal-leaf descriptive bootstrap intervals are retained per condition; they do not support a uniform harmful effect of distant shuffling. These are few clusters with no multiplicity correction, and trained seeds are not independent manuscripts.

**Working inference:** some benefit could come from the distant inventory/frequency of symbols or page/style context, rather than precise distant order. This is a new hypothesis suggested by the result, not a registered conclusion or proof of bag-of-symbol generation. Short ordered context is preserved, corruption/truncation changes the input distribution, and only 192 targets were sampled. Proper matched prefix replacements and conditional-frequency baselines are needed before assigning a mechanism.

## Final-position head interventions, seed 42

Effects are absolute bits per sampled target. Positive clean/mismatched patch gain improves prediction on the corrupted input. Positive zero-ablation damage worsens prediction on clean input. Layer/head labels are implementation indices, not cross-seed identities.

| Head | Clean patch gain | Mismatched patch gain | Zero-ablation damage |
| --- | ---: | ---: | ---: |
| L0H0 | +0.00861 | +0.00030 | +0.02205 |
| L0H1 | −0.00729 | −0.03087 | +0.00743 |
| L0H2 | +0.00303 | −0.00841 | +0.00515 |
| L0H3 | +0.00163 | −0.01309 | +0.00667 |
| L1H0 | −0.00262 | −0.08977 | +0.01844 |
| L1H1 | −0.00237 | −0.06709 | +0.04019 |
| L1H2 | −0.00140 | −0.06981 | +0.02430 |
| L1H3 | +0.00976 | −0.18074 | +0.13251 |

L1H3 has the largest zero-ablation effect in this exploratory sweep; its equal-leaf descriptive 95% interval is +0.08047 to +0.17189 bits. This localizes a component worth investigating, but does not tell us whether it implements copying, frequency adjustment, syntax, or anything semantic. It was selected after examining eight heads, only in one seed, and zeroing is off-distribution. No head-function assignment or unique circuit claim.

The very small clean/corrupt gap makes normalized recovery misleading, so the report preserves absolute effects. Identity and full final-residual restoration each had maximum logit error **0.0**. These controls verify intervention computation, not a scientific interpretation.

## Reproduction and next discriminating test

`scripts/run_interpretation_round.py --execute` runs the fixed profiles/head sweep, followed by independent synthetic calibration. Individual CLI commands and source are in that script and `src/voynich/context_analysis.py`. Results include exact checkpoint/data/source hashes, target coordinates, losses, all head effects and descriptive intervals. `scripts/archive_interpretation_results.py` audits and retains them.

Next test should hold length and the last 16 units fixed while replacing distant context with frequency-matched same-section versus different-section text and explicit histogram baselines. Record it as a new experiment before execution; preserve this small exploratory sample and untouched manuscript test status. No filler, language or decipherment result follows from EXP-0003.
