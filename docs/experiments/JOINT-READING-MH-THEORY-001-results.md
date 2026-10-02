# Joint reading correction: finite qualification passed; recovery unmeasured

The single frozen finite check passed. We can correct a stochastic proposal that changes an entire reading and its used cipher entries together, without calculating the sum over every possible reading under a complete key. This is a mathematical and implementation result on fully enumerable fixtures. No trained proposal was sampled, no original source was scored, and no manuscript was read.

## What actually ran

Revision `6055bc1a5ef6bcec611092c23f9094ed784e886d` was committed, pushed and verified at exact origin before ONE invocation of `scripts.check_joint_reading_mh001 --freeze 6055bc1a5ef6bcec611092c23f9094ed784e886d` (session3730, terminal0). There was no retry, adaptive panel extension or failure marker. The [registration](JOINT-READING-MH-THEORY-001.md) and [derivation](../research/joint-reading-independence-2026-10-02.md) remain frozen.

144 panels covered36 ordered two-record binary observations ×iid/contextual source ×uniform/weighted positive action policy. Each independently enumerated all36 full two-row dictionaries and compatible source strings. Fourteen additional one-row binary observations preserved possible dead ends. These are exact rational fixtures, not trained-network or historical data.

| Check | Actual result |
| --- | ---: |
| Complete reading/used-key states, summed across panels | 928 |
| Full dictionary/reading states | 1,408 |
| Used-key joint detailed-balance equations | 9,216 |
| Full-key joint detailed-balance equations | 28,096 |
| Unused-row cancellation witnesses | 576 |
| Duplicate-unit witnesses | 256 |
| Different joint versus marginal key densities | 480 |
| Balance failures when substituting a key marginal | 18,864 |
| Balance failures when omitting the visited-row prior | 1,984 |
| Maximum floating acceptance discrepancy | 1.7763568394e−15 |

All probability laws normalized including failures; full and collapsed targets agreed exactly; every stationary distribution equation and finite minorization bound held. All importance means equaled the observation evidence, and population efficiency was at most proposal success mass.64of144 two-record panels had nonzero proposal failure mass. Success conditioning under the SAME frozen, state-independent policy was also balanced: its common normalizer cancels, while its transition clock and workload change. No universal claim that any success conditioning is invalid was made.

## Why it could help

Earlier empirical banks lacked required emission units. Swapping existing labels cannot create those units. A new complete reading-action proposal can change units and segmentation together. Keeping that reading in the inference state lets the correction use a direct source-path probability instead of constructing an expensive full-key reading lattice. The finite checks show that this shortcut can preserve the intended terminal model distribution; they do not show that a neural model proposes good changes often enough.

Wrong probability accounting is not a harmless implementation detail: the negative witnesses actually changed stationary fluxes. The key probability sums multiple readings, while the proposed state contains ONE reading. Unused rows also need the correct prior/completion factors. Our helpers refuse incomplete states, nonfinite scores, intermediate key temperatures and unsupported nonuniform-prior claims.

## What remains unresolved

The current neural trainer learns from64–224-letter corpus windows, whereas the fixed inference source has its own contextual probabilities and geometric stopping law. Its posterior can assign mass to other compatible lengths. That mismatch can create extremely large importance weights and poor mixing. Full support in exact arithmetic alone gives no useful time-to-recovery promise. Finite numerical support, actual sampled-path densities, unchanged-source direct scores, cost and equal-time recovery need separate prospective admissions.

The first1000-update neural checkpoint has improved guided-path loss but still zero exact recoveries and worse greedy completion; see the [live training record](SOURCE-ACTION-TRAIN-002-live.md). This mathematical PASS does not change that result or the remaining paired training gates. Literal reproduction and correct model probabilities cannot establish historical language or meaning: a misspecified language/channel model can still prefer a wrong reading. Mechanistic circuits and Voynich decipherment remain unqualified.

## Resources and validation

Actual finite stage0.387278wall/0.387112CPU seconds, peak host212,484,096bytes,99,807-byte result, zero paid cost. All600wall/500absoluteCPU/1536MiBhost/2MiBresult bounds held. No GPU, empirical corpus, original model probabilities, reserved author or manuscript access. Full2960tests+23subtests passed/13skips240.91s before publication; final moment assertions then17focused tests passed. Scoped three-file lint passed; whole-tree lint retains five pre-existing findings.

The [actual result](../../results/JOINT-READING-MH-THEORY-001/result.json) preserves every panel. The [read-only closure](../../results/JOINT-READING-MH-THEORY-001/closed-check001.json) verifies ten frozen file identities, receipt/resource/count/fraction arithmetic and73unchanged live-training inputs, without repeating the full checker or model calls. This is same-author qualification, not independent expert review. Initial negative fixture construction treated a key marginal repeated across readings as a joint candidate law; the normalization guard caught it, and the fixture was corrected BEFORE the actual registered run. All preparation history remains in NB378.
