# Shared-key particles finish, but fail synthetic text recovery

2026-10-01. Registered source/protocol/tests
`793e3996df2ed90ce268886f757936bceee6979a` were pushed and exactremote verified
before one fixed32-cell run. All32populations finished and passed literal,
ancestry and scalar probability checks. **None of the64selected passages
exactly matched its generating source.** Completion is not recovery.

## Fixed observations

The following sums pair the two inference seeds. Letters are the total true
source letters for those two runs, not independent new data. Fixtures are
artificial; repetitive records are duplicates, and forced lengths differ from
the decoder's unconditional geometric population.

| Fixture | Schedule | Particles | Edit errors / true letters | Gold-used row matches | Distinct readings in each seed |
| --- | --- | ---: | ---: | ---: | --- |
| Repetitive64 | Sequential | 512 | 228/256 | 1/2 | 10,5 |
| Repetitive64 | Sequential | 4096 | 235/256 | 0/2 | 64,26 |
| Repetitive64 | Balanced | 512 | 221/256 | 1/2 | 10,11 |
| Repetitive64 | Balanced | 4096 | 225/256 | 2/2 | 28,86 |
| Repetitive224 | Sequential | 512 | 782/896 | 2/2 | 2,4 |
| Repetitive224 | Sequential | 4096 | 787/896 | 2/2 | 19,21 |
| Repetitive224 | Balanced | 512 | 778/896 | 2/2 | 2,4 |
| Repetitive224 | Balanced | 4096 | 771/896 | 2/2 | 19,21 |
| Markov64 | Sequential | 512 | 377/256 | 0/42 | 21,8 |
| Markov64 | Sequential | 4096 | 384/256 | 0/42 | 668,43 |
| Markov64 | Balanced | 512 | 351/256 | 1/42 | 5,3 |
| Markov64 | Balanced | 4096 | 379/256 | 0/42 | 43,69 |
| Markov224 | Sequential | 512 | 1156/896 | 1/40 | 1,1 |
| Markov224 | Sequential | 4096 | 1240/896 | 0/40 | 16,42 |
| Markov224 | Balanced | 512 | 1215/896 | 0/40 | 15,9 |
| Markov224 | Balanced | 4096 | 1267/896 | 2/40 | 548,8 |

Across all32cells:10,396unit-cost edits/9,216true source letters,0/64exact
records. Edit ratios can exceed100% because predictions contain insertions
and can be longer than the original. The varied-source subset has6,369edits/
4,608letters and4/328gold-used row matches. Its larger populations give
3,270/2,304edits versus3,099/2,304for512particles. That does not prove larger
populations generally worsen inference; there are only two source fixtures,
paired seeds, and a limited decision rule. No recovery/null gate is passed.

All16varied-source cells descend from one initial particle index after
resampling. Some still contain many distinct late readings. This is a
genealogical diagnostic, not by itself proof that the posterior has one mode
or that ancestry caused the recovery failure. Finite evidence estimates vary
substantially across seeds; their expectation identity is not convergence.

## Integrity and resources

One benchmark session62473 terminal0,36.318371stage wall/36.211142CPU seconds,
920,944,640peak resident bytes. One auditor session53978 terminal0,
21.820678wall/21.791132CPU seconds,959,856,640resident bytes. Both obeyed the
600wall/500absoluteCPU/2GiB bounds. Ignored population/prediction/fixture
archives total97,002,811bytes, below1GiB. NumPy2.5.3/Python3.12.13, BLAS1.
Zero GPU/neural inference, additional training, paid use or new historical
holdout. The trained statistical source was used for prediction.

All32compact cells bind every population array and prediction bank. The
auditor regenerated all fixture inputs and RNG outcomes, checked every stored
array/shape/dtype, reconstructed terminal ancestors, checked literal emission
and independent scalar source/first-binding/stop scores, and replayed diagnostic
arithmetic. Source arrays remained unchanged. This is same-author/same-large-
algorithm audit, not independent scientific confirmation. Exact tiny full-key
enumeration tests provide an independent counting algorithm only at small scale.

Prelaunch full2,567tests plus23subtests passed/13skipped in152.15s; final31
focused tests passed. The test transport caught and fixed a tuple/string metric
interface error before empirical execution. Scoped lint/whitespace checks pass;
five earlier unrelated full-tree lint findings remain. No retries/extensions.

Compact result7,720bytes SHA256
`217513e8cb89abfa7b579f4a8cdcfa34de69efb4c0ceded42a46634471b1218e`
and the complete audit are in `results/SOURCE-PARTICLE-SYSTEMS-001/`.
Large ancestor arrays and prediction banks remain ignored with individual
checksums. The separate95M/four-fit campaign is unchanged and still live.

## Next distinction

The solver now produces complete consistent guesses for these fixtures, but
its current sampling method and decision rule are inadequate for recovery.
The result does not isolate inference error from source-prior preference or
reading ambiguity. The registered post-outcome DIAG-A will compare known
generating-leaf mass, visited-bank mass, population evidence estimates and exact
candidate-reading masses after integrating compatible used dictionaries.
It will not change this outcome or treat a fluent guess as a manuscript reading.
