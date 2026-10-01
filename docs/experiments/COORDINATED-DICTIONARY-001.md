# COORDINATED-DICTIONARY-001 — Fresh-key initialization and coordinated moves

Exploratory synthetic solver experiment. Registration is complete only after exact Git commit and remote verification, before fixture generation/fitting. **One run and one full audit; no retries, resumption, adaptive extensions, seed replacement or post-outcome retuning.** See [method and own proofs](../research/coordinated-dictionary-2026-10-01.md).

## Question and controls

Can corrected support-oriented initialization and coordinated reversible code moves recover source dictionaries/readings better than single-row moves? Three fixed arms: `prior_row` = uniform prior initialization plus row MH; `covered_row` = corrected half-prior/half-covered initialization plus row MH; `covered_coordinated` = the identical corrected initial law plus coordinated MH. The two covered arms have identical initial banks per seed. All use 32 particles, balanced observation stride4, four mutations per stage, multinomial resampling every stage and a separate final EOS stage. Target source/prior are unchanged. No IID surrogate, annealing, neural training, gold repair or extinction refill.

## Inputs, assumptions, split and identity

Use the original LATIN-SOURCE-COMPACT-001 selected order12 source: 23 letters, 1,447,724 states, immutable float64 probabilities/uint32 transitions, expected399,571,824 array bytes. Counts file `outputs/LATIN-SOURCE-COMPACT-001/large.npz`, 19,566,187 bytes, SHA256 `9769415ac460c56778a461af7646635abf5906fa726484ba84d98d4f401815f6`. Original qualified native library `outputs/NATIVE-SUFFIX-001/build/suffix_marginal.dylib`, 36,888 bytes, SHA256 `d8d0fbdeb183b292594248f95c1a56eeea843d3c3ba26a0eb05548fb7bcaf30f`; source/library/array hashes rechecked on loading. Require CENSORED-CONTEXT-001 complete qualification and audit, plus exact finite coordinated-law receipt. All inherited195 dependency paths and new transitive code/tests/docs are frozen, not just this runner.

Four fresh **development** fixtures, generation seeds82601/82602/82603/82604; each uses a uniform42^23 dictionary and two independent source strings of forced64/64/224/224 letters respectively, starting at root and sampled from the original contextual source. Cipher alphabet `ABCDEF`, all one/two-glyph units, no spaces/no noise. Generation's forced length is not the solver's length model: inference retains geometric EOSρ=1/225 and receives no true lengths, key or source strings. Sampler seeds82611/82612, shared across arms. Fit24 fixed calls in case→seed→arm order. These keys are new but generator/model choice is adaptively developed; there is no language/mechanism or historical holdout and no confirmation claim. Equal-label unidentifiable null and finite identifiable/barrier controls are in pre-run tests; the real panel tests this one restrictive generator only.

The generator serializes gold separately. Solver query contains exactly case ID, two cipher records and all-unknown partial23 dictionary. Fit and predict functions reject extra generator fields. Highest final sampled dictionary likelihood wins, first-index tie. Gold-free fixed-key Viterbi yields two readings, with literal re-encoding, independent scalar source score and native/Python marginal checks. All24 prediction archives are sealed together before gold metrics. No top-k truth selection. Gold is used only for post-seal scoring, never proposals or acceptance. Ground truth need not be identifiable under all keys/sources; exact generator recovery is an operational metric, not a uniqueness proof.

## Primary decision and diagnostics

Exploratory coordinated recovery support requires **all eight covered-coordinated and all eight covered-row calls complete**, at least one selected complete used generator key in the new arm, at least+.10 absolute used-row match fraction relative to covered-row, and at least10% lower total reading edit errors. Failed fits/readers count zero matched rows/full true message length edit errors (successful edit distance can exceed true length). Criteria fixed now, not relaxed after outcomes. Report prior-row separately to isolate initialization; all complete calls have the identical (1+4)×32×stage key-table budget, not guaranteed identical CPU.

Report all statuses, used-row matches, selected/bank complete used keys, exact records and edit errors; two-seed log-evidence spread per case; incremental ESS/maxweight; operation/proposal/acceptance counts; unique parents/keys; actual tables/native calls/nodes/edges; full archived banks/traces. Stable calibration per arm requires all8 complete, every two-seed spread≤2nats and all incremental maximum weights≤.5. Two seeds/four keys cannot support narrow uncertainty intervals or broad generalization. A living population or a correct elementary MH law alone is not recovery.

## Concrete resource and stop contract

Local CPU only; no new source fit, GPU allocation, paid API or downloads. Expect roughly0.4–0.7million key tables and5–25minutes per run/audit based on the prior403,456-table/~702s run; unknown new inventories can increase graphs. Each run/audit has hard1800s wall,1600s CPU,2GiB host peak,128MiB total ignored bulk. Alarm/cap failures create failure receipts and stop with no retry. 32MiB particle/trace envelope; existing pinned~381MiB source separate; bridge owned-work128MiB. Per-record300,000nodes/2,000,000edges, no pruning or partial score.

**Each fixed fitting call** has a separate1,000,000,000edge work allowance. `StrictCensoredNativeBridge` makes cumulative exhaustion a distinct `work_cap`, an explicitly allowed failed-call outcome here; per-record bounds yield `graph_cap`. Both fail recovery and continue only the remaining predefined calls. This explicitly differs from CONTEXTUAL-GIBBS-001's global-stop contract. Native counters can charge the one edge that trips a limit. **Global fitting campaign bound16,000,000,016edges** accommodates sixteen such one-edge overshoots; refuse to launch another call unless≥1,000,000,001remain, and stop on any global/time/memory/bulk cap. No substitution of a global abort as a graph outcome.

Each selected-key reader has up to two native and two Python graphs capped100,000nodes/400,000edges, prefix expansion50,000. Reader failures return no partial reading. Conservatively charge at most800,002 native plus800,002 Python reader edges per completed call (≤38,400,096 combined for24); this is separate from fitting's recorded edges. This upper bound includes failed attempts whose partial native graph counters are unavailable. Reader work/time/RSS remain within global process bounds. Exact integer initialization rejection cap64 per draw, no biased fallback.

Finite probability proofs use the ideal uncapped integer draw. The capped implementation aborts with probability at most736×2^−64<4×10^−17 per call under ideal byte randomness; completed-run conditioning has that explicit censoring caveat. No integer-draw failure is replaced or retried.

## Verification and commands

Before publication: focused finite/null/cap/transport tests, scoped/full-tree lint classification, complete test suite, dependency/old-freeze checks and diff hygiene. Finite qualification command `.venv/bin/python -m scripts.check_coordinated_dictionary001 --save`; not empirical corpus fitting.

After registration commit `<freeze>` is verified at origin/main:

```
.venv/bin/python -m scripts.run_coordinated_dictionary001 --freeze <freeze>
.venv/bin/python -m scripts.audit_coordinated_dictionary001
```

The single audit must replay fresh fixture generation and every native/RNG/stage/counter/archive/selection/reader/recovery/gate result; independently score every final key of every completed call using the Python contextual FixedKeyBridge (≤1e−7nats tolerance). Audit CPU/RSS/caps independently bounded. Same-author replay shares search/native code and is not independent expert review. Hash/archive/publication checks after this audit are read-only and must not rerun the empirical sampler. Record failures and actual cost, update notebook/project memory and publish the coherent checkpoint. Voynich remains unsolved unless separate decipherment evidence actually passes.
