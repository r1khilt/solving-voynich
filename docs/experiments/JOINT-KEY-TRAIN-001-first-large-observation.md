# First large fit finished; key recovery remains poor

2026-10-01. A partial campaign observation, not its completion audit.
Original freeze9053abf667fb296ef41e7eb5d9b7eedd1da94533 unchanged.
Parent session45401 observed first large child exiting0 after14,255.70outerwall
seconds, then continued into the registered small-72209 arm. No restart/resume,
extra GPU model, altered objective or new validation data.

The94,981,674-parameter large-72203 model completed20,000updates/80,000
episodes/1,840,000 row targets/23,037,420 source letters in14,254.64stagewall
seconds (3h57m35s),694.03CPU seconds,1,960,673,280peakRSSbytes. Its lowest
stored proper validation loss selected step20000:3.500445738nats/row versus
initial3.879034721. It matched54/1248 used key rows (4.33%) and0/64 complete
keys on the exposed development selection episodes. Complete-key accuracy is
limited by unused-row ambiguity; the poor USED-row score is still unresolved.

The same-seed5,423,146-parameter small model selected3.407782929nats/row and
49/1248 used rows. The large model therefore has worse proper development
loss and only five additional used-row matches in this first pair. This is a
single-seed, development-selected observation, not a general scaling law,
fresh recovery qualification or proof that larger architectures cannot help.
The stored data digest and source-letter counts agree across these paired
fits. Full episode/optimizer reconstruction remains for the registered auditor.

launch-observation-005.json independently checks CLOSED input/selected-weight/
validation/ledger/trace byte hashes, selected-checkpoint arithmetic and stored
64-key loss aggregation. Existing short CPU/MPS discrepancy7.41087e-6,
duplicate-MPS2.38419e-6 and duplicate-CPU2.66454e-15 meet registered bounds.
No new neural inference/tensor comparison/ledger replay was performed. The
four-fit completion audit is deliberately pending until all fits terminate.
All selection episode scores are exposed development data, not fresh test data.

Selected weight379,981,527bytes SHA256
1490d2793ee4dee50d4ec69fb60e0f5d6e0038d69c9e0d7b568e72e96e9d0e9b;
validation12,585bytes SHA256
d97e07e5c9a166b04fa670ebe35774e0c8a08a0ce4b085c29e4defbef485b502.
No weights/corpora enter Git; compact closed metadata and observation tracked.
Two seed72209 arms and final audit remain. The available behavior does not
qualify a cipher-recovery mechanism for interpretation. Voynich unsolved.
