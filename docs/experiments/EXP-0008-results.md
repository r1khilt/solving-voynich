# EXP-0008 results — Blind partitions help familiar keys, but transfer fails

Completed 2026-09-21 from clean, published source `72f632804a162009b488e6de5d8550493670f937`, under the [preregistered design](EXP-0008.md). Four models and 160 text-only abstraction fits finished before final latent-state diagnostics. No Voynich text, manuscript final test, paid API or cloud training was used.

## Plain-English finding

The larger model learned more useful patterns for artificial encodings it had encountered during training. We then found groups in its internal activity without supplying the correct hidden states. That is a useful improvement over the earlier supervised diagnostic experiments.

However, the improvement did **not** carry reliably to unfamiliar encoding keys, and the completely omitted XOR-based family failed. Clustering the model's ordinary output probabilities was usually as good as or better than clustering its hidden activity. We therefore have a useful blind benchmark and limited predictive partitions, not a general mechanism-recovery method or a decipherment.

The comparison also caught a subtle failure: the copying control looked successful on an average across three future positions, but most of its success came from the very next symbol. Its second- and third-symbol predictions remained poor. Passing our registered average-score criterion does not establish recovery of the whole future-generating process.

## Execution and integrity

- Four complete fits: compact 405,376 parameters and large 3,220,224 parameters, each seeds 42/43, 4,000 updates per run. Total **16,000 updates / 131,072,000 sampled training targets**, with repeated exposure to the fixed synthetic training corpus.
- Twenty diagnostic datasets: four keyed/stream variants for each cycle/null, branch/null, lag-copy and IID family; four keys for the entirely omitted RRXOR family. Four models × 20 datasets × two feature representations = **160 partitions**, with K selected only by observed future text on validation.
- End-to-end elapsed time **1,320.414 seconds (22.01 minutes)** while other campaign tracks ran concurrently. Summed measured training time was approximately 20.28 minutes; data preparation and blind extraction account for the remainder. This is a measured concurrent run, not a standalone throughput benchmark.
- Peak process RSS **558,350,336 bytes (about 0.52 GiB)**. Maximum recorded training-checkpoint Metal driver allocation **1,287,356,416 bytes (about 1.20 GiB)**. The latter is sampled allocation, not a measured whole-run GPU-memory peak, and these figures should not be added as independent physical-memory totals on a unified-memory machine.
- All **320 frozen abstraction/prediction/decision files**, **83 dataset files**, four selected checkpoint hashes and source/config hashes verified. Frozen record written at 1,319.369 seconds; final diagnostics finished at 1,320.414 seconds. No fitting or model selection occurs after that record in the registered runner.
- Truth is known to the generator while creating sealed files. Fitting interfaces consume visible strings/features and actual future observations only. Final true states, Bayesian probabilities and nuisance labels are loaded for scoring after every extraction artifact is frozen. File checksums are provenance checks, not earlier semantic use of the truth arrays.
- Post-run key-equivalence audit, added after the freeze without changing selection: compare every held-out cycle/branch key 8/9 with every training key 0–7, testing exact edge-emission tensors and initial priors under all 3!/4! latent-state relabelings. **Zero aliases among 32 comparisons.** A stronger visible check also finds different stationary two-symbol probability tables in **all 32 pairs**. The archive records a concrete pair of symbols and its unequal probabilities for each comparison. This finite observable difference rules out equal observable distributions for these particular stationary keys; it does not establish general hidden-state identifiability.

Reproduction command and resource environment are in the registration and archived run manifest. Full data, weights, partitions and per-context loss arrays remain in ignored local storage with digests. Compact means, per-case outcomes, all selection candidates, histories, manifests and key-equivalence audit are archived under `results/EXP-0008/`.

## Larger models improved prediction on the development mixture

These scores use the fixed LM-validation mixture, not final mechanism-recovery diagnostics. Both model sizes scored identical target strings. All four selected their final step 4,000 under the predeclared minimum-validation-loss rule.

| Model | Seed | Validation bits per next symbol | Training seconds |
| --- | ---: | ---: | ---: |
| Compact | 42 | 3.021789 | 130.548 |
| Compact | 43 | 3.028753 | 136.802 |
| Large | 42 | 2.917309 | 472.451 |
| Large | 43 | 2.923561 | 477.027 |

Mean compact **3.025271**, large **2.920435**, an improvement of **0.104836 bits** on this particular mixture and update budget. This is evidence that additional capacity helped that prediction task; it does not rank architectures generally or establish unknown-key transfer.

## Blind partitions on familiar and unfamiliar encodings

The table reports the **larger model's residual-based** partitions, averaging both model seeds. “Bits” averages observed prediction losses at offsets 1, 2 and 3; these are three marginal predictions, not a joint future-sequence likelihood. “MAP ARI” measures agreement with the oracle's most likely hidden state after observing the context. ARI is chance corrected and invariant to state names; it is **not an accuracy percentage**. Copying uses the next copy-source symbol as its diagnostic label, not a complete process state.

| Family / evaluation | Cases | Residual bits | Last-symbol baseline bits | Residual MAP ARI | Last-symbol MAP ARI | Strong criterion passes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Cycle/null, familiar keys | 4 | 3.440747 | 3.494396 | 0.572523 | 0.219752 | 4/4 |
| Cycle/null, new keys | 4 | 3.540906 | 3.469762 | 0.154107 | 0.218941 | 0/4 |
| Branch/null, familiar keys | 4 | 3.538235 | 3.544796 | 0.526039 | 0.364285 | 1/4 |
| Branch/null, new keys | 4 | 3.595162 | 3.542960 | 0.123061 | 0.365147 | 0/4 |
| Lag copying, fresh streams | 8 | 2.980025 | 3.673479 | 0.777430 | −0.000048 | 8/8 |
| IID, fresh streams | 8 | 3.594412 | 3.664944 | 0 | −0.000665 | 0/8 |
| Omitted RRXOR family | 8 | 3.591224 | 3.665049 | 0 | 0.033656 | 0/8 |

The registered stronger criterion requires at least 0.03 bits improvement over the last-symbol predictor and at least 0.05 additional MAP ARI. All outcomes are retained. Each cycle/branch row combines only two keys and two model seeds; these are descriptive results, not four independent historical-language replications.

On familiar cycle keys, larger residuals improved substantially over compact residuals: MAP ARI **0.572523 versus 0.206484**, and prediction loss **3.440747 versus 3.526433**. Familiar branch keys showed a smaller/less consistent behavioral gain. On genuinely held-out cycle/branch keys, **neither representation at either size passed the stronger criterion in any of the 16 model–key cases**. Most measured abstraction losses were worse than the simple last-symbol conditional. This is a real transfer failure under the frozen-weight, unlabeled-adaptation protocol. It is consistent with insufficient key invariance; the test does not uniquely prove literal key memorization.

Neural next-symbol predictions also deteriorated on new encodings: the large cycle models averaged **3.204663 bits** on familiar keys and **3.530004** on unseen keys, with oracle KL increasing from **0.068190** to **0.383265 bits**. Thus the failure is not solely an unfortunate clustering choice; the frozen predictor itself is substantially less faithful to the held-out process. These one-step scores have a different denominator from the three-offset abstraction table and should not be directly compared as the same metric.

## Output-information and delayed-future controls

On familiar large-model cycle keys, output-probability clustering produced MAP ARI **0.791691** and **3.418321 bits**, beating residual clustering's 0.572523 / 3.440747. On familiar branch keys, the corresponding output figures were **0.708953 ARI / 3.510475 bits**, versus residual 0.526039 / 3.538235. This provides no evidence that the selected residual partition reveals a uniquely hidden algorithm beyond information already present near the output.

Copying passed the average-score and label-association criteria at both sizes, but the control is deliberately narrow. Larger residual partitions had observed prediction bits **1.567390 / 3.688996 / 3.683688** at offsets 1/2/3; oracle KL was **0.224354 / 2.366288 / 2.368502 bits**. The method mostly identified the symbol about to be copied, while losing information relevant to subsequent copies. The output-based partition showed the same pattern. We retain the registered pass and explicitly reject the stronger interpretation of full state-machine recovery.

No genuine new-key claim is made for copying or IID: their distributions are unchanged by alphabet permutations, so these cases are fresh-stream controls. Both representations at both sizes chose **K=1 on every IID and every RRXOR dataset**, producing MAP ARI zero. The IID outcome is an appropriate negative control. The RRXOR outcome is a negative transfer result: the pretraining mixture did not equip this analysis to discover the omitted dependency family. Direct neural RRXOR forecasting also remained far from the oracle, with large-model one-step KL **0.349060 bits**.

Independent nuisance-state associations remained very small: residual ARI ranged from **−0.001215 to 0.002959**, mean **0.000152**; forecast ARI ranged **−0.001225 to 0.002136**, mean **0.000104**. These small positive values are retained as chance-scale warnings rather than treated as recovered hidden information. The report includes per-case permutation references; they are not multiplicity-corrected significance claims.

## What this establishes and what remains open

This round establishes a working **label-free fitting and selection pipeline**, limited predictive abstractions on some familiar synthetic encodings, useful IID behavior, and clear held-out key/family failures. It does not establish semantic decoding, filler-role recovery, a minimal number of hidden states, transition-rule extraction, or a faithful causal circuit. In ambiguous HMMs the predictive belief state need not be a discrete generating state; matching K to the generator would have been the wrong target anyway.

The extraction method is deliberately limited to one final-layer representation, a fixed PCA projection and k-means. Failure cannot exclude useful information distributed across earlier layers or a continuous belief representation. Three separate future marginals also cannot express every dependency in a joint future sequence, including parity constraints. These limitations qualify the failed transfer result; they do not justify changing its registered scores after observing them.

Possible next hypotheses, not new scheduled work: expose a broader random-key distribution or a recurrence-based representation to test key invariance; require success separately at each future offset and on joint continuations; test explicit transition consistency or state-machine extraction against simple observed-text models. Any adaptive successor requires fresh held-out keys/families/contexts because these final diagnostics are now exposed. No thresholds or fits were retroactively changed in this experiment.
