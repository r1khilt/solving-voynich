# Progress explained, and useful ways to scale the experiments

Updated 2026-09-21 after all three tracks of CAMPAIGN-0001 completed. This guide separates manuscript findings from artificial-cipher calibration. The original larger-compute proposals are preserved below as history.

## Where we actually are

We have built and checked a working research pipeline, trained small models that predict Voynich transcription, and tested interpretation methods on an artificial process whose rules we know. We have **not** decoded a Voynich word, identified its language, established which characters are filler, or recovered the historical encoding algorithm.

Those distinctions matter: a system can become good at completing strings without knowing what they mean. Our current manuscript models are specialized autocomplete systems. They receive previous transcription symbols and predict the next one; we never supplied translations.

## What the parallel campaign added

**Blind recovery:** previously, our diagnostics were taught the correct artificial hidden-state labels. This time, fitting did not receive those answers. It found useful groups on some familiar artificial cipher keys, but failed the stronger criterion on every unfamiliar cycle/branch key. A completely omitted rule family also failed. The copying control mostly recovered which symbol would be copied next; it did not recover the whole copying rule. Bigger models improved familiar-key prediction without solving transfer. See [EXP-0008 results](experiments/EXP-0008-results.md).

**Causal interpretation:** changing an earlier computation in the toy-cipher models affected later predictions, beating random and wrong-example controls across all three trained models. Merely changing the final answer could not do that. The same registered test failed across all three Voynich models. This validates an experimental tool on a known toy; it does not identify the manuscript's mechanism. See [EXP-0009 results](experiments/EXP-0009-results.md).

**Longer context and boundaries:** we trained 21 Voynich models with exactly matched training examples. None of the longer-context comparisons met the registered criterion. The 2,048-unit models used extra history when it was available, but their separately trained 256-unit counterparts still predicted better. Hiding the transcription's line/record-boundary identity hurt overall, with mixed evidence when looking only at ordinary glyphs. This does not mean the text lacks long dependencies or that we found historical line rules. See [EXP-0010 results](experiments/EXP-0010-results.md).

The campaign trained **25 new models for 41,200 updates**, processing about **219 million sampled targets** in **54.62 minutes elapsed** on the Mac. Those targets include repeated synthetic and manuscript examples; they are not 219 million independent manuscript characters. Three jobs ran together, followed by the remainder of the longest track. No paid research API or cloud training was used, and the final manuscript test remains unscored. [Visual overview](../results/CAMPAIGN-0001/overview.png), [campaign report](experiments/CAMPAIGN-0001-results.md), [agent handoff](CURRENT_STATUS.md).

Finding a pattern, controlling a prediction, and recovering a reusable rule are different achievements. Our next methods need to handle unfamiliar encodings and predict complete future sequences under explicit rules. [Candidate source reviews](research/PREDICTIVE_RULES.md) and an [explicit hidden-state model review](research/BELIEF_NET_REVIEW.md) describe possible next tests; no additional experiment is automatically scheduled.

## What the model reads

The input is the Zandbergen–Landini ZL3b transcription, not manuscript photographs. EVA is a convention for writing manuscript shapes using keyboard characters. Its Latin letters do not imply Latin language, pronunciation or meaning, and an EVA character is not always one historical glyph.

We prepared 226 nonempty modeling pages. The frozen split is 177 training pages, 24 validation pages and 25 final-test pages, with related sides/panels kept together. There are about 185,000 token positions in training, including page markers. Repeating them millions of times would not create more independent evidence.

In the model, **vocabulary** means the list of input/output token types, not a dictionary of translated words. Our vocabulary has 112 entries: 27 ordinary transcription characters, 75 atomic rare-form codes, and 10 control/spacing/uncertainty entries. For an illustrative string `qokeedy`, the ordinary portion is read as `q | o | k | e | e | d | y`. Nothing in that representation says what the string means. Rare codepoints and transcription controls account for much of the inventory; 112 is not a finding about the manuscript's historical alphabet size.

Definite spaces and source-segment boundaries are also tokens. A source segment can be a line, label, title or other locus, so a newline in our data is not always a physical manuscript line. Uncertain readings remain marked; uncertain prediction targets are excluded from the primary score. Details and reproducible counts: [data contract](research/DATA.md).

## Terminology

| Term | Meaning in this project |
| --- | --- |
| Corpus | The collection of text being studied. |
| Token / unit | One item the model reads or predicts: usually one normalized EVA character here, sometimes a marker or rare-form code. |
| Vocabulary | The menu of token types the model can represent. It does not contain translations. |
| Parameter / weight | An adjustable number inside the model. The selected model has 430,720 of them. |
| Architecture | How those numbers and computations are arranged: layers, attention heads, memory pathways, and so on. |
| Training update / step | One round of predict, measure error, and adjust weights using a batch of examples. It is not necessarily a full pass over the manuscript. |
| Batch | Several text windows processed together for an update. |
| Epoch | Roughly one pass through all training examples; our random overlapping windows make update counts and token exposure more precise than a simple epoch count. |
| Seed | A setting controlling randomized initialization/sampling. Three seeds are three independently randomized runs, not three new manuscripts. |
| Context | The preceding tokens visible for a prediction. A 256-token context here means transcription units, not 256 words. |
| Baseline | A simpler method the neural model must beat to justify its complexity. |
| Unigram | Guess from overall token frequencies, ignoring preceding text. |
| Five-gram | A frequency-table predictor using up to the previous four tokens to predict the fifth, with smoothing/backoff for sparse counts. |
| Validation set | Pages used to compare settings and choose checkpoints. Repeatedly consulting these pages makes them development data. |
| Test set / holdout | Reserved pages intended for an eventual frozen evaluation. Manuscript model scores have not been computed on these pages. |
| Overfitting | Learning details of seen examples that do not improve predictions on excluded examples. |
| Activation | Temporary numbers the model computes while processing a particular input; unlike weights, they change with the input. |
| Probe / readout | A small diagnostic fitted to predict a known label from activations. Successful reading does not prove the model uses that information causally. |
| Intervention / patch | Deliberately changing an internal computation and measuring what behavior changes. |
| Mechanistic interpretability | Trying to explain the computations that cause a model's behavior, with tests rather than attention pictures alone. |
| Synthetic | Artificial data produced by rules we wrote, so we can check answers against known truth. |
| Hidden state | An unobserved internal condition of a generating process. Our artificial generators use different state counts; none establishes that Voynich uses such states. |
| Cipher key | In the synthetic tests, a particular assignment of visible symbols to the generator's categories. A new key changes that assignment. No Voynich key has been found. |
| Blind fitting | Finding groups or predictors without using correct hidden-state labels. Observed text and following symbols are still training information. |
| Cluster / partition | A grouping of examples treated as similar. A cluster is not automatically a letter, word, state or meaning. |
| ARI | Adjusted Rand index: a chance-corrected comparison between groupings. It is not an accuracy or decipherment percentage. |
| Prediction horizon | How far ahead we inspect a prediction. In the causal tests, later horizons follow additional shared observed symbols. |
| Marginal versus joint prediction | Predicting each future symbol separately versus predicting a whole sequence and its dependencies. Success at the first does not establish the second. |
| Filler / null | In our toy, an emission that leaves its hidden state unchanged. Such symbols can still affect an observer's uncertainty; historical filler is an unresolved hypothesis. |
| Control | A comparison that can expose a misleading interpretation, such as random directions or deliberately shuffled labels. |
| Ablation | Removing/zeroing a component to measure its contribution. Damage does not by itself identify the component's function. |
| KL divergence | A measure of how different two probability distributions are. In the steering tests it measures how far predictions are from the desired donor's distribution; smaller is better. |
| MPS | The PyTorch backend that runs our computations on the Mac's GPU. |

## What bits per token means

For each actual next token, the score measures how surprised the model was. Giving the correct token probability 50% costs one bit; 25% costs two; 12.5% costs three. We average this penalty across eligible targets. A confident wrong prediction is penalized heavily. Lower is better.

The whole-validation comparisons are approximately 4.00 bits for unigram, 2.09 for five-gram and 1.84 for the selected neural model. This means the neural model assigns better probabilities to unseen-page transcription than these baselines. It does **not** mean 84% accuracy, 1.84% error or any percentage of decipherment. Values from later 192/768-target samples cannot be directly compared with the whole-validation values as if training had improved or regressed.

## What each completed experiment established

| Work | What we did | What the result means |
| --- | --- | --- |
| Preparation and literature | Reviewed prior neural Voynich work, decipherment methods and architecture ideas; acquired and audited transcription; implemented training, baselines and intervention hooks. | We have reproducible machinery and provenance. This is research infrastructure. |
| EXP-0001 | Initial 200-update reference-model pilot plus a smaller smoke run. | The pipeline works and initially beats the five-gram baseline. |
| EXP-0002 | Six architecture variants, each with three seeds: 18 runs. | All beat five-gram. Extra architectural features offer small gains under this schedule. The 430,720-parameter model is the chosen compact near-tie. |
| EXP-0003 | Remove or shuffle parts of context; intervene on attention heads. | Distant context helps. One influential head was localized, without an identified function. The small sample's weak order finding was revised by EXP-0005. |
| EXP-0004 | Train six models on a four-state artificial generator and a random-text control; fit diagnostics afterward. | Known toy state is readable at 86.25%; signal/filler at 81.58%. Impossible labels stay near chance/majority guessing. Patching probe directions fails against random control. |
| EXP-0005 | Use a larger 768-target sample; replace distant prefixes with same/other illustration-category text; approximately match symbol frequencies. | Distant order has a small consistent effect. Symbol composition explains part of category replacement effects, with residual confounding. No category-specific cipher discovered. |
| EXP-0006 | Fit supervised interventions inside frozen toy models on fresh artificial contexts. | Late interventions select the desired next category about 97% of the time. Direct output-weight directions work almost as well; early true supervision fails against shuffled supervision. We can control outputs, but have not recovered a unique algorithm. |
| EXP-0007 | Move complete equal-length transcription groups versus scrambling their characters at matched positions. | The difference is small and uncertain. Boundaries and recency are candidate explanations requiring separate tests. |
| EXP-0008 | Train four models; fit 160 blind partitions before opening correct-state diagnostics. | Familiar-key improvements fail to transfer to unfamiliar stateful keys. The copying control captures mainly the next symbol; output-only groups generally match or beat hidden-vector groups. |
| EXP-0009 | Map earlier components and prefix positions; select on discovery examples, then test later predictions independently. | All three trained toy models pass; the untrained model and all three Voynich models fail. Broad component changes carry useful information without revealing a state-only variable or transition rule. |
| EXP-0010 | Train 21 models with contexts from 256 to 2,048 units, identical sampled target exposure, and a boundary-marker ablation. | No longer-context comparison passes. Removing boundary identity hurts overall, but glyph-only effects vary. Bigger context is not a reliable improvement at this budget; historical layout rules remain unknown. |

The 86%/82%/97% numbers concern **our artificial generator**, not Voynich decoding. Its rules are a calibration challenge, not an assertion that the manuscript was generated this way. The 295 passing software tests and 23 subtests check implementation behavior, not a historical hypothesis.

The main 18 Voynich plus six synthetic model runs used 40,300 training updates and about 11.93 minutes of summed measured training time. Two earlier pilots, process startup, data preparation, evaluation, interpretation and coding/research time are additional. EXP-0006's separate 5,600 updates changed intervention bases while keeping language-model weights frozen. No paid research APIs were used.

## What this Mac can contribute

A read-only hardware inspection on 2026-09-21 UTC reports Apple M5 Pro, 18 CPU cores, 20 GPU cores and 64 GB installed memory. PyTorch 2.14 MPS is available. `torch.mps.recommended_max_memory()` reports 51.8400 GiB. This is Metal's recommended maximum working set, **not currently free memory**, a guaranteed safe allocation, or a measured training throughput. API meaning: [PyTorch documentation](https://docs.pytorch.org/docs/2.14/generated/torch.mps.recommended_max_memory.html).

The earlier bounded reference-model benchmark measured about 0.0195 seconds/update on MPS versus 0.0659 on CPU, at batch 8/context 256. It is not valid to extrapolate that timing directly to much larger models or contexts. EXP-0006's 1.32 GB peak process RSS also did not measure total GPU allocations or GPU utilization. Small memory use alone does not establish unused compute capacity.

RAM is working space; GPU throughput determines how quickly arithmetic happens. Useful scaling can mean longer sequences, many independent hypotheses, larger batches or reusable activation caches. Allocating all RAM is not itself scientific progress, and tiny-data memorization remains possible on powerful hardware.

## Original larger-experiment proposals — historical planning record

The bounded implementations are EXP-0008/0009/0010. This earlier proposal had a broader scope; not every candidate family, model size or representation below was implemented. Follow the registrations and results for actual work.

### 1. Blind recovery across many artificial cipher families

This is my first choice. Generate diverse processes: substitution/homophony, state-dependent alphabets, structured nulls, line/locus resets, copy/mutate production, plus deliberately unrecoverable controls. Train predictors on visible text. Develop an analysis procedure that infers predictive states or rules **without being given the true states/filler labels**. Use known truth only for a separately frozen evaluation. Hold out keys, parameter settings and entire generator families, and include ordinary-language controls where appropriate.

Why useful: our current success depends on knowing what labels/interventions to ask for. Voynich gives us no such labels. Reliable blind recovery on unfamiliar artificial systems would establish a much stronger reason to apply that procedure to the manuscript; failure would show exactly what needs improvement.

Why compute-heavy: many independent models/tasks, fresh generated data and repeated intervention checks. A candidate model-size range is 2–20 million parameters, with smaller controls; benchmark before fixing the upper size. Synthetic text can be regenerated without rereading the same tiny historical corpus forever. Pretraining on synthetic material would be a separately labeled branch, compared with our Voynich-only baseline rather than silently changing it.

Related primary evidence: [Shai et al., NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/8936fa1691764912d9519e1b5673ea66-Abstract-Conference.html) report representations of predictive belief states in controlled transformer tasks. [Weiss, Goldberg and Yahav, ICML 2018](https://proceedings.mlr.press/v80/weiss18a.html) extract automata from RNNs using queries/counterexamples. These are precedents, not demonstrations of our proposed blind unknown-cipher recovery. Primary abstracts checked this turn; full method review is required before implementation.

### 2. Systematic causal mapping

Across multiple trained seeds, test interventions throughout the model: layers, attention heads, positions and candidate directions. Compare realistic matched substitutions, output-weight shortcuts, random interventions and toy ground truth. Require proposed mechanisms to predict multiple future consequences, not just alter one next-symbol answer. Extra RAM can cache activations so thousands of tests reuse expensive computations.

Why useful: the current late intervention may merely manipulate the final answer. This tests whether we can identify an earlier computation with consequences that its proposed explanation predicts. Broad searching requires a frozen development/confirmation policy and multiple-comparison discipline; every favorable head is not a discovery.

### 3. Longer context and representation comparison

Compare contexts such as 256, 512, 1,024 and 2,048 units, within pages, under controlled parameter/compute budgets and common eligible scoring targets. Compare glyph/transcription-unit choices and verified layout features, plus recency/frequency controls. Retain models with shorter context and stronger simple baselines. Some pages are too short for every condition; define eligibility before seeing outcomes and report coverage.

Why useful: our current main training context is only 256 units. This could test whether models benefit from information across larger portions of a page, including layout or repeated forms. Longer contexts genuinely increase work; in ordinary dense attention, an eightfold length increase produces 64 times as many attention-pair scores per head, although total runtime and optimized memory use do not necessarily scale by 64. Our current intervention/cache path materializes attention scores; ordinary training uses PyTorch's scaled-dot-product attention path. Profile these paths separately before a large sweep.

This remains limited by one manuscript. Better scores alone would not establish a language or cipher. Separate uncertainty/alternative-transcription sensitivity could be valuable, but independent transcriptions need provenance and alignment before comparing them.

## Original proposed compute budget

Recommend a short scaling benchmark followed by a registered **8-hour local campaign** focused on proposal 1, with checkpoints and automatic stopping. Eight hours is a chosen budget, not a prediction that the whole proposed benchmark will finish. The short benchmark must determine how many settings fit that budget and whether larger models help enough to keep them.

Start with a measured workload using roughly 24–40 GiB total working memory if that improves throughput or analysis reuse, keeping the system responsive; do not allocate padding merely to hit a memory target. Monitor GPU allocator/driver statistics and system memory pressure without summing overlapping unified-memory accounting. Model size, batch, contexts and number of trials follow measured throughput. Run the same scientific comparisons on smaller models; stop scaling a branch when its extra cost provides no useful improvement.

The original explanation/planning turn launched no experiments. The subsequent authorized campaign is documented above and in its registrations/results. Future work requires method review and a bounded registration; no recurring background automation is configured.
