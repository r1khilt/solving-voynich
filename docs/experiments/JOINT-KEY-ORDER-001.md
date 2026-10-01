# JOINT-KEY-ORDER-001: does the first trained inverse use order beyond counts?

2026-10-01, exploratory input intervention. The prior goal turn was concrete
progress: completed/published a fixed manuscript-image review and the first
20k fit's weak recovery outcome. The training parent remains live. This separate
diagnostic is CPU-only and leaves all22 frozen training files unchanged.

Question: does selected small-72203 assign the true whole dictionary higher
probability for original source order than for a matched order-destroyed input?
Distinguish source-unit order from within-unit glyph order. This precedes neuron
or representation claims and is not a new key-recovery qualification.

Inputs: published completed small-72203 initial0 and selected20000 checkpoints,
first-fit metadata at training freeze9053abf667fb296ef41e7eb5d9b7eedd1da94533,
same64 Pliny development episodes and allowed prepared source. Hash/state/config
checks apply. These are exposed selection episodes; not fresh independent data.
The full four-fit ledger/checkpoint completion audit is pending, so this study
is explicitly provisional with respect to that audit. It independently replays
own-input CPUdouble true/old-greedy logq and old-choice deficits, not all training
ledgers. If the later training audit fails, conclusions require re-evaluation.

For every episode and original record, use gold source segmentation only to
re-encode its true one/two-glyph units and prove exact original reconstruction.
Find the shortest whole-unit prefix containing every glyph seen in that record.
Freeze that prefix. This preserves all within-record first-occurrence names and
therefore the pair's global canonicalization, including incomplete alphabets.
No gold segmentation, prefix mask, plaintext, source lengths or used-row mask
enters the encoder. Every encoder input is only the unsegmented canonical
ciphertext pair duplicated A,B,A,B as in training.

Three conditions, fixed in order for each case at each checkpoint:

1. Original input; independently replay stored MPS score references.
2. Shuffle the remaining whole encoded source units. Preserve the true unit
   inventory, source-letter counts, glyph counts and lengths; destroy suffix
   source order. Gold boundaries assist control construction only.
3. Shuffle the remaining individual glyphs using the identical prefix boundary.
   Preserve glyph counts and lengths but generally destroy codeword structure.

NumPy seed72341+zero-based episode index, independently reset per condition.
One permutation per condition, not best-of selection. Identity permutations,
short/empty mutable suffixes and ambiguous inputs are retained and counted;
report actual changed positions and frozen/mutable lengths. The entire true
canonical dictionary and diagnostic used-row set remain unchanged. The glyph
shuffle is often outside the original channel support; its effect alone cannot
prove language-order reasoning. Whole-unit shuffle is a valid same-key encoding
of a permuted source, but the permuted source is not a natural-language positive.
Both interventions preserve an ordered prefix; failure to detect an effect
does not prove the network ignores all order or learns only counts.

Exactly2checkpoints×64episodes×3conditions=384 cells. CPUfloat64 whole-key teacher
logq (complete23-row proper score) and free-running greedy key/logq are saved for
every cell. Used-row accuracy is diagnostic, never loss/input. Original references:
true-key and saved-greedy whole-logq error<=.002 nats; saved-greedy CPUargmax
deficit<=1e-4 allows numerical near ties. Report any actual CPU/MPS greedy changes.

Primary: selected mean `control-minus-original NLL` per row for the unit shuffle,
and difference versus the initial checkpoint's same contrast. Fixed descriptive
95% episode-bootstrap interval,2048 samples seed72353, paired across conditions
and checkpoints. Resampling is a description of this64-case exposed panel,
not a confidence interval for manuscript language or causal mechanisms.
Exploratory order diagnostic SUPPORTED only if selected unit contrast>=.01
nats/row, lower interval>0 and its effect exceeds initial. Report every contrast,
all denominators, per-case values, greedy shifts/used matches and glyph control,
including when NOT_SUPPORTED. This rule does not qualify recovery/circuits.

Prior work: PRIOR_WORK.md preserves Voynich predictive/saliency precedents and
their limits. [Kambhatla et al.2023](https://aclanthology.org/2023.findings-eacl.160/)
uses recurrence encoding for synthetic/historical substitution tasks;
[ALICEv1 §1/2/5](https://arxiv.org/html/2509.07282v1) studies explicit bijective
substitution and layer interpretation. Their cipher assumptions/supervision
differ from this duplicate-allowing, variable-unit Latin teacher. Attention or
row-label readability alone does not identify causal key reasoning. Our prior
proper-key memo and weak first-fit result motivate matched input controls.

Single invocation after source/protocol/test publication and exact remote check.
CPUthreads2/BLAS1, no MPS inference,1200wall/1000absoluteCPU seconds (CPU cap
includes startup),8GiB sampled hostRSS, zero paid APIs. Rough estimate2–10minutes
before measured workload, at most20minutes stage wall; no retry/extension/training.
Per-cell flushed ignored JSONL, compact metadata/results Git; preserve any partial
failure. Mechanical completion requires complete grid, original numeric gates,
construction invariants and finite probabilities. A separate read-only artifact
audit will replay construction, arithmetic, hashes and the whole grid, not repeat
model inference; same-author audit is not independent scientific confirmation.
The one metadata auditor has120stagewall/100absoluteCPU seconds and2GiB sampled
hostRSS, estimated under30seconds; it also starts with an exclusive marker.

```
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONPATH=.:src \
.venv/bin/python -u scripts/run_joint_key_order001.py --freeze <source-commit>
```
