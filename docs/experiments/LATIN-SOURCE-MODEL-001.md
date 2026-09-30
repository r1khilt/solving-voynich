# LATIN-SOURCE-MODEL-001: paired data and source-family comparison

2026-09-30. Prospective bounded source-learning experiment, after the failed
SUFFIX-READER-001/002 qualifications and failed length-only repair. Those
failures remain unchanged. This is source development, not a decipherment test.

## Question and source-based rationale

Does broader Latin text improve generalization more than changing the source
architecture, and can an ordinary recurrent source outperform our finite
context model on the same source-selection author? Prediction loss is an
intermediate capability check, not evidence of Voynich language or semantics.

[Kambhatla et al. 2018](https://aclanthology.org/D18-1102.pdf), sections 2–4,
provides primary precedent for neural character source scoring in substitution
cipher search. Its 4096-unit multiplicative LSTM, much larger English data,
author-specific material and homophonic substitution task differ from this
ordinary two-layer LSTM and variable-length Latin channel. It motivates a
normalized incremental source, not a transferred accuracy claim. Ordinary
LSTM hidden/cell states support later incremental candidate scoring without
recomputing each full prefix. Distinct histories cannot be merged merely
because they have the same observed-glyph offset; later beam search will be
approximate and must separate search failure from source preference.

[Chen and Goodman 1996](https://aclanthology.org/P96-1041.pdf) motivates common
data and held-out smoothing selection; our support-filtered interpolated counts
are a bounded control, not their best estimator or modified Kneser–Ney.
[Hauer and Kondrak 2016](https://aclanthology.org/Q16-1006.pdf) and the earlier
[corpus review](../research/latin-source001-data-plan.md) document why language
scores and assumptions about units do not establish a Voynich reading. No
pretrained LatinBERT weights, masked-token pseudolikelihood, external LLM
translations, embeddings trained on reserved authors, or cipher labels enter
this comparison. The support cutoff and all training hyperparameters below
are practical choices fixed here, not claimed theoretical optima.

## Fixed inputs, population and isolation

Use [LATIN-SOURCE-001](../research/latin-source001-data-results.md), immutable
corpus manifest SHA256
`fcade8e816fac499284c103808be3b3e7c88e66711cbb6f426be089ec8ecc9c5`.
Pinned Perseus `cc843833e101992ba54549005273cfe61a33504b`; original notices,
conservative TEI handling, 64-letter overlap exclusions and catalogue-author
limitations remain. The prepared corpus is 6,817,650 training letters.

Full 512-letter training windows require segments of at least 512 letters.
**Both source families therefore use the same eligible 97,850 small or
6,497,939 large letters.** Small is a subset from Caesar/Virgil; large covers
12 training groups. Drop shorter segments from both fits; preserve all eligible
segment boundaries. This loses 2,150/319,711 letters respectively. Selection
scores all 359,276 Pliny letters, including short segments. The loader can read
only the registered training and Pliny roles. It rejects Nepos/Apuleius roles
before archive access; those archives remain unopened by this experiment.
Earlier Cicero/Sallust/Tacitus are not loaded. Training inputs, exact text hashes,
eligible segment IDs/counts and boundary identities are saved separately.

The two families have matched text, not identical likelihood weighting:
statistical counts observe each eligible segment once; neural windows sample
starts uniformly, so interior letters occur in more windows than edge letters.
The fixed compute comparison is not an equal-epoch or equal-parameter comparison.

## Neural fits and exposure

Four fits in order small/31103, large/31103, small/31109, large/31109. Each has
23 letter outputs, a separate BOS input ID 23, 96-dimensional embedding,
two ordinary LSTM layers of width 768, no dropout, and a linear softmax head:
7,405,079 parameters. Zero hidden/cell state at each window; never carry state
between records/batches. The hardware probe used this shape and found about
0.1213 seconds/update before full instrumentation; this is a planning estimate.

Pair initialization with `torch.manual_seed(seed)` and independently use
NumPy generator `seed + 100003` for training-window starts. Hash initial weights
to verify paired equality. Seeds do not guarantee bitwise reproducibility on
all devices/library releases. Sample all legal full-window starts uniformly,
16 windows × 512 targets/update. Input is BOS followed by targets except the
last. Exactly 6,000 updates: **49,152,000 target letters per fit**, 196,608,000
over four fits. No early performance stop, extra restart or budget extension.

AdamW: peak learning rate .001; betas (.9,.999); epsilon 1e-8;
weight decay .01 on all parameters; global gradient norm clipped at 1.
For steps 1..100, lr=.001*step/100. Thereafter lr=.0001 +
.5*(.001-.0001)*(1+cos(pi*(step-100)/5900)); final lr=.0001.
Training is float32 MPS, two PyTorch host threads; no CPU fallback.
Check finite loss and gradients; retain every update's loss, gradient norm,
learning rate, token exposure and elapsed time in the ignored trace.

Save weights and full Pliny scores at steps 0,100,500,1000,2000,4000,6000.
Every checkpoint is retained with hash and size; no training resumption is
supported. Select lowest Pliny bits/character, then earlier step on exact tie.
Initial step is eligible: failure to beat initialization must remain visible.

## Statistical controls

Fit an order-12 suffix source independently to each same eligible dataset.
Count all context→next-letter occurrences within segments, no cross-boundary
counts. Keep root always and nonroot contexts whose total successor count is
at least 4. Keep every successor count for each retained context, even if that
successor occurred once. This threshold preserves both prefix and suffix
closure: each long occurrence with a successor implies an occurrence with a
successor for its prefix and suffix. Fixed-width base-23 codes through length
13 fit uint64; tests compare every count against literal enumeration, including
leading zero IDs and largest alphabet symbols. Bound contexts at 1,200,000;
if exceeded, fail rather than change the cutoff.

Root p(c)=(N(c)+.5)/(N+.5*23). At each retained nonempty context h,
p(c|h)=(N(hc)+tau*p(c|suffix(h)))/(N(h)+tau). Unretained contexts back off.
Select tau from 16,64,256,1024 on the same Pliny scoring chunks; smaller tau
breaks exact ties. Save counts once and every candidate score. This is a new
support-filtered control, not a rerun or reclassification of earlier sources.

## Scoring and decision criteria

For both families, split every Pliny segment into nonoverlapping chunks of
at most 512 characters; reset source history/BOS at each chunk. Score every
letter once, with no EOS or padding included in the denominator. Evaluate
neural logits with float64 host log-softmax and summation. Weight by letters,
not by equal author/segment length. Report every checkpoint and all four
neural fits, not only the best seed. Report seed-paired data improvements,
statistical data improvement, and large neural versus large statistical.

Predeclared practical priority screen: **each** neural seed must improve at
least .10 bits/character with large versus small data, and each large neural
fit must beat the selected large statistical source by at least .02 bits/char.
These effect-size thresholds are workflow choices, not a significance test or
reader qualification. Pliny is used for selection and this screen, so scores
are development scores with selection optimism. Two seeds and one author are
not enough for a population confidence interval. No claim of general superiority
across Latin, medieval Latin, languages or cipher families.

After source selection and publication, a separately registered known-key
reader experiment on unused author allocations is needed. It must preserve
variable-unit ambiguity, measure inference approximation, compare literal
correct-path and returned-candidate source scores, and freeze predictions
before scoring accuracy. No cipher panel is created in this experiment.

## Resources, attempts and verification

All training and source code, tests, protocol, corpus/audits and hardware profile
are commit-bound before any fit. Push and verify the source revision remotely.
Commands (with the three BLAS thread variables set to 1, `PYTHONPATH=.:src`):

```
.venv/bin/python scripts/run_latin_source_model001.py train --dataset small --seed 31103 --freeze REV
.venv/bin/python scripts/run_latin_source_model001.py statistics --dataset small --freeze REV
.venv/bin/python scripts/audit_latin_source_model001.py neural --dataset small --seed 31103 --freeze REV
.venv/bin/python scripts/audit_latin_source_model001.py statistical --dataset small --freeze REV
```

Repeat only for the other predeclared dataset/seed combinations, once each.
Exclusive attempt marker precedes loading/fitting; exclusive outputs prevent
silent replacement. One GPU fit at a time, up to one separate CPU statistics
job may overlap. Expected four-fit training roughly 49 minutes before validation,
checkpoint I/O, clipping/synchronization and thermal overhead. Hard per-fit
2,400 wall/4,800 CPU seconds, four-fit wall sum at most 9,600 seconds.
Statistical fits each 1,800 wall/CPU seconds. Each of six audits 900 wall/1,200
CPU seconds. GPU driver allocation checked per update, hard 8 GiB limit;
host memory planning 8 GiB. Four complete checkpoint sets approximately 0.84 GB,
additional traces/counts kept ignored. Zero paid API/cloud spend; local energy
not metered. A cap, nonfinite value or interruption is a retained failed attempt;
never restart it under the same experiment or quietly extend its budget.

Before empirical execution: full regression and changed-file lint; synthetic
causality, incremental-state, explicit gate equations, padding/reset, count
closure, role access, schedule and checkpoint tests; random-token CPU/MPS
forward/backward/optimizer validation. After execution: all hashes, every
trace row and checkpoint inventory, paired inputs/initialization, exact selection
argmins; full selected neural validation replay from disk with changed batch
size; explicit float64 gate equations on first 128 Pliny inputs; independent
root and 48 overlapping-substring context counts plus raw-history recursive
statistical score replay. Algorithmically separate checks are not independent
researcher replication. Record any failures and limitations without omitting runs.
