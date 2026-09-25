# NAIBBE-003: source-only backoff selection and final-block recovery

Registered 2026-09-25, before any new source-model scores or final-block decoding.
Status: source selection and recovery not yet run. Exploratory model development
motivated by exposed NAIBBE-002; prospectively frozen source check and final
published Pliny block. No Voynich text is involved.

## Question and stop rule

Can a normalized lower-order backoff prior remove the sparse-context pathology
identified in NAIBBE-002 and improve recovery without tuning to its known key?
Run one bounded source-calibrated repair. Regardless of its outcome, do not spend
another experiment tuning to these rare letters: subsequent work must withdraw
supplied codebook/role structure or test a historical ciphertext.

## Source basis and prior applications

- [MacKay and Peto 1995](https://www.cs.toronto.edu/pub/gh/MacKay%2BPeto-1995.pdf),
  equation15, supplies the Dirichlet predictive mean. Our recursive plug-in
  lower-order distribution and source-development choice of concentration are
  not their full hierarchical inference procedure.
- [Chen and Goodman 1999](https://u.cs.biu.ac.il/~yogo/courses/mt2014/papers/chen-goodman-99.pdf),
  section2.6, gives interpolated absolute discount. Raw lower-order counts are
  used here, so this is not Kneser--Ney. Their comparative word-level results do
  not guarantee character-level or out-of-author decoding improvement.
- [Teh 2006](https://aclanthology.org/P06-1124/) motivates principled hierarchical
  backoff but averages latent seating arrangements; we do not implement that
  posterior or claim its uncertainty calibration.
- [Nuhn et al. 2014](https://aclanthology.org/D14-1184/) is relevant prior
  character-language-model homophonic decipherment. Its cipher assumptions
  differ from our supplied local code classes and exact token lattice. Earlier
  Voynich/statistical decipherment review and limits are recorded in NAIBBE-001,
  NAIBBE-002 and the research notes; no accepted Voynich decoding follows from
  language-model fit. This repair concerns an explicit synthetic-control flaw.

## Fixed inputs and source model selection

Reuse the two SHA-pinned normalized 317,326-character Caesar corpora in
`data/manifests/naibbe003_data.json`, alphabet `abcdefghilmnopqrstuvxyz`. These
source texts were already fully used by the old model; this is a new frozen
partition for selecting new priors, not acquisition of previously unseen works.
Train on [0,220000), select on [220000,268000), check once on [268000,317326).
Reset context at every split, retain all within-split contexts, fit no counts on
development/check text. Same-author adjacent blocks do not test domain transfer.

Order four throughout. Let C(hc) be raw training ngram counts, C(h) their sum
over next letters, and T(h) the number of positive next-letter counts. Unigram
probability is (C(c)+0.1)/(N+0.1K) with K=23. For higher orders:

1. Recursive Dirichlet: p(c|h)=(C(hc)+tau*p(c|suffix(h)))/(C(h)+tau), with shared
   tau in {0.25,1,4,16,64,256}.
2. Interpolated absolute discount: p(c|h)=max(C(hc)-d,0)/C(h)
   +d*T(h)*p(c|suffix(h))/C(h), d in {0.25,0.5,0.75,0.9}. Unseen histories use
   lower-order probability exactly. No modified-discount estimation or
   continuation-count transformations.
3. Unchanged legacy additive/mixed four-gram, as source-calibration baseline.

All distributions must be finite, positive and sum to one. Select separately
for Latin and English by lowest development bits/character; exact ties follow
the listed implementation order (legacy first, ascending tau then d). Evaluate
only the selected model and legacy on the source check. Report paired mean gain
and 2,000 weighted resamples of nonoverlapping 1,000-character blocks (final
326-character block included), seed920301, percentile95% interval. These are
within-work block uncertainty estimates, not independent-document intervals.

Latin source gate: a new family selected, check gain at least0.005 bits/character,
and bootstrap lower bound strictly positive. If it fails, stop cipher003 without
opening the final block or replacing the grid. English uses the same selection;
its source gate is reported but does not veto the wrong-language control.
Freeze selected configurations and independent audit before any new cipher fit.
Refit selected priors on full original source corpora for cipher inference.

## Cipher phase, conditional on Latin source gate

Inherit all138 local-class identities and six unknown23-letter permutations
from002. Fit remains published ciphertext tokens[0,8192); new transfer is the
entire remaining published block[26624,34764):8,140 tokens/12,296 gold characters.
The builder mechanically verifies plaintext support but prints only counts and
hashes; it does not choose or score a decoding. This is unused decoding material
from the same Pliny publication, not an independent author or cipher mechanism.
True within-table role linkage, group membership, grammar, alphabet and token
boundaries remain supplied. No original table-draw trace exists.

Two fits: selected Latin prior and selected English prior, each using unchanged
002 local+coordinated key/parse search, seed920201,8 starts x7 cycles, per-arm
deadline1,200 seconds. Keys then frozen remotely before either model sees
transfer or answers. The unchanged002 Latin joint key/legacy prior is an extra
baseline on the fresh block, with no refit. Gold-key oracles for all three source
priors are evaluated only after freeze. Viterbi scores are search objectives,
not marginal model evidence or posterior key probabilities.

Same primary competence gate as002: Latin transfer CER<=0.02, within0.005
absolute CER of its gold-key oracle, macro key accuracy>=0.95, uniquely assigned
class-weighted key accuracy>=0.99. Exclude unresolved class positions from the
weighted denominator and report their number. Separately report improvement,
tie or regression in CER/key count versus frozen002 baseline, with no new
success threshold fitted after observing results. Report all rare-letter errors
without redefining the prior002 FAIL. Parse/chunk accuracy and gold-vs-learned
source objective are diagnostics, not evidence of the historical mechanism.

## Resources, isolation and verification

Source grid uses two languages x11 small dense models; estimated under one
minute and <200MB per process. Two cipher arms may run concurrently, expected
5–10minutes each and bounded20minutes; no paid API or external compute.
No neural training or manuscript final holdout access. Freeze source/code/data
before scores, selection before fits, and keys/auditor before final evaluation.
Independently reconstruct conditional probabilities, lattice decoding, metrics
and frozen hashes; retain immutable outputs. Toy normalization/unseen-history,
exhaustive lattice, source-isolation and data-integrity tests must pass first.
Record failures rather than replacing trials silently. Broader tests follow
integration; preserve unrelated worktree material.
