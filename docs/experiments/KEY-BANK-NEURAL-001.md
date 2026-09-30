# KEY-BANK-NEURAL-001: source-only reranking of exposed candidate readings

2026-09-30. Exploratory development diagnostic, registered before new neural
scores or predictions. Runs separately from the frozen CONFIRM-002 campaign;
no changes to that campaign's methods, data, budgets, or success criteria.

## Question, prior work, and exact boundary

READ-002's remaining errors were mostly model preference failures: the true
tuple had key-bank support, but the large statistical source preferred another
reading. Can either of the two already trained, independently seeded neural
sources rank the **same saved candidate readings** more accurately?

[Kambhatla, Bigvand and Sarkar (2018)](https://aclanthology.org/D18-1102.pdf),
sections 2–4 inspected, use full-sequence neural language scoring inside
supplied-symbol substitution beam search. Their partial-completion heuristic
and English/Zodiac-domain training differ from this Latin unknown-unit family.
Their result motivates testing sequence scores, not an optimality or transfer
claim for our method. The [existing review](../research/unknown-unit-search-alternatives.md)
and [Reddy–Knight Voynich review](https://aclanthology.org/W11-1511.pdf) cover
the unknown-unit/transcription gap. Neural language scoring is prior art.
No architecture, new source training, cipher-informed fine-tuning, or seed
selection is introduced. Existing LATIN-SOURCE-MODEL-001 provenance and
NEURAL-READER-001 supplied-key qualification are inherited with their limits.

## Inputs and model

Use exactly the eight exposed positive cases from READ-002, all of which were
previously evaluated. No new CONFIRM-002 artifact is accessed. Retain the whole
union of each key's already saved top-eight complete two-record tuples; do
not generate more candidates, take recordwise cross-products, trim poor
candidates, insert gold, or give true lengths to the scorer. Include every
finite-weight bank key when determining whether a tuple has a consistent key.

Metadata-only counting before this registration found 104,290 distinct
case-tuples, 79,264 casewise distinct records, and 18,516,050 record characters;
maximum candidate length270. No neural score was computed during that count.
Cap each whole case at40,000tuples/20,000records; a cap failure stops the run,
with no partial-list renormalization or selective retry.

For source model m and tuple x, use

```text
score_m(x) = sum_r log q_m(x_r) + log sum_{K: encode_K(x)=Y} w_fit_stat(K).
```

The generating key is shared across records. A compatible key must encode
every complete record exactly. `w_fit_stat` is unchanged from the statistical
fit bank. This is a conditional model with a fixed learned key distribution
and a changed transfer source; it is not a refit under the neural source.
`q_m` includes the frozen geometric length law, rho=1/225, and complete
autoregressive character histories starting at BOS independently per record.
Never split a candidate into512-token source chunks. Even duplicate record
strings at two tuple positions contribute their probability twice.

Run both large neural checkpoints, seeds31103and31109, as separate equal-status
arms. Both were selected on Pliny before any decipherment use and remain
byte-for-byte unchanged. Each has7.405Mparameters. Reconstruct the original
statistical decision from the same candidate union and key masses before
neural scoring. This must match its frozen tuple and score within1e-7nats.

The old unseen-candidate bound is specific to the statistical source and
**cannot be reused** for a neural source. The new optimum is only over this
fixed candidate set. Full neural evidence and neural posterior confidence
are not computed. This experiment has no shuffled-candidate arm and cannot
establish a semantic/noise discrimination result.

## Computation and checks

One separate Mac GPU process, two CPU numerical threads, no paid APIs or new
training. Fixed batches of64; MPS float32 network forward with host float64
log-softmax/sums.37,032,100characters across both sources before audits.
Expected5–12wallminutes; prediction2400wall/1800CPU seconds, evaluation300/240,
validation300CPU seconds, total component allocation2340CPU seconds under a
separate40CPU-minute planning ceiling. Sampled peak host RSS4GiB and MPS driver
allocation2GiB, additional to CONFIRM-002's four CPU workers. These are sampled
memory guards, not OS reservations. The sandbox does not expose MPS; a
read-only device check confirmed access outside it. Use that approved device
context at launch; no silent fallback or extra retry after empirical scoring.

Exclusive attempt markers and immutable output archives. If the one process
fails, preserve all completed case/model predictions and count every missing
record as224deletion errors during later evaluation. All prediction artifacts
or failure records must be committed/pushed before evaluation reopens the old
answers. No reading of fresh CONFIRM-002 answers is permitted in this branch.

Checks before answers:

- Exact candidate inventory and original-statistical decision replay.
- Full-bank literal encoding replay for eight evenly spaced tuples plus the
  original winner and each new neural winner, compared with the bitmask matcher.
- Sixteen evenly spaced distinct record scores plus winner/runner-up records
  compared to CPU float64 forward, tolerance1e-3nats per complete record.
- Each winning path independently scored one character at a time in float64,
  matching full-forward within1e-7nats. A winner/runner reversal greater than
  2e-3nats fails the numerical gate. Smaller differences are reported.
- Complete saved-score arithmetic replay; independent Levenshtein evaluation
  only after prediction freeze. Numerical sampling is not a global floating
  point certificate for every candidate's ordering.

Artificial tests use tiny random sources, exact shared-key compatibility,
tampered inputs, ranking/ties, full-history padding beyond512characters,
stepwise numerical comparison, and full eight-case staged execution with an
answer-access guard. Run the small MPS test separately with device access.

## Prospective development criterion and interpretation

All sixteen case/model predictions must complete and pass the checks. Both
seeds must independently reduce total errors below the frozen43/3,584 baseline,
support all16records, and have no key above5%CER. Report every case and both
seeds even if one worsens. A better single seed does not pass this replication
criterion. Report full denominators, exact records, score margins, numerical
deltas, supported tuples, costs and failures. No tuned interpolation weights,
ensemble selection, or posterior-based uncertainty claim.

A pass would support testing improved source scores in a separately frozen
fresh unknown-key experiment. It would not repair or change CONFIRM-002's
outcome, prove neural global MAP, or produce a Voynich reading. A failure can
reflect source preference, missing proposal readings, or both. Gold may be
used for descriptive candidate availability after prediction freeze, never
to amend the registered candidate union.

```text
.venv/bin/python -u scripts/run_key_bank_neural001.py predict --freeze SOURCE_COMMIT
# Commit/push terminal predictions and checks before answers.
.venv/bin/python -u scripts/run_key_bank_neural001.py evaluate --freeze PREDICTION_COMMIT
```
