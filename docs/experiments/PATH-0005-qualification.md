# PATH-0005: two-hop task competence diagnosis (qualification pilot)

Registered 2026-09-23 before the analysis script scored PATH-0004 baseline rows. This is a retrospective, exploratory diagnosis on **already exposed** PATH-0004 source/donor baselines, not a fresh confirmation or a causal intervention. It cannot rescue PATH-0004's failed gate. No manuscript text or final test is read.

## Question and prior basis

PATH-0004's confirmation panel had only 10/24 eligible composed source/donor pairs, despite 12/12 unrelated-copy competence. Why? Distinguish wrong final object, wrong instruction/format, and source/donor asymmetry, and inspect whether failures cluster by lexical bundle, template, or tokenization. A pretrained chat model's next-token behavior may be sensitive to surface format and word choice; the causal-routing literature does not license treating an unreliable two-hop task as a verified circuit. Earlier PATH-0003 likewise stopped at a copy gate; PATH-0002's direct lookup had much higher competence. These are the relevant local applications. See PATH-0002/0003/0004 and the method review in PATH-0004 for circuit-tracing context. This pilot adds no new model architecture or historical hypothesis.

## Fixed inputs and unit

Use only `results/PATH-0004/{discovery,confirmation}-baselines.json`, frozen `src/voynich/workspace/path4_tasks.py`, and `outputs/PATH-0004/rendered-inputs.json` if needed. The scripts do not load the 8B model. Preserve source/donor orientation and name-query variants; **bundle**, not prompt, is the independent unit. Scores are descriptive. No split is held out: both PATH-0004 splits were exposed by the prior experiment. The raw baseline arrays are unnecessary.

## Predeclared analysis and decision

For each of the 48 composed source/donor pairs across both splits, classify each clean answer into: exact expected object, exact alternate object from the same second table, or other. The existing `answer_correct` function decides exactness. Within `other`, report literal output and manual interpretation, but do not post-hoc relabel correctness. Count failures on source and donor separately and their intersection. Tabulate by split, bundle, template, and query slot. Count `first_token_differs` among both-correct pairs. Inspect tokenized answer lengths using the frozen tokenizer only if available without the 8B model; otherwise omit this diagnostic.

The qualification finding is **surface competence failure concentrated** if at least 75% of incorrect source/donor answers in confirmation are the alternate object; otherwise report a mixed failure. The threshold is a diagnostic label only, not statistical significance. A candidate format is ready for a future mechanistic experiment only after a **fresh** panel reaches at least 18/24 both-correct and first-token-distinct composed records and at least 11/12 copy records in *each* of two disjoint six-bundle splits. No candidate can pass that gate in this CPU-only archival pilot. A future model run must predeclare prompt candidates and select on a separate calibration split, then freeze one before confirmation; do not tune on PATH-0004's confirmation words.

Budget: CPU-only, under two minutes, no network, model download, training, new inference, GPU/MPS, paid API, or manuscript final scoring. Save compact JSON and a result memo under unique PATH-0005 qualification paths. Stop if any frozen task/baseline ID or expected answer fails to reconcile.
