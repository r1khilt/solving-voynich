# TEACH-0012 independent-audit hardening amendment

Date: 2026-09-23 UTC. Status: prospective post-launch audit change made while replicate 0
`raw_looped` was still training and before `report.json` existed or any TEACH-0012 confirmation
score was read.

## Reason

A blind static review found that the frozen trainer archives the per-item fields needed for every
registered diagnostic, but its top-level score table does not directly summarize all required
slices. In particular, the registration promises accuracy by sequence length, distractor count,
marker dropout, answer-row location and serialized query distance, plus wrong-answer destinations.
The existing independent auditor verified the complete rows and primary decision but did not
derive all of those diagnostics.

The static review also found verification opportunities that do not change any scientific choice:
loss-history summaries can be recomputed from the archived8,000 losses; report/benchmark/config
and status hashes can be cross-checked; resource ceilings can be rechecked; and milestone byte
counts can be verified in addition to hashes.

## Change

The independent auditor now additionally:

- records and verifies its own committed revision and source hashes separately from the launch
  source revision;
- checks the config, benchmark file, conservative projection, numerical qualification, suite item
  count, final status/decision and registered resource ceilings;
- recomputes every250-step loss-history entry from the loss archive;
- verifies milestone checkpoint byte counts as well as hashes;
- independently reconstructs semantic relation endpoint positions from tokens and
  `serialized_rows`;
- reports accuracy by length, distractor count, marker dropout, zero-based serialized answer-row
  index and query-to-answer-source token distance;
- classifies incorrect predictions as F values, G values, distractor values, query copies or
  other symbols;
- retains the original independent suite/oracle/score/decision regeneration.

Three focused tests compare the independent and native suite generators on a small development
size, verify semantic-position/wrong-destination diagnostics, and verify loss-history
reconstruction.

## Scientific invariants

No generator, model, training data, seed, optimizer, checkpoint, confirmation suite, behavioral
threshold, decision rule or saved campaign process changed. The already running process loaded
the original frozen trainer and is unaffected by this auditor edit. The auditor continues to
verify the launch-time analyzer blob recorded by the campaign, while its new self-provenance
fields identify the later committed audit code actually used on the completed report.

This amendment cannot rescue or change a primary verdict. It makes preregistered descriptive
reporting and integrity checks complete before outcome access.
