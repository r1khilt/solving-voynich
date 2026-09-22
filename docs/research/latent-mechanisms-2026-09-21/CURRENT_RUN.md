# What this investigation is trying to establish

**Status: completed; both registered causal hypotheses failed.** See [JSPACE results](../../experiments/JSPACE-0001-results.md) and [neuron results](../../experiments/NEURON-0001-results.md). The original running checkpoint below is retained as history; its pending-state statements are superseded by those reports. Live progress is in `outputs/JSPACE-0001/fit-progress.json` and `outputs/JSPACE-0001/pipeline/state.json`. Both are ignored local files. Refresh those before quoting a completion count.

The earlier small model could produce somewhat better-looking reconstructions while failing to recover executable explanations. It was too weak, and the accepted copy-based explanations too permissive, to support the mechanistic claims we wanted. That failure is recorded, not discarded.

We have moved this method investigation to an existing eight-billion-parameter model that can already answer most of our factual control questions. We are testing whether its internal country representation can be changed in a precise way, and whether that change can be used by several different kinds of question.

| Test | Concrete example | What success would support |
| --- | --- | --- |
| Restricted Jacobian-lens intervention | Change the internal France/China coordinates while asking about a landmark, then inspect the resulting capital, currency, language, or continent | These fixed directions exert useful causal control in this model and panel |
| Neuron transfer across questions | Measure the internal country change between two currency questions; apply selected neuron changes to a capital question | Some country-change information transfers across query functions |
| Matched controls | Compare random changes of equal size, raw output directions, full donor states, and unchanged execution | A result depends on which internal information was changed, beyond merely disturbing the model |
| Unrelated copying | Keep an unrelated requested word intact despite the country edit | Limited nuisance preservation; this is not a complete grammar or general-capability test |

The first job measures32 selected token directions at nine depths over512 distinct articles. It has already exceeded an hour of real computation. This is a deliberately restricted adaptation of the [Jacobian-lens research](https://transformer-circuits.pub/2026/workspace/index.html), not a replication of a full-vocabulary workspace or its consciousness-related claims.

The next studies are actually queued, sequentially. They must pass a numerical check on the8B model before causal scoring. The queue stops on execution errors, records negative findings, and has fixed time/memory limits. It does not automatically tune the model until something looks good. No paid API or additional model download is involved.

## What is established so far

- The factual pilot passed its registered capability gate:28/32 indirect facts,26/32 anchored aliases,16/16 copying records. Some mistakes are answer-format violations; others are incorrect answers. These small rates do not establish general competence.
- The derivative implementation passed finite-difference and other numerical checks. The cached intervention path passed12 tests on a small model with the same architecture; its actual8B qualification remains pending.
- Independent input rendering exactly matched the69 unique pilot prompts. The final evaluations contain repeated recipient questions with different intended donors, shared facts, and only three country pairs. We will not treat every row as an independent discovery.
- Landmark country changes also change prompt length. Codebook country changes preserve length exactly. Separate family results will help assess this confound, but do not fully isolate every cause.

## What is still unknown

We do not yet know whether either registered causal test passes. Similar-looking directions or large neuron changes cannot answer that question. A failed test can reveal a poor intervention method or query-dependent representation, rather than absence of country knowledge.

Even a successful result would concern known modern concepts in a pretrained model. The manuscript has no corresponding verified semantic labels. A useful next decipherment system would still have to recover a constrained encoding process, demonstrate it on known systems, beat nonsemantic alternatives, and predict genuinely excluded manuscript evidence. Internal interpretability can help find or debug that process; it cannot supply a missing historical translation by itself.

Registrations: [JSPACE-0001](../../experiments/JSPACE-0001.md), [evaluation details](../../experiments/JSPACE-0001-evaluation.md), [NEURON-0001](../../experiments/NEURON-0001.md). Detailed derivations and limits: [causal geometry notes](CAUSAL_GEOMETRY_NOTES.md), [numerical precision audit](PRECISION_AUDIT.md).
