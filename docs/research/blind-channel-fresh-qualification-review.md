# Independent review: fresh variable-unit recovery qualification

2026-09-29. Design review before final-author acquisition, key generation or
empirical scoring. The proposed experiment is a useful next qualification of
a frozen decoder under known Latin and a restricted channel family. It does
not yet qualify language identification, stateful channels, historical Borg,
or Voynich decipherment. Approval of the design is conditional on the data
and runner checks below; no runner has been approved by this document yet.

Reviewed the current project memory, NB-216–218, [the research protocol](PROTOCOL.md),
[DEV003](../experiments/BLIND-CHANNEL-DEV-003.md),
[DEV004](../experiments/BLIND-CHANNEL-DEV-004.md),
[DEV005](../experiments/BLIND-CHANNEL-DEV-005.md), their reported conclusions,
the original development builder, and the current search/refinement APIs.
No final-author text, fresh key, answer, or empirical candidate was accessed.
A second read-only reviewer independently checked the original B generator
and confirmed the segmentation ambiguity described below.

## Why this experiment, and what is fixed

DEV003–005 reused two exposed keys while changing search, source context and
dictionary refinement. Their progress motivates a frozen fresh test, not
another repair of those cases. DEV005's agreement with the true-key transfer
decoder did not recover every literal row or establish that its remaining
errors were unavoidable.

The adopted panel is **eight fresh B keys and eight paired within-record
glyph shuffles**. Each positive is fit on four Sallust records and evaluated
on two Tacitus records under its unchanged learned dictionary. Every underlying
record contains 224 normalized source letters. Eligible body-contained windows
are enumerated at stride 256; key k uses candidate indices k*16+[0,1,2,3] for
Sallust and k*16+[0,1] for Tacitus, for k=0,…,7. This preserves disjoint text
between keys and authors. It is a systematic window sample from two fixed
works, not eight independently sampled authors. Require at least 116 eligible
Sallust windows and 114 Tacitus windows; insufficient qualified text stops the
panel rather than shrinking it or borrowing another author.

The pipeline starts from scratch on each ciphertext:

1. Frozen order1/tau64 source; existing unit search with at most 16 starts and
   300 seconds, the complete 42-unit pool, and the registered remaining search
   settings and seed schedule.
2. Its selected dictionary, rescored under the frozen order3/tau256 source,
   initializes one 300-second refinement with at most 20 sweeps. Retain the
   best fully scored candidate, including this initial dictionary.

Both source tables remain exactly the already selected Caesar/Virgil tables.
No Sallust/Tacitus counts update them. Use the existing literal channel code,
not a cheaper code for the now-known deterministic family. All source letters
and all six declared glyphs remain eligible; the decoder receives no true
unit boundaries, key rows, per-record source length constraint, or oracle
initialization. Known Latin, source/glyph inventories, record boundaries,
one-state deterministic one/two-glyph emissions and rho=1/225 are assistance.
Fixed 224-letter excerpts are not draws from this geometric stopping law.

Stage1 selects under its own order1 objective; stage2 selects under order3.
Do not compare those numerical totals to choose a stage. Rescore both saved
dictionaries under order3 for a comparable diagnostic. Source tables are fixed
side information here; these scores are not evidence comparing languages or
Bayesian integration over source models. A 300-second budget permits fewer
than 16 completed starts, and a local certificate never means global search.

## Generator ambiguity and isolation

The existing B generator chooses all six singleton glyph units and 17 distinct
digram units, then permutes their assignment to the 23 source letters. Use this
distribution unchanged with new, independently derived key streams. **Distinct
rows do not make the code uniquely decodable:** each digram also equals the
concatenation of two singleton codewords. Recovery therefore depends on the
source model and text, even with the generating dictionary. Neither source
positivity nor a complete search eliminates that ambiguity.

Never redraw a key or replace a passage because the oracle, initializer,
search, or transfer decoder performs poorly. Do not select singleton assignment
to frequent letters, require particular letters to occur, or enforce a
post-result identifiability screen. Report repeated or rare row identities and
source-letter exposure only after unblinding. The learner may return duplicate
units even though generating rows are distinct; that existing larger search
family must remain disclosed.

Freeze solver, source hashes, generator distribution, window policy, metrics,
seeds/seed commitments, limits and audit code before acquisition. The evaluator
then acquires and hashes the files, audits Latin body spans and records exact
extraction metadata before generation. Edition-specific span manifests can be
produced during this mechanical audit; they must not depend on cipher scores.
If extraction policy requires revision, stop and document it before generating
keys or running the solver.

Separate builder/evaluator and fit-process inputs. The fit process should open
only its allowlisted fitting ciphertext, coding context and frozen source
tables. Keep source text, transfer ciphertext, gold channels, key-generation
seeds and positive/null labels outside that input path. Opaque case IDs and
evaluator-only seed commitments strengthen this separation; public deterministic
seeds and locally accessible answers provide procedural isolation only, not
cryptographic blindness. Search seeds are independent from key/window/null
streams. No agent or LLM supplies literary guesses to the finite-state decoder.

Freeze **all sixteen selected pipelines**, both stages and baseline dictionaries,
full trace hashes and campaign outcomes before exposing transfer data or answers
to evaluation. All failures remain in the panel. A numerical/hash mismatch is
an audit failure; a cooperative timeout with a retained model is a budgeted
result. A hard child failure is retained as failure, never silently replaced.

## Data acquisition blockers

The [corpus plan](blind-channel-corpus-plan.md) identifies Sallust PG7402 and
Tacitus PG9090 as unopened editions containing Latin and English material.
Their authorial spans and eligible lengths are not established by catalogue
metadata. The evaluator must distinguish original Latin from introductions,
notes, translations, quotations and apparatus before alphabet filtering.
Use the frozen development normalization, preserve raw/derived hashes, body
boundaries, excluded intervals, token-to-source maps and rights/provenance.
Windows must not join separate bodies or splice across excluded authorial
passages. A frozen rule may reconnect the surrounding original body when
removing an inserted editorial note.

Apply a predeclared exact-duplicate/quotation audit against the admitted
Caesar/Virgil/Cicero material and between final roles. The existing 64-character
exact-window check can be retained as a mechanical exclusion rule, with its
limited sensitivity stated; it cannot prove absence of shorter or paraphrased
overlap. Freeze the handling of an excluded interval and candidate reindexing
before acquisition. Report exclusion counts and reasons without displaying
plaintext to the solver. No adaptation to final letter statistics, no easy
passage selection, and no model changes after acquisition.

## Adopted recovery gates and separate screening

Let C_k be total transfer Levenshtein edits divided by 448 true letters for
positive key k; O_k is the corresponding true-channel/order3 CER. Let F_k be
the same metric using that run's original frequency-only initialization with
the **same final order3 decoder**. Do not cap CER at one. The eight-key recovery
qualification passes only if all of these predeclared conditions hold:

| Condition | Threshold |
| --- | --- |
| Every positive transfers adequately | C_k <= 0.05 for all eight keys |
| Equal-key average is near the true text | mean(C_k) <= 0.02 |
| No key substantially trails its generating-channel decoder | C_k - O_k <= 0.02 for every key |
| Search improves the registered uninformed dictionary | C_k < F_k for every key |

These are practical literal-recovery tolerances chosen before fresh data, not
significance tests or guarantees of universal reliability. Poor oracle reading
does not excuse the absolute gates or authorize a redraw. An unsupported
transfer record receives 224 deletion errors, null log likelihood, and a
distinct missing support-floor value; it cannot disappear from denominators.
A baseline that is already exact cannot satisfy the strict improvement gate;
record that outcome rather than changing the comparison.

Report every key's fit/transfer edits, decoded length, exact records, literal
row matches out of 23, row exposure, dictionary support floor, oracle gap and
conditional MAP surprisal. The support floor is answer-only. A zero floor
means truth is permitted, not correctly ranked; a positive floor identifies a
dictionary limitation. Conditional surprisal fixes the selected source/key
and does not quantify model uncertainty or correctness. Literal row equality
and exact text are separate from the CER qualification.

The frequency initializer uses only singleton units, so it is a weak structural
baseline for mostly-digram B keys. Keep it for continuity without overstating
the relative gain. Report 1-mean(C_k)/mean(F_k) as a descriptive macro reduction
when the denominator is positive; otherwise report it undefined. Also decode
the saved stage1 dictionary under order3 after the common freeze. That controls
for merely changing the final source/decision when assessing refinement.

Separately preserve the old iid diagnostic without changing thresholds:
language-channel fit saving at least 32 bits and transfer gain at least 0.05
bits per emitted glyph versus its fit-trained, coded iid-glyph baseline.
Charge the same one-bit family selector to both families. The screening
criterion is all eight positives flagged and none of eight shuffles flagged.
Report recovery and screening outcomes separately. Require all sixteen jobs
to complete for either qualification. An unsupported model is unflagged;
a hard-failed null search cannot count as successful negative-control rejection.

Within-record glyph shuffles exactly match each positive's record lengths and
glyph histograms, and run through the identical pipeline/budget. They destroy
local structure and are therefore a relatively easy null. Passing this screen
does **not** qualify rejection of structured gibberish or establish semantic
confidence. No plaintext accuracy exists for these shuffles. Even zero flags
among eight independent draws would have a one-sided 95% binomial upper bound
of about 31.2%; this small panel cannot establish a 5% false-declaration rate.
Dependence and the fixed author pair further limit broad population claims.

## Computation and required runner checks

At two single-thread workers, 16 cases times 600 cooperative seconds gives
160 nominal CPU minutes and about 80 wall minutes if every fit uses its full
allocation. Actual time depends on atomic inference overruns, auditing and
serialization. Each disposable child has an 840-CPU-second cap and a
960-second parent wall cap. Before any freeze or acquisition, the campaign
limit was revised to **five CPU hours**: 16*840 + 1800 evaluation + 300 build
+ 900 test/replay seconds = 16,440 seconds, below 18,000. Evaluation has an
1800-CPU-second and 2700-wall-second cap. Track actual aggregate consumption
as well as enforcing the component caps; include disposable-process overhead
and preserve every failed attempt. Use the 8 GiB planning memory limit, record
actual RSS and avoid describing summed process peaks as whole-host usage.
No paid compute or API use. Never extend a poor or slow case after seeing it.

Before launch, the independent runner review should verify source/table hashes,
access allowlists, opaque case ordering, seed derivation, body/window checks,
exact no-overwrite behavior and all sixteen failure paths. Verify both search
stages' ordered neighborhoods, complete versus partial sweeps, retained best
finished candidate, actual literal costs, trajectory and certificate ownership.
Use independent numeric replay for every selected fitting model and all frozen
transfer marginal/MAP predictions, edit metrics, support floors and iid flags.
Trace accounting is not numeric replay of every neighbor likelihood. Inspect
the newly root-authored refiner independently before this qualification; the
DEV005 record explicitly disclosed that this fresh review was absent there.

## Interpretation and the next bridge

A pass establishes approximate new-key, new-author recovery for this known-Latin
memoryless family at the fixed budget. A failure identifies a qualification
failure and a preserved diagnostic; it does not show that Voynich is nonlanguage.
Do not tune this panel and rerun it as fresh confirmation. Reserve any further
author windows and keys before unblinding.

The next bridge is a separately frozen harder-family or historical control,
with realistic transcription uncertainty and structured nonlanguage controls.
Stateful C still tests a missing capability; Borg still needs a qualified unit
parser and exposure handling. A manuscript attempt must make a constrained,
independently testable prediction on protected physical folios or external
evidence. This eight-key success, if obtained, cannot choose Latin for Voynich
or turn fluent decoded text into a decipherment.

Primary literature checked for this review: [Nuhn, Schamper and Ney (2013)](https://aclanthology.org/P13-1154.pdf),
abstract and sections 3/6, demonstrates that stronger character context can
outperform exact optimization of a weaker source model in supplied-unit
substitution; it does not validate our unknown-unit or Latin assumptions.
[Berg-Kirkpatrick and Klein (2013)](https://aclanthology.org/D13-1087.pdf),
sections 1–3, documents restart dependence and divergence between the best
likelihood and best reading in a supplied-symbol homophonic model. It supports
reporting search coverage and actual recovery, not a claim that 16 starts are
sufficient here. [Reddy and Knight (2011)](https://aclanthology.org/W11-1511.pdf),
introduction and section 2, discusses unresolved manuscript structure and
transcription ambiguity; its statistical analysis establishes no Latin key.
These papers motivate controls. The adopted numerical gates are this project's
prospective engineering criteria, not thresholds inherited from those papers.

## Pre-data implementation review

The independently authored
[orchestration tests](../../tests/test_blind_channel_confirm001_independent.py)
use only hand-constructed two-letter sources, tiny ciphertexts and temporary
artificial archives. They exercise real stage1 search, stage2 refinement and
both trace auditors; rational enumeration independently checks selected scores.
They also test fit-only reads, mixed and entirely failed sixteen-case panels,
record denominators and unsupported deletion errors, actual iid selector/gain
arithmetic, campaign timeouts, duplicate-key stopping, equal-key aggregation,
and a complete tolerance-stopped neighborhood whose returned improvement must
not inherit its parent's local certificate. No final data was read or generated.

The first integration run passed ten such checks. Five additional regression
tests exposed pre-data guard gaps: construction, direct fitting and evaluation
could repeat after an early failure; failed null searches could count toward a
screening pass; and a blocked corpus manifest was not explicitly rejected before
attempting derived-data reads. These findings were sent to the implementation
author. They must be repaired and the full focused set passed before approval.
No empirical attempt or result was affected. The numerical refiner review found
no new objective, best-retention or certificate-ownership discrepancy.

### Root resolution, 2026-09-30

Before any data acquisition, root added exclusive attempt markers before
construction, search and answer evaluation; required a prepared, correctly
assigned corpus before derived reads; and made screening require every job to
complete. All15 independently authored orchestration tests now pass, including
the five formerly failing regressions. With seven root tests,44 corpus tests
and32 parallel Borg-parser tests, all98 new focused checks pass. This records
root's execution of the independent tests, not an additional reviewer approval.
The interrupted work left no live fitting or validation process on resumption;
no empirical recovery attempt or final-author acquisition occurred before
these repairs. The formal registration is
[BLIND-CHANNEL-CONFIRM-001](../experiments/BLIND-CHANNEL-CONFIRM-001.md).
