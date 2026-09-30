# BLIND-CHANNEL-CONFIRM-001: fresh keys and new-author transfer

2026-09-30. **Prospective conditional recovery qualification.** Freeze this
registration, every solver/source/generator choice and extraction policy before
acquiring Sallust or Tacitus. No final author has been acquired, no case generated
and no new score observed at registration. This does not replace the larger
[blind-channel program](../research/blind-channel-recovery-design.md), qualify
stateful familyC or identify Voynich's language or encoding.

## Pre-text transport amendment, 2026-09-30

The original pre-data freeze `e142a59f8c87bb2f16eb74343cca7c874aa56b68`
was published before acquisition. The first request stopped at an HTTP302
because its Location downgraded to HTTP. No book bytes, source statistics,
keys or scores were read. Failure remains at
`data/manifests/blind_channel_confirmation_acquisition.json`.
Subsequent HEAD-only checks showed the same origin/path responds200 over HTTPS:
`https://www.gutenberg.org/cache/epub/7402/pg7402.txt` (503,378bytes) and
`https://www.gutenberg.org/cache/epub/9090/pg9090.txt` (436,490bytes).

Before any text acquisition, publish this explicit transport-only amendment
and freeze the pipeline again. A separate acquisition attempt uses those exact
HTTPS paths and writes `blind_channel_confirmation_acquisition_v2.json`.
Preserve the original failure and retain HTTPS verification, no automatic retry,
the same edition IDs, byte caps, all extraction rules, source tables, solver,
key streams, windows, limits and decision gates. This is not a replacement of
a scored experiment or an outcome-dependent change. Any second failure is
preserved rather than silently retried.

## Question and rationale

Can the complete pipeline developed on Cicero and two exposed B keys recover
new text under eight independently generated keys, fitting on Sallust and
transferring to Tacitus? DEV003–005 improved development readings to0/8 edits
per448 letters, matching the true-key transfer decoder, but reused cases.
Stop that tuning. This test measures actual plaintext recovery with no supplied
unit boundaries or generating dictionary.

Method rationale, independent review and source limitations are recorded in
[fresh qualification review](../research/blind-channel-fresh-qualification-review.md),
[source-context analysis](../research/fixed-channel-source-context.md), and
[unit-search alternatives](../research/unknown-unit-search-alternatives.md).
Nuhn et al2013 supports testing stronger source context separately from search;
Berg-Kirkpatrick/Klein2013 motivates restart and likelihood/recovery diagnostics.
Their supplied-unit settings differ. Reddy/Knight2011 and Hauer/Kondrak2016
do not establish our Latin or channel assumptions for Voynich. These primary
sources motivate controls, not the prospective numerical tolerances below.

## Frozen inputs, construction and disclosure

Use the existing DEV004 source archive without refitting: order1/tau64 and
order3/tau256, both trained only on the pinned first50,000 Caesar and50,000
Virgil letters. Source-selection SHA256
`1a0136627f04e3b68b7e2b3d801e5d8cdebc617fc1e5f3f0fa7b7e344ab74c60`;
archive SHA256 `a314c7120a14f6636a7b0a129e803fc517c0488016422a3fd84764b092179615`.
Ordered plaintext alphabet: `abcdefghiklmnopqrstuxyz`.

Acquire fixed PG7402 Sallust for fitting and PG9090 Tacitus for transfer under
the [corpus policy](../research/final-corpus-provenance-plan.md). Both editions
contain English and Latin. Evaluator-only bibliographic inspection establishes
all original-Latin authorial body spans in publication order and explicit
editorial exclusions; do not use linguistic-model or cipher scores for selection.
Preserve raw bytes, rights notice, URLs, hashes, body/token mappings and reviewed
exclusion hashes. Freeze that span plan before mechanical preparation. If
separation needs a methodological revision, stop and disclose it before any
generation. No alternate author/edition or score-dependent span replacement.

Normalization remains the prior NFKD/lowercase/ligature policy, j→i,v→u,
bracket/numeric exclusions, omitted whitespace and strict alphabet rejection.
Use first50,000 normalized eligible letters per final author, preserving body
boundaries. Enumerate224-letter windows at body-relative stride256, resetting
at each body. For key index k=0,…,7, use candidate indices16k+[0,1,2,3] for
Sallust fit and16k+[0,1] for Tacitus transfer. No window crosses a body/cap;
insufficient windows or repeated selected records stops generation.

Before generation, exact64-letter overlap screening compares full eligible
bodies of all five authors, never forming windows across bodies. Any overlap
involving a final author stops preparation with a preserved blocked manifest;
do not silently remove quotations or reindex candidates. Report within-author
repeats. Zero overlap does not establish absence of short/fuzzy quotation,
genre dependence or LLM pretraining exposure. Source models never ingest these
final bodies, and no LLM supplies guesses to the finite-state decoder.

Generate eight B keys using the unchanged algorithm: all six singleton glyphs
ABCDEF plus17 distinct digrams sampled from36, then permute onto23 source rows.
Key seed=60129+104729k. A digram is also representable by two singleton units;
this distribution is not uniquely decodable. Do not reject a key for poor
oracle accuracy or rare-letter coverage. Duplicate keys stop, never redraw.
For each positive, paired nulls shuffle glyphs separately within each record
using one split-ordered RNG seeded key_seed+3,000,000. Fit then transfer ordering
is fixed. All record histograms and lengths are retained.

Case order is B-key1, B-key1-shuffle,…,B-key8,B-key8-shuffle. Search seed for
zero-based case index i is63101+257i. Each positive has896 fit and448 transfer
letters. Eight keys share two fixed authors but disjoint windows; they are not
eight independent author pairs. Answer/key seed isolation is procedural rather
than cryptographic: seeds are public, case IDs disclose null status, and local
answer files exist. No fit operation branches on positivity or reads the answer,
transfer ciphertext, corpus plaintext or generating seed. The runner validates
metadata and opens only the assigned fitting artifact plus frozen source tables.

Disclosed assistance: Latin/source alphabet, ABCDEF inventory, record boundaries,
one deterministic state, maximum unit length2, and geometric stop rho=1/225.
The fixed-length generator differs from the geometric decoder prior. Learners
may reuse units even though generating rows are distinct. No homophonic emission
alternatives, state discovery, source-language identification or manuscript data
are tested.

## Fixed algorithm and controls

For every positive and null, from scratch:

1. Order1 source; `search_unit_channel` with16starts,80sweeps/start,300cooperative
   seconds total, batch256, complete42-unit pool, tolerance1e-8bits. All unequal
   row swaps and single-row replacements. First initialization ranks source/glyph
   frequencies into singleton units; remaining starts use existing seeded
   singleton-coverage plus uniform-pool initialization.
2. Take its selected dictionary without comparing objectives across sources.
   Rescore under order3 and run one `refine_units` search with20sweeps,
   300cooperative seconds and tolerance1e-8bits. Same complete move family.
   Retain best completed candidate, including the initial dictionary; a partial
   sweep never supplies a local certificate for an unscanned returned candidate.

Exact finite marginals and MAP decoding use no beam. Sparse inference retains
the existing3million-node hard cap. A numeric/state-cap/hash discrepancy is a
failed run, not a candidate silently dropped. Actual literal channel code is
unchanged. Local optimality is not global optimality. Save all trace hashes,
stage1 and final dictionaries, frequency initializer, coded iid-glyph baseline,
stop reasons and measured resources before the common key freeze.

After that freeze, evaluate each final dictionary, its saved stage1 dictionary
and original frequency initializer under the **same order3 decoder**. On positives
also evaluate the true-channel oracle. The oracle is not an initializer or
selection criterion. Report all fit/transfer edit counts, uncapped CER, decoded
lengths, exact records, support and marginal/MAP scores. Compute answer-only
dictionary edit floors for learned/oracle arms. Literal row/exposure and
conditional best-reading surprisal are descriptive post-evaluation checks;
neither controls another fit nor implies correctness confidence.

## Prospective gates and failure accounting

Let C_k, O_k and F_k be final learned, true-channel and frequency-initializer
transfer edits divided by448 for key k. Recovery passes only if:

- All16 jobs complete without hard failure.
- Every C_k≤.05 and the equal-key mean of C_k≤.02.
- Every C_k−O_k≤.02.
- Every C_k<F_k.

These stricter gates target near-reading competence beyond the earlier10%
development diagnostic. They are engineering tolerances chosen before fresh
data, not significance tests. Report descriptive macro reduction
1−mean(C_k)/mean(F_k), undefined if the denominator is zero; an already exact
frequency baseline does not satisfy strict improvement. Poor oracle decoding
does not relax the gates or allow a redraw. Report eight individual outcomes;
the sample/author allocation does not estimate broad reliability precisely.

Screening is separate: preserve the earlier flag requiring≥32 fit bits saved
and≥.05 transfer bits/glyph gained over its fit-trained iid-glyph geometric
baseline. Charge the same one-bit family selector to both alternatives. The
screen passes only if all16 jobs complete, all8 positives flag and no shuffle
flags. This tests discrimination from easy matched shuffles, not semantics or
structured-gibberish rejection. A failed null cannot count as a successful
rejection. No semantic confidence threshold is introduced.

An unsupported positive transfer record incurs all224 deletion errors, null
likelihood and missing support floor. A hard-failed pipeline similarly incurs
full-record deletions for its unavailable learned/stage1 readings and forces
qualification failure; its fixed baseline and oracle remain evaluable. Never
drop cases/records from denominators, retry a failed run, extend a budget after
outcomes, overwrite attempted construction/fitting/evaluation, or relabel an
adaptive repair as this fresh result. Keep the case panel even if failures
occur. No progress-file observation timeout alone authorizes a replacement run.

## Freeze sequence, validation and resources

Publish/remote-verify source and protocol before acquisition; publish reviewed
raw hashes/span plan before preparation; publish the prepared corpus manifest
before key generation; publish the full16-case manifest before fitting; publish
all selected dictionaries, attempt/process records and campaign outcomes before
the single evaluation. Hash-check ignored bulk data against tracked manifests.
Freeze all implementation files in runner `SOURCE_PATHS`. A later span-manifest
commit cannot modify the original extraction method silently.

Independent artificial tests exercise both real search stages, rational selected
score checks, trace accounting, fit-only reads, failures/denominators, control
arithmetic, freeze/no-repeat guards and certificate ownership. Independently
authored backward inference replays both final fitting models and every scored
post-freeze arm. Separate integer edit and shortest-path references check edits
and floors. Existing literal-code and iid references check those fields.
Trace accounting checks moves, costs, trajectory, completeness and retention;
it does not independently numerically replay every candidate likelihood.

At most2 single-threadCPUworkers and8GiB aggregate planning memory. Each case
has840CPU seconds and960parent-wall seconds, including its two300-second core
limits. Evaluation1800CPU/2700wall seconds, construction300CPU seconds. Five
CPU-hour total planning cap:16*840+1800+300+900validation/replay=16,440seconds
below18,000. Record aggregate actual resources and failed attempts; no paid
compute/API use. No open-ended loop. Archive cap200MiB compressed per artifact.
If resource or service failures stop the block, preserve state and report it.

Final authors consumed here cannot be reused as untouched methodological
development; any future confirmation requires a separately declared unused
allocation. StatefulC is still ungenerated/untested. A pass permits the next
broader-family/historical test, not a Voynich translation claim. A failure
requires diagnosis without retuning and republishing this panel as fresh.
