# Rolling support compilation with protected-root collection

Own engineering successor to the [failed whole-record compiler](../experiments/COMPILED-INVENTORY-001-results.md). Its successful final joint functions used3,267/2,673 of101,512/78,468 retained nodes. That motivates reclamation, but does not establish that the six capped calls had small live graphs or that collection will make them pass. The original compiler, counts, sampler, failed outcomes and source remain frozen.

## Source-based design and prior-work limits

The prior Voynich/decipherment review and positive-source support/occupancy derivation in [the original method](compiled-inventory-sampling-2026-10-01.md) apply unchanged. No paper cited there qualifies Latin or the synthetic channel as historical fact. Bryant's annotated Apply complexity correction and the compilation/query distinction remain reasons to preserve work and memory caps.

[Fabio Somenzi, CUDD2.7 Programmer's Manual source](https://raw.githubusercontent.com/ivmai/cudd/master/doc/cudd.tex.in), Basic Data Structures/Nodes and Cache, read2026-10-01, explains reclamation of unused decision diagrams using internal/external references and the performance tradeoff of computed-result caches. This is primary manual text, copyright1995–2015 University of Colorado, accessed in a maintained repository. We do not install CUDD, borrow its implementation, use its floating ADD counts or assume its performance. Our explicit rooted traversal/remapping is an independently written finite algorithm; collection is standard engineering, not claimed novel.

## Exact invariants

For maximum emission length two, the backward word-break recurrence at position i only needs functions F_(i+1),F_(i+2). Store those two roots instead of retaining a pointer to every suffix. Previous completed record roots remain protected until the joint AND is complete. Protect all code-variable functions and Boolean terminals as well, including variables absent from the current record. Variable ordering and code identities do not change.

At fixed16-position intervals, or after a position when25,000nodes/50,000Apply entries are retained, traverse from these protected roots. Build the induced reachable graph with new IDs in old topological-ID order; every node's children precede it. An injective old→new ID map rewrites every edge and returned root, preserving each node's variable and low/high choices. Rebuild the canonical unique table. All Apply-cache entries are discarded before IDs can be reused. Never infer liveness from a node's individual reference count alone or leave an unrewritten external root. Final collection also protects both original record functions and the joint root.

Induction over nodes shows identical Boolean functions at every protected root before/after remapping. The rolling recurrence therefore produces the same closed/nonclosed support function as the frozen all-suffix compiler. The frozen exact polynomial, subset unranking and occupancy draw operate on that unchanged function; counts and same-seed draws should agree for a fixed variable order. No probability correction is added because this is an exact representation change, not a proposal-law change.

Collection is allowed only inside unprofiled compilation. A sealed graph rejects later collection, preventing stale polynomial-cache IDs. Rebuild allocation is admitted conservatively before scratch is created; new nodes/table are committed atomically only after successful traversal/remapping. A collection-work cap returns only counters through the same typed compiler failure; no partially remapped graph or partial count is exposed.

## Separate work and allocation bounds

Original150,000node/300,000retained-cache limits now constrain the live manager, not cumulative historical allocations. Neither limit is enlarged. Three-million cumulative Apply invocations remain bounded and can grow because clearing caches sacrifices reuse. Report created/collected/final/max-live nodes, maximum retained Apply entries, completed collections and cumulative collection work separately. Node conservation is `created_nodes+2−collected_nodes=final_nodes` on completed graphs.

Twenty-million collection-work units charge cache entries discarded, old-node scans, traversal stack pops and retained-node rebuilds. They are deterministic accounting units, not CPU instructions. Before collection, an additional512bytes per old node reserves traversal/mapping/new-manager scratch beside the original512bytes/node+256bytes/cache-entry manager envelope; the unchanged512MiB owned cap still applies. Process RSS remains a separate measured2GiB bound. Collection cannot eliminate exponential live Boolean functions or expensive Apply operations.

## Qualification, transport and scientific boundary

[New finite tests](../../tests/test_rolling_inventory_bdd.py) use a tracking collector to compare protected-root structural signatures around every finite collection, exhaust original Boolean masks/known-binding/cardinality/subset/assignment/SMC law checks at intervals1and16, compare six-glyph exact counts and seeded draws, and test longer rolling suffixes, dead-node removal, collection caps, invalid records and sealing. [Transport tests](../../tests/test_compiled_inventory002.py) replay the entire eight-cell tiny native/Python panel under ordinary/compiler/native/collection caps, plus duplicate refusal and exception-safe runtime binding restoration.

The new runner reuses the frozen001 transport and auditor through an explicit temporary namespace adapter in one isolated process: separate experiment ID/output paths/freeze dependencies, new compiler/settings/proof admission. Every substituted binding is restored on exit, including exceptions. Original file bytes and output archives remain unchanged. This is explicit component reuse, not a second run of the failed001 experiment.

[COMPILED-INVENTORY-002](../experiments/COMPILED-INVENTORY-002.md) reuses the same exposed ciphertext, fixed orders, seeds,32conditional draws/64prior controls and first-key source checks for a controlled engineering comparison. No generator answers, old fitted bank or new holdout enters. If it passes, it qualifies this initializer's implementation and measured costs only. Correct full-support intermediate targets, posterior concentration, global moves, fresh recovery gates and manuscript adequacy remain separate unresolved work.
