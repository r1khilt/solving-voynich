# Review scope and evidence audit

Reviewed 2026-09-21. This file records the research process, important corrections and the limits of the resulting dossier. It is documentation, not an experiment report.

## Coverage and method

Four reading tracks were developed in parallel: world/action models, diffusion/inference, linguistics/decipherment, and mechanism/identification theory. The synthesis was written against those distinctions and the repository's research protocol. Existing project memory, recent notebook entries, the knowledge index, original charter, and relevant earlier reviews were consulted before substantive work.

Searches used primary papers, proceedings, author-hosted manuscripts, official projects, and primary announcements. Discovery queries covered method names and combinations with Voynich, decipherment, cryptanalysis, language, latent actions and mechanistic interpretation. The literature was followed through relevant references and adjacent methods. This is a deep targeted review, **not a formally exhaustive systematic review** with a complete database-export/search-screening protocol.

Reading depth is recorded per source: selected technical sections, abstract/metadata only, primary announcement, or access-limited material. A source record is not a claim that every page, proof or experiment was read. Long author lists are explicitly abbreviated where necessary. No reported result was independently replicated.

Recent works were checked at the stated versions and access date. Web rendering, metadata and paper headers sometimes disagree; records retain these differences. Stable versioned URLs are preferred. Mutable web projects and official announcements are explicitly identified. Links to public artifacts do not certify their licenses, executability, or availability for a future training run.

## Corrections and qualifications that affect the proposal

| Issue | What the review preserves | Consequence |
| --- | --- | --- |
| V-JEPA 2 “unlabeled” robot adaptation | The inspected method includes measured end-effector state and action-conditioned prediction | Do not claim passive pictures alone supply executable control |
| LAPA pretraining | Latent actions precede later action-grounded adaptation | Video pretraining is not an entirely ungrounded pipeline |
| MuZero hidden state | Planning sufficiency is different from attached environmental semantics | A useful search state need not be a historically faithful world |
| Masked diffusion | Arbitrary-order filling can freeze committed tokens | Explicit remasking or edit mechanisms are needed for revision |
| Discrete versus continuous state | Embedding-space proximity does not enforce valid symbols, keys or alignments | Use a declared discrete realization and rule executor |
| Dang–Ermon constrained decoding | Exact per-step constrained mean-field inference; NFA accepting-path multiplicity; limited Sudoku constraint | Do not call it exact full posterior sampling or a general constraint solver |
| Diffusion time | Artificial corruption time differs from event time and research time | Do not interpret denoising dynamics as historical writing operations |
| Oracle-bone diffusion | Known glyph correspondences and cross-era visual assistance | Evidence for an adjacent task, not blind unknown-language decipherment |
| Direct Voynich proposal site | A public diffusion/CLIP proposal exists; inspected material supplies no validated decoding result | Avoid a universal novelty claim or treating project marketing as evidence |
| Othello interpretation | Large known synthetic task, truth-labeled probes and interventions | An existence proof under favorable conditions, not manuscript-scale transfer proof |
| Causal/representation theorems | Explicit generation, intervention, task and support assumptions | Use their conditions to design training; do not claim those conditions hold for Voynich |
| Linguistic reviews and statistical structure | Language hypotheses coexist with structured nonsemantic alternatives | Statistical regularity does not settle whether there is plaintext |
| Historical world knowledge | Authors can describe obsolete or false theories coherently | Modern physical truth is not the sole semantic verifier |
| Scope of concurrent work | Other repository tasks can change code/data or run independently | Only documentation and research validation belong to this review |

The root review independently reopened the September 2026 MoWAM and ZimaBlue records and the June 2026 LaWAM record to check their dates and identities. Their recent abstract-level claims remain explicitly tentative. DLM-Scope and the diffusion concept paper were inspected at versioned full-text sections. Multimodal identification was checked through the accessible arXiv PDF after OpenReview returned a browser challenge. These checks do not imply full-paper replication.

## Synthesis boundaries

The joint generative graph, proposed posterior, architecture interfaces, causal atlas, program-level deliverables and ranking are **our proposals**. Cited papers motivate components and constrain extrapolation. No paper is represented as implementing the entire proposed system. A mathematically written target is still conditional on its factorization, priors and observation models.

The semantic graph is not assumed known. Neither “botanical” page classifications nor guessed plant names become translation labels. Pictures may be weakly related to local text. Scribes, layout, sections and topics are possible common causes. Independent evidence must remain independent; AI descriptions of the same source cannot be multiplied into separate observations.

No plaintext, filler assignment, language identification, hidden transition, or historical encoding rule was established in this review. No model, sampler, evaluator, data pipeline or experiment configuration was implemented. No manuscript split was opened for scoring. No paid research API, cloud job or model training was launched.

## Remaining literature and resource gaps

- The broad sweep includes abstract-only leads. Their complete methods, ablations and artifacts require later focused review before architectural adoption.
- General linguistic coverage is a foundation for this program, not a comprehensive treatment of phonology, formal semantics, language acquisition or historical linguistics.
- Historical shorthand, abbreviation traditions, manuscript image conventions and expert palaeographic alternatives need deeper dedicated scholarship before choosing a production grammar.
- Comparative corpora and grounded procedure resources are proposed inputs; no assembled, licensed dataset with all required correspondences is claimed to exist.
- Diffusion bridge methods, advanced posterior corrections, and inference-time scaling need further algorithm-specific analysis if selected. These are not interchangeable just because all use stochastic refinement.
- Every proposed scale range is architectural context, not a measured runtime, procurement estimate or spend authorization. Reported model/API credits are not verified compute capacity.
- Bounded search did not locate a validated diffusion/world-model/VLA Voynich decipherment. Search absence cannot establish that no unpublished or obscure attempt exists.

## Validation record

Final delivery checks verify Markdown link targets within the repository, source-ID resolution, nonempty completed chapters/ledgers, whitespace, and the unchanged original charter. They also inspect the staged file set so unrelated data, code and bulk artifacts are excluded. These are document and repository checks, not scientific tests. Exact observed outcomes are recorded in the associated notebook checkpoint.

The completed dossier has 11 Markdown documents, approximately 35,000 words, and 98 source records (W:23, D:27, L:32, M:16). Records are not a deduplicated paper count: for example, W23 and D27 describe the same proposal website. No local link target was missing, no source ID was undefined or duplicated within a ledger, and no trailing whitespace was found in the dossier at the final content check. The original charter matched the committed bytes; SHA-256: `ba74631f91464449097cd230bc7045cd58876c90b56be1292f723c185d407604`.

An independent cross-review of the synthesis and mechanism chapter caught an overly strong claim that identical denoising randomness was necessary for causal attribution. This was corrected: paired comparisons can hold noise fixed, while randomized repeated comparisons can establish effects without identical noise. Neither strategy by itself makes an intervention historically meaningful. The cross-review reported no further substantive issue within its reviewed scope; it does not certify all literature claims or prove the proposed architecture.
