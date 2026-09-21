# World models, diffusion, linguistics, and ambitious decipherment

Research date: **2026-09-21**. A substantial literature review and architectural synthesis, written in response to the request to reconsider the project at much greater ambition. **Research and ideation only: no model implementation, training, manuscript scoring, or experiment was performed for this review.**

The dossier contains approximately **35,000 words across 11 documents and 98 annotated source records**: 23 world/action, 27 diffusion, 32 linguistics, and 16 mechanism/identification records. These are records, not 98 distinct fully read papers: some overlap across tracks, some group resources, and reading depth varies as declared in the ledgers.

**Our strongest proposed direction is a foundation model of document-producing systems, combined with globally revisable inference and an explicit rule executor.** It would infer how content, language, notation, illustration and scribal conventions jointly produce a manuscript. Diffusion is a promising inference tool within that system. World/action models supply structured state and transitions. Linguistics defines the distinct layers that connect marks to communication. Mechanistic interpretation could extract candidate decoding rules from the learned computation.

This is a research judgment, not a published demonstration or a forecast of successful decipherment. The review identifies capabilities that already exist, assumptions those capabilities require, and larger systems we propose building upon them. It does not claim that any reviewed method has solved Voynich.

## Reading routes

For the complete proposal, begin with [SYNTHESIS.md](SYNTHESIS.md). For the foundational reading requested separately, begin with [LINGUISTICS_GROUNDING.md](LINGUISTICS_GROUNDING.md), then compare the two model families.

| Document | What it covers |
| --- | --- |
| [Synthesis and five major research bets](SYNTHESIS.md) | A joint inference target, concrete architecture interfaces, ambitious endpoints, alternative explanations, and ranking of the proposals |
| [World models and world-action models](WORLD_MODELS_ACTION.md) | Predictive state, World Models, PlaNet, Dreamer, MuZero, object-centric dynamics, JEPA, Genie, latent actions, VLAs and recent world-action proposals |
| [Diffusion and structured inference](DIFFUSION_INFERENCE.md) | Continuous/discrete diffusion, masked LMs, revision, edit flows, constraints, posterior sampling, planning and diffusion world models |
| [Linguistics and grounding](LINGUISTICS_GROUNDING.md) | Writing systems, typology, morphology, syntax, discourse, semantics, acquisition, pragmatics, action semantics and historical decipherment |
| [Mechanisms and identifiability](MECHANISMS_AND_IDENTIFIABILITY.md) | What a recovered world model would mean, causal state extraction, diffusion-time interpretation, transferable update rules and semantic ambiguity |
| [World/action source ledger](WORLD_SOURCES.md) | W-series primary sources, versions, reading depth, supervision and access limits |
| [Diffusion source ledger](DIFFUSION_SOURCES.md) | D-series primary sources, versions, exactness qualifications and decipherment precedents |
| [Linguistics source ledger](LINGUISTICS_SOURCES.md) | L-series linguistic and historical sources, with explicit access and reading limits |
| [Mechanism source ledger](MECHANISM_SOURCES.md) | M-series sources for causal interpretation, identification and shared inference methods |
| [Review audit and scope](REVIEW_AUDIT.md) | Coverage, source corrections, cross-review checks, validation and remaining gaps |

Source IDs are local to this directory. The earlier [deep review](../deep-review-2026-09-21/README.md) has its own IDs; an `M01` there is not this review's `M01`.

## The five major bets

1. **Learn how unfamiliar communication systems work.** Externally train across linguistic, graphic and production systems; adapt that inference ability to a short unknown artifact.
2. **Revise whole explanations.** Jointly search the global encoding rules, segmentation, alignment, latent message and semantic structure while preserving the observed marks.
3. **Recover executable relations before attaching names.** Use entity, part, event and procedure models where justified, including historical belief systems and nonprocedural genres.
4. **Extract a decoder from the learned computation.** Discover state variables and update rules, compile them into an inspectable account, and separate neural faithfulness from historical truth.
5. **Choose evidence that separates hypotheses.** Give the researcher a model of uncertainty and explicit information-seeking actions.

These are integrated program-level ideas, not another schedule of small experiments. Their detailed failure conditions specify what would make the ambition informative.

## Findings that materially change architectural choices

- Passive video learning and action-grounded control are different stages. “Unlabeled” robot data can still contain measured state/action signals.
- A world model can use diffusion; a diffusion model need not represent a world. The terms refer to different properties of a system.
- Ordinary absorbing-mask diffusion often commits revealed tokens. Iterative correction and variable-length alignment require explicit mechanisms.
- Per-step exact constrained decoding is not automatically exact sampling from a full conditional generative model.
- A useful linguistic representation distinguishes writing, language, discourse and referent state. Semantic relations can be more identifiable than their lexical names.
- Successful neural re-decipherment often receives related-language evidence, constrained correspondences or execution feedback that Voynich does not supply.
- Interpreting a model can reveal its learned algorithm. Independent observations are still needed to connect that algorithm to the historical source.

Each point is developed with primary links in the relevant chapter and the matching source ledger. Recent preprints and corporate demonstrations have lower evidential weight than established methods with clear assumptions; neither category is treated as a manuscript result.

## Relation to repository work

The [research protocol](../PROTOCOL.md) governs eventual scientific claims. The original [charter](../../RESEARCH_CHARTER.md) remains unchanged. Existing experiments inform the problem framing but are not rerun here, and this review does not adopt concurrent unreviewed results as its own. Its proposals do not update historical hypotheses to “confirmed” or authorize a scheduled run.
