# Knowledge base

## Status and evidence labels

This knowledge base organizes the charter and the project's research. Targeted prior-work and architecture reviews are now available below; they are not exhaustive surveys or independent replications. Source claims remain distinct from our implementation results.

Use these labels in future records:

- **Source report:** what a named source claims, with a precise citation; not automatically accepted as fact.
- **Verified observation:** a directly inspected or reproducible result, with provenance and scope.
- **Hypothesis:** a proposed explanation that remains under test.
- **Inference:** an interpretation of identified evidence; state competing explanations.
- **Unverified context:** user or model statements not yet checked.
- **Rejected / superseded:** a claim or method whose status changed, with the reason and replacement.

## Source register

| ID | Source | Provenance | Status / permitted use |
| --- | --- | --- | --- |
| SRC-0001 | [Original research charter](../RESEARCH_CHARTER.md) | User attachment received 2026-09-20; preserved verbatim | Primary record of user intent and hypotheses; not empirical support |
| SRC-0002 | Initial user instructions | Summarized in [project memory](../../MEMORY.md) and [NB-0001](../../NOTEBOOK.md) | Authority for scope, workflow preferences, Git checkpoints, and reported resources |
| SRC-0003 | [The Beinecke Cipher (Voynich) Manuscript](https://beinecke.library.yale.edu/beinecke/collections/beinecke-cipher-voynich-manuscript) | Yale Library; undated page, accessed 2026-09-20; About and Access sections | Custodian describes the text as undeciphered and provides access to scans; no scans downloaded, and other page claims not audited |
| SRC-0004 | [Deciphering a mysterious manuscript](https://news.yale.edu/2025/02/21/deciphering-mysterious-manuscript) | Oscar Sweeting, Yale News, 2025-02-21; accessed 2026-09-20; direct interview quote from Claire Bowern | Expert emphasizes the many unknowns and possibility of studying structure without knowing meanings; not a primary experimental paper |
| SRC-0005 | [What We Know About The Voynich Manuscript](https://aclanthology.org/W11-1511/) | Sravana Reddy and Kevin Knight, LaTeCH 2011, pp. 78-86; accessed 2026-09-20 | Bibliographic record and introduction checked only; historical research context, not a current survey or a fully reviewed paper |
| SRC-0006 | [An Intoxicating 500-Year-Old Mystery](https://www.theatlantic.com/magazine/archive/2024/09/decoding-voynich-manuscript/679157/) | The Atlantic, 2024-08-08; accessed 2026-09-20; direct interview reporting | Davis interprets physical condition as evidence of use; reported daily function is tentative, not established consensus |

When adding an external source, record title, authors/maintainer, date/version, exact URL or identifier, access date, relevant page/section, claims supported, limitations, rights/access constraints, and local path/checksum if downloaded. Prefer original manuscript images, transcription documentation, primary papers, and reproducible code over summaries.

## Questions needing external grounding later

These are deferred discovery topics, not established findings or an active research assignment:

- Manuscript custody, dating evidence, foliation, missing/reordered leaves, and available scans.
- Transcription systems, disputed glyph readings, layout preservation, annotator disagreement, and redistribution terms.
- Currier classifications, scribal distinctions, manuscript sections, and how securely each annotation is established.
- Position effects, repetition, near-neighbor forms, entropy, dependence, and the effect of transcription/segmentation choices on each.
- Prior cipher, language, abbreviation, null/filler, and copy/mutate proposals; which tests actually discriminate among them.
- Earlier neural modeling and decipherment attempts, their datasets, leakage risks, available code, and claims that survived independent testing.

## Active research records

- [Implemented communication-system workbench](../COMMUNICATION_SYSTEM.md): executable document worlds, joint denoising and compiler search, action dynamics, causal controls, anonymous relational grounding and active evidence; [WMD-0001 qualification](../experiments/WMD-0001.md) and [measured results](../experiments/WMD-0001-results.md). All valid explanations in this qualification came from compiler search; no decipherment is established.
- [World models, diffusion, linguistics and ambitious decipherment](../research/world-models-diffusion-2026-09-21/README.md): approximately 35,000 words, 98 annotated source records, independent linguistics foundations, action/VLA transfer, causal interpretation and five integrated research directions. [Synthesis](../research/world-models-diffusion-2026-09-21/SYNTHESIS.md); [evidence audit](../research/world-models-diffusion-2026-09-21/REVIEW_AUDIT.md). Research and ideation only; no implementation or decipherment result.
- [Expanded architecture, interpretation, decipherment and theory review](../research/deep-review-2026-09-21/README.md): 93 records /92 works/resources, honest reading-depth labels, audited recent claims and proposed next design. [Searchable catalog](../research/deep-review-2026-09-21/CATALOG.md); run `python3 scripts/research_catalog.py --query <term>` from the repository root.
- [Prior Voynich models and neural decipherment](../research/PRIOR_WORK.md): original reports, supervision distinctions, evaluation limitations and source links.
- [Modern architecture and interpretability review](../research/ARCHITECTURE_REVIEW.md): verified primary-source claims and adopt/ablate/defer decisions.
- [Data provenance and representation](../research/DATA.md): official transcription, usage terms, parsing, uncertainty and split policy; machine-readable manifests live in `data/manifests/` at the repository root.
- [Implemented architecture](../research/ARCHITECTURE.md): exact design and its tradeoffs.
- [EXP-0001](../experiments/EXP-0001.md): bounded implementation-validation pilot, registered before training.
- [EXP-0002 results](../experiments/EXP-0002-results.md): six model families × three seeds; compact model selected by the registered near-tie rule.
- [EXP-0003 results](../experiments/EXP-0003-results.md): initial paired context perturbations and head interventions; the inconsistent distant-order hint was revised by EXP-0005.
- [EXP-0004 registration](../experiments/EXP-0004.md) and [results](../experiments/EXP-0004-results.md): primary-source synthetic/causal-method rationale, positive hidden-state readout calibration, IID control, and negative causal steering result.
- [EXP-0005 registration](../experiments/EXP-0005.md) and [results](../experiments/EXP-0005-results.md): prior page-heterogeneity/cache work, matched distant donors, approximate symbol-frequency controls and revised order sensitivity.
- [EXP-0006 registration](../experiments/EXP-0006.md) and [results](../experiments/EXP-0006-results.md): primary causal-alignment paper, subspace-steering critique and reply; supervised synthetic late steering passes, output-weight control matches it, early shuffled-supervision control fails.
- [EXP-0007 registration](../experiments/EXP-0007.md) and [results](../experiments/EXP-0007-results.md): word-form/conditional-entropy motivation, matched group/character corruptions, weak contrast and untested boundary/recency hypothesis.
- [Completed parallel campaign](../experiments/CAMPAIGN-0001-results.md): all three tracks, finite resources, verified source/holdouts and combined interpretation.
- [EXP-0008 results](../experiments/EXP-0008-results.md): blind fitting helps familiar synthetic keys but unfamiliar-key/family transfer fails; narrow copy control and output-information limitations retained.
- [EXP-0009 results](../experiments/EXP-0009-results.md): earlier synthetic computations affect later predictions; untrained/manuscript criteria fail; broad replacement is not state-machine recovery.
- [EXP-0010 results](../experiments/EXP-0010-results.md): matched-exposure longer contexts fail registered criteria; boundary information and sparse full-prefix coverage remain qualified.
- [Latest scientific overview](../../results/CAMPAIGN-0001/overview.png): plotted directly from archived EXP-0008/0009/0010 reports. [Previous overview](../../results/research-round-2-2026-09-20/overview.png) retains EXP-0005/0006/0007.
- [Predictive-rule extraction notes](../research/PREDICTIVE_RULES.md) and [Belief Net review](../research/BELIEF_NET_REVIEW.md): primary-method reviews added after campaign registration; unimplemented candidates, not empirical manuscript evidence.
- [AI on hard problems and Voynich analogues](../research/ai-hard-problems-2026-09-21/README.md): sourced NS/Erdős/formal-math mechanisms, analogy map, ranked experiments; [source list](../research/ai-hard-problems-2026-09-21/SOURCES.md). Not a decipherment.

The original source register above is preserved for early discussions. The targeted reviews contain their own source/version/access records; follow the exact citation for each claim rather than treating a review as proof of a manuscript hypothesis.

## Navigation and retrieval

Use stable IDs (`SRC-`, `HYP-`, `EXP-`, `NB-`) to connect sources, hypotheses, experiments, and notebook entries. Search this repository with `rg` before creating duplicate records. Create per-source notes and per-experiment records only as actual work occurs. If a retrieval index is added later, treat versioned source documents as authoritative and make the index rebuildable.
