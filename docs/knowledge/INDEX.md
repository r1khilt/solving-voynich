# Knowledge base

## Status and evidence labels

This initial knowledge base organizes the charter. It is **not a completed literature review**. No external historical, paleographic, cryptanalytic, or machine-learning claim has yet been independently checked for this project.

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

When adding an external source, record title, authors/maintainer, date/version, exact URL or identifier, access date, relevant page/section, claims supported, limitations, rights/access constraints, and local path/checksum if downloaded. Prefer original manuscript images, transcription documentation, primary papers, and reproducible code over summaries.

## Questions needing external grounding later

These are deferred discovery topics, not established findings or an active research assignment:

- Manuscript custody, dating evidence, foliation, missing/reordered leaves, and available scans.
- Transcription systems, disputed glyph readings, layout preservation, annotator disagreement, and redistribution terms.
- Currier classifications, scribal distinctions, manuscript sections, and how securely each annotation is established.
- Position effects, repetition, near-neighbor forms, entropy, dependence, and the effect of transcription/segmentation choices on each.
- Prior cipher, language, abbreviation, null/filler, and copy/mutate proposals; which tests actually discriminate among them.
- Earlier neural modeling and decipherment attempts, their datasets, leakage risks, available code, and claims that survived independent testing.

## Navigation and retrieval

Use stable IDs (`SRC-`, `HYP-`, `EXP-`, `NB-`) to connect sources, hypotheses, experiments, and notebook entries. Search this repository with `rg` before creating duplicate records. Create per-source notes and per-experiment records only as actual work occurs. If a retrieval index is added later, treat versioned source documents as authoritative and make the index rebuildable.
