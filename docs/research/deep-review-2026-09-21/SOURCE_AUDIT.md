# Source and coverage audit

Audit date: 2026-09-21. Three topic drafts plus root theory synthesis were consolidated; a separate bounded review checked high-impact recent claims and incomplete metadata. This was not an independent reread of every source or a replication of any paper.

## Corrections incorporated

| Record | Correction or qualification | Evidence |
| --- | --- | --- |
| D14 | Authors corrected to Jeff Shen and Lindsay M. Smith; upgraded to v2, 2025-09-25. Method and ablation sections inspected in the newer version. | [Primary v2](https://arxiv.org/html/2509.07282v2) |
| D13/D15/D19/D25 | Filled previously missing author lists; retained month-level date where only a month is established; pinned D15 v1 and D19 v3. | Primary records in the ledger |
| D22/T13 | Full author list; 2020 preprint versus 2021 publication distinguished; grouped as one work. | [DreamCoder](https://arxiv.org/abs/2006.08381v1) |
| T09 | Author names corrected from an unverified expansion to Binyamin Perets and Mark Kozdoba, with Shie Mannor. | [PMLR record](https://proceedings.mlr.press/v202/perets23a.html) |
| A02 | Official September release and report details cross-checked; local PDF hash independently matched. | [Official release](https://deepseek.com/en/news/deepseek-v4-1-flash/), [report](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/DeepSeek_V41_Tech_Report.pdf) |
| M12 | Replaced blanket reconstruction/stability improvement with regime- and metric-dependent claim; real-activation stability can worsen while reconstruction improves. | [Tables 1–2](https://arxiv.org/pdf/2605.31245) |
| M13/M17 | Dates, authors and bounded experimental settings checked; no general mechanism-recovery claim inferred. | URLs and limitations in interpretation ledger |
| D20/D21 | Confirmed known-key image setup and invariant symbol-meaning pool respectively. D20 venue status recorded as conference-paper preprint. | [Image paper](https://arxiv.org/html/2606.27700v1), [LSTM paper](https://arxiv.org/html/2606.05078v1) |

## Version and access limits

- A01: identifier `2606.19348v1` and displayed April 26 date disagree. Both are retained; no inferred correction.
- D23: identifier `2609.20835` and displayed July 28 submission date disagree, including on the primary submission history. Unresolved.
- A16: a later rendered footer is not evidence for a newly published version.
- A02: official PDF is mutable. Local temporary artifact `/private/tmp/voynich-architecture-review/DeepSeek_V41_Tech_Report.pdf`, SHA-256 `ba68e2e40408125ae6d2f63a9a241b61c73910691c74ec1a2a7023c851eac08d`. Temporary availability is not a permanent backup; source URL and hash are retained, PDF excluded from Git.
- A07 and author repositories are mutable resources; an access date does not identify an immutable revision.
- Some theoretical PDFs use an unversioned primary URL. When an exact revision date was not verified, the ledger leaves it null rather than inventing precision.
- M24 cites an accessible original author report; an attempted later OpenReview source was inaccessible. No claim to have read that blocked source.
- Reading depths describe actual selected sections or retrieved excerpts. Only M07, a short note, is marked full text. Long proof appendices were not comprehensively checked.
- Search-result snippets and discovery-level abstracts do not justify implementation adoption without follow-up methods review.

## Coverage and reproducibility

The catalog validator checks required fields, identifiers, author/claim/limitation lists, depth/decision labels, URL form, checksum format, duplicate works and generated-index freshness. This validates record structure, **not citation truth or website availability**. The source audit above supplies targeted semantic checking.

The review includes primary papers, official technical reports, original preliminary research notes, model cards, manuscript-custodian pages and author repositories. They are not equivalent levels of evidence. An `adopt_candidate` decision is a research priority, not an empirical success.

No source PDFs, large corpora, model weights, API credentials or generated research datasets are added to Git. Existing scientific source and original research charter remain unchanged. No paid research API was used; conversational model credit usage was not measured.

Remaining work includes full proofs where a theorem will support a concrete claim, implementation-specific source review for currently abstract-only candidates, replication of selected methods, and independently verified external manuscript anchors.
