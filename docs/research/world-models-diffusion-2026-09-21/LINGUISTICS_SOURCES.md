# Linguistics and decipherment source ledger

Access date for every live check: **2026-09-21**. This ledger contains **32 records**: original research, author surveys/theoretical articles, authored typology chapters, and original accompanying materials. It is a targeted foundation, not an exhaustive literature review or 32 full-paper reads. Sources were not replicated. The accompanying [chapter](LINGUISTICS_GROUNDING.md) distinguishes reported work from our architectural synthesis.

**Depth labels:** `selected_sections` means the named source passages were inspected; it never means the complete paper was read. `abstract_or_excerpt` means the abstract and/or limited indexed excerpts informed the narrow claim. `documentation` means the named original resource documentation was inspected. `bibliographic_pointer` means the original text was not retrievable and no independent methodological assessment is claimed. A PDF being opened by the browser is not counted as a full read.

Publisher publication dates, preprint dates and web migration dates are distinguished below. Mutable resource pages are not pinned to a commit in this review. No downloaded paper was added to Git, and no source-content checksum is claimed.

## Direct Voynich research and competing accounts

### L01 — The Linguistics of the Voynich Manuscript

- **Authors:** Claire L. Bowern; Luke Lindemann.
- **Date/version:** *Annual Review of Linguistics* 7 (2021), 285–308; DOI 10.1146/annurev-linguistics-011619-030613. An earlier September 2020 author preprint is described in the repository's prior PDF assessment; this record's live check used the published landing page.
- **Primary URL:** [Publisher article](https://www.annualreviews.org/content/journals/10.1146/annurev-linguistics-011619-030613).
- **Depth:** `abstract_or_excerpt`. Published abstract, bibliographic details and article context; no fresh full reread in this pass.
- **Supports:** A linguistic synthesis and the authors' preference for treating the text as encoded language.
- **Limits:** That preference is an author interpretation, not established consensus. The present chapter does not treat it as proof against structured nonsemantic production. Prior repository assessment supplied continuity, not a substitute for claiming a new full read.

### L02 — Character Entropy in Modern and Historical Texts: Comparison Metrics for an Undeciphered Manuscript

- **Authors:** Luke Lindemann; Claire Bowern.
- **Date/version:** Author-hosted PDF dated October 27, 2020; 51 PDF pages. Associated arXiv record 2010.14697 has later revisions; the date printed on the inspected author PDF identifies the version used here.
- **Primary URLs:** [Author PDF](https://www.lukelindemann.com/docs/voy_entropy_wp_2020.pdf); [arXiv record](https://arxiv.org/abs/2010.14697).
- **Depth:** `selected_sections`. Abstract; §§1–3.1.1, especially transcription composition, historical normalization, section/hand distinctions; contents and summary location also inspected. The full entropy tables and all appendix data were not audited.
- **Supports:** Treating script conventions and representation as important variables in entropy comparisons; examined transcription changes do not by themselves normalize the reported Voynich pattern.
- **Limits:** Results concern the compared corpora and transformations. They do not prove a unique channel, a phonemic interpretation, or an underlying language. Some historical interpretation in the introduction is stronger than the measurements alone.

### L03 — What We Know About The Voynich Manuscript

- **Authors:** Sravana Reddy; Kevin Knight.
- **Date/version:** LaTeCH 2011, 78–86; June 24, 2011 proceedings PDF.
- **Primary URLs:** [Paper](https://aclanthology.org/W11-1511/); [PDF](https://aclanthology.org/W11-1511.pdf).
- **Depth:** `selected_sections`. §§2–5, including transcription, letter classes, word frequencies, morphological signatures, local word prediction and correlation discussion.
- **Supports:** Multi-level analysis reveals substantial structure and representation-dependent behavior; the authors discuss abjad-like alternatives.
- **Limits:** Inferred classes lack known meanings; an abjad is a hypothesis. Their dated background claims and manuscript counts are not used as current authority. Their corpus and comparison languages differ from the repository's ZL3b setup.

### L04 — Topic Modeling in the Voynich Manuscript

- **Authors:** Rachel Sterneck; Annie Polish; Claire Bowern.
- **Date/version:** arXiv:2107.02858, July 2021; accessed PDF through the unversioned endpoint, 18 pages. No claim of a pinned revision.
- **Primary URLs:** [Record](https://arxiv.org/abs/2107.02858); [PDF](https://arxiv.org/pdf/2107.02858).
- **Depth:** `selected_sections`. Topic/hand/illustration analysis around Figures 16–19 and §5 discussion, PDF pp.12–15; abstract and metadata.
- **Supports:** Topic partitions, hands, illustration groupings and Currier labels have relationships that are not identical.
- **Limits:** The authors' semantic explanation is an interpretation of associations. These analyses do not make section, scribe and topic independently randomized factors or supply translated topics.

### L05 — Keywords and Co-Occurrence Patterns in the Voynich Manuscript: An Information-Theoretic Analysis

- **Authors:** Marcelo A. Montemurro; Damián H. Zanette.
- **Date/version:** *PLOS ONE* 8(6):e66344, June 21, 2013; DOI 10.1371/journal.pone.0066344.
- **Primary URLs:** [Publisher article](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0066344); [original full-text archive](https://pmc.ncbi.nlm.nih.gov/articles/PMC3689824/).
- **Depth:** `abstract_or_excerpt`. Abstract, opening rationale and indexed passages on co-occurrence and semantic interpretation; full numerical procedure not independently audited.
- **Supports:** Nonuniform word distributions and co-occurrence networks are serious structural targets.
- **Limits:** Calling the networks semantic presupposes an interpretation not independently established by their topology. The chapter makes no replicated numerical claim and does not infer that all hoax accounts are excluded.

### L06 — Decoding Anagrammed Texts Written in an Unknown Language and Script

- **Authors:** Bradley Hauer; Grzegorz Kondrak.
- **Date/version:** *TACL* 4 (2016), 75–86; published April 11, 2016; DOI 10.1162/tacl_a_00084.
- **Primary URLs:** [Paper](https://aclanthology.org/Q16-1006/); [PDF](https://aclanthology.org/Q16-1006.pdf).
- **Depth:** `selected_sections`. Abstract; §5.2 conclusions and §§5.3–5.4 on alphagrams and Voynich application, including the source-language score context.
- **Supports:** Controlled language identification and anagram/substitution decoding can work under specified channel assumptions; their Voynich scores prefer Hebrew in some comparisons.
- **Limits:** Those results do not establish a Voynich translation. Candidate-language selection, channel assumptions and modern comparison corpora shape the score. Exact benchmark percentages are not transferred to the unknown-manuscript setting.

### L07 — A possible generating algorithm of the Voynich manuscript / SelfCitationTextgenerator materials

- **Authors:** Torsten Timm; Andreas Schinner; accompanying repository maintained by Torsten Timm.
- **Date/version:** Article: *Cryptologia* 44(1) (2020), 1–19, first online 2019; DOI 10.1080/01611194.2019.1596999. Materials: mutable GitHub default branch accessed September 21, 2026; no commit pinned.
- **Primary URLs:** [Article DOI](https://doi.org/10.1080/01611194.2019.1596999); [author materials](https://github.com/TorstenTimm/SelfCitationTextgenerator).
- **Depth:** `documentation`. README identification, network-graph categories and generator/source-code availability. Article abstract discovered, not a complete paper or code audit.
- **Supports:** A concrete structured self-citation alternative exists and has original comparison materials.
- **Limits:** This review does not reproduce it, adjudicate all its claims, or assert that the manuscript was historically meaningless. The repository's separately recorded simplified generator is not equated with the published method.

### L08 — How Many Glyphs and How Many Scribes? Digital Paleography and the Voynich Manuscript

- **Author:** Lisa Fagin Davis.
- **Date/version:** *Manuscript Studies* 5(1) (2020), 164–180; DOI 10.1353/mns.2020.0011.
- **Primary URLs:** [Publisher DOI](https://doi.org/10.1353/mns.2020.0011); [original institutional PDF endpoint](https://repository.upenn.edu/cgi/viewcontent.cgi?article=1082&context=mss_sims).
- **Depth:** `bibliographic_pointer`. DOI access was blocked and the institutional endpoint failed. The paper's identity and five-hand attribution were encountered in the bibliographies/discussion of L02 and L04; no independent full-text examination here.
- **Supports:** An essential paleographic research branch to integrate, with explicitly qualified attribution of the five-hand proposal.
- **Limits:** Do not count this as a fresh verification of the classification or use it to infer semantics, community identity, geographical origin or communicative intent.

## Historical decipherment and its supplied assumptions

### L09 — A Statistical Model for Lost Language Decipherment

- **Authors:** Benjamin Snyder; Regina Barzilay; Kevin Knight.
- **Date/version:** ACL 2010, 1048–1057.
- **Primary URLs:** [Paper](https://aclanthology.org/P10-1107/); [PDF](https://aclanthology.org/P10-1107.pdf).
- **Depth:** `selected_sections`. Input specification around §4; §§5.1–5.2 modeling rationale and hierarchical generation; introduction/result summary.
- **Supports:** Coupling character and morpheme correspondences with sparse mappings under a known-related-language setup.
- **Limits:** Input includes a morphologically analyzed lexicon in the known language and segmented word types in the lost one. No paired lexicon is supplied as supervision, but external related-language information is substantial. Ugaritic was already deciphered by humans.

### L10 — Neural Decipherment via Minimum-Cost Flow: From Ugaritic to Linear B

- **Authors:** Jiaming Luo; Yuan Cao; Regina Barzilay.
- **Date/version:** ACL 2019, 3146–3155; DOI 10.18653/v1/P19-1303.
- **Primary URLs:** [Paper](https://aclanthology.org/P19-1303/); [PDF](https://aclanthology.org/P19-1303.pdf).
- **Depth:** `selected_sections`. Introduction, related work and §3 assumptions: distributional correspondence, monotonicity, sparsity and coverage; reported Linear B result definition.
- **Supports:** Neural alignment combined with global lexical constraints can recover cognates across alphabetic and syllabic settings without paired training examples.
- **Limits:** A known related-language corpus and constrained correspondence structure remain essential. The 67.3% reported Linear B result concerns cognates, not unrestricted translation of an undeciphered archive. Later detailed optimization tables not replicated.

### L11 — Deciphering Undersegmented Ancient Scripts Using Phonetic Prior

- **Authors:** Jiaming Luo; Frederik Hartmann; Enrico Santus; Regina Barzilay; Yuan Cao. This follows the published TACL author order; the arXiv listing differs in its final two names.
- **Date/version:** *TACL* 9 (2021), 69–81; DOI 10.1162/tacl_a_00354; published PDF.
- **Primary URLs:** [Paper](https://aclanthology.org/2021.tacl-1.5/); [PDF](https://aclanthology.org/2021.tacl-1.5.pdf).
- **Depth:** `selected_sections`. Introduction and §§2–3 setup/phonological constraints; indexed results/conclusion passages and the caveat about true isolates.
- **Supports:** Joint segmentation and cognate alignment using phonetic features, sound preservation and monotonicity; partial inquiry even when closest related language is uncertain.
- **Limits:** Iberian does not acquire a validated translation; Basque relatedness is not established. Success on existing known-script/known-language controls does not remove the need for relevant external phonological evidence.

### L12 — The Copiale Cipher

- **Authors:** Kevin Knight; Beáta Megyesi; Christiane Schaefer.
- **Date/version:** BUCC 2011, 2–9; June 2011 proceedings paper.
- **Primary URLs:** [Paper](https://aclanthology.org/W11-1202/); [PDF](https://aclanthology.org/W11-1202.pdf).
- **Depth:** `selected_sections`. §§4–5 decipherment narrative, failed symbol-role assumptions, German evidence, homophonic reasoning and the boundary role of Roman letters.
- **Supports:** Historically successful decipherment can require multiple symbol roles and iterative correction of the channel model using language evidence.
- **Limits:** Computational and human inference were intertwined; this is not a demonstration of autonomous unknown-language recovery. Its early dating statements are not used as current historical authority.

## General linguistic representations

### L13 — Fusion of Selected Inflectional Formatives

- **Authors:** Balthasar Bickel; Johanna Nichols.
- **Date/version:** WALS Online, chapter 20, current authored online chapter accessed September 21, 2026; the chapter belongs to the WALS reference tradition, not a new 2026 study.
- **Primary URL:** [WALS chapter 20](https://wals.info/chapter/20).
- **Depth:** `selected_sections`. §1 definitions; §2 sample/feature explanation; §3 discussion.
- **Supports:** Phonological fusion is a distinct typological variable; isolating, concatenative and nonlinear realizations can be distinguished.
- **Limits:** Sample counts concern the selected formatives and languages. They are not a universal ranking of languages or a ready-made distribution over historical manuscript channels.

### L14 — Exponence of Selected Inflectional Formatives

- **Authors:** Balthasar Bickel; Johanna Nichols.
- **Date/version:** WALS Online, chapter 21; current authored online text accessed September 21, 2026.
- **Primary URL:** [WALS chapter 21](https://wals.info/chapter/21).
- **Depth:** `selected_sections`. Opening and §1 definition, including cumulative versus separative category realization and independence from phonological fusion.
- **Supports:** One formative may express several grammatical categories; exponence must not be conflated with fusion.
- **Limits:** Does not establish how any Voynich substring functions or require that its graphic groups correspond to morphemes.

### L15 — Universal Dependencies

- **Authors:** Marie-Catherine de Marneffe; Christopher D. Manning; Joakim Nivre; Daniel Zeman.
- **Date/version:** *Computational Linguistics* 47(2) (2021), 255–308; DOI 10.1162/coli_a_00402.
- **Primary URLs:** [Paper](https://aclanthology.org/2021.cl-2.11/); [PDF](https://aclanthology.org/2021.cl-2.11.pdf).
- **Depth:** `selected_sections`. Framework abstract; §2.2 discussion of content/function words, grammaticalization, clitics, segmentation and word classes, especially PDF pp.5–7.
- **Supports:** Crosslinguistic dependency and morphological representations, with acknowledged annotation tradeoffs.
- **Limits:** It is an annotation framework for analyzed languages, not a proof that all latent categories or boundaries in an unknown script must match UD labels. The full 54-page article was not read.

### L16 — The Taxonomy of Writing Systems: How to Measure How Logographic a System Is

- **Authors:** Richard Sproat; Alexander Gutkin.
- **Date/version:** *Computational Linguistics* 47(3) (2021), 477–528; accepted June 3, 2021; DOI 10.1162/coli_a_00409.
- **Primary URLs:** [Paper](https://aclanthology.org/2021.cl-3.16/); [PDF](https://aclanthology.org/2021.cl-3.16.pdf).
- **Depth:** `selected_sections`. Abstract, §1 and opening §2 discussion: phonography versus morphography, mixed systems, the unit-size issue and scope of the proposed measure.
- **Supports:** Separating the phonological size of represented units from degree of morphographic information rather than enforcing exclusive alphabet/syllabary/logography bins.
- **Limits:** Their operational measurement requires pronunciation and spelling data and primarily evaluates modern systems. It cannot directly classify unknown Voynich glyphs or provide their readings. Attention-based measurement is not cited as causal interpretability.

### L17 — Distributional Structure

- **Author:** Zellig S. Harris.
- **Date/version:** *WORD* 10(2–3) (1954), 146–162; DOI 10.1080/00437956.1954.11659520. Publisher online date 2015 is digitization/publication metadata, not the original research year.
- **Primary URLs:** [Publisher](https://doi.org/10.1080/00437956.1954.11659520); [original article PDF hosted by BFSU](https://corpus.bfsu.edu.cn/Zellig_Harris_Distributional_Structure_Word_1954.pdf).
- **Depth:** `abstract_or_excerpt`. Publisher metadata and selected indexed passages from §3.1 and the discourse/substitutability discussion; full argument not newly reread.
- **Supports:** Distributional analysis can concern several linguistic levels and environments, not only modern word embeddings.
- **Limits:** Similar environments do not independently name the semantic category of an unknown sign. The chapter's modern architecture proposals are ours, not attributed to Harris.

### L18 — The Symbol Grounding Problem

- **Author:** Stevan Harnad.
- **Date/version:** *Physica D* 42 (1990), 335–346; DOI 10.1016/0167-2789(90)90087-6. Inspected PDF is a 15-page print of the author's text hosted by Oxford, not the original journal pagination.
- **Primary URLs:** [DOI](https://doi.org/10.1016/0167-2789(90)90087-6); [author-text mirror](https://www.cs.ox.ac.uk/activities/ieg/e-library/sources/harnad90_sgproblem.pdf).
- **Depth:** `selected_sections`. Abstract and §§1.1–1.4, especially explicit versus interpreted rules and symbolic-system definition.
- **Supports:** A clear conceptual distinction between manipulating forms and grounding their interpretation.
- **Limits:** A theoretical proposal about cognition, not a mathematical impossibility theorem for all text-derived semantic inference or a settled verdict on contemporary networks.

### L19 — Climbing towards NLU: On Meaning, Form, and Understanding in the Age of Data

- **Authors:** Emily M. Bender; Alexander Koller.
- **Date/version:** ACL 2020, 5185–5198; DOI 10.18653/v1/2020.acl-main.463.
- **Primary URLs:** [Paper](https://aclanthology.org/2020.acl-main.463/); [PDF](https://aclanthology.org/2020.acl-main.463.pdf).
- **Depth:** `selected_sections`. Abstract, introduction and §§3–4 around the meaning/form distinction and octopus thought experiment.
- **Supports:** The authors' explicit meaning/communicative-intent distinction and the danger of mistaking fluent form prediction for independently grounded interpretation.
- **Limits:** Position paper and thought experiment, not empirical proof that arbitrary relational knowledge cannot be inferred from text. Our identifiability formulation is separately labeled synthesis.

## Acquisition, pragmatics and grounding

### L20 — Statistical Learning by 8-Month-Old Infants

- **Authors:** Jenny R. Saffran; Richard N. Aslin; Elissa L. Newport.
- **Date/version:** *Science* 274(5294) (1996), 1926–1928; December 13, 1996; DOI 10.1126/science.274.5294.1926.
- **Primary URLs:** [DOI](https://doi.org/10.1126/science.274.5294.1926); [original-paper academic mirror](https://people.uleth.ca/~fangfang.li/Psychology3850/readings/Saffrane1996l_StatisticalLearning.pdf).
- **Depth:** `abstract_or_excerpt`. Abstract and indexed experiment/limitations passages; PDF retrieved but the entire experimental/statistical argument was not reviewed.
- **Supports:** Controlled evidence for learning regularities relevant to segmentation in artificial speech streams.
- **Limits:** Does not establish semantic word learning, unknown-script recovery, or the general adequacy of one statistical-learning mechanism for natural acquisition. No replication or methodological meta-analysis here.

### L21 — Rapid Word Learning Under Uncertainty via Cross-Situational Statistics

- **Authors:** Chen Yu; Linda B. Smith.
- **Date/version:** *Psychological Science* 18(5) (2007), 414–420; DOI 10.1111/j.1467-9280.2007.01915.x.
- **Primary URLs:** [DOI](https://doi.org/10.1111/j.1467-9280.2007.01915.x); [original abstract record](https://pubmed.ncbi.nlm.nih.gov/17576281/).
- **Depth:** `abstract_or_excerpt`. Original abstract and bibliographic record; full paper not read in this pass. Later reopening returned no body, so no improved depth is claimed.
- **Supports:** Cross-situational mappings can be learned from repeated ambiguous exposures under controlled conditions.
- **Limits:** The controlled referent inventory and repeated pairings differ sharply from Voynich images. Generalization to unknown segmentation, uncertain label placement and absent action traces is our speculation.

### L22 — Pragmatic Language Interpretation as Probabilistic Inference

- **Authors:** Noah D. Goodman; Michael C. Frank.
- **Date/version:** *Trends in Cognitive Sciences* 20(11) (2016), 818–829; online September 28, 2016; DOI 10.1016/j.tics.2016.08.005.
- **Primary URLs:** [DOI](https://doi.org/10.1016/j.tics.2016.08.005); [original abstract record](https://pubmed.ncbi.nlm.nih.gov/27692852/).
- **Depth:** `abstract_or_excerpt`. Abstract and bibliographic record.
- **Supports:** RSA as a probabilistic framework connecting literal meaning, speaker choices, context and interpretation.
- **Limits:** Does not justify any particular medieval speaker goal, cost model, common ground or lexicon. Nested-agent architecture and manuscript application in our chapter are proposals rather than reported decipherment results.

### L23 — A World Unto Itself: Human Communication as Active Inference

- **Authors:** Jared Vasil; Paul B. Badcock; Axel Constant; Karl Friston; Maxwell J. D. Ramstead.
- **Date/version:** *Frontiers in Psychology* 11 (2020), article 417; DOI 10.3389/fpsyg.2020.00417.
- **Primary URL:** [Publisher full text](https://www.frontiersin.org/journals/psychology/articles/10.3389/fpsyg.2020.00417/full).
- **Depth:** `abstract_or_excerpt`. Authorship, introductory framing and selected indexed passages on acquisition, precision/attention and alignment of expectations; no complete derivation audit.
- **Supports:** An explicit theoretical connection among communication, generative models, actions and belief alignment.
- **Limits:** Broad theoretical synthesis. Examples of empirical support include work by other authors and are not treated as fresh experiments by Vasil et al. No evidence here that active inference is necessary, uniquely correct or a decipherment solution.

## Procedural semantics and executable interpretation

### L24 — Weakly Supervised Learning of Semantic Parsers for Mapping Instructions to Actions

- **Authors:** Yoav Artzi; Luke Zettlemoyer.
- **Date/version:** *TACL* 1 (2013), 49–62; DOI 10.1162/tacl_a_00209.
- **Primary URLs:** [Paper](https://aclanthology.org/Q13-1005/); [PDF](https://aclanthology.org/Q13-1005.pdf).
- **Depth:** `selected_sections`. §6 semantic representation, spatial reference and event types; §6.3 CCG connection; experimental excerpts specifying action/final-state supervision and seed lexicon.
- **Supports:** Joint contextual parsing and execution with typed entities/events provides a concrete bridge between linguistic structure and actions.
- **Limits:** Defined navigation worlds, observable validation and a seed lexicon are supplied. Their weak supervision is much richer than unlabeled manuscript marks. Exact numerical gains are not imported into our proposal.

### L25 — Mise en Place: Unsupervised Interpretation of Instructional Recipes

- **Authors:** Chloé Kiddon; Ganesa Thandavam Ponnuraj; Luke Zettlemoyer; Yejin Choi.
- **Date/version:** EMNLP 2015, 982–992; DOI 10.18653/v1/D15-1114.
- **Primary URLs:** [Paper](https://aclanthology.org/D15-1114/); [PDF](https://aclanthology.org/D15-1114.pdf).
- **Depth:** `selected_sections`. Abstract, §§1–2 and Figure 1: action graphs, argument flow, implicit arguments and task representation.
- **Supports:** Explicit procedural graphs can connect omitted arguments and evolving materials; redundancy across known-language recipes can support unsupervised graph induction.
- **Limits:** Known English text, a specified domain and structured assumptions are supplied. It does not infer an unknown script or demonstrate physical simulation of arbitrary recipes. Our partial-order/reversible-channel architecture is an extension.

### L26 — Simulating Action Dynamics with Neural Process Networks

- **Authors:** Antoine Bosselut; Omer Levy; Ari Holtzman; Corin Ennis; Dieter Fox; Yejin Choi.
- **Date/version:** ICLR 2018; initial arXiv:1711.05313 submitted November 14, 2017. Inspected arXiv PDF displays the ICLR 2018 publication header; endpoint not pinned to a revision.
- **Primary URLs:** [Record](https://arxiv.org/abs/1711.05313); [PDF](https://arxiv.org/pdf/1711.05313).
- **Depth:** `selected_sections`. §§2.2–2.5 sentence/action/entity modules and state updates; §§3.1–3.3 training, ingredient information and weak supervision; Figure 2.
- **Supports:** Modeling actions as transformations of persistent entity representations rather than treating procedural text solely as a string.
- **Limits:** State-change knowledge, heuristically derived labels and entity lists provide supervision. A neural state transformer need not be an identified physical mechanism; the cited work is not unsupervised ancient-text grounding.

### L32 — Tracking State Changes in Procedural Text: A Challenge Dataset and Models for Process Paragraph Comprehension

- **Authors:** Bhavana Dalvi Mishra; Lifu Huang; Niket Tandon; Wen-tau Yih; Peter Clark.
- **Date/version:** NAACL-HLT 2018; arXiv:1805.06975, first submitted May 17, 2018.
- **Primary URL:** [Original preprint record](https://arxiv.org/abs/1805.06975).
- **Depth:** `abstract_or_excerpt`. Abstract and bibliographic entry; dataset/model details not audited.
- **Supports:** Entity existence and location changes as explicit targets for procedural-text understanding.
- **Limits:** Known-language benchmark and supplied annotations; no unknown-script claim. Included as an adjacent task reference, not evidence that all linguistic meaning reduces to physical state tracking.

## Transfer, composition and alternative symbol systems

### L27 — On the Limitations of Unsupervised Bilingual Dictionary Induction

- **Authors:** Anders Søgaard; Sebastian Ruder; Ivan Vulić.
- **Date/version:** ACL 2018, 778–788; DOI 10.18653/v1/P18-1072; arXiv:1805.03620 first submitted May 9, 2018.
- **Primary URLs:** [Paper](https://aclanthology.org/P18-1072/); [PDF](https://aclanthology.org/P18-1072.pdf).
- **Depth:** `abstract_or_excerpt`. Original abstract and author presentation conclusions; PDF opened, but detailed result sections not audited in this pass.
- **Supports:** Sensitivity of unsupervised dictionary induction to morphology, domain mismatch and embedding choices; crosslinguistic geometric alignment is not guaranteed.
- **Limits:** Does not prove that all multilingual transfer fails or directly measure Voynich. Romanization and geneaological/genre confounding discussion in our chapter is a broader methodological synthesis.

### L28 — Generalization without Systematicity: On the Compositional Skills of Sequence-to-Sequence Recurrent Networks

- **Authors:** Brenden M. Lake; Marco Baroni.
- **Date/version:** ICML 2018, *PMLR* 80; initial arXiv:1711.00350 (2017).
- **Primary URLs:** [Proceedings](https://proceedings.mlr.press/v80/lake18a.html); [PDF](https://proceedings.mlr.press/v80/lake18a/lake18a.pdf).
- **Depth:** `abstract_or_excerpt`. Abstract and task motivation; proceedings PDF retrieved, individual split tables not audited.
- **Supports:** Distinguishing performance on familiar combinations from systematic compositional generalization in a controlled command/action domain.
- **Limits:** Restricted synthetic task and then-current models. It does not establish a permanent architectural impossibility or suffice as a benchmark for historical decipherment.

### L29 — Human-like systematic generalization through a meta-learning neural network

- **Authors:** Brenden M. Lake; Marco Baroni.
- **Date/version:** *Nature* 623 (2023), 115–121; DOI 10.1038/s41586-023-06668-3.
- **Primary URL:** [Publisher full text](https://www.nature.com/articles/s41586-023-06668-3).
- **Depth:** `selected_sections`. Abstract/overview and Methods opening on the few-shot task, primitive/function mappings and changing mappings; indexed interpretation-grammar and meta-learning descriptions. Not a complete supplement audit.
- **Supports:** Task-distribution design and meta-learning can improve controlled compositional behavior.
- **Limits:** Structured episodes and behaviorally informed training are supplied. This is not arbitrary unsupervised grammar discovery, an unknown historical lexicon, or evidence that all forms of human systematicity are solved.

### L30 — Grounded language acquisition through the eyes and ears of a single child

- **Authors:** Wai Keen Vong; Wentao Wang; A. Emin Orhan; Brenden M. Lake.
- **Date/version:** *Science* 383 (2024), 504–511; February 2, 2024; DOI 10.1126/science.adi1374.
- **Primary URLs:** [Publisher DOI](https://doi.org/10.1126/science.adi1374); [original abstract record](https://pubmed.ncbi.nlm.nih.gov/38300999/).
- **Depth:** `abstract_or_excerpt`. Indexed original abstract and bibliographic record. Direct reopening of the PubMed page failed; no full-paper read is claimed.
- **Supports:** A relevant line of work connecting visual experience and transcribed language for grounded word learning.
- **Limits:** No numeric result, raw-audio learning claim, broad grammar acquisition claim or manuscript-transfer claim is made. The temporally coupled observational setting differs from static illustrations.

### L31 — A Statistical Comparison of Written Language and Nonlinguistic Symbol Systems

- **Author:** Richard Sproat.
- **Date/version:** *Language* 90(2) (2014), 457–481; DOI 10.1353/lan.2014.0042. Cambridge's migrated page displays January 1, 2026 as an online date; that is not the original study date.
- **Primary URL:** [Publisher record via DOI](https://doi.org/10.1353/lan.2014.0042); [current publisher abstract](https://www.cambridge.org/core/journals/language/article/abs/statistical-comparison-of-written-language-and-nonlinguistic-symbol-systems/9D2C4213767B8A8DEBEC765AB6517955).
- **Depth:** `abstract_or_excerpt`. Publisher abstract and bibliographic information only.
- **Supports:** The empirical distinction between language-writing and nonlinguistic symbol systems needs more than unqualified entropy-based assertions.
- **Limits:** We do not reproduce its comparison statistics or apply its findings directly to Voynich. Semasiographic and mixed-notation hypotheses in our chapter are explicit proposals, not results of this paper.

## Scope and reading audit

The deepest newly inspected passages concern writing-system taxonomy, linguistic category boundaries, Voynich representation, topic/hand relationships, historical decipherment assumptions, and procedural state/action interfaces. Acquisition, RSA, recent child-perspective grounding and some transfer/compositionality works are intentionally labeled as limited-depth references rather than fully reviewed evidence. L08 remains an access-limited bibliographic pointer. These differences matter: broad conceptual coverage is not the same claim as an exhaustive systematic review.

Shared ideas with the other chapters—world-state identifiability, causal mechanism recovery, structured diffusion and amortized inference—are developed in the review's joint synthesis. This ledger does not double-count those papers as independently read linguistic sources. No source provides evidence of an actual Voynich translation; no architecture in the accompanying chapter was implemented or evaluated.
