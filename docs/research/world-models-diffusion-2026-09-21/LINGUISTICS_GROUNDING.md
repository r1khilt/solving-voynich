# Linguistics as the interface between symbols and worlds

Research and ideation only, 2026-09-21. This is a conceptual foundation and a set of ambitious architectural proposals, not an experiment registration or implementation. No training, manuscript scoring, corpus transformation or translation was performed. The [source ledger](LINGUISTICS_SOURCES.md) distinguishes inspected sections, abstract-level discovery and access failures. Our extensions are labeled as proposals; none is an established account of Voynich.

## 1. The central idea: learn a theory of communication, not just a cipher

A powerful decipherment system should explain how someone represents a world, chooses what to communicate, expresses that content, and writes it through a graphic convention. Those are different transformations. A world model constrains possible events and relations; a linguistic model constrains possible expressions; a writing-channel model explains the visible marks. The ambition is to infer their coupling, rather than ask a generative model for an attractive English interpretation.

Our proposed target is a posterior over **world structures, discourse plans, linguistic realizations and writing channels**. Its latent vocabulary should include objects, events, relations, quantities, times, reference, modality and speaker goals. It must also admit families in which the marks encode a domain notation directly, or arise from structured nonsemantic production. This broadens the search beyond “which European language was substituted?” without giving an unconstrained translator permission to explain anything.

The linguistic foundation matters before architecture selection. A syllable model, morphological model, dependency parser, event simulator and pragmatic reasoner solve different problems. Adding them together is useful only if their interfaces match the information preserved in the manuscript. Diffusion might search uncertain structures jointly; world models might impose relational and temporal constraints; VLA methods might suggest shared action representations. None supplies the missing interface by its name alone.

## 2. General linguistics: the levels are distinct, and their boundaries do not line up

The following is a working glossary and modeling synthesis. It deliberately separates observable objects from inferred linguistic units.

| Level | What the unit means | Why it matters to an unknown script |
| --- | --- | --- |
| Stroke and glyph instance | A particular visible mark and its component pen traces | Different scribes can draw the same unit differently; joined strokes can be missegmented |
| Grapheme and allograph | A contrastive unit of a writing system and its graphical variants | Two shapes need not be two letters; one shape can combine several units |
| Phonetic segment and phoneme | A produced sound versus a language-specific sound contrast | Neither is directly observed in a silent manuscript |
| Syllable and mora | Units of phonological organization and timing | A written group can encode a syllable, consonant sequence or mora rather than a letter |
| Morpheme and allomorph | A recurring grammatical or lexical contribution and its surface variants | A function may have several spellings, or several functions may share one spelling |
| Lexeme and word form | A vocabulary item versus a particular inflected realization | Inflections inflate the observed vocabulary without creating new meanings |
| Syntactic word, clitic, phrase | Units of grammatical combination | Spaces can split one grammatical unit or join several |
| Clause and construction | A predicate structure or a conventional form–function pairing | Meaning can belong to a pattern larger than a single word |
| Discourse and pragmatics | Reference, information flow and communicative use across utterances | What is left unsaid can depend on shared knowledge and genre |

Several mappings can be many-to-many. A single grammatical distinction might appear as an affix, a separate particle, word order, tone, or no obligatory overt marker. One visible ending might jointly signal number and case. An unchanged written form might cover multiple pronunciations, lexemes or grammatical roles. A model that assumes a neat chain of one-to-one conversions can score well while learning its own convenient fiction.

Writing systems should not be divided into “alphabetic” versus “pure pictures.” Sproat and Gutkin distinguish the phonological size of represented units from the degree of morphographic information; they emphasize mixed systems. Their proposed quantitative measure needs known pronunciations and contextual spellings. It is a useful conceptual decomposition, not a classifier we can apply to Voynich without those inputs. [L16](https://aclanthology.org/2021.cl-3.16/)

**Our implication:** infer alternative unitizations jointly with content. Keep at least three representations conceptually separate: a faithful visual trace, a lattice of plausible graphic units, and a hypothesis about linguistic units. A high-confidence semantic guess should not silently rewrite the first layer. Conversely, a fixed EVA token should not become an ontological assertion that the medieval writer perceived that exact unit.

### Orthography is a channel with conventions, not a transparent recording

A writing channel can contain ligatures, context-conditioned letterforms, abbreviations, suspensions, contractions, spelling variation, vowel omission, phonographic complements and nonalphabetic signs. Some phenomena change appearance without changing lexical content; others intentionally omit information that a competent reader reconstructs. The right mathematical object is often a transduction with context and uncertainty, not a permutation of characters.

Lindemann and Bowern explicitly compare Voynich transcriptions and historical diplomatic versus normalized text. Their study connects character entropy to segmentation, script size, word position and scribal conventions. Its finding that the examined manipulations did not normalize Voynich's unusual predictability constrains those tested manipulations; it does not eliminate all composite writing channels or establish a plaintext language. [L02](https://www.lukelindemann.com/docs/voy_entropy_wp_2020.pdf)

Our design consequence is more radical than preprocessing hygiene. The channel should be a first-class explanatory component. Otherwise, the world model may spend its capacity predicting spelling artifacts, and a morphology detector may mistake a cipher suffix for a grammatical ending. Separating these components creates the possibility of learning a meaning-preserving rewrite across hands, sections or orthographic conventions.

### Semasiography is an important alternate branch

A meaningful notation need not encode a spoken sentence. A map, equation, calendar, ritual diagram or domain-specific ledger can specify relations directly. It may support several verbal renderings without one recoverable original utterance. Such a system is not thereby meaningless, and interpreting it is not identical to phonetic decipherment.

Sproat's statistical comparison of linguistic and nonlinguistic symbol systems directly challenges simple entropy-based discrimination. This review inspected its publisher abstract rather than the full study, so the result motivates an alternative family rather than a detailed quantitative claim. [L31](https://doi.org/10.1353/lan.2014.0042)

**Our proposal:** let the historical hypothesis choose between a phonographic route, a morphographic route, a direct concept/action notation, and mixtures. Under the direct route, success could be a stable interpretation of operations, quantities or correspondences with no pronunciation. We should state that as interpretation of a notation, not invent a reconstructed language to make the output resemble a conventional translation.

## 3. Morphology and typology: learn factors, not language stereotypes

Morphology concerns systematic relations among word forms, their components and grammatical functions. An inflection can express tense, aspect, number, person, case or agreement; derivation can produce a new lexical item; compounding can build another word from existing ones. These are useful distinctions even if the visible word boundaries are unreliable. The system needs to represent both the composition and the distribution of resulting forms.

Traditional labels such as isolating, agglutinative and fusional compress several variables. Bickel and Nichols separate **fusion**—how a formative connects to a host—from **exponence**—how many categories a formative expresses. Their WALS chapters show why these axes should not be treated as a single universal scale. [L13](https://wals.info/chapter/20), [L14](https://wals.info/chapter/21)

Our generative prior should therefore sample these factors independently where linguistically reasonable: concatenative versus nonconcatenative structure; cumulative versus separative marking; degree of allomorphy; cliticization; affix placement; templatic patterns; optional omission. It should include mixed systems within one language. A family-level label is a soft distribution over these choices, not a verdict inferred from a few suffix counts.

An essential distinction is **form similarity versus paradigm membership**. Two strings differing by one symbol could be singular and plural, two homophones of one message unit, neighboring outputs of a copy–mutate generator, or unrelated words in a small alphabet. A paradigm hypothesis gains substance when the same transformation recurs across a class with consistent distributional or semantic consequences. A suffix cluster alone does not tell us whether it marks case, tense, a writer's preference, or nothing linguistic.

Our ambitious representation would combine a lexeme graph with a realization grammar. Lexeme nodes carry uncertain semantic types; edges carry recurring grammatical transformations; the realization grammar permits several surface outputs. In a genuine language case, this allows the system to pool evidence across sparse forms. In an encoding case, a separate graphic transducer can explain superficially similar regularities. Model competition happens at the correct level, instead of requiring “all morphology” or “all cipher.”

### Typology constrains possibilities but cannot simply identify a language

Order and dependency are separate. A language can vary constituent order while preserving case or agreement cues. A short technical register can omit arguments and articles that occur in other genres. A noun-heavy inventory can make a language look unlike its conversational form. A corpus written by one institution can overrepresent formulae. Therefore a broad multilingual reference is valuable, but a modern news or Wikipedia distribution should not become the default template for every historical text.

Universal Dependencies offers an explicit crosslinguistic inventory of grammatical relations and morphological features. Its own discussion acknowledges difficult boundaries among function words, clitics and affixes and the distortion introduced by categorical annotation. It is a valuable source of structured supervision for known languages, not an oracle for the unknown segmentation and grammar of Voynich. [L15](https://aclanthology.org/2021.cl-2.11/)

**Our synthesis:** use linguistic annotations as representational examples during external training, then allow alternate analyses at inference. A graph over relations can be more transferable than a fixed English constituency tree, but neither should be declared universal simply because it is convenient to annotate. Particularly for terse procedural or tabular material, graph fragments and unresolved arguments may be more faithful than complete sentences.

## 4. Syntax, composition and discourse as structured state updates

A grammar characterizes possible combinations and dependencies; semantics describes their interpretations. A finite-state grammar can model local templates, a context-free grammar nested structure, and richer formalisms additional dependencies. These are useful computational distinctions, not a reason to assume the manuscript needs the most expressive grammar available. With finite text, an expressive grammar can memorize the corpus. The problem is to identify reusable relations with a constrained description.

A linguistic world model needs two different states. **World state** tracks objects and events. **Discourse state** tracks what has been introduced, what is salient, which descriptions refer to the same object, and what the intended reader already knows. “Grind the roots. Add them to the vessel” changes both, but in different ways. The second instruction needs a discourse referent before a simulator can determine which physical object changes. A model containing only a physical latent state may invent the wrong referent while still generating a physically plausible continuation.

Composition is also not just concatenation. Negation can deny an event; quantification can range over entities; modality can describe possibility, obligation or prescription; conditionals describe branches that need not actually occur. A historical medical text can report beliefs or recommendations whose claimed causal effects are false. Thus a competent solver must distinguish the **world described by the author** from the world accepted by a modern simulator. A modern medical or botanical prior that rejects obsolete theories could erase exactly the historical content we want to recover.

The original SCAN work demonstrates how success on familiar command combinations can diverge from systematic composition on novel ones. Lake and Baroni's later meta-learning study shows that deliberately training across changing compositional tasks can produce stronger systematic behavior in its controlled setting. The latter supplies structured episodes and behaviorally informed training; it does not demonstrate unsupervised discovery of a lost lexicon. [L28](https://proceedings.mlr.press/v80/lake18a.html), [L29](https://www.nature.com/articles/s41586-023-06668-3)

Our use of this evidence is architectural: train inference over how new symbolic systems compose, not merely over many documents sharing a fixed vocabulary. The latent factors should include what an operator does, what argument types it accepts, and how a surface grammar expresses it. That provides a route for a VLA-inspired learner to transfer “combine two materials, then treat the result” even when the words, word order and script all change.

Discourse introduces additional structure beyond recipes. A catalogue may repeatedly instantiate a schema. An astronomical section may define cyclic relations. A commentary may explain a diagram rather than narrate events. A liturgical text may encode actions and participants but derive its ordering from convention. An ambitious system should support these as distinct document-generating programs with shared lower-level components.

## 5. Distributional semantics and the grounding gap

Distributional analysis exploits the environments in which forms occur. Harris's original framework spans multiple linguistic scales rather than merely putting words into vectors. Similar distributions can reveal classes and substitution relations without an initial translation. That is useful structural evidence; assigning an English label to the resulting class is an additional inference. [L17](https://doi.org/10.1080/00437956.1954.11659520)

A word embedding can place two objects together because they are similar, because one acts on the other, because both appear in one topic, or because the same scribe favors their spellings. A visual-text contrastive model has analogous ambiguities. If botanical pages have one hand and bathing pages another, image–text agreement may reflect style rather than object meaning. These are different causal explanations for the same association.

Harnad formulates the symbol-grounding problem as the gap between formal manipulation and an interpretation connected to objects and categories. Bender and Koller make a related argument using their definition of meaning as a relation to communicative intent. These are influential theoretical positions, not blanket impossibility theorems for every kind of inference from text. [L18](https://doi.org/10.1016/0167-2789(90)90087-6), [L19](https://aclanthology.org/2020.acl-main.463/)

**Our practical version of the problem:** a model can recover a network of distinctions before knowing their names. If two interpretations preserve all observed relations, the manuscript may identify an equivalence class rather than a unique semantic assignment. External knowledge can favor one member, but the solver must disclose where that preference came from. “Latent type 7 behaves like a material transformed by operation 2” can be a substantive intermediate representation; translating it as a particular herb requires another constraint.

There is real upside here. Objects have relational signatures, not just names. An unknown item can be constrained by its position in a process, permitted transformations, quantities, interactions and visual motifs. Many weak constraints can jointly narrow possibilities. The proposed system should exploit that conjunction while preserving which constraints are independent. Ten plausible captions generated by one pretrained model are not ten independent anchors.

## 6. Acquisition: what can a learner extract before it has a dictionary?

Human acquisition suggests several different learning signals: repetition and transitional structure; recurring form–referent co-occurrence; shared attention; social intention; distributional categories; known event schemas; and feedback from successful action. The interesting question is how these signals reduce each other's ambiguity.

Saffran, Aslin and Newport's artificial-stream study supports learning of speech-sequence regularities under a highly controlled exposure. Yu and Smith's cross-situational work shows that repeated ambiguous word–object encounters can support mappings. Neither result equates sequence segmentation with knowing a word's referent, and neither recreates a silent historical manuscript with unknown segmentation and uncontrolled illustrations. [L20](https://doi.org/10.1126/science.274.5294.1926), [L21](https://doi.org/10.1111/j.1467-9280.2007.01915.x)

Our extension is **cross-context relational grounding**. A putative symbol should be inferred from a collection of contexts whose intersections constrain its role. If a visual motif appears in multiple image arrangements and its candidate textual correlate varies with a hypothesized action or quantity, the evidence is richer than one picture–word pairing. The learner should represent an uncertain set of candidate referents, including attributes, parts, relations, whole events and discourse functions. It should not assume every recurrent label names the nearest object.

The contrast with a VLA dataset is decisive. A robot often receives a command, observed initial state, actions and outcomes. A static illustration may show an object type, idealized arrangement, symbolic association, final state, several times at once, or something unrelated to nearby text. We do not know the observation-selection process. The manuscript is missing much of the temporal and action information that makes robot grounding powerful.

Vong and colleagues offer a contemporary bridge: their study uses a single child's visual experience paired with transcribed language for grounded word learning. This review checked its bibliographic abstract only. Its value here is that grounded transfer can be investigated with temporally coupled everyday observations; it does not establish that a comparable signal exists in manuscript pictures. [L30](https://doi.org/10.1126/science.adi1374)

**Our foundation-level idea:** learn an acquisition engine over many unfamiliar communication environments. Each environment changes the lexicon, morphology, graphic channel and available sensory evidence. Some expose actions; some only static scenes; some supply descriptions with omitted arguments; some include distractor images. The central capability is inference about what supervision is present. Transferring such an engine to Voynich is a serious research bet; treating every illustrated page as already aligned training data is not.

## 7. Pragmatics and active inference: why this message, for this reader?

A speaker chooses an utterance among alternatives. That choice can convey information beyond literal content: what is assumed known, what is relevant, which distinction matters, and how much precision is necessary. A technical document may be compressed because its readers share a practice. A label may identify only the distinguishing part of an object. Repetition can be pedagogical, ritual, tabular or rhetorical.

The rational speech act framework models interpretation through nested reasoning about a speaker's informative choices. Goodman and Frank review quantitative applications in known-language situations. Vasil and colleagues propose active inference as an account of communication and the alignment of agents' expectations. The first is a modeling framework with task-specific empirical results; the second is a broad theoretical synthesis. Neither supplies a decipherment algorithm or evidence that Voynich encodes cooperative instructions. [L22](https://doi.org/10.1016/j.tics.2016.08.005), [L23](https://doi.org/10.3389/fpsyg.2020.00417)

Our mathematical sketch makes the added assumptions visible. Let a document producer choose a message `u` given intended content `m`, common knowledge `k`, genre `g` and an expression cost. A reader inverts that choice. Such a model can explain omitted arguments when they are already recoverable, and repeated identifiers when ambiguity would otherwise be high. But an unknown cost function can explain arbitrary omissions. We would need restricted alternatives, explicit common-ground variables and competing producer goals.

The useful active-inference analogy is two-sided. In the modeled historical community, actions may change the world and utterances may change a reader's beliefs. In our present investigation, actions mean acquiring better evidence: a paleographic judgment, a versioned image annotation, a historical parallel, or an independently checked reconstruction. A posterior sampler's internal steps are neither historical actions nor new observations. Calling all three “actions” would obscure the very distinctions we need.

A radical extension is a **communication ecology model**: infer conventions that would let a hypothetical community perform a recurring task using a compact notation. Such a model could compare expressive efficiency, learnability and ambiguity across candidate systems. These are conditional priors over plausible usage, not proof that the manuscript was practical or that its producers optimized communication.

## 8. Event semantics and procedural worlds: the strongest bridge to VLA

The most concrete connection between language and world models is that many utterances describe or prescribe state changes. An event representation can record a process, its participants, time, location, instrument, manner, preconditions and effects. It separates the event from the sentence used to describe it. That separation is useful across languages and writing systems.

Artzi and Zettlemoyer connect grounded semantic parsing to execution using typed logical forms and contextual interpretation. Their training uses external validation from actions or final states, a defined navigation environment, and a seed lexicon. This is powerful evidence for combining language with execution; it is not unsupervised bootstrapping from arbitrary marks. [L24](https://aclanthology.org/Q13-1005/)

Kiddon and colleagues model recipe action graphs with ingredient flow and implicit arguments. Bosselut and colleagues instead learn action operators that update entity representations; their training uses weak labels, ingredient information and state-change knowledge. These works show two complementary interfaces: explicit argument-flow structure and learned state transformation. [L25](https://aclanthology.org/D15-1114/), [L26](https://arxiv.org/abs/1711.05313)

ProPara further formulates procedural understanding through changes to entity existence and location. We inspected the primary abstract only, so it supplies a task reference rather than detailed benchmark claims here. [L32](https://arxiv.org/abs/1805.06975)

### Our proposed reversible procedural semantics

Consider a deliberately invented instruction, not a Voynich reading: “Crush the dried material; warm the liquid; combine them; strain the mixture.” Its semantic object is better represented as a dependency graph than as one mandatory action string. Crushing and warming can proceed independently; combining needs their products; straining needs the mixture. Several sentence orders and surface languages can express that same partial order.

The forward direction generates a linguistic realization from the event graph, then applies a writing channel. The inverse direction infers graph and channel jointly from the observations. A renderer might generate expected illustrations or schematic motifs from selected portions of the graph. “Reversible” here means the architecture exposes both generative and inferential directions with provenance; it does not mean that boiling is physically reversible or that one text uniquely determines its history.

The representation should carry object identity through transformations. “Root,” “powder,” “mixture” and “residue” may refer to related material at different states, not four independent entities. A flow graph can preserve that relationship when language omits it. Quantity and containment constraints can rule out candidate parses before a particular noun is identified. Instrument and location types constrain verbs. This is a genuine route from relational meaning to unknown vocabulary.

But an executable interpretation is not enough. Many arbitrary symbol mappings can describe valid procedures. A simulator with generic mixing and heating will accept huge numbers of plausible recipes. The evidence must come from whether one compact mapping repeatedly predicts manuscript distinctions and independently anchored relations. The world model should supply discriminative constraints, not a theater in which any generated story can be enacted.

### A historically situated world model

For historical texts, physical causality is only one component. The model needs social and cultural relations: named categories, seasonal cycles, ritual roles, symbolic correspondences, attributed properties and conventional classifications. It must permit the author to believe something false. A four-element cosmology can organize a text coherently without being a scientifically correct physical simulator.

Our proposal is to distinguish **physical affordance**, **historical belief**, **document convention** and **illustration convention**. An action might be physically possible, believed therapeutic, expressed as a terse formula and pictured through a traditional motif. Each factor can constrain the others without collapsing them into modern common sense. This is more demanding than adapting a robot policy, but also much closer to what historical understanding requires.

## 9. What direct Voynich research actually licenses

The sources below provide constraints and competing explanations, not a decoding key.

| Work | Useful evidence or method | Limit of the inference |
| --- | --- | --- |
| Bowern and Lindemann, L01 | Linguistic synthesis and strong interest in an encoded-language account | Their preference for language over hoax is an author position, not settled consensus |
| Reddy and Knight, L03 | Analyses at character, word, syntax and page levels; inferred classes and positional regularities | A latent class is not a translated part of speech; their abjad suggestion is a hypothesis |
| Sterneck, Polish and Bowern, L04 | Topic structure considered alongside scribal hand, illustration and Currier labels | Association does not uniquely identify semantic content or eliminate shared causes |
| Montemurro and Zanette, L05 | Information-based word selection and co-occurrence networks | Their semantic interpretation is stronger than identifying nonuniform lexical organization |
| Hauer and Kondrak, L06 | Strong controlled performance for particular substitution/anagram assumptions | A language preference under those assumptions is not an accepted Voynich decipherment |
| Timm and Schinner materials, L07 | Concrete self-citation generator and comparison materials | Matching selected statistics does not establish historical nonsemantic production |
| Davis, L08 | Digital paleography and a five-hand proposal, discussed in the linguistic sources | Hand attribution is neither semantic annotation nor proof of a linguistic code |

Primary links: [L01](https://doi.org/10.1146/annurev-linguistics-011619-030613), [L03](https://aclanthology.org/W11-1511/), [L04](https://arxiv.org/abs/2107.02858), [L05](https://doi.org/10.1371/journal.pone.0066344), [L06](https://aclanthology.org/Q16-1006/), [L07](https://github.com/TorstenTimm/SelfCitationTextgenerator), [L08](https://doi.org/10.1353/mns.2020.0011). Davis's original text was not retrievable in this pass; the ledger marks that access limit explicitly.

Our conclusion from the disagreement is constructive. Build a model of **manuscript production** broad enough to represent a linguistic document, a direct notation, a copy-derived pseudotext and a mixed artifact. A producer state could track a semantic event, a graphic template or a recently copied source. A world-model architecture is capable of learning all three; calling its hidden variables a “world” cannot adjudicate among them.

Illustrations are potentially independent evidence only after accounting for how they were selected, arranged and produced. Repeated forms might refer to the same object, same action, same category, same label template, or same copy source. Sections, hands, layouts and topics are correlated rather than cleanly crossed experimental factors. A joint model can represent this structure; it cannot manufacture missing combinations that the manuscript never contains.

The useful scale increase is therefore not merely more parameters on the same text. It is a wider, historically grounded model of the processes that could produce text, images and layout together. That model should preserve the possibility that one layer is meaningful while another is decorative, conventional or mechanically generated.

## 10. Historical and machine decipherment: the external bridge is part of the method

Successful computational re-decipherment is encouraging precisely when its supplied information is named. Snyder, Barzilay and Knight use unanalysed words from a lost language together with a morphologically analysed lexicon in a known related language, plus sparse correspondences and morpheme structure. Luo, Cao and Barzilay relax parts of that setup using neural correspondences and a global minimum-cost-flow alignment; they still receive a related-language corpus and constrain cognate matching. Their reported Linear B percentage is cognate recovery, not automatic translation of an unknown archive. [L09](https://aclanthology.org/P10-1107/), [L10](https://aclanthology.org/P19-1303/)

Luo and colleagues subsequently jointly model segmentation and cognate alignment using phonetic geometry and sound-change constraints. Their Iberian analysis does not establish a full decipherment or a Basque relationship. The method illustrates useful partial inference and rejection under specified assumptions, not the disappearance of the need for relatedness or interpretable phonetic structure. [L11](https://aclanthology.org/2021.tacl-1.5/)

Copiale supplies a different lesson. Its historical decipherment combined computational evidence with language hypotheses, manual reasoning and revised symbol roles. Initially plausible assumptions about which symbols carried information had to change. The successful account included homophony and boundary-like roles, rather than a universal one-symbol/one-letter table. [L12](https://aclanthology.org/W11-1202/)

**Our synthesis:** the generalizable achievement is coordinated constraint solving across levels. A useful architecture should let a lexical hypothesis improve segmentation, a channel hypothesis improve lexical matching, and a grammatical hypothesis reject an otherwise plausible alignment. It should retain alternative explanations rather than committing to the first language favored by one score. The human historical parallel is cumulative convergence of constrained evidence, not one miraculous semantic guess.

### Multilingual transfer and romanization can manufacture apparent regularity

Søgaard, Ruder and Vulić show limitations of unsupervised embedding alignment across language properties, domains and embedding choices. Comparable geometry is an assumption to inspect, not a default property of arbitrary monolingual corpora. Their results directly weaken the idea that rotating Voynich word vectors into English space automatically supplies a dictionary. [L27](https://aclanthology.org/P18-1072/)

Our corpus design implication is to preserve orthography, pronunciation and normalized text as different views. Diacritic stripping can merge contrastive units. Romanization can replace one character by several, collapse multiple characters into one spelling, insert vowels or boundaries, and change length and entropy. Consonant-only text removes information that a modern language model might confidently hallucinate back. A fair multilingual comparison needs the transform recorded and uncertainty retained.

Likewise, a held-out language can still resemble training languages through genealogy, borrowed vocabulary, shared script, register or translationese. Conversely, a difficult result can reflect a poor romanizer rather than a linguistic limitation. An ambitious external training universe should vary genealogical relationship, typological structure, domain, channel and supervision separately. That is a foundation-model problem of learning transferable inference, not a contest among a few modern language names.

## 11. Four ambitious architectural bets emerging from the linguistic foundation

These are original proposals derived from the distinctions above, not claims that the cited papers implement them. They are research programs, not requests to launch small experiments.

**A. A communication-world inverse model.** Learn a generative account in which event structures and historical beliefs produce discourse plans, which produce language or notation, which produces marks. Inference conditions on manuscript images, transcription alternatives and layout. A shared latent relational structure connects the modalities, while separate selection models decide which information each modality exposes. This lets a picture constrain a noun, an event constrain an omitted argument, and an orthographic model explain recurring graphic expansions.

The hard problem is joint identifiability. If the event prior is too flexible, it absorbs any text. If the channel is too flexible, it encodes any proposed message. If the discourse selector is too flexible, missing evidence becomes expected. The architecture therefore needs compact, inspectable interfaces and uncertainty over whole explanations. Scale should improve its ability to compare constrained hypotheses, not the freedom to produce one fluent hypothesis.

**B. A multilingual procedural foundation model with opaque surface systems.** Build representations of object flow, state changes, event composition and reference from diverse known materials, with independent script and language variation. At inference, treat the unknown writing as a new emission system. The transferable competence is relational composition and inference about a code; lexical labels are secondary. Nonprocedural domains should share the entity/relation layer while using distinct document programs.

Its historical value could exceed Voynich: unknown notations, abbreviated medical writing, partially deciphered archives and diagrams with unfamiliar labels all present versions of the interface problem. The main risk is that modern task corpora impose the ontology and genre we later “discover.” Diversity must include static descriptions, calendars, catalogues, counterfactual discourse, unreliable beliefs and nonsemantic symbol systems.

**C. Joint denoising of linguistic and world hypotheses.** A diffusion-inspired solver could revise segmentation, lexicon, grammatical graph, event graph and channel together. Partial evidence clamps some variables while others remain uncertain. Unlike a left-to-right translation, a late entity constraint can revise an early boundary or morphological analysis. The source of power is global structured inference, not denoising as a metaphor for removing “filler.”

The denoising process should operate over explicitly chosen hypothesis objects. Corrupting a graph edge, a unit boundary and a graphic mark are different operations. A sampler trained to reverse artificial corruption approximates its learned task distribution; it does not automatically sample the historical posterior. Competing complete explanations and uncertainty calibration are central to making this proposal scientific. The diffusion chapter develops the algorithmic options; linguistics determines what the variables mean.

**D. Mechanistic interpretation across representation layers.** Interpret a model only after it demonstrably uses the intended relational structures on externally grounded tasks. The exciting target is a causal path from perceived marks through unitization, reference resolution and event updates to predictions. A useful intervention would change a grammatical operator or participant identity while preserving unrelated content, then induce the corresponding downstream behavior.

For an opaque script, there are at least three different claims: the network implements a reusable inference algorithm; its inferred graph explains the corpus; and that graph captures the historical meaning. Mechanistic evidence can strongly support the first and help interrogate the second. The third still needs independent anchors. A beautiful “heating” circuit that reflects modern pretraining might reveal the model's prior more clearly than the manuscript's content.

## 12. What going big should mean here

The strongest opportunity is a system that learns how **unfamiliar symbolic conventions communicate structured worlds**. That integrates acquisition, linguistic typology, historical channel modeling, event semantics, multimodal inference and generative search. It aims at the mechanism connecting marks to meaning, which a Voynich-only next-token predictor never has to learn explicitly.

The manuscript could remain semantically underdetermined even if such a system becomes very capable. That does not make the program pointless: it could identify a stable grammar, a family of channels, an interpretable notation or the precise missing anchor required to distinguish two readings. But the ultimate output should remain a constrained account that can be independently applied to unfamiliar manuscript material, with unresolved ambiguity retained.

The ambitious bet is therefore conditional but concrete: world models provide relational and causal structure; linguistics provides the representation and realization of that structure; diffusion provides a possible machinery for joint uncertain inference; and mechanistic interpretation can test which bridge the model actually uses. Bringing them together could attack a substantially different problem from generating Voynich-like text. This review proposes that synthesis; it does not claim to have built or validated it.
