# Frozen tasks for the natural-language workspace audit

Designed 2026-09-21, before model scoring. This is a tokenizer-independent benchmark definition for the planned cached Qwen3-8B-4bit audit. No model was loaded or scored by the task-construction subtask. The tasks assess a proposed interpretability method; they contain no Voynich transcription or historical semantic labels.

## Scope and interface

`src/voynich/workspace/tasks.py` provides `build_tasks(seed=51021, split=None)`, `validate_tasks(records)`, and `task_manifest(records)`. Supported split names are `pilot`, `development`, and `final`. The default dataset contains **440 records / 359 unique prompt strings**. Its canonical JSON record SHA-256 is:

`11a61cf1833ec2c69c19670ce6096fff286867b720879ffb555bef5a24e83e42`

The version is `workspace-tasks-v1-20260921`. Every record is a plain JSON object. **Only `prompt` is model input.** `answer`, `answers`, `latent_concept`, `donor_concept`, `counterfactual_answer(s)`, `supports`, and grouping fields are evaluator metadata and must never be appended to model prompts. `donor_prompt` is a separate controlled input, not extra recipient evidence.

Positions are declared as `{"kind":"last_prompt_position","token_index":-1}`. The runner must define and freeze its chat-template rendering, then resolve this position; the task module does not assume a tokenizer, token count, thinking mode, or model-specific assistant suffix.

| Family | Records | Intended observation |
| --- | ---: | --- |
| `indirect_fact` | 176 | Infer an unnamed country from a landmark, then answer one of four relations. |
| `anchored_alias` | 176 | A local three-entry codebook binds artificial names to landmark descriptions; answer a relation for one named entry. |
| `surface_copy` | 88 | Ignore a country/landmark background note and copy an unrelated literal word. |

Five country concepts—France, Germany, Brazil, Egypt, and China—form eight unordered pairs, each tested in both directions. Relations are national capital, currency unit, continent containing the capital under the seven-continent convention, and principal nationwide official/standard language. All four factual answer sets differ within every selected pair. France/Germany is excluded because its currency and continent answers coincide. This is four different query functions, not proof of four independent neural mechanisms.

Four template families each have two semantic-preserving paraphrases. Artificial codebook entries and row order remain fixed between the two paraphrases. Changing the separate surface seed changes alias names and copy words while preserving pair/split assignment. Alias names carry no built-in geographical meaning; their mappings are supplied explicitly.

Example indirect task: infer the country containing the Eiffel Tower and return its capital. The target is `Paris`; its latent label `France` is absent from both prompt and accepted answer strings. The matched donor can concern the Great Wall, with latent `China` and target `Beijing`. All indirect records exclude both country labels from the prompt by case-insensitive normalized substring checks. Answers also exclude both labels; explicit answer aliases are checked for word-boundary leakage into indirect prompts.

An absent intermediate label **does not establish a multi-hop computation**. Direct landmark-to-answer associations could solve these questions. The planned causal intervention and flexible-use controls must distinguish that explanation from a shared intermediate country representation. These tasks are deliberately small capability-calibration material, not a new general reasoning benchmark.

## Frozen split and pairing policy

Pair ordering is a deterministic SHA-256 ordering with split seed 51021, fixed in code. Pair assignments and template families are both disjoint:

| Split | Pairs | Template IDs | Records |
| --- | --- | --- | ---: |
| Pilot | Germany/Egypt; China/Egypt | 0 | 80 |
| Development | France/Brazil; Germany/Brazil; Germany/China | 1 | 120 |
| Final | France/Egypt; China/Brazil; France/China | 2, 3 | 240 |

Pilot is for capability calibration and formatting checks. Discovery/selection belongs to development. Final may be scored only after the runner's analysis choices and thresholds are frozen. This task submodule builds records but cannot enforce a runner's scientific final-scoring gate. Concept identities, relations, and individual facts are shared between splits; the held-out factors are pair combinations and template families, **not unseen concepts**. A concept-holdout claim would be false.

Every `paired_id` resolves to a reverse-direction record in the same split, family, relation, and template. `donor_prompt` and `counterfactual_answers` exactly match that reverse record. `reuse_group` joins the same source/donor combination across all four relations. `paraphrase_group` joins equivalent questions across the two paraphrases. `prompt_group` identifies repeated prompt text. Repeated prompts occur when the same recipient is compared against different donors; they are not additional independent observations. Statistical resampling should respect pair and prompt grouping rather than treating 440 rows as 440 independent worlds.

`expected_effect="change"` applies to factual and alias tasks and requires disjoint source/donor answer sets. Copy controls use `expected_effect="preserve"`: changing the background country must leave the copied answer unchanged. These controls must not enter a denominator that expects every counterfactual answer to change.

## Fact support and ambiguity decisions

Primary-source material was checked on 2026-09-21 using retrieved page text or indexed excerpts. No source text or third-party dataset is copied into the repository; the module stores short facts and URLs. Full-document reading or independent archival verification is not claimed. The Chinese embassy currency PDF exceeded the browser retrieval limit; only its indexed fact-file excerpt, explicitly identifying renminbi/RMB/yuan, was consulted. UN country profiles were available as indexed excerpts despite direct-open failures. `SOURCES` preserves the exact URLs.

| Concept | Frozen labels | Primary support |
| --- | --- | --- |
| France | Paris; euro; Europe; French | [EU country profile](https://european-union.europa.eu/principles-countries-history/eu-countries/france_en), [UN profile](https://data.un.org/en/iso/fr.html), [official tourism Paris/Eiffel Tower page](https://www.france.fr/fr/destination/paris/). |
| Germany | Berlin; euro; Europe; German | [EU country profile](https://european-union.europa.eu/principles-countries-history/eu-countries/germany_de), [government-supported country facts](https://www.deutschland.de/en/germany-facts-information-figures-worth-knowing), [Berlin city landmark page](https://www.berlin.de/en/attractions-and-sights/3560266-3104052-brandenburg-gate.en.html). |
| Brazil | Brasilia/Brasília; real; South America; Portuguese | [Anvisa visitor facts](https://www.gov.br/anvisa/pt-br/icmra2024/about-brasilia), [UN profile](https://data.un.org/en/iso/br.html), [government Rio/Christ the Redeemer page](https://www.gov.br/g20/en/about-the-g20/host-cities/rio-de-janeiro-rj). |
| Egypt | Cairo; pound; Africa; Arabic | [GAFI facts](https://gafi.gov.eg/en/contenttemplate/b383871b-ee1c-4e10-a79d-120505cba5ec), [UN profile](https://data.un.org/en/iso/eg.html), [UNESCO Giza site record](https://whc.unesco.org/en/list/86). |
| China | Beijing; yuan/renminbi; Asia; Mandarin/Putonghua | [Constitution, Article 143](https://english.www.gov.cn/archive/lawregulations/201911/20/content_WS5ed8856ec6d0b3f0e9499913.html), [embassy currency fact file](https://in.china-embassy.gov.cn/eng/xwfw/zgxw/201909/P020210622244496199876.pdf), [standard-language definition](https://en.moe.gov.cn/documents/laws_policies/201506/t20150626_191388.html), [official geography description](https://us.mofcom.gov.cn/AboutChina/art/2020/art_94836791fc4448cf9f0221c1bd054314.html), [UNESCO Great Wall record](https://whc.unesco.org/en/list/438). |

Geographical continent labels are the benchmark's conventional mapping from the source locations; asking about the capital avoids assigning every overseas or transcontinental territory to one continent. Cairo follows the cited national/UN profiles; no claim that every administrative office remains there is needed. The language relation allows the explicitly listed equivalent Chinese/Mandarin/Putonghua labels rather than assuming every resident speaks one language.

Currency questions explicitly request the short unit without a national adjective: `pound`, not `Egyptian pound`, and `real`, not `Brazilian real`. This prevents country-label leakage in accepted targets. ISO currency codes are listed as additional acceptable labels. The record's accepted-answer list is finite and must be frozen before scoring; it does not justify treating arbitrary paraphrases as correct after inspecting model outputs. `normalize_answer` only folds case, accents, and whitespace. It is not a permissive semantic scorer.

## Validation and limits

`tests/test_workspace_tasks.py` passes **21 tests** covering deterministic construction and hashes, JSON round trips, pair/template/prompt split isolation, exact counterfactual reversals, country/answer leakage, four-relation reuse, paraphrase equivalence, explicit alias anchors, copy invariance, fact-label distinctions, invalid input rejection, source/position metadata, and deliberate corruption. Ruff passes on the two task files. No model inference or scientific outcome is tested or asserted.

The anchored extension is a finite in-context binding exercise with explicit landmark supervision. It is not evidence of discovering an unknown alphabet without anchors, learning medieval language, or recovering an historical encoding. Quantization, tokenization, sequence-position alignment, natural-language answer scoring, and capability thresholds belong to the separately registered model runner. Successful task answers alone cannot establish a shared workspace, and successful interventions in Qwen would still not establish a Voynich interpretation.
