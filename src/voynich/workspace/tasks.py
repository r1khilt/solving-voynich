"""Frozen, tokenizer-independent tasks for a bounded workspace-method audit.

Only ``prompt`` is model input. Labels, donors, supports, and grouping metadata
are evaluator material. An absent intermediate word does not prove a model uses
that intermediate concept: direct landmark-to-answer associations remain possible.
"""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
import hashlib
import json
import re
import unicodedata


SCHEMA_VERSION = 1
DATASET_VERSION = "workspace-tasks-v1-20260921"
DEFAULT_SEED = 51021
SOURCE_DATE = "2026-09-21"
SPLITS = ("pilot", "development", "final")
RELATIONS = ("capital", "currency", "capital_continent", "language")
FAMILIES = ("indirect_fact", "anchored_alias", "surface_copy")

# These sources were checked before any model scores. Retrieval depths and
# ambiguity conventions are documented in TASK_DESIGN.md; there is no network I/O.
SOURCES = {
    "fr_profile": "https://european-union.europa.eu/principles-countries-history/eu-countries/france_en",
    "fr_region": "https://data.un.org/en/iso/fr.html",
    "fr_landmark": "https://www.france.fr/fr/destination/paris/",
    "de_profile": "https://european-union.europa.eu/principles-countries-history/eu-countries/germany_de",
    "de_region": "https://www.deutschland.de/en/germany-facts-information-figures-worth-knowing",
    "de_landmark": "https://www.berlin.de/en/attractions-and-sights/3560266-3104052-brandenburg-gate.en.html",
    "br_profile": "https://www.gov.br/anvisa/pt-br/icmra2024/about-brasilia",
    "br_region": "https://data.un.org/en/iso/br.html",
    "br_landmark": "https://www.gov.br/g20/en/about-the-g20/host-cities/rio-de-janeiro-rj",
    "eg_profile": "https://gafi.gov.eg/en/contenttemplate/b383871b-ee1c-4e10-a79d-120505cba5ec",
    "eg_region": "https://data.un.org/en/iso/eg.html",
    "eg_landmark": "https://whc.unesco.org/en/list/86",
    "cn_capital": "https://english.www.gov.cn/archive/lawregulations/201911/20/"
                  "content_WS5ed8856ec6d0b3f0e9499913.html",
    "cn_currency": "https://in.china-embassy.gov.cn/eng/xwfw/zgxw/201909/P020210622244496199876.pdf",
    "cn_language": "https://en.moe.gov.cn/documents/laws_policies/201506/t20150626_191388.html",
    "cn_region": "https://us.mofcom.gov.cn/AboutChina/art/2020/art_94836791fc4448cf9f0221c1bd054314.html",
    "cn_landmark": "https://whc.unesco.org/en/list/438",
}

CONCEPTS = {
    "France": {
        "landmark": "the Eiffel Tower",
        "answers": {"capital": ("Paris",), "currency": ("euro", "EUR"),
                    "capital_continent": ("Europe",), "language": ("French",)},
        "sources": ("fr_profile", "fr_region", "fr_landmark"),
    },
    "Germany": {
        "landmark": "the Brandenburg Gate",
        "answers": {"capital": ("Berlin",), "currency": ("euro", "EUR"),
                    "capital_continent": ("Europe",), "language": ("German",)},
        "sources": ("de_profile", "de_region", "de_landmark"),
    },
    "Brazil": {
        "landmark": "the Christ the Redeemer statue on Corcovado",
        "answers": {"capital": ("Brasilia", "Brasília"), "currency": ("real", "BRL"),
                    "capital_continent": ("South America",), "language": ("Portuguese",)},
        "sources": ("br_profile", "br_region", "br_landmark"),
    },
    "Egypt": {
        "landmark": "the Great Pyramid of Giza",
        "answers": {"capital": ("Cairo",), "currency": ("pound", "EGP"),
                    "capital_continent": ("Africa",), "language": ("Arabic",)},
        "sources": ("eg_profile", "eg_region", "eg_landmark"),
    },
    "China": {
        "landmark": "the Great Wall built during the Ming dynasty",
        "answers": {"capital": ("Beijing",), "currency": ("yuan", "renminbi", "RMB", "CNY"),
                    "capital_continent": ("Asia",),
                    "language": ("Mandarin", "Mandarin Chinese", "Chinese", "Putonghua", "Standard Chinese")},
        "sources": ("cn_capital", "cn_currency", "cn_language", "cn_region", "cn_landmark"),
    },
}

# France/Germany is intentionally absent: its currency and continent are equal.
CONCEPT_PAIRS = (
    ("France", "China"), ("France", "Brazil"), ("France", "Egypt"),
    ("Germany", "China"), ("Germany", "Brazil"), ("Germany", "Egypt"),
    ("China", "Brazil"), ("China", "Egypt"),
)

# Four template families, each with two semantic-preserving surface forms.
TEMPLATES = (
    ("For {subject}, give {relation}. Reply with the answer only.",
     "Identify {relation} for {subject}. Return only the answer."),
    ("What is {relation} for {subject}? Provide only the requested value.",
     "Consider {subject}. What is {relation}? Output just that value."),
    ("A geography question concerns {subject}. State {relation}; add nothing else.",
     "Use {subject} as the referent. Supply {relation}, without explanation."),
    ("Find {relation} associated with {subject}. Give a short answer only.",
     "Answer this lookup about {subject}: {relation}. Do not explain."),
)
RELATION_TEXT = {
    "capital": "its present-day national capital city",
    "currency": "its main currency unit, using the short unit name without a national adjective",
    "capital_continent": "the continent containing its national capital, in the seven-continent convention",
    "language": "its principal nationwide official or standard spoken language",
}
ALIASES = ("dax", "wug", "zarp", "tiv", "nup", "kiv", "bep", "fep", "suv", "lom", "vot", "paz")
COPY_WORDS = ("amber", "velvet", "cobalt", "ribbon", "marble", "violet", "copper", "lantern")
COPY_TEMPLATES = (
    "Background note: this item concerns {subject}. Ignore that note.\n"
    "Copy the following word exactly, without explanation: {word}",
    "Ignore this irrelevant context: {subject}.\n"
    "Your only task is to repeat the literal word {word}. Output nothing else.",
    "A note mentions {subject}, but it is unrelated to the copying task.\n"
    "Return only this literal word: {word}",
    "Copying exercise. Unrelated subject: {subject}.\n"
    "The required output is the literal word {word}; give that word alone.",
)


def _digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode()).hexdigest()


def normalize_answer(value: str) -> str:
    """Normalize labels only; this is not an open-ended generated-answer scorer."""
    text = unicodedata.normalize("NFKD", value).casefold()
    return " ".join("".join(c for c in text if not unicodedata.combining(c)).split())


def _pair_assignments() -> dict[str, str]:
    # The split seed is frozen separately from optional alias/surface seeds.
    ordered = sorted(range(len(CONCEPT_PAIRS)), key=lambda i: _digest([DEFAULT_SEED, "split", i]))
    return {f"pair-{i:02}": "pilot" if rank < 2 else "development" if rank < 5 else "final"
            for rank, i in enumerate(ordered)}


PAIR_SPLITS = _pair_assignments()
SPLIT_TEMPLATES = {"pilot": (0,), "development": (1,), "final": (2, 3)}


def _source_records(concepts) -> list[dict]:
    ids = sorted({s for concept in concepts for s in CONCEPTS[concept]["sources"]})
    return [{"kind": "primary_source", "source_id": source_id,
             "url": SOURCES[source_id], "checked_on": SOURCE_DATE} for source_id in ids]


def _alias_context(source, donor, pair_id, template, paraphrase, seed):
    # Hold anchors fixed across the two paraphrases; alias renaming is a separate seed control.
    pool = sorted(ALIASES, key=lambda value: _digest([seed, pair_id, template, value]))
    extra = next(concept for concept in sorted(CONCEPTS) if concept not in (source, donor))
    rows = [(pool[0], source), (pool[1], donor), (pool[2], extra)]
    rows.sort(key=lambda row: _digest([seed, pair_id, template, "row", row[0]]))
    text = "Local codebook, valid only in this question:\n" + "\n".join(
        f"{alias} means the country containing {CONCEPTS[concept]['landmark']}."
        for alias, concept in rows)
    return text, pool[0], [{"alias": alias, "concept": concept} for alias, concept in rows]


def _prompt(family, source, donor, relation, pair_id, template, paraphrase, seed):
    subject = f"the country containing {CONCEPTS[source]['landmark']}"
    supports = []
    if family == "anchored_alias":
        context, subject, bindings = _alias_context(source, donor, pair_id, template, paraphrase, seed)
        supports.append({"kind": "constructed_alias_codebook", "bindings": bindings,
                         "scope": "explicit local anchors, not an inferred historical cipher"})
    else:
        context = ""
    if family == "surface_copy":
        index = int(_digest([seed, pair_id, template, relation, "copy"])[:8], 16) % len(COPY_WORDS)
        word = COPY_WORDS[index]
        text = COPY_TEMPLATES[template].format(subject=subject, word=word)
        return text, [word], [{"kind": "constructed_copy_instruction", "literal": word}]
    question = TEMPLATES[template][paraphrase].format(subject=subject, relation=RELATION_TEXT[relation])
    return (f"{context}\n\n{question}" if context else question), list(CONCEPTS[source]["answers"][relation]), supports


def build_tasks(*, seed: int = DEFAULT_SEED, split: str | None = None) -> list[dict]:
    """Return 440 records, or one split, without loading a model or tokenizer.

    ``seed`` changes artificial aliases and neutral copy words, never split
    membership. Scientific runs must record seed plus ``task_manifest``. Only the
    pilot split may be used for capability calibration before frozen evaluation.
    """
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if split is not None and split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}")
    records = []
    for pair_index, pair in enumerate(CONCEPT_PAIRS):
        pair_id = f"pair-{pair_index:02}"
        assigned_split = PAIR_SPLITS[pair_id]
        if split is not None and assigned_split != split:
            continue
        for template in SPLIT_TEMPLATES[assigned_split]:
            for relation in RELATIONS:
                for family in FAMILIES:
                    for paraphrase in ((0,) if family == "surface_copy" else (0, 1)):
                        for direction in (0, 1):
                            source, donor = pair[direction], pair[1 - direction]
                            prompt, answers, constructed = _prompt(
                                family, source, donor, relation, pair_id, template, paraphrase, seed)
                            donor_prompt, donor_answers, _ = _prompt(
                                family, donor, source, relation, pair_id, template, paraphrase, seed)
                            stem = f"{family}/{pair_id}/t{template}/p{paraphrase}/{relation}"
                            row = {
                                "schema_version": SCHEMA_VERSION, "dataset_version": DATASET_VERSION,
                                "id": f"{stem}/d{direction}", "family": family,
                                "prompt": prompt, "answer": answers[0], "answers": answers,
                                "latent_concept": source, "donor_concept": donor,
                                "counterfactual_answer": donor_answers[0],
                                "counterfactual_answers": donor_answers, "donor_prompt": donor_prompt,
                                "paired_id": f"{stem}/d{1 - direction}",
                                "positions": {"kind": "last_prompt_position", "token_index": -1,
                                              "scope": "after the runner's fixed chat-template rendering"},
                                "split": assigned_split, "pair_id": pair_id, "relation": relation,
                                "template_id": template, "paraphrase_id": paraphrase,
                                "seed": seed, "split_seed": DEFAULT_SEED,
                                "latent_absent_from_prompt": True, "latent_absent_from_answer": True,
                                "answer_absent_from_prompt": family != "surface_copy",
                                "expected_effect": "preserve" if family == "surface_copy" else "change",
                                "supports": _source_records((source, donor)) + constructed,
                                "prompt_group": _digest(prompt),
                                "paraphrase_group": f"{family}/{pair_id}/t{template}/{relation}/d{direction}",
                                "reuse_group": f"{family}/{pair_id}/t{template}/p{paraphrase}/d{direction}",
                                "claim_scope": "natural-language method calibration, not manuscript semantics",
                            }
                            records.append(row)
    validate_tasks(records)
    return records


def validate_tasks(records: list[dict]) -> dict:
    """Check pairing, exact labels, asserted leak exclusions, and split isolation."""
    if not records:
        raise ValueError("task collection is empty")
    ids = [row["id"] for row in records]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate task id")
    lookup = {row["id"]: row for row in records}
    prompt_splits, pair_splits, template_splits = {}, {}, {}
    for row in records:
        if row["family"] not in FAMILIES or row["relation"] not in RELATIONS or row["split"] not in SPLITS:
            raise ValueError("unknown family, relation, or split")
        source, donor = row["latent_concept"], row["donor_concept"]
        if source not in CONCEPTS or donor not in CONCEPTS or source == donor:
            raise ValueError("invalid concept pair")
        pair_index = int(row["pair_id"].removeprefix("pair-"))
        if not 0 <= pair_index < len(CONCEPT_PAIRS) or set(CONCEPT_PAIRS[pair_index]) != {source, donor}:
            raise ValueError("concepts do not match the frozen pair")
        if row["split_seed"] != DEFAULT_SEED or row["paraphrase_id"] not in (0, 1):
            raise ValueError("invalid frozen split seed or paraphrase")
        if (row["latent_absent_from_prompt"] is not True or row["latent_absent_from_answer"] is not True
                or row["answer_absent_from_prompt"] is not (row["family"] != "surface_copy")):
            raise ValueError("incorrect leak-assertion metadata")
        if row["paired_id"] not in lookup:
            raise ValueError("missing paired donor record")
        paired = lookup[row["paired_id"]]
        if (paired["paired_id"] != row["id"] or paired["prompt"] != row["donor_prompt"]
                or paired["latent_concept"] != donor or paired["answers"] != row["counterfactual_answers"]
                or paired["split"] != row["split"]):
            raise ValueError("inconsistent donor pairing")
        if row["answer"] != row["answers"][0] or row["counterfactual_answer"] != row["counterfactual_answers"][0]:
            raise ValueError("canonical answer must be the first accepted answer")
        for concept in (source, donor):
            if normalize_answer(concept) in normalize_answer(row["prompt"]):
                raise ValueError("latent concept leaked into indirect prompt")
        for answer in row["answers"] + row["counterfactual_answers"]:
            if any(normalize_answer(c) in normalize_answer(answer) for c in (source, donor)):
                raise ValueError("latent concept leaked into answer")
        own = {normalize_answer(a) for a in row["answers"]}
        other = {normalize_answer(a) for a in row["counterfactual_answers"]}
        if row["family"] == "surface_copy":
            if own != other or row["expected_effect"] != "preserve":
                raise ValueError("copy control must preserve its literal answer")
        else:
            expected = list(CONCEPTS[source]["answers"][row["relation"]])
            if row["answers"] != expected or own & other or row["expected_effect"] != "change":
                raise ValueError("incorrect or ambiguous fact/counterfactual label")
            # Short currency-code aliases can occur within ordinary words, so
            # assert literal answer absence at word boundaries, not substrings.
            for answer in row["answers"]:
                if re.search(r"(?<!\w)" + re.escape(normalize_answer(answer)) + r"(?!\w)",
                             normalize_answer(row["prompt"])):
                    raise ValueError("answer leaked into indirect prompt")
        if row["positions"].get("token_index") != -1 or row["positions"].get("kind") != "last_prompt_position":
            raise ValueError("unsupported intervention position specification")
        if PAIR_SPLITS.get(row["pair_id"]) != row["split"]:
            raise ValueError("pair is assigned to the wrong split")
        if row["template_id"] not in SPLIT_TEMPLATES[row["split"]]:
            raise ValueError("template is assigned to the wrong split")
        for collection, key in ((prompt_splits, row["prompt_group"]), (pair_splits, row["pair_id"]),
                                (template_splits, row["template_id"])):
            if key in collection and collection[key] != row["split"]:
                raise ValueError("cross-split reuse of prompt, pair, or template")
            collection[key] = row["split"]
        if row["prompt_group"] != _digest(row["prompt"]):
            raise ValueError("prompt identity mismatch")
        if not row["supports"]:
            raise ValueError("missing support record")
    # JSON round-trip is part of the public schema contract.
    if json.loads(json.dumps(records, allow_nan=False)) != records:
        raise ValueError("records must be plain JSON values")
    return {"records": len(records), "unique_prompts": len(prompt_splits),
            "split_counts": dict(sorted(Counter(r["split"] for r in records).items())),
            "family_counts": dict(sorted(Counter(r["family"] for r in records).items()))}


def task_manifest(records: list[dict] | None = None, *, seed: int = DEFAULT_SEED) -> dict:
    """Return immutable-content identities and disclosed grouping limitations."""
    if records is None:
        records = build_tasks(seed=seed)
    counts = validate_tasks(records)
    return {"schema_version": SCHEMA_VERSION, "dataset_version": DATASET_VERSION,
            "records_sha256": _digest(records), "source_date": SOURCE_DATE,
            "split_seed": DEFAULT_SEED, "surface_seeds": sorted({r["seed"] for r in records}),
            "pair_splits": deepcopy(PAIR_SPLITS),
            "split_templates": {name: list(values) for name, values in SPLIT_TEMPLATES.items()},
            **counts,
            "limitations": ["Five familiar countries; concepts are shared across splits.",
                            "Pairs and template families are held out, not concepts or relations.",
                            "Repeated prompts and paraphrases are correlated; group by pair and prompt.",
                            "Four relations do not prove independent internal consumers.",
                            "An absent intermediate string does not prove multi-hop computation.",
                            "Artificial aliases receive explicit landmark anchors in the prompt.",
                            "No manuscript language, symbols, or historical semantics are tested."]}
