"""Integrity tests for pre-score task construction; no model execution."""

from collections import defaultdict
from copy import deepcopy
import json
import re

import pytest

from voynich.workspace.tasks import (
    ALIASES, CONCEPT_PAIRS, CONCEPTS, DEFAULT_SEED, RELATIONS, SOURCES,
    build_tasks, normalize_answer, task_manifest, validate_tasks,
)


@pytest.fixture(scope="module")
def tasks():
    return build_tasks()


def test_frozen_sizes_serialization_and_content_identity(tasks):
    manifest = task_manifest(tasks)
    assert manifest["records"] == 440
    assert manifest["unique_prompts"] == 359
    assert manifest["split_counts"] == {"pilot": 80, "development": 120, "final": 240}
    assert manifest["family_counts"] == {"anchored_alias": 176, "indirect_fact": 176, "surface_copy": 88}
    assert json.loads(json.dumps(tasks, ensure_ascii=False)) == tasks
    assert json.loads(json.dumps(manifest)) == manifest
    assert build_tasks() == tasks
    assert task_manifest(build_tasks())["records_sha256"] == manifest["records_sha256"]


def test_splits_hold_out_pairs_and_template_families_not_concepts(tasks):
    fields = {name: defaultdict(set) for name in ("pair_id", "template_id", "prompt_group")}
    for row in tasks:
        for name in fields:
            fields[name][row["split"]].add(row[name])
    for grouping in fields.values():
        assert not grouping["pilot"] & grouping["development"]
        assert not grouping["pilot"] & grouping["final"]
        assert not grouping["development"] & grouping["final"]
    assert {r["latent_concept"] for r in tasks if r["split"] == "development"} & {
        r["latent_concept"] for r in tasks if r["split"] == "final"}
    for split in ("pilot", "development", "final"):
        selected = build_tasks(split=split)
        assert selected == [row for row in tasks if row["split"] == split]
        assert validate_tasks(selected)["records"] == len(selected)


def test_every_counterfactual_has_exact_reverse_pair(tasks):
    by_id = {r["id"]: r for r in tasks}
    for row in tasks:
        donor = by_id[row["paired_id"]]
        assert donor["paired_id"] == row["id"]
        assert donor["prompt"] == row["donor_prompt"]
        assert donor["answers"] == row["counterfactual_answers"]
        assert donor["latent_concept"] == row["donor_concept"]
        assert donor["template_id"] == row["template_id"]
        assert donor["relation"] == row["relation"]
        if row["expected_effect"] == "change":
            assert not set(map(normalize_answer, row["answers"])) & set(
                map(normalize_answer, row["counterfactual_answers"]))
        else:
            assert row["family"] == "surface_copy"
            assert row["answers"] == row["counterfactual_answers"]


def test_no_hidden_country_or_literal_answer_in_indirect_inputs(tasks):
    for row in tasks:
        prompt = normalize_answer(row["prompt"])
        for concept in (row["latent_concept"], row["donor_concept"]):
            assert normalize_answer(concept) not in prompt
            assert all(normalize_answer(concept) not in normalize_answer(answer)
                       for answer in row["answers"] + row["counterfactual_answers"])
        if row["family"] != "surface_copy":
            for answer in row["answers"]:
                assert not re.search(r"(?<!\w)" + re.escape(normalize_answer(answer)) + r"(?!\w)", prompt)


def test_same_concept_is_reused_across_four_relations_and_two_paraphrases(tasks):
    reuse, paraphrases = defaultdict(list), defaultdict(list)
    for row in tasks:
        reuse[row["reuse_group"]].append(row)
        if row["family"] != "surface_copy":
            paraphrases[row["paraphrase_group"]].append(row)
    for rows in reuse.values():
        assert {r["relation"] for r in rows} == set(RELATIONS)
        assert len({r["latent_concept"] for r in rows}) == 1
        assert len({r["donor_concept"] for r in rows}) == 1
    for rows in paraphrases.values():
        assert {r["paraphrase_id"] for r in rows} == {0, 1}
        assert rows[0]["answers"] == rows[1]["answers"]
        assert rows[0]["prompt"] != rows[1]["prompt"]
        if rows[0]["family"] == "anchored_alias":
            assert rows[0]["prompt"].split("\n\n")[0] == rows[1]["prompt"].split("\n\n")[0]


def test_aliases_have_three_explicit_unique_anchors_and_swap_target_binding(tasks):
    for row in tasks:
        if row["family"] != "anchored_alias":
            continue
        codebook = next(s for s in row["supports"] if s["kind"] == "constructed_alias_codebook")
        bindings = codebook["bindings"]
        assert len(bindings) == 3
        assert len({b["alias"] for b in bindings}) == 3
        assert len({b["concept"] for b in bindings}) == 3
        assert all(b["alias"] in ALIASES for b in bindings)
        target = next(b["alias"] for b in bindings if b["concept"] == row["latent_concept"])
        assert target in row["prompt"].split("\n\n", 1)[1]
        donor = next(r for r in tasks if r["id"] == row["paired_id"])
        donor_bindings = next(s["bindings"] for s in donor["supports"]
                              if s["kind"] == "constructed_alias_codebook")
        assert next(b["concept"] for b in donor_bindings if b["alias"] == target) == row["donor_concept"]


def test_copy_preserves_output_despite_different_background_concept(tasks):
    for row in tasks:
        if row["family"] == "surface_copy":
            literal = next(s["literal"] for s in row["supports"] if s["kind"] == "constructed_copy_instruction")
            assert row["answer"] == literal == row["counterfactual_answer"]
            assert row["prompt"] != row["donor_prompt"]
            assert literal in row["prompt"] and literal in row["donor_prompt"]


def test_fact_labels_and_pair_constraints_are_unambiguous():
    assert len(CONCEPT_PAIRS) == 8 and len(CONCEPTS) == 5
    assert CONCEPTS["Brazil"]["answers"]["capital"] == ("Brasilia", "Brasília")
    assert CONCEPTS["Egypt"]["answers"]["capital"] == ("Cairo",)
    assert CONCEPTS["China"]["answers"]["currency"][0] == "yuan"
    for source, donor in CONCEPT_PAIRS:
        for relation in RELATIONS:
            own = set(map(normalize_answer, CONCEPTS[source]["answers"][relation]))
            other = set(map(normalize_answer, CONCEPTS[donor]["answers"][relation]))
            assert own.isdisjoint(other)
    assert normalize_answer(" Brasília ") == "brasilia"


def test_sources_and_position_are_explicit(tasks):
    for row in tasks:
        assert row["positions"]["token_index"] == -1
        assert row["positions"]["kind"] == "last_prompt_position"
        assert row["split_seed"] == DEFAULT_SEED
        for support in row["supports"]:
            if support["kind"] == "primary_source":
                assert SOURCES[support["source_id"]] == support["url"]
                assert support["url"].startswith("https://")
                assert support["checked_on"] == "2026-09-21"


def test_seed_changes_aliases_and_controls_but_never_fact_splits(tasks):
    alternate = build_tasks(seed=DEFAULT_SEED + 1)
    assert [(r["id"], r["split"], r["answer"]) for r in tasks if r["family"] == "indirect_fact"] == [
        (r["id"], r["split"], r["answer"]) for r in alternate if r["family"] == "indirect_fact"]
    assert any(a["prompt"] != b["prompt"] for a, b in zip(tasks, alternate)
               if a["family"] == "anchored_alias")
    assert {r["seed"] for r in alternate} == {DEFAULT_SEED + 1}
    assert task_manifest(tasks)["records_sha256"] != task_manifest(alternate)["records_sha256"]


@pytest.mark.parametrize("bad_seed", [-1, True, 2.5, "42"])
def test_invalid_seeds_refused(bad_seed):
    with pytest.raises(ValueError):
        build_tasks(seed=bad_seed)


def test_invalid_split_refused():
    with pytest.raises(ValueError):
        build_tasks(split="test")


@pytest.mark.parametrize("mutation", ["leak", "wrong_answer", "wrong_donor", "wrong_split", "duplicate"])
def test_integrity_checks_reject_corruption(tasks, mutation):
    damaged = deepcopy(tasks)
    row = damaged[0]
    if mutation == "leak":
        row["prompt"] += " " + row["latent_concept"]
    elif mutation == "wrong_answer":
        row["answers"] = ["London"]
        row["answer"] = "London"
    elif mutation == "wrong_donor":
        row["counterfactual_answers"] = row["answers"]
    elif mutation == "wrong_split":
        row["split"] = "pilot"
    else:
        damaged.append(deepcopy(row))
    with pytest.raises(ValueError):
        validate_tasks(damaged)


def test_builder_returns_independent_mutable_records(tasks):
    changed = build_tasks()
    changed[0]["answers"].append("corruption")
    changed[0]["supports"][0]["url"] = "corruption"
    assert build_tasks() == tasks
