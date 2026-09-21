"""Independent execution, leakage, invariance and failure tests for synthetic worlds."""

import json
from dataclasses import replace

import pytest

from voynich.communication.worlds import (
    FAMILIES,
    GRAMMARS,
    MORPHOLOGIES,
    Action,
    Entity,
    EntityState,
    Relation,
    SurfaceClause,
    WorldConfig,
    WorldExample,
    decode,
    encode,
    equivalent_entity_renaming,
    execute_action,
    generate_world,
    infer_procedure_trace,
    infer_taxonomy_graph,
    parse_document,
    reencode,
    rename_observation,
    state_projection,
)


def _plain_boundaries(example):
    return tuple(boundary for boundary, keep in zip(example.word_boundary, example.keep) if keep)


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("grammar", GRAMMARS)
@pytest.mark.parametrize("morphology", MORPHOLOGIES)
def test_worlds_decode_parse_and_json_roundtrip(family, grammar, morphology):
    config = WorldConfig(73, family=family, grammar=grammar, morphology=morphology, homophones=2)
    example = generate_world(config)
    decoded = decode(example.observation, example.inverse_key, example.keep, example.word_boundary)
    assert decoded.plaintext == example.plaintext
    assert decoded.alignment == example.alignment
    assert reencode(example.plaintext, example.inverse_key, example.alignment,
                    symbols=example.observation) == example.observation
    assert parse_document(example.plaintext, _plain_boundaries(example), entities=config.entities,
                          grammar=grammar, morphology=morphology, family=family) == example.clauses
    assert WorldExample.from_dict(json.loads(json.dumps(example.to_dict()))) == example
    assert all(2 <= value < config.alphabet_size + 2 for value in example.observation)
    assert len(example.observation) == len(example.keep) == len(example.roles) == len(example.word_boundary)


@pytest.mark.parametrize("seed", range(12))
def test_seed_determinism_and_execution_without_gold(seed):
    example = generate_world(WorldConfig(seed, events=20))
    assert generate_world(example.config) == example
    witness = infer_procedure_trace(example.clauses, example.config.entities)
    state = witness.initial_state
    for step in witness.action_trace:
        assert step.before == state
        state = execute_action(witness.entities, state, step.action)
        assert state == step.after
        assert sum(row.full for row in state) == 1
        assert all(value in (0, 1, 2) for value in state_projection(state))


def test_safe_export_is_an_observation_allowlist():
    example = generate_world(WorldConfig(813))
    exported = example.observation_dict()
    assert exported == {"observation": list(example.observation), "alphabet_size": 24}
    encoded = json.dumps(exported)
    for forbidden in ("seed", "key", "grammar", "morphology", "family", "trace", "plaintext", "keep"):
        assert forbidden not in encoded
    exported["observation"].append(200)
    assert 200 not in example.observation


def test_hand_computed_channel_and_accountable_nulls():
    # Observed 2 and 4 both encode canonical 2; observed 3 is globally null.
    key = (2, 0, 2, 3)
    result = decode((3, 2, 4, 3, 5), key, word_boundary=(False, True, False, False, True))
    assert result.plaintext == (2, 2, 3)
    assert result.keep == (False, True, True, False, True)
    assert result.alignment == (-1, 0, 1, -1, 2)
    with pytest.raises(ValueError, match="unlicensed deletion"):
        decode((2, 2), key, keep=(True, False))
    with pytest.raises(ValueError, match="null positions"):
        decode((3, 2), key, word_boundary=(True, True))
    with pytest.raises(ValueError, match="first retained"):
        decode((3, 2), key, word_boundary=(False, False))
    with pytest.raises(ValueError, match="alignment"):
        reencode((2, 2, 3), key, (-1, 0, 0, -1, 2), symbols=(3, 2, 4, 3, 5))
    with pytest.raises(ValueError, match="plaintext"):
        reencode((2, 3, 2), key, result.alignment, symbols=(3, 2, 4, 3, 5))


def test_channel_does_not_silently_drop_unknown_plaintext_or_accept_reserved_glyphs():
    for glyph in (0, 1, 9, -1, True, 2.0):
        with pytest.raises(ValueError):
            decode((glyph,), (2, 0, 2, 3))
    for key in ((1, 0), (-1, 2), (4, 2), (2.0, 2), (True, 2), ()):
        with pytest.raises(ValueError):
            decode((), key)
    with pytest.raises(ValueError, match="unsupported"):
        encode((4,), (2, 0, 2, 3))
    with pytest.raises(ValueError, match="globally null"):
        encode((2,), (2, 3), null_rate=0.5)
    with pytest.raises(ValueError, match="safety cap"):
        encode((2,), (2, 0), null_rate=0.9999999)


def test_budget_accounts_for_morphology_homophones_and_null_reserve():
    example = generate_world(WorldConfig(1, alphabet_size=19, homophones=2))
    assert example.config.canonical_count == 9
    assert example.inverse_key.count(0) == 1
    for symbol in range(2, 11):
        assert example.inverse_key.count(symbol) == 2
    with pytest.raises(ValueError, match="requires 19"):
        WorldConfig(1, alphabet_size=18, homophones=2)
    no_nulls = generate_world(WorldConfig(1, alphabet_size=18, homophones=2, null_rate=0))
    assert all(no_nulls.keep)
    assert 0 not in no_nulls.inverse_key


@pytest.mark.parametrize("change", [
    {"stateful": True}, {"family": "unknown"}, {"grammar": "SSV"}, {"morphology": "infix"},
    {"entities": 2}, {"events": 0}, {"seed": -1}, {"seed": True}, {"homophones": 0},
    {"null_rate": -0.1}, {"null_rate": 1.0}, {"null_rate": float("nan")}, {"null_rate": True},
])
def test_unsupported_or_invalid_configuration_is_explicitly_rejected(change):
    arguments = {"seed": 1, **change}
    with pytest.raises(ValueError):
        WorldConfig(**arguments)


def test_typed_preconditions_and_effects_are_independent_of_generation():
    entities = (Entity(0, "agent"), Entity(1, "vessel"), Entity(2, "vessel"))
    initial = (EntityState(0, False, False), EntityState(1, True, False), EntityState(2, False, False))
    hot = execute_action(entities, initial, Action("heat", 0, 1))
    moved = execute_action(entities, hot, Action("transfer", 1, 2))
    assert state_projection(initial) == (0, 1, 0)
    assert state_projection(hot) == (0, 2, 0)
    assert state_projection(moved) == (0, 0, 2)
    assert state_projection(execute_action(entities, moved, Action("cool", 0, 2))) == (0, 0, 1)
    for state, action in (
        (initial, Action("transfer", 2, 1)),  # empty source
        (initial, Action("transfer", 1, 0)),  # wrong target type
        (initial, Action("heat", 1, 2)),      # wrong actor type
        (initial, Action("heat", 0, 2)),      # empty target
        (initial, Action("cool", 0, 1)),      # already cold
        (hot, Action("heat", 0, 1)),          # already hot
        (hot, Action("transfer", 1, 1)),      # same argument
    ):
        with pytest.raises(ValueError):
            execute_action(entities, state, action)


def test_semantically_impossible_trace_rejected_without_gold():
    with pytest.raises(ValueError, match="no typed initial state"):
        infer_procedure_trace((SurfaceClause(1, 0, 1), SurfaceClause(1, 0, 1)), 3)
    with pytest.raises(ValueError, match="no typed initial state"):
        infer_procedure_trace((SurfaceClause(0, 1, 2), SurfaceClause(0, 1, 2)), 3)
    with pytest.raises(ValueError, match="no typed initial state"):
        infer_procedure_trace((SurfaceClause(2, 0, 1),), 3)  # cold initial prior
    assert infer_procedure_trace((SurfaceClause(2, 0, 1),), 3, initial_hot=None).action_trace


@pytest.mark.parametrize("seed", range(6))
def test_taxonomy_inference_needs_no_gold_tree(seed):
    example = generate_world(WorldConfig(seed, family="taxonomy", entities=5, events=20))
    graph = infer_taxonomy_graph(example.clauses, 5)
    # A different compatible graph is allowed; only the expressed claims constrain it.
    replace(example, relations=graph).validate()


def test_taxonomy_search_rejects_cycles_false_siblings_and_conflicting_parents():
    impossible = [
        (SurfaceClause(1, 0, 1), SurfaceClause(1, 1, 0)),
        (SurfaceClause(0, 0, 1), SurfaceClause(0, 2, 1)),
        (SurfaceClause(0, 0, 1), SurfaceClause(2, 0, 1)),
        (SurfaceClause(2, 0, 0),),
    ]
    for clauses in impossible:
        with pytest.raises(ValueError):
            infer_taxonomy_graph(clauses, 3)
    with pytest.raises(ValueError, match="at most 6"):
        infer_taxonomy_graph((SurfaceClause(0, 0, 1),), 7)


def test_parser_cannot_explain_wrong_case_markers_or_arbitrary_segmentation():
    example = generate_world(WorldConfig(13))
    args = dict(entities=4, grammar="SOV", morphology="suffix", family="procedure")
    with pytest.raises(ValueError, match="morphology"):
        parse_document(example.plaintext, _plain_boundaries(example), **{**args, "morphology": "prefix"})
    with pytest.raises(ValueError):
        parse_document(example.plaintext, (True,) * len(example.plaintext), **args)
    with pytest.raises(ValueError):
        parse_document(example.plaintext[:-1], _plain_boundaries(example)[:-1], **args)
    wrong = list(example.plaintext)
    wrong[1] = 10  # Replace subject case by object case.
    with pytest.raises(ValueError, match="morphology"):
        parse_document(wrong, _plain_boundaries(example), **args)


def test_alphabet_renaming_is_exact_and_does_not_regenerate_semantics():
    example = generate_world(WorldConfig(37, homophones=2))
    permutation = tuple(reversed(range(2, example.config.alphabet_size + 2)))
    renamed = rename_observation(example, permutation)
    assert renamed.observation != example.observation
    assert renamed.plaintext == example.plaintext
    assert renamed.action_trace == example.action_trace
    assert renamed.keep == example.keep
    assert rename_observation(renamed, permutation) == example
    assert generate_world(WorldConfig(38)).inverse_key != generate_world(WorldConfig(37)).inverse_key
    with pytest.raises(ValueError):
        rename_observation(example, (2,) * 24)


@pytest.mark.parametrize("family", FAMILIES)
def test_entity_identity_ambiguity_has_identical_observations(family):
    example = generate_world(WorldConfig(72, family=family))
    alternative = equivalent_entity_renaming(example, (3, 2, 1, 0))
    assert alternative.observation_dict() == example.observation_dict()
    assert alternative.plaintext != example.plaintext
    assert alternative.inverse_key != example.inverse_key
    assert equivalent_entity_renaming(alternative, (3, 2, 1, 0)) == example
    if family == "procedure":
        assert infer_procedure_trace(alternative.clauses, 4).action_trace


def test_copy_control_does_not_invent_world_states():
    example = generate_world(WorldConfig(91, family="copy", events=40))
    assert not example.entities
    assert not example.initial_state
    assert not example.action_trace
    assert not example.relations
    assert not example.anonymous_trace()
    assert len(set(example.clauses)) < len(example.clauses)
    with pytest.raises(ValueError, match="referent semantics"):
        replace(example, entities=(Entity(0, "agent"),)).validate()


def test_gold_validation_detects_corrupt_state_trace_taxonomy_and_channel():
    example = generate_world(WorldConfig(2))
    step = example.action_trace[0]
    with pytest.raises(ValueError, match="effects"):
        replace(example, action_trace=(replace(step, after=step.before), *example.action_trace[1:])).validate()
    with pytest.raises(ValueError, match="null count"):
        replace(example, config=replace(example.config, null_rate=0.3)).validate()
    taxonomy = generate_world(WorldConfig(2, family="taxonomy"))
    with pytest.raises(ValueError, match="cyclic"):
        replace(taxonomy, relations=(Relation("parent", 0, 1), Relation("parent", 1, 0),
                                     Relation("parent", 0, 3))).validate()
    with pytest.raises(ValueError, match="false"):
        invalid = replace(taxonomy, clauses=(SurfaceClause(0, 1, 1), *taxonomy.clauses[1:]))
        # A false semantic statement cannot be smuggled via a self-consistent text.
        from voynich.communication.worlds import _realize
        plain, boundaries, roles = _realize(invalid.config, invalid.clauses)
        rendered = encode(plain, invalid.inverse_key, seed=invalid.config.seed, null_rate=invalid.config.null_rate,
                          word_boundary=boundaries, roles=roles)
        replace(invalid, plaintext=plain, observation=rendered.observation, keep=rendered.keep,
                alignment=rendered.alignment, roles=rendered.roles, word_boundary=rendered.word_boundary).validate()


def test_roundtrip_rejects_unknown_fields_and_boolean_integer_confusion():
    example = generate_world(WorldConfig(51))
    data = example.to_dict()
    data["covert_target"] = example.plaintext
    with pytest.raises(ValueError, match="fields"):
        WorldExample.from_dict(data)
    data = example.to_dict()
    data["action_trace"][0]["before"][0]["entity"] = False
    with pytest.raises(ValueError, match="integer"):
        WorldExample.from_dict(data)
    data = example.to_dict()
    data["keep"][0] = int(data["keep"][0])
    with pytest.raises(ValueError, match="booleans"):
        WorldExample.from_dict(data)
