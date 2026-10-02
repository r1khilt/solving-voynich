"""Exhaustive permutation witnesses independently qualify the orbit diagnostic."""
import itertools

import pytest

from voynich.reading_label_orbit import label_orbit_obstructions
from voynich.source_action_proposal import ReadingEnvironment, ReadingState


def enumerate_complete(env):
    agenda, result = [env.initial], []
    while agenda:
        state = agenda.pop()
        if env.selected_record(state) is None:
            result.append(state)
        else:
            agenda.extend(env.advance(state, action) for action in env.legal_actions(state))
    assert len(set(result)) == len(result)
    return result


def permute(state, mapping):
    key = [-1]*len(state.key)
    for old, new in enumerate(mapping):
        key[new] = state.key[old]
    return ReadingState(tuple(key), state.offsets,
                        tuple(tuple(mapping[row] for row in text) for text in state.texts))


@pytest.mark.parametrize('rows', [2, 3])
@pytest.mark.parametrize('records', [((0,), (0, 0)), ((0, 1), (0, 1))])
def test_orbit_and_binding_ceiling_against_every_permutation(rows, records):
    env = ReadingEnvironment(records, rows=rows, glyphs=2)
    states = enumerate_complete(env)
    permutations = tuple(itertools.permutations(range(rows)))
    for old in states:
        orbit = [permute(old, p) for p in permutations]
        for point in orbit:
            env.validate(point)
        for known in states:
            result = label_orbit_obstructions(env, old, known)
            assert result['label_orbit_reachable'] == (known in orbit)
            used = [row for row, k in enumerate(known.key) if k >= 0]
            ceiling = max(sum(p.key[row] == known.key[row] for row in used) for p in orbit)
            assert result['binding_match_ceiling'] == ceiling
            assert result['inventory_can_cover_known_used_rows'] == (ceiling == len(used))
            if result['label_orbit_reachable']:
                mapping = dict(result['permutation_of_used_rows'])
                assert all(mapping[a] == b for x, y in zip(old.texts, known.texts, strict=True)
                           for a, b in zip(x, y, strict=True))


def test_equal_inventory_does_not_fix_repeated_letter_partition():
    env = ReadingEnvironment(((0, 0, 0),), rows=3, glyphs=2)
    key = (env.pool.index((0,)), env.pool.index((0,)), -1)
    old = ReadingState(key, (3,), ((0, 1, 0),))
    known = ReadingState(key, (3,), ((0, 0, 1),))
    result = label_orbit_obstructions(env, old, known)
    assert result['inventory_can_cover_known_used_rows'] and result['unit_lengths_match']
    assert not result['cross_record_equality_pattern_matches'] and not result['label_orbit_reachable']


def test_equal_lengths_and_inventory_do_not_fix_boundaries():
    env = ReadingEnvironment(((0, 0, 0),), rows=3, glyphs=2)
    one, two = env.pool.index((0,)), env.pool.index((0, 0))
    old = ReadingState((one, two, -1), (3,), ((0, 1),))
    known = ReadingState((two, one, -1), (3,), ((0, 1),))
    result = label_orbit_obstructions(env, old, known)
    assert result['inventory_can_cover_known_used_rows'] and result['source_lengths_match']
    assert result['cross_record_equality_pattern_matches'] and not result['unit_lengths_match']
    assert not result['label_orbit_reachable'] and result['length_edit_lower_bound'] == 0


def test_complete_literal_inputs_required():
    env = ReadingEnvironment(((0,),), rows=3, glyphs=2)
    complete = env.advance(env.initial, 0)
    with pytest.raises(ValueError):
        label_orbit_obstructions(env, env.initial, complete)
    with pytest.raises(ValueError):
        label_orbit_obstructions(env, ReadingState((0, -1, -1), (1,), ((2,),)), complete)
