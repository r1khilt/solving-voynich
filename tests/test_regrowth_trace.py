"""End-to-end JSON receipts and replay, with independently accumulated state."""
import json
from fractions import Fraction

import pytest

from tests.test_reading_regrowth import fixture_source, independent_target
from voynich.reading_regrowth import RegrowthConfig, SourceRegrowth
from voynich.regrowth_trace import chain, path_receipt
from voynich.source_action_proposal import ReadingEnvironment


def test_json_chain_replay_and_independent_summary():
    env = ReadingEnvironment(((0, 1), (0, 1)), rows=2, glyphs=2)
    sampler = SourceRegrowth(fixture_source(True), env, RegrowthConfig(grid_bits=8))
    result = json.loads(json.dumps(chain(sampler, (1, 1), seed=71, steps=40)))
    repeated = chain(sampler, (1, 1), seed=71, steps=40)
    assert result == repeated
    current = sampler.path(forced_actions=(1, 1))
    totals = {'accepted': 0, 'accepted_inventory_changes': 0, 'accepted_key_changes': 0,
              'accepted_length_changes': 0, 'failed_proposals': 0, 'acceptance_raw64_blocks': 0, 'root_cuts': 0}
    for row in result['proposals']:
        candidate = sampler.path(forced_actions=tuple(row['candidate']['actions']))
        assert path_receipt(candidate) == row['candidate']
        assert candidate.actions[:row['cut']] == current.actions[:row['cut']]
        if candidate.complete:
            assert Fraction(*sampler.target_integers(candidate)) == independent_target(
                fixture_source(True), env, candidate.state, sampler.config.stop)
        totals['accepted'] += row['accepted']
        totals['accepted_inventory_changes'] += row['accepted'] and (set(current.state.key)-{-1}) != (
            set(candidate.state.key)-{-1})
        totals['accepted_key_changes'] += row['accepted'] and current.state.key != candidate.state.key
        totals['accepted_length_changes'] += row['accepted'] and len(current.actions) != len(candidate.actions)
        totals['failed_proposals'] += row['failed']
        totals['acceptance_raw64_blocks'] += row['acceptance_blocks']
        totals['root_cuts'] += row['cut'] == 0
        if row['accepted']:
            current = candidate
    assert path_receipt(current) == result['final']
    assert all(result['summary'][k] == v for k, v in totals.items())
    assert result['summary']['attempts'] == 40
    assert result['final']['not_chain_state_sampling_density']
    for steps in (0, -1, True):
        with pytest.raises(ValueError):
            chain(sampler, (1, 1), seed=71, steps=steps)
