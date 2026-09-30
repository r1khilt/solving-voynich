import itertools
import math

import pytest

from voynich.expanded_key_bank import expand_key_bank
from voynich.local_key_bank import one_move_bank
from voynich.sparse_suffix_source import SuffixSource, decode


def objective(target):
    return lambda key: {'log_weight_unnormalized': -10. * sum(a != b for a, b in zip(key, target, strict=True))}


def test_multiple_rounds_reach_distant_key_and_keep_every_scored_alternative():
    target = ('xx', 'yy', 'xy')
    parent = ('x', 'y', 'x')
    scored, round_events = [], []
    result = expand_key_bank(objective(target), parent, 'xy', max_rounds=4,
                             progress=scored.append, round_progress=round_events.append)
    assert result['best_units'] == list(target)
    assert target not in one_move_bank(parent, 'xy')
    assert result['complete_rounds'] == 4 and result['stop_reason'] == 'local_tolerance_stop_at_center'
    rows = result['bank']
    assert len(rows) == len({tuple(row['units']) for row in rows}) == len(scored)
    for report in result['rounds']:
        neighborhood = one_move_bank(rows[report['parent_index']]['units'], 'xy')
        assert [tuple(rows[i]['units']) for i in report['member_indices']] == list(neighborhood)
    assert round_events == result['rounds']
    assert all(row['index'] == i for i, row in enumerate(rows))
    assert math.fsum(math.exp(row['log_weight']) for row in rows) == pytest.approx(1.)


def test_one_round_is_exactly_original_complete_bank_policy():
    parent = ('x', 'y', 'x')
    result = expand_key_bank(objective(('xx', 'yy', 'xy')), parent, 'xy', max_rounds=1)
    assert [tuple(row['units']) for row in result['bank']] == list(one_move_bank(parent, 'xy'))
    assert result['stop_reason'] == 'round_limit' and not result['global_optimality_claimed']


def test_tolerance_certificate_is_for_center_not_a_slightly_better_bank_row():
    result = expand_key_bank(lambda key: {'log_weight_unnormalized': 0. if key == ('x', 'y') else 1e-10},
                             ('x', 'y'), 'xy', tolerance=1e-6)
    assert result['search_center_index'] == 0 and result['best_index'] != 0
    assert result['stop_reason'] == 'local_tolerance_stop_at_center'


def test_real_likelihood_strict_local_barrier_does_not_claim_global_optimum():
    # The analytic example in unknown-unit-search-alternatives: every single
    # move is worse, but replacing both rows with xy improves total objective.
    class Source:
        alphabet = ('a', 'b')
        probabilities = [[.99, .01], [.01, .99], [.99, .01]]
        def state(self, history): return 0
        def step(self, state, letter): return letter + 1
    class Rows:
        def __getitem__(self, pair): return Source.probabilities[pair[0]][pair[1]]
    source = Source()
    source.probabilities = Rows()
    def score(key):
        likelihood = decode(source, key, 'xyxy', .5).log_likelihood
        return {'log_weight_unnormalized': None if likelihood == -math.inf else
                2*likelihood - (2 + sum(map(len, key))) * math.log(2)}
    result = expand_key_bank(score, ('x', 'y'), 'xy')
    assert result['best_units'] == ['x', 'y']
    assert result['stop_reason'] == 'local_tolerance_stop_at_center'
    assert score(('xy', 'xy'))['log_weight_unnormalized'] > result['bank'][0]['log_weight_unnormalized']
    assert not result['global_optimality_claimed']


def test_unreachable_keys_are_retained_with_exact_zero_mass():
    source = SuffixSource('ab', 0, 1., {'': {'a': 7, 'b': 3}})
    def score(key):
        value = decode(source, key, 'xy', .2).log_likelihood
        return {'log_weight_unnormalized': None if value == -math.inf else value}
    result = expand_key_bank(score, ('x', 'y'), 'xy')
    assert any(row['log_weight'] is None for row in result['bank'])
    assert result['finite_keys'] < result['bank_size']
    assert math.fsum(math.exp(r['log_weight']) for r in result['bank'] if r['log_weight'] is not None) == pytest.approx(1.)


def test_interruptions_preserve_progress_but_never_return_a_complete_bank():
    stream = []
    def fail_after_three(key):
        if len(stream) == 3:
            raise RuntimeError('resource limit')
        return {'log_weight_unnormalized': -1.}
    with pytest.raises(RuntimeError, match='resource limit'):
        expand_key_bank(fail_after_three, ('x', 'y'), 'xy', progress=stream.append)
    assert len(stream) == 3 and all('log_weight' not in row for row in stream)


@pytest.mark.parametrize('weight', [float('nan'), float('inf'), -float('inf'), True])
def test_nonfinite_or_boolean_weights_are_not_silently_unsupported(weight):
    with pytest.raises(ValueError):
        expand_key_bank(lambda key: {'log_weight_unnormalized': weight}, ('x',), 'xy')


def test_numerical_scores_equal_independent_full_enumeration_on_one_row():
    model = SuffixSource('a', 0, 1., {'': {'a': 1}})
    pool = [''.join(s) for n in (1, 2) for s in itertools.product('xy', repeat=n)]
    def score(key):
        value = decode(model, key, 'xxxx', .2).log_likelihood
        return {'log_weight_unnormalized': None if value == -math.inf else value}
    result = expand_key_bank(score, ('x',), 'xy')
    wanted = max((u for u in pool if score((u,))['log_weight_unnormalized'] is not None),
                 key=lambda u: score((u,))['log_weight_unnormalized'])
    assert result['best_units'] == [wanted]
    assert {tuple(row['units']) for row in result['bank']} == {(u,) for u in pool}


@pytest.mark.parametrize('offset', [-1e100, 1e100])
def test_large_common_score_offset_does_not_destroy_normalization(offset):
    result = expand_key_bank(lambda key: {'log_weight_unnormalized': offset}, ('x',), 'xy')
    assert all(row['log_weight'] == pytest.approx(-math.log(6)) for row in result['bank'])
    assert math.fsum(math.exp(row['log_weight']) for row in result['bank']) == pytest.approx(1.)
