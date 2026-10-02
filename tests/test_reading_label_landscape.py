"""Full pair allocation and exact choice, not only an advantageous neighbor."""
import itertools
from fractions import Fraction

from voynich.reading_label_landscape import complete_label_landscape, target_fingerprint
from voynich.reading_regrowth import SourceRegrowth
from voynich.source_action_proposal import ReadingEnvironment
from tests.test_reading_label_transport import exact_target, source_fixture


def test_all_pairs_and_target_choice_without_reference_replay(monkeypatch):
    source = source_fixture(3, True)
    env = ReadingEnvironment(((0, 1), (1, 0, 1)), rows=3, glyphs=2)
    sampler = SourceRegrowth(source, env)
    path = sampler.path(forced_actions=(2, 4, 2, 4, 4))
    monkeypatch.setattr(sampler, 'path', lambda **_: (_ for _ in ()).throw(AssertionError('No q replay')))
    pairs, best, summary = complete_label_landscape(sampler, path)
    assert [p['pair'] for p in pairs] == [[0, 1], [0, 2], [1, 2]]
    old = exact_target(source, env, path.state, sampler.config.stop)
    expected = []
    for a, b in itertools.combinations(range(3), 2):
        mapping = tuple(b if row == a else a if row == b else row for row in range(3))
        from tests.test_reading_label_orbit import permute
        state = permute(path.state, mapping)
        expected.append((exact_target(source, env, state, sampler.config.stop), [a, b], state))
    largest = max([old]+[e[0] for e in expected])
    assert summary['uphill_pairs'] == sum(e[0] > old for e in expected)
    assert summary['downhill_pairs'] == sum(e[0] < old for e in expected)
    assert summary['equal_target_pairs'] == sum(e[0] == old for e in expected)
    assert summary['reference_q_replay_calls'] == 0
    if largest > old:
        expected_best = next(e for e in expected if e[0] == largest)
        assert summary['best_uphill_pair'] == expected_best[1] and best.state == expected_best[2]
        assert Fraction(best.numerator, best.denominator) == largest
    else:
        assert best is None and summary['best_uphill_pair'] is None


def test_integer_fingerprint_handles_long_targets_and_framing():
    assert target_fingerprint(2**40000, 3**30000) == target_fingerprint(2**40000, 3**30000)
    assert target_fingerprint(1, 258) != target_fingerprint(257, 2)
    assert target_fingerprint(2, 4) != target_fingerprint(1, 2)  # Exact unreduced identity.
