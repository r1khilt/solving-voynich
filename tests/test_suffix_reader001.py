import itertools
import math
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.audit_suffix_reader001 import counts_reference, infer_reference, reading_score
from scripts.run_suffix_reader001 import gate, window_offsets
from voynich.sparse_suffix_source import SuffixSource, collect_counts, decode


def test_fixed_window_allocation_has_no_overlap_and_is_inside_one_known_body():
    offsets = [o for pair in window_offsets() for o in pair]
    assert len(offsets) == 32
    assert min(offsets) == 40000 and max(offsets) + 224 == 48160
    assert all(a + 224 <= b for a, b in zip(offsets, offsets[1:]))
    assert all(35839 <= a and a + 224 <= 50000 for a in offsets)


def test_integer_gates_retain_every_key_and_exact_boundary():
    assert all(gate([8] * 16, [16] * 16).values())
    assert not gate([9] * 16, [20] * 16)['overall_cer_at_most_002']
    assert not gate([0] * 15 + [23], [20] * 16)['every_key_cer_at_most_005']
    assert not gate([8] * 16, [10] * 16)['at_least_25pct_relative_edit_reduction']
    with pytest.raises(ValueError):
        gate([0] * 15, [1] * 16)


@pytest.mark.parametrize('order', [0, 3, 8, 12])
def test_reverse_reference_and_direct_path_score_agree_on_all_small_observations(order):
    records, alphabet = ('abbabbabab', 'baa', 'ab'), 'ab'
    counts = collect_counts(records, alphabet, order)
    assert counts == counts_reference(records, order)
    source = SuffixSource(alphabet, order, 4., counts)
    raw = source.to_dict()
    for units in (('x', 'xy'), ('xy','xy'), ('xy','yx'), ('x','y')):
        for length in range(5):
            for letters in itertools.product('xy', repeat=length):
                observed = ''.join(letters)
                result = decode(source, units, observed, .25)
                total, best, nodes = infer_reference(raw, units, observed, .25)
                assert nodes == result.reachable_nodes
                assert total == pytest.approx(result.log_likelihood, abs=2e-14)
                assert best == pytest.approx(result.joint_log_probability, abs=2e-14)
                assert reading_score(raw, units, observed, result.plaintext, .25) == pytest.approx(best, abs=2e-14)
    with pytest.raises(ValueError, match='re-encoding'):
        reading_score(raw, ('x','y'), 'x', 'b', .25)
    assert math.isinf(reading_score(raw, ('x','y'), 'z', None, .25))


def test_reference_cap_is_a_failure_not_a_pruned_result():
    raw = SuffixSource('ab', 3, 4., collect_counts(('ababbb',), 'ab', 3)).to_dict()
    with pytest.raises(RuntimeError, match='cap'):
        infer_reference(raw, ('x','y'), 'xy', .5, max_nodes=1)
