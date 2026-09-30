import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.audit_suffix_reader001 import infer_reference, reading_score
from voynich.compact_suffix_adapter import CompactSuffixAdapter
from voynich.compact_suffix_reference import LiteralCountView
from voynich.compact_suffix_source import fit_compact
from voynich.sparse_suffix_source import collect_counts, decode


@pytest.mark.parametrize('order', [0, 3, 8, 12])
def test_literal_view_and_separate_backward_match_original_counts(order):
    training = ['aaabbabbabbabb', 'bbabaa']
    model = fit_compact(training, 'ab', order, 2)
    view = LiteralCountView(model, cache_size=2)
    expected = {c: row for c, row in collect_counts(training, 'ab', order).items()
                if not c or sum(row.values()) >= 2}
    assert dict(view) == expected
    assert len(view) == len(expected) and 'z' not in view and 'a' * 15 not in view
    raw = {'alphabet': 'ab', 'order': order, 'tau': 16., 'counts': view}
    adapter = CompactSuffixAdapter(model, 16.)
    for observed in ('', 'x', 'xxxx', 'xxxxxx'):
        result = decode(adapter, ('x', 'xx'), observed, .2)
        total, best, nodes = infer_reference(raw, ('x', 'xx'), observed, .2)
        assert result.log_likelihood == pytest.approx(total, abs=1e-12)
        assert result.joint_log_probability == pytest.approx(best, abs=1e-12)
        assert result.reachable_nodes == nodes
        assert reading_score(raw, ('x', 'xx'), observed, result.plaintext, .2) == pytest.approx(best, abs=1e-12)
