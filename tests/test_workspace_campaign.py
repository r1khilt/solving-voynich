import numpy as np
import pytest

from voynich.workspace.campaign import answer_correct, article_windows, canonical_digest, write_json
from voynich.workspace.geometry import sparse_nonnegative_projection


def test_article_windows_never_cross_or_reuse_articles():
    text = '\n'.join(f' = article {i} = \n' + ' '.join([str(i)] * 20) for i in range(6))
    rows, eligible = article_windows(text, lambda s: [int(x) for x in s.split()], count=5, length=8, seed=9)
    assert eligible == 6
    assert len({r['article_id'] for r in rows}) == 5
    assert all(len(set(r['token_ids'])) == 1 and len(r['token_ids']) == 8 for r in rows)
    assert (rows, eligible) == article_windows(text, lambda s: [int(x) for x in s.split()], count=5, length=8, seed=9)
    with pytest.raises(ValueError, match='independent articles'):
        article_windows(text, lambda s: s.split(), count=7, length=8, seed=9)


def test_answer_policy_keeps_extra_explanation_incorrect():
    assert answer_correct(' "Brasília." ', ['Brasilia'])
    assert not answer_correct('Paris is the capital.', ['Paris'])
    assert not answer_correct('Paris.', ['Paris'], strict=True)
    assert not answer_correct('Cairo or Beijing', ['Cairo'])


def test_atomic_json_and_canonical_hash(tmp_path):
    assert canonical_digest({'a': 1, 'b': 2}) == canonical_digest({'b': 2, 'a': 1})
    write_json(tmp_path / 'x.json', {'value': 2})
    with pytest.raises(ValueError):
        write_json(tmp_path / 'x.json', {'value': float('nan')})
    assert '2' in (tmp_path / 'x.json').read_text()


def test_sparse_nonnegative_projection_retains_unexplained_component():
    pytest.importorskip("scipy", reason="Optional workspace-analysis dependency")
    reconstructed, report = sparse_nonnegative_projection([2., -3., 4.], np.eye(3), max_features=2)
    np.testing.assert_allclose(reconstructed, [2., 0., 4.])
    assert report['residual_fraction'] == pytest.approx(9 / 29)
    assert all(w >= 0 for w in report['coefficients'])
