"""Causal field scopes and fresh two-hop task integrity."""

import numpy as np

from voynich.workspace.path3_tasks import BUNDLES as OLD
from voynich.workspace.path4_campaign import CONDITIONS, LAYERS, make_fields
from voynich.workspace.path4_tasks import BUNDLES, path4_tasks


def test_fresh_panels_and_counterfactuals():
    old_words = {word.lower() for group in OLD.values() for bundle in group for word in bundle}
    new_words = {word.lower() for group in BUNDLES.values() for bundle in group for word in bundle}
    assert not old_words & new_words
    assert len(LAYERS) == 3 and len(CONDITIONS) == 7
    for split in BUNDLES:
        rows = path4_tasks(split)
        assert len(rows) == 36
        assert sum(r['family'] == 'composed' for r in rows) == 24
        assert sum(r['family'] == 'copy' for r in rows) == 12
        assert len({r['prompt'] for r in rows}) == 36
        for row in rows:
            assert row['prompt'] != row['donor_prompt']
            assert row['prompt'].split('Second table' if row['template'] == 0 else 'Lookup 2')[1] == (
                row['donor_prompt'].split('Second table' if row['template'] == 0 else 'Lookup 2')[1])
            if row['family'] == 'composed':
                assert row['answer'] != row['donor_answer']


def test_suffix_partition_and_random_norm():
    delta = np.arange(36, dtype=np.float32).reshape(9, 4)
    fields = make_fields(delta, [2, 3], seed=19)
    np.testing.assert_array_equal(fields['value'][2:4], delta[2:4])
    assert not np.any(fields['value'][:2])
    np.testing.assert_array_equal(fields['suffix'][4:8], delta[4:8])
    assert not np.any(fields['suffix'][:4])
    np.testing.assert_array_equal(fields['value_suffix'], fields['value']+fields['suffix'])
    np.testing.assert_array_equal(fields['whole'], delta)
    np.testing.assert_array_equal(fields['all_earlier'][-1], 0)
    np.testing.assert_allclose(np.linalg.norm(fields['suffix_random'][4:8], axis=1),
                               np.linalg.norm(delta[4:8], axis=1), atol=1e-5)
