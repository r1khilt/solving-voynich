import numpy as np
import pytest

from voynich.workspace.route_backend import positional_field
from voynich.workspace.route_campaign import route_tasks


def test_position_fields_partition_a_complete_donor_change():
    delta = np.arange(24,dtype=np.float32).reshape(6,4)
    earlier,last = positional_field(delta,'earlier'),positional_field(delta,'last')
    np.testing.assert_array_equal(earlier+last,delta)
    assert not earlier[-1].any() and not last[:-1].any()
    np.testing.assert_array_equal(positional_field(delta,'all'),delta)
    with pytest.raises(ValueError):
        positional_field(delta,'unknown')


def test_route_panel_keeps_development_pairs_and_equal_copy_labels():
    rows=route_tasks()
    assert len(rows)==60
    assert all(r['split']=='development' for r in rows)
    lookup={r['id']:r for r in rows}
    for row in rows:
        assert lookup[row['paired_id']]['prompt']==row['donor_prompt']
        if row['expected_effect']=='preserve':
            assert row['answers']==row['counterfactual_answers']
            assert row['answer']==row['answers'][0]
            assert not row['answer_absent_from_prompt']
