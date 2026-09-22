"""Frozen composed-task and selection checks."""

from voynich.workspace.path3_campaign import WINDOWS, changed_positions, choose_window
from voynich.workspace.path3_tasks import BUNDLES, path3_tasks


def test_fresh_panel_and_pairing():
    first = path3_tasks('discovery')
    second = path3_tasks('confirmation')
    assert len(first) == len(second) == 24
    assert len({word for group in BUNDLES.values() for bundle in group for word in bundle}) == 56
    assert {r['prompt'] for r in first}.isdisjoint({r['prompt'] for r in second})
    for group in (first, second):
        assert sum(r['family'] == 'composed' for r in group) == 16
        assert sum(r['family'] == 'copy' for r in group) == 8
        for row in group:
            assert row['prompt'] != row['donor_prompt']
            if row['family'] == 'composed':
                assert row['answer'] != row['donor_answer']


def test_changed_positions_and_frozen_selection():
    assert changed_positions([0]*10+[1, 2]+[0]*25, [0]*10+[2, 1]+[0]*25) == [10, 11]
    rows = [{'task_id': 'x', 'condition': 'upstream', 'desired_answer': True}]
    for name in WINDOWS:
        rows.append({'task_id': 'x', 'condition': name,
                     'donor_first_token': name not in ('16-19', '20-23')})
    selected, ranking = choose_window(rows, {'x'})
    assert selected == '16-19'
    assert len(ranking) == 5
