"""Critical panel and selection invariants for PATH-0006."""

from voynich.workspace.path6_campaign import WINDOWS, changed_positions, select_window
from voynich.workspace.path6_tasks import path6_tasks


def test_fresh_disjoint_panel_and_inverse_pairs():
    discovery, confirmation = path6_tasks('discovery'), path6_tasks('confirmation')
    assert len(discovery) == len(confirmation) == 48
    assert len({r['prompt'] for r in discovery + confirmation}) == 96
    for group in (discovery, confirmation):
        lookup = {r['prompt']: r for r in group}
        assert sum(r['family'] == 'binding' for r in group) == 32
        assert sum(r['family'] == 'copy' for r in group) == 16
        for row in group:
            opposite = lookup[row['donor_prompt']]
            assert opposite['donor_prompt'] == row['prompt']
            assert opposite['answer'] == row['donor_answer']


def test_position_guard_and_selection_tie_break():
    assert changed_positions([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
                             [1, 4, 5, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]) == [1, 2]
    rows = [{'task_id': 't', 'condition': 'upstream', 'desired_answer': True}]
    rows += [{'task_id': 't', 'condition': name, 'donor_first_token': False} for name in WINDOWS]
    selected, ranking = select_window(rows, {'t'})
    assert selected == '24-27'
    assert all(item['donor_removed'] == 1 for item in ranking)
