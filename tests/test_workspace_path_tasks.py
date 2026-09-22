from voynich.workspace.path_tasks import path_tasks


def test_path_bundles_and_reverse_interventions_are_disjoint():
    discovery = path_tasks('discovery')
    confirmation = path_tasks('confirmation')
    assert len(discovery) == len(confirmation) == 24
    assert len({r['id'] for r in discovery+confirmation}) == 48
    left_words = {r['answer'] for r in discovery}
    right_words = {r['answer'] for r in confirmation}
    assert not left_words & right_words
    assert all(r['answer'] != r['donor_answer'] for r in discovery+confirmation if r['family'] == 'binding')
    assert all(r['answer'] == r['donor_answer'] for r in discovery+confirmation if r['family'] == 'copy')


def test_only_the_two_record_values_change_in_donor_prompt():
    for split in ('discovery', 'confirmation'):
        rows = path_tasks(split)
        by_prompt = {r['prompt']: r for r in rows}
        for row in rows:
            paired = by_prompt[row['donor_prompt']]
            assert paired['donor_prompt'] == row['prompt']
            assert paired['family'] == row['family']
            assert paired['bundle'] == row['bundle']
