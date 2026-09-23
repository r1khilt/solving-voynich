"""PATH-0008 panel isolation, structured-control mapping and frozen selection."""

from voynich.workspace.path6_tasks import BUNDLES as PATH6_BUNDLES, path6_tasks
from voynich.workspace.path8_campaign import HEAD_LAYERS, select_heads
from voynich.workspace.path8_tasks import CONFIRMATION_BUNDLES, path8_tasks


def test_exposed_discovery_and_fresh_confirmation_pairs():
    discovery = path8_tasks('discovery')
    confirmation = path8_tasks('confirmation')
    original = path6_tasks('discovery')
    assert [r['prompt'] for r in discovery] == [r['prompt'] for r in original]
    assert len(discovery) == len(confirmation) == 48
    assert len({r['prompt'] for r in discovery + confirmation}) == 96
    old = {word.casefold() for group in PATH6_BUNDLES.values()
           for bundle in group for word in bundle}
    new = {word.casefold() for bundle in CONFIRMATION_BUNDLES for word in bundle}
    assert old.isdisjoint(new)
    for group in (discovery, confirmation):
        assert sum(r['family'] == 'binding' for r in group) == 32
        assert sum(r['family'] == 'copy' for r in group) == 16
        lookup = {r['prompt']: r for r in group}
        by_id = {r['id']: r for r in group}
        for row in group:
            opposite = lookup[row['donor_prompt']]
            assert opposite['donor_prompt'] == row['prompt']
            assert opposite['answer'] == row['donor_answer']
            parts = row['id'].split('/')
            mismatch = '/'.join(parts[:2]+[str((row['bundle']+1) % 8)]+parts[3:])
            assert mismatch in by_id
            assert by_id[mismatch]['family'] == row['family']


def test_head_selection_ties_and_top_four_sum():
    rows = []
    for layer in HEAD_LAYERS:
        for head in range(32):
            rows.append({'task_id': 't', 'layer': layer, 'head': head,
                         'margin_drop': 5.0 if layer == 29 and head < 4 else 1.0})
    layer, heads, ranking = select_heads(rows, {'t'})
    assert layer == 29
    assert heads == (0, 1, 2, 3)
    assert ranking[1]['top_four_sum'] == 20.0
    for row in rows:
        row['margin_drop'] = 1.0
    layer, heads, _ = select_heads(rows, {'t'})
    assert layer == 28
    assert heads == (0, 1, 2, 3)
