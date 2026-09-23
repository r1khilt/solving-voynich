"""Exposed PATH-0006 discovery and disjoint direct-lookup confirmation bundles."""

from .path6_tasks import BUNDLES as PATH6_BUNDLES, path6_tasks, render


CONFIRMATION_BUNDLES = (
    ('Rafik', 'Vemol', 'bamboo', 'nickel', 'candle'),
    ('Saron', 'Welik', 'ginger', 'canvas', 'lantern'),
    ('Tavik', 'Xelor', 'granite', 'almond', 'pocket'),
    ('Umer', 'Yalor', 'oyster', 'basil', 'towel'),
    ('Varin', 'Zemol', 'cherry', 'khaki', 'basket'),
    ('Warin', 'Alofik', 'satin', 'onyx', 'ticket'),
    ('Xavon', 'Belik', 'lemon', 'beige', 'tablet'),
    ('Yelor', 'Cavon', 'mustard', 'denim', 'saddle'),
)


def path8_tasks(split):
    if split == 'discovery':
        return [{**task, 'id': task['id'].replace('path6/', 'path8/', 1)}
                for task in path6_tasks('discovery')]
    if split != 'confirmation':
        raise ValueError('Unknown PATH-0008 split')
    old_words = {word.casefold() for group in PATH6_BUNDLES.values()
                 for bundle in group for word in bundle}
    new_words = [word.casefold() for bundle in CONFIRMATION_BUNDLES for word in bundle]
    if len(new_words) != len(set(new_words)) or old_words.intersection(new_words):
        raise AssertionError('Confirmation vocabulary is not disjoint from PATH-0006')
    rows = []
    for bundle, (left, right, first, second, copy_word) in enumerate(CONFIRMATION_BUNDLES):
        template = bundle // 4
        for orientation in (0, 1):
            values = (first, second) if orientation == 0 else (second, first)
            reverse = values[::-1]
            for slot, name in enumerate((left, right)):
                rows.append({
                    'id': f'path8/confirmation/{bundle}/{orientation}/{slot}',
                    'split': 'confirmation', 'bundle': bundle, 'template': template,
                    'family': 'binding', 'prompt': render(template, (left, right), values, name),
                    'donor_prompt': render(template, (left, right), reverse, name),
                    'answer': values[slot], 'donor_answer': reverse[slot],
                })
            rows.append({
                'id': f'path8/confirmation/{bundle}/{orientation}/copy',
                'split': 'confirmation', 'bundle': bundle, 'template': template,
                'family': 'copy', 'prompt': render(template, (left, right), values,
                                                 copy_word=copy_word),
                'donor_prompt': render(template, (left, right), reverse,
                                       copy_word=copy_word),
                'answer': copy_word, 'donor_answer': copy_word,
            })
    if len(rows) != 48 or len({r['prompt'] for r in rows}) != 48:
        raise AssertionError('Unexpected PATH-0008 confirmation panel')
    lookup = {r['prompt']: r for r in rows}
    if any(lookup[r['donor_prompt']]['donor_prompt'] != r['prompt'] for r in rows):
        raise AssertionError('Source/donor pairing failed')
    return rows
