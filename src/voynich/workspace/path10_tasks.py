"""Fresh direct-lookup lexical bundles for a full four-block factorial grid."""

from .path6_tasks import BUNDLES as PATH6_BUNDLES, render
from .path8_tasks import CONFIRMATION_BUNDLES as PATH8_BUNDLES


BUNDLES = (
    ('Averik', 'Bovra', 'crimson', 'straw', 'compass'),
    ('Cemaro', 'Dunil', 'emerald', 'oatmeal', 'ladder'),
    ('Evikor', 'Foral', 'navy', 'cement', 'wallet'),
    ('Galenik', 'Havor', 'plaster', 'hazel', 'camera'),
    ('Ivoran', 'Jemor', 'maroon', 'pearl', 'puzzle'),
    ('Kavor', 'Lemik', 'teal', 'ash', 'bicycle'),
    ('Mavro', 'Nerin', 'cream', 'mauve', 'soccer'),
    ('Ovel', 'Parik', 'umber', 'lilac', 'tulip'),
)


def path10_tasks():
    previous = {word.casefold() for groups in PATH6_BUNDLES.values()
                for bundle in groups for word in bundle}
    previous.update(word.casefold() for bundle in PATH8_BUNDLES for word in bundle)
    words = [word.casefold() for bundle in BUNDLES for word in bundle]
    if len(words) != len(set(words)) or previous.intersection(words):
        raise AssertionError('PATH-0010 vocabulary overlaps PATH-0006/0008')
    rows = []
    for bundle, (left, right, first, second, copy_word) in enumerate(BUNDLES):
        template = bundle // 4
        for orientation in (0, 1):
            values = (first, second) if orientation == 0 else (second, first)
            reverse = values[::-1]
            for slot, name in enumerate((left, right)):
                rows.append({
                    'id': f'path10/{bundle}/{orientation}/{slot}', 'bundle': bundle,
                    'template': template, 'family': 'binding',
                    'prompt': render(template, (left, right), values, name),
                    'donor_prompt': render(template, (left, right), reverse, name),
                    'answer': values[slot], 'donor_answer': reverse[slot],
                })
            rows.append({
                'id': f'path10/{bundle}/{orientation}/copy', 'bundle': bundle,
                'template': template, 'family': 'copy',
                'prompt': render(template, (left, right), values, copy_word=copy_word),
                'donor_prompt': render(template, (left, right), reverse, copy_word=copy_word),
                'answer': copy_word, 'donor_answer': copy_word,
            })
    if len(rows) != 48 or len({r['prompt'] for r in rows}) != 48:
        raise AssertionError('Unexpected PATH-0010 panel')
    lookup = {r['prompt']: r for r in rows}
    if any(lookup[r['donor_prompt']]['donor_prompt'] != r['prompt'] for r in rows):
        raise AssertionError('PATH-0010 inverse pairing failed')
    return rows
