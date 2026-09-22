"""Fresh, explicit two-slot bindings for attention-route qualification."""

from .campaign import canonical_digest


# Each bundle is a split unit. Names and answer words never cross splits.
DISCOVERY = (
    ('Dax', 'Miro', 'amber', 'velvet', 'horizon'),
    ('Taren', 'Luma', 'cobalt', 'ivory', 'harbor'),
    ('Neris', 'Veko', 'maple', 'cedar', 'window'),
    ('Sorin', 'Pavo', 'coral', 'indigo', 'planet'),
)
CONFIRMATION = (
    ('Bex', 'Rilo', 'copper', 'saffron', 'garden'),
    ('Kavi', 'Zeno', 'marble', 'orchid', 'lantern'),
    ('Feron', 'Tula', 'raven', 'lilac', 'pocket'),
    ('Jori', 'Navo', 'willow', 'pebble', 'ribbon'),
)


def path_tasks(split):
    if split not in ('discovery', 'confirmation'):
        raise ValueError('Unknown path-task split')
    bundles = DISCOVERY if split == 'discovery' else CONFIRMATION
    rows = []
    for bundle, (left, right, first, second, copy_word) in enumerate(bundles):
        for orientation in (0, 1):
            values = (first, second) if orientation == 0 else (second, first)
            opposite = values[::-1]

            def prompt(assigned, query):
                if split == 'discovery':
                    setup = f'Records:\n{left}: {assigned[0]}\n{right}: {assigned[1]}\n'
                    ending = (f'Question: Which value belongs to {query}?\n'
                              'Reply with the value word only.')
                else:
                    setup = (f'Use this little registry:\n'
                             f'Name {left} has value {assigned[0]}.\n'
                             f'Name {right} has value {assigned[1]}.\n')
                    ending = (f'Look up {query} in the registry. '
                              'Output its value as one word.')
                return setup + ending

            for slot, query in enumerate((left, right)):
                row = {
                    'id': f'{split}/{bundle}/{orientation}/{slot}',
                    'split': split, 'bundle': bundle,
                    'family': 'binding', 'expected_effect': 'change',
                    'prompt': prompt(values, query),
                    'donor_prompt': prompt(opposite, query),
                    'answer': values[slot], 'donor_answer': opposite[slot],
                }
                rows.append(row)
            # Literal copying is queried under the same swapped registry.
            if split == 'discovery':
                copy_prompt = (f'Records:\n{left}: {values[0]}\n{right}: {values[1]}\n'
                               f'Question: Ignore those records and copy {copy_word}.\n'
                               'Reply with that one word only.')
                donor_copy = (f'Records:\n{left}: {opposite[0]}\n{right}: {opposite[1]}\n'
                              f'Question: Ignore those records and copy {copy_word}.\n'
                              'Reply with that one word only.')
            else:
                copy_prompt = (f'Use this little registry:\n'
                               f'Name {left} has value {values[0]}.\n'
                               f'Name {right} has value {values[1]}.\n'
                               f'Ignore the registry. Repeat {copy_word} as one word.')
                donor_copy = (f'Use this little registry:\n'
                              f'Name {left} has value {opposite[0]}.\n'
                              f'Name {right} has value {opposite[1]}.\n'
                              f'Ignore the registry. Repeat {copy_word} as one word.')
            rows.append({
                'id': f'{split}/{bundle}/{orientation}/copy',
                'split': split, 'bundle': bundle,
                'family': 'copy', 'expected_effect': 'preserve',
                'prompt': copy_prompt, 'donor_prompt': donor_copy,
                'answer': copy_word, 'donor_answer': copy_word,
            })
    if len(rows) != 24 or len({r['id'] for r in rows}) != 24:
        raise AssertionError('Unexpected path-task count')
    for row in rows:
        row['prompt_sha256'] = canonical_digest(row['prompt'])
    return rows
