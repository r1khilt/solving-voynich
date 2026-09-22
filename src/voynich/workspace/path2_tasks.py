"""Fresh multi-template binding bundles for positional-route interventions."""


BUNDLES = (
    ('Pira', 'Venso', 'jade', 'bronze', 'island'),
    ('Malu', 'Korin', 'lemon', 'basil', 'forest'),
    ('Tega', 'Brin', 'crimson', 'silver', 'bridge'),
    ('Rava', 'Niko', 'oak', 'pine', 'mountain'),
    ('Dela', 'Suvi', 'pearl', 'denim', 'vessel'),
    ('Horan', 'Peli', 'azure', 'ochre', 'ladder'),
    ('Fira', 'Zaku', 'glass', 'paper', 'mirror'),
    ('Mena', 'Roku', 'tulip', 'birch', 'rocket'),
    ('Tala', 'Bevo', 'granite', 'satin', 'compass'),
    ('Jena', 'Vori', 'cherry', 'olive', 'basket'),
    ('Reko', 'Davi', 'falcon', 'otter', 'picture'),
    ('Vira', 'Solu', 'candle', 'meadow', 'ticket'),
)


def render(template, names, values, query, *, copy_word=None):
    left, right = names
    first, second = values
    if template == 0:
        setup = f'Records:\n{left}: {first}\n{right}: {second}\n'
        ending = (f'Question: Which value belongs to {query}?\nReply with the value word only.'
                  if copy_word is None else
                  f'Question: Ignore those records and copy {copy_word}.\nReply with that one word only.')
    elif template == 1:
        setup = (f'Use this little registry:\nName {left} has value {first}.\n'
                 f'Name {right} has value {second}.\n')
        ending = (f'Look up {query} in the registry. Output its value as one word.'
                  if copy_word is None else
                  f'Ignore the registry. Repeat {copy_word} as one word.')
    elif template == 2:
        setup = (f'Lookup list:\n{left} maps to {first}.\n'
                 f'{right} maps to {second}.\n')
        ending = (f'Return only the value mapped from {query}.'
                  if copy_word is None else
                  f'Ignore the lookup list. Return only the literal word {copy_word}.')
    else:
        raise ValueError('Unknown path2 template')
    return setup+ending


def path2_tasks():
    rows = []
    for bundle, (left, right, first, second, copy_word) in enumerate(BUNDLES):
        template = bundle//4
        for orientation in (0, 1):
            values = (first, second) if orientation == 0 else (second, first)
            reverse = values[::-1]
            for slot, name in enumerate((left, right)):
                rows.append({
                    'id': f'path2/{bundle}/{orientation}/{slot}', 'bundle': bundle,
                    'template': template, 'family': 'binding', 'expected_effect': 'change',
                    'prompt': render(template, (left, right), values, name),
                    'donor_prompt': render(template, (left, right), reverse, name),
                    'answer': values[slot], 'donor_answer': reverse[slot],
                })
            rows.append({
                'id': f'path2/{bundle}/{orientation}/copy', 'bundle': bundle,
                'template': template, 'family': 'copy', 'expected_effect': 'preserve',
                'prompt': render(template, (left, right), values, None, copy_word=copy_word),
                'donor_prompt': render(template, (left, right), reverse, None, copy_word=copy_word),
                'answer': copy_word, 'donor_answer': copy_word,
            })
    if len(rows) != 72 or len({r['id'] for r in rows}) != 72:
        raise AssertionError('Unexpected path2 task panel')
    lookup = {r['prompt']: r for r in rows}
    if len(lookup) != 72 or any(lookup[r['donor_prompt']]['donor_prompt'] != r['prompt'] for r in rows):
        raise AssertionError('Path2 pairing failed')
    return rows
