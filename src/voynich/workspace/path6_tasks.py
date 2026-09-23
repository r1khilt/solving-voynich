"""Fresh direct-lookup bundles for a downstream attention mediation test."""

BUNDLES = {
    'discovery': (
        ('Arola', 'Mivik', 'amber', 'copper', 'pepper'),
        ('Benro', 'Silan', 'velvet', 'marble', 'pencil'),
        ('Cavik', 'Lomra', 'violet', 'golden', 'hammer'),
        ('Darin', 'Nevok', 'walnut', 'cotton', 'window'),
        ('Elaro', 'Puvin', 'silver', 'clover', 'button'),
        ('Femon', 'Ralik', 'orange', 'indigo', 'pillow'),
        ('Gavin', 'Torel', 'cobalt', 'maple', 'garden'),
        ('Havro', 'Zemik', 'rubber', 'plum', 'bottle'),
    ),
    'confirmation': (
        ('Imero', 'Balin', 'saffron', 'pewter', 'ribbon'),
        ('Javon', 'Cemik', 'coral', 'cedar', 'folder'),
        ('Kerin', 'Dovra', 'scarlet', 'linen', 'bucket'),
        ('Lavin', 'Ebor', 'ivory', 'cinnamon', 'glove'),
        ('Maron', 'Fivik', 'bronze', 'mint', 'napkin'),
        ('Nerik', 'Govar', 'quartz', 'wool', 'kettle'),
        ('Omal', 'Hesin', 'yellow', 'slate', 'spoon'),
        ('Pavin', 'Jelor', 'tangerine', 'tin', 'cushion'),
    ),
}


def render(template, names, values, query=None, *, copy_word=None):
    left, right = names
    first, second = values
    if template == 0:
        body = f'Records:\n{left}: {first}\n{right}: {second}\n'
        end = (f'Question: Which value belongs to {query}?\nReply with the value word only.'
               if copy_word is None else
               f'Question: Ignore those records and copy {copy_word}.\nReply with that one word only.')
    elif template == 1:
        body = f'Lookup list:\n{left} maps to {first}.\n{right} maps to {second}.\n'
        end = (f'Return only the value mapped from {query}.' if copy_word is None else
               f'Ignore the lookup list. Return only the literal word {copy_word}.')
    else:
        raise ValueError('Unknown template')
    return body + end


def path6_tasks(split):
    if split not in BUNDLES:
        raise ValueError('Unknown split')
    rows = []
    for bundle, (left, right, first, second, copy_word) in enumerate(BUNDLES[split]):
        template = bundle // 4
        for orientation in (0, 1):
            values = (first, second) if orientation == 0 else (second, first)
            reverse = values[::-1]
            for slot, name in enumerate((left, right)):
                rows.append({
                    'id': f'path6/{split}/{bundle}/{orientation}/{slot}', 'split': split,
                    'bundle': bundle, 'template': template, 'family': 'binding',
                    'prompt': render(template, (left, right), values, name),
                    'donor_prompt': render(template, (left, right), reverse, name),
                    'answer': values[slot], 'donor_answer': reverse[slot],
                })
            rows.append({
                'id': f'path6/{split}/{bundle}/{orientation}/copy', 'split': split,
                'bundle': bundle, 'template': template, 'family': 'copy',
                'prompt': render(template, (left, right), values, copy_word=copy_word),
                'donor_prompt': render(template, (left, right), reverse, copy_word=copy_word),
                'answer': copy_word, 'donor_answer': copy_word,
            })
    if len(rows) != 48 or len({r['prompt'] for r in rows}) != 48:
        raise AssertionError('Unexpected PATH-0006 task panel')
    lookup = {r['prompt']: r for r in rows}
    if any(lookup[r['donor_prompt']]['donor_prompt'] != r['prompt'] for r in rows):
        raise AssertionError('Source/donor pairing failed')
    return rows
