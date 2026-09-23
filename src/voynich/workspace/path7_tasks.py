"""Balanced fresh prompt-format qualification for composed lookups."""

BUNDLES = {
    'discovery': (
        ('Afen', 'Boril', 'sage', 'chalk', 'pear', 'fork', 'cloth'),
        ('Cerol', 'Davin', 'moss', 'clay', 'bell', 'chair', 'thread'),
        ('Emina', 'Foval', 'seed', 'wax', 'book', 'spoon', 'globe'),
        ('Gorin', 'Halen', 'salt', 'flax', 'wheel', 'crown', 'brush'),
        ('Iven', 'Joral', 'vine', 'coal', 'lamp', 'plate', 'bucket'),
        ('Kalen', 'Leris', 'sand', 'bark', 'clock', 'cup', 'pillow'),
    ),
    'confirmation': (
        ('Malen', 'Noril', 'milk', 'dust', 'ring', 'stone', 'glove'),
        ('Ovira', 'Pelor', 'rain', 'iron', 'fish', 'table', 'ribbon'),
        ('Qanor', 'Riven', 'snow', 'wood', 'door', 'apple', 'window'),
        ('Saler', 'Tovin', 'foam', 'rope', 'hat', 'key', 'paper'),
        ('Ulor', 'Vena', 'peat', 'silk', 'bag', 'boot', 'mirror'),
        ('Wenor', 'Xalia', 'steam', 'glass', 'nest', 'flag', 'hammer'),
    ),
}


def render(style, names, assigned, keys, outputs, *, query=None, copy_word=None):
    left, right = names
    if style == 'colon':
        body = (f'First table (name to key):\n{left}: {assigned[0]}\n{right}: {assigned[1]}\n'
                f'Second table (key to object):\n{keys[0]}: {outputs[0]}\n{keys[1]}: {outputs[1]}\n')
    elif style == 'arrow':
        body = (f'First table (name to key):\n{left} -> {assigned[0]}\n{right} -> {assigned[1]}\n'
                f'Second table (key to object):\n{keys[0]} -> {outputs[0]}\n{keys[1]} -> {outputs[1]}\n')
    else:
        raise ValueError('Unknown style')
    if copy_word is None:
        ending = f'Question: Follow both tables for {query}.\nFinal object (one word):'
    else:
        ending = f'Question: Ignore both tables and copy {copy_word}.\nFinal object (one word):'
    return body + ending


def path7_tasks(split):
    if split not in BUNDLES:
        raise ValueError('Unknown split')
    rows = []
    for bundle, (left, right, first_key, second_key, first_out, second_out, copy_word) in enumerate(BUNDLES[split]):
        keys, outputs = (first_key, second_key), (first_out, second_out)
        for style in ('colon', 'arrow'):
            for orientation in (0, 1):
                assigned = keys if orientation == 0 else keys[::-1]
                reverse = assigned[::-1]
                for slot, name in enumerate((left, right)):
                    rows.append({
                        'id': f'path7/{split}/{bundle}/{style}/{orientation}/{slot}',
                        'split': split, 'bundle': bundle, 'style': style, 'family': 'composed',
                        'prompt': render(style, (left, right), assigned, keys, outputs, query=name),
                        'donor_prompt': render(style, (left, right), reverse, keys, outputs, query=name),
                        'answer': outputs[keys.index(assigned[slot])],
                        'donor_answer': outputs[keys.index(reverse[slot])],
                    })
                rows.append({
                    'id': f'path7/{split}/{bundle}/{style}/{orientation}/copy',
                    'split': split, 'bundle': bundle, 'style': style, 'family': 'copy',
                    'prompt': render(style, (left, right), assigned, keys, outputs, copy_word=copy_word),
                    'donor_prompt': render(style, (left, right), reverse, keys, outputs, copy_word=copy_word),
                    'answer': copy_word, 'donor_answer': copy_word,
                })
    if len(rows) != 72 or len({r['prompt'] for r in rows}) != 72:
        raise AssertionError('Unexpected PATH-0007 panel')
    lookup = {r['prompt']: r for r in rows}
    if any(lookup[r['donor_prompt']]['donor_prompt'] != r['prompt'] for r in rows):
        raise AssertionError('Source/donor pairing failed')
    return rows
