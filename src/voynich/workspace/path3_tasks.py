"""Fresh composed name-to-key-to-output tasks for attention mediation."""

BUNDLES = {
    'discovery': (
        ('Arel', 'Banu', 'teal', 'plum', 'shell', 'drum', 'cloud'),
        ('Ceto', 'Darin', 'mint', 'gold', 'horse', 'spoon', 'flame'),
        ('Evar', 'Fomi', 'rose', 'sand', 'bread', 'chair', 'whale'),
        ('Giri', 'Halu', 'lime', 'snow', 'clock', 'feather', 'boat'),
    ),
    'confirmation': (
        ('Iven', 'Joru', 'salt', 'moss', 'spider', 'brick', 'star'),
        ('Karo', 'Lemi', 'wheat', 'coal', 'tiger', 'pillow', 'moon'),
        ('Mavi', 'Nero', 'ink', 'dust', 'knife', 'flower', 'train'),
        ('Odin', 'Pelu', 'rain', 'clay', 'piano', 'beach', 'frog'),
    ),
}


def render(template, names, assigned, keys, outputs, query=None, *, copy_word=None):
    left, right = names
    if template == 0:
        first = f'Table A (person to key):\n{left}: {assigned[0]}\n{right}: {assigned[1]}\n'
        second = f'Table B (key to object):\n{keys[0]}: {outputs[0]}\n{keys[1]}: {outputs[1]}\n'
        question = (f'Question: Which object belongs to {query} after following both tables?'
                    if copy_word is None else f'Question: Ignore both tables. Copy {copy_word}.')
        return first+second+question+'\nReply with one object word only.'
    if template == 1:
        first = f'First lookup, name to label:\n{left} -> {assigned[0]}\n{right} -> {assigned[1]}\n'
        second = f'Second lookup, label to item:\n{keys[0]} -> {outputs[0]}\n{keys[1]} -> {outputs[1]}\n'
        question = (f'Follow both lookups starting at {query}. Return only the final item word.'
                    if copy_word is None else f'Ignore the lookups and return only {copy_word}.')
        return first+second+question
    raise ValueError('Unknown template')


def path3_tasks(split):
    if split not in BUNDLES:
        raise ValueError('Unknown split')
    rows = []
    for bundle, (left, right, key_a, key_b, out_a, out_b, copy_word) in enumerate(BUNDLES[split]):
        template = bundle % 2
        keys, outputs = (key_a, key_b), (out_a, out_b)
        for orientation in (0, 1):
            assigned = keys if orientation == 0 else keys[::-1]
            reverse = assigned[::-1]
            for slot, name in enumerate((left, right)):
                index = keys.index(assigned[slot])
                donor_index = keys.index(reverse[slot])
                rows.append({
                    'id': f'path3/{split}/{bundle}/{orientation}/{slot}', 'split': split,
                    'bundle': bundle, 'template': template, 'family': 'composed',
                    'prompt': render(template, (left, right), assigned, keys, outputs, name),
                    'donor_prompt': render(template, (left, right), reverse, keys, outputs, name),
                    'answer': outputs[index], 'donor_answer': outputs[donor_index],
                })
            rows.append({
                'id': f'path3/{split}/{bundle}/{orientation}/copy', 'split': split,
                'bundle': bundle, 'template': template, 'family': 'copy',
                'prompt': render(template, (left, right), assigned, keys, outputs, copy_word=copy_word),
                'donor_prompt': render(template, (left, right), reverse, keys, outputs, copy_word=copy_word),
                'answer': copy_word, 'donor_answer': copy_word,
            })
    if len(rows) != 24 or len({r['id'] for r in rows}) != 24:
        raise AssertionError('Unexpected PATH-0003 panel')
    lookup = {r['prompt']: r for r in rows}
    if len(lookup) != 24 or any(lookup[r['donor_prompt']]['donor_prompt'] != r['prompt'] for r in rows):
        raise AssertionError('Source/donor pairing failed')
    if any(r['prompt'].split('Table B' if r['template'] == 0 else 'Second lookup')[1]
           != r['donor_prompt'].split('Table B' if r['template'] == 0 else 'Second lookup')[1]
           for r in rows):
        raise AssertionError('Downstream table or question changed')
    return rows
