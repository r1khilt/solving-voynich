"""Fresh composed lookup panel with stronger unrelated-copy controls."""

BUNDLES = {
    'discovery': (
        ('Qira', 'Romi', 'cove', 'ridge', 'apple', 'needle', 'orbit'),
        ('Sena', 'Tuko', 'berry', 'root', 'anchor', 'goat', 'lamp'),
        ('Ulan', 'Viro', 'dawn', 'dusk', 'cup', 'kite', 'ladle'),
        ('Weka', 'Xoni', 'peach', 'stone', 'rabbit', 'fork', 'marsh'),
        ('Yaro', 'Zemi', 'leaf', 'flint', 'wagon', 'hat', 'rope'),
        ('Abel', 'Bira', 'fog', 'honey', 'sword', 'cupcake', 'owl'),
    ),
    'confirmation': (
        ('Cavi', 'Doro', 'lake', 'hill', 'coin', 'duck', 'path'),
        ('Eron', 'Falu', 'ruby', 'chalk', 'melon', 'brush', 'gate'),
        ('Geno', 'Hiro', 'steam', 'frost', 'pencil', 'shoe', 'tower'),
        ('Iram', 'Jova', 'grain', 'mud', 'bottle', 'fox', 'road'),
        ('Kira', 'Lavo', 'reed', 'flour', 'helmet', 'book', 'prism'),
        ('Meko', 'Nalu', 'smoke', 'dew', 'crown', 'fish', 'harp'),
    ),
}


def render(template, names, assigned, keys, outputs, *, query=None, copy_word=None):
    left, right = names
    if template == 0:
        start = (f'First table, person to key:\n{left}: {assigned[0]}\n{right}: {assigned[1]}\n'
                 f'Second table, key to object:\n{keys[0]}: {outputs[0]}\n{keys[1]}: {outputs[1]}\n')
        if copy_word is None:
            return start+f'Follow both tables for {query}. Reply with only the final object word.'
        return start+f'Ignore both tables. Copy exactly this word: {copy_word}. Reply with only {copy_word}.'
    if template == 1:
        start = (f'Lookup 1:\n{left} -> {assigned[0]}\n{right} -> {assigned[1]}\n'
                 f'Lookup 2:\n{keys[0]} -> {outputs[0]}\n{keys[1]} -> {outputs[1]}\n')
        if copy_word is None:
            return start+f'Start at {query}, follow lookup 1 and then lookup 2. Return the resulting word only.'
        return start+f'Exception: do not use either lookup. Return exactly {copy_word} and nothing else.'
    raise ValueError('Unknown template')


def path4_tasks(split):
    if split not in BUNDLES:
        raise ValueError('Unknown split')
    rows = []
    for bundle, (left, right, key_a, key_b, out_a, out_b, copy_word) in enumerate(BUNDLES[split]):
        keys, outputs = (key_a, key_b), (out_a, out_b)
        template = bundle % 2
        for orientation in (0, 1):
            assigned = keys if orientation == 0 else keys[::-1]
            reverse = assigned[::-1]
            for slot, name in enumerate((left, right)):
                index, donor_index = keys.index(assigned[slot]), keys.index(reverse[slot])
                rows.append({
                    'id': f'path4/{split}/{bundle}/{orientation}/{slot}', 'split': split,
                    'bundle': bundle, 'template': template, 'family': 'composed',
                    'prompt': render(template, (left, right), assigned, keys, outputs, query=name),
                    'donor_prompt': render(template, (left, right), reverse, keys, outputs, query=name),
                    'answer': outputs[index], 'donor_answer': outputs[donor_index],
                })
            rows.append({
                'id': f'path4/{split}/{bundle}/{orientation}/copy', 'split': split,
                'bundle': bundle, 'template': template, 'family': 'copy',
                'prompt': render(template, (left, right), assigned, keys, outputs, copy_word=copy_word),
                'donor_prompt': render(template, (left, right), reverse, keys, outputs, copy_word=copy_word),
                'answer': copy_word, 'donor_answer': copy_word,
            })
    if len(rows) != 36 or len({r['id'] for r in rows}) != 36:
        raise AssertionError('Unexpected PATH-0004 task count')
    return rows
