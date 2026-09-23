"""Independent compact-result audit and format diagnosis for PATH-0007."""

import json
from pathlib import Path

import numpy as np

from voynich.workspace.campaign import answer_correct, canonical_digest, digest, write_json
from voynich.workspace.path7_tasks import path7_tasks

ROOT = Path('results/PATH-0007')
RAW = Path('outputs/PATH-0007')
STYLES = ('colon', 'arrow')


def read(path):
    return json.loads(path.read_text())


def score(rows, style):
    chosen = [r for r in rows if r['style'] == style]
    return {
        'eligible': sum(r['family'] == 'composed' and r['source_correct']
                        and r['donor_correct'] and r['first_token_differs'] for r in chosen),
        'source_copy': sum(r['family'] == 'copy' and r['source_correct'] for r in chosen),
        'source_composed_correct': sum(r['family'] == 'composed' and r['source_correct'] for r in chosen),
        'donor_composed_correct': sum(r['family'] == 'composed' and r['donor_correct'] for r in chosen),
    }


def audit():
    inputs = read(ROOT/'inputs.json')
    tasks = {split: path7_tasks(split) for split in ('discovery', 'confirmation')}
    assert canonical_digest(tasks) == inputs['tasks_sha256']
    assert digest(RAW/'rendered-inputs.json') == inputs['rendered_inputs_sha256']
    assert digest('results/JSPACE-0001/inputs.json') == inputs['model_input_sha256']
    for name, expected in inputs['source_sha256'].items():
        assert digest(name) == expected, name
    assert inputs['seed'] == 51091 and inputs['max_generated_tokens'] == 16
    rendered = read(RAW/'rendered-inputs.json')
    summaries, details = {}, {}
    for split in tasks:
        rows = read(ROOT/f'{split}-baselines.json')
        lookup = {task['id']: task for task in tasks[split]}
        assert [row['id'] for row in rows] == [task['id'] for task in tasks[split]]
        for row in rows:
            task = lookup[row['id']]
            assert row['split'] == split and row['bundle'] == task['bundle']
            assert row['style'] == task['style'] and row['family'] == task['family']
            assert row['source_correct'] == answer_correct(row['source_text'], [task['answer']])
            assert row['donor_correct'] == answer_correct(row['donor_text'], [task['donor_answer']])
            for prefix, prompt in (('source', task['prompt']), ('donor', task['donor_prompt'])):
                assert canonical_digest(prompt) in rendered
                local = RAW/'baselines'/(canonical_digest(prompt)+'.npz')
                assert digest(local) == row[f'{prefix}_arrays_sha256']
                with np.load(local) as arrays:
                    assert np.isfinite(arrays['first_logits']).all()
                    assert arrays['generated_tokens'].ndim == 1
            first = read_tokens(task['prompt'])
            second = read_tokens(task['donor_prompt'])
            assert row['first_token_differs'] == bool(first and second and first[0] != second[0])
            for prefix, expected, alternate in (
                ('source', task['answer'], task['donor_answer']),
                ('donor', task['donor_answer'], task['answer']),
            ):
                phrase = row[f'{prefix}_text']
                label = ('correct' if answer_correct(phrase, [expected]) else
                         ('alternate_object' if task['family'] == 'composed'
                          and answer_correct(phrase, [alternate]) else 'other'))
                assert row[f'{prefix}_class'] == label
        summaries[split] = {style: score(rows, style) for style in STYLES}
        assert read(ROOT/f'{split}-summary.json') == summaries[split]
        details[split] = {
            style: {
                'composed_wrong_alternate': sum(row['family'] == 'composed' and row['style'] == style
                                                and row[prefix+'_class'] == 'alternate_object'
                                                for row in rows for prefix in ('source', 'donor')),
                'composed_wrong_other': sum(row['family'] == 'composed' and row['style'] == style
                                            and row[prefix+'_class'] == 'other'
                                            for row in rows for prefix in ('source', 'donor')),
            } for style in STYLES
        }
    selected = sorted(STYLES, key=lambda style: (-summaries['discovery'][style]['eligible'],
                                                 -summaries['discovery'][style]['source_copy'],
                                                 STYLES.index(style)))[0]
    assert read(ROOT/'selection.json')['style'] == selected
    qualified = all(summaries[split][selected]['eligible'] >= 18
                    and summaries[split][selected]['source_copy'] >= 11
                    for split in ('discovery', 'confirmation'))
    decision = read(ROOT/'decision.json')
    assert decision['qualified'] == qualified and decision['selected_style'] == selected
    assert decision['status'] == ('qualified' if qualified else 'unqualified')
    assert decision['summary'] == summaries
    report = {'passed': True, 'status': decision['status'], 'selected_style': selected,
              'summary': summaries, 'wrong_answer_classes': details,
              'limitation': 'Prompt style is crossed with bundles, but fixed model, scorer, and answer prefix restrict claim to these two styles.'}
    write_json(ROOT/'audit.json', report)
    return report


def read_tokens(prompt):
    local = RAW/'baselines'/(canonical_digest(prompt)+'.npz')
    with np.load(local) as arrays:
        return arrays['generated_tokens'].tolist()


if __name__ == '__main__':
    print(json.dumps(audit(), indent=2))
