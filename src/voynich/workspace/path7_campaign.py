"""Fresh, bounded two-hop prompt-format competence qualification."""

from pathlib import Path
import subprocess
import time

import numpy as np

from .campaign import answer_correct, canonical_digest, digest, load_config, write_json
from .path7_tasks import path7_tasks
from .route_backend import RouteWorkspace

STYLES = ('colon', 'arrow')
SEED = 51091
CAP_SECONDS = 1200
MEMORY_CAP = 45_000_000_000


def counts(rows, style):
    selected = [r for r in rows if r['style'] == style]
    return {
        'eligible': sum(r['family'] == 'composed' and r['source_correct'] and r['donor_correct']
                        and r['first_token_differs'] for r in selected),
        'source_copy': sum(r['family'] == 'copy' and r['source_correct'] for r in selected),
        'source_composed_correct': sum(r['family'] == 'composed' and r['source_correct'] for r in selected),
        'donor_composed_correct': sum(r['family'] == 'composed' and r['donor_correct'] for r in selected),
    }


def choose(summary):
    return sorted(STYLES, key=lambda style: (-summary[style]['eligible'],
                                             -summary[style]['source_copy'], STYLES.index(style)))[0]


def run():
    output, result = Path('outputs/PATH-0007'), Path('results/PATH-0007')
    output.mkdir(parents=True, exist_ok=True)
    result.mkdir(parents=True, exist_ok=True)
    if (result/'inputs.json').exists():
        raise FileExistsError('No automatic repeat of PATH-0007')
    started = time.monotonic()
    config = load_config('configs/jspace0001.json')
    model = RouteWorkspace(config['model_snapshot'], dense_transport=True)
    model.mx.set_memory_limit(min(config['memory_bytes_cap'], MEMORY_CAP))

    def budget(completed):
        elapsed = time.monotonic()-started
        if elapsed >= CAP_SECONDS or model.mx.get_peak_memory() > MEMORY_CAP:
            write_json(result/'incomplete.json', {'elapsed_seconds': elapsed,
                                                   'completed_prompts': completed,
                                                   'reason': 'resource_cap'})
            raise RuntimeError('PATH-0007 resource cap')

    tasks = {split: path7_tasks(split) for split in ('discovery', 'confirmation')}
    prompts = {row[key] for group in tasks.values() for row in group for key in ('prompt', 'donor_prompt')}
    ids = {prompt: model.encode(prompt, chat=True) for prompt in prompts}
    write_json(output/'rendered-inputs.json', {canonical_digest(prompt): token_ids for prompt, token_ids in ids.items()})
    sources = [Path('src/voynich/workspace')/name for name in
               ('path7_tasks.py', 'path7_campaign.py', 'route_backend.py')]
    sources.append(Path('docs/experiments/PATH-0007.md'))
    write_json(result/'inputs.json', {
        'source_revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'source_sha256': {str(path): digest(path) for path in sources},
        'tasks_sha256': canonical_digest(tasks), 'rendered_inputs_sha256': digest(output/'rendered-inputs.json'),
        'model_input_sha256': digest('results/JSPACE-0001/inputs.json'),
        'seed': SEED, 'styles': STYLES, 'cap_seconds': CAP_SECONDS, 'memory_bytes_cap': MEMORY_CAP,
        'max_generated_tokens': 16,
    })

    baselines = {}
    baseline_dir = output/'baselines'
    baseline_dir.mkdir(exist_ok=True)

    def baseline(prompt):
        if prompt not in baselines:
            budget(len(baselines))
            phrase, tokens, _, logits = model.generate_field(ids[prompt], max_tokens=16)
            local = baseline_dir/(canonical_digest(prompt)+'.npz')
            np.savez_compressed(local, first_logits=logits, generated_tokens=np.asarray(tokens, dtype=np.int32))
            baselines[prompt] = {'text': phrase, 'tokens': tokens, 'arrays_sha256': digest(local)}
        return baselines[prompt]

    summaries = {}
    selected = None
    for split in ('discovery', 'confirmation'):
        rows = []
        for task in tasks[split]:
            source, donor = baseline(task['prompt']), baseline(task['donor_prompt'])
            source_correct = answer_correct(source['text'], [task['answer']])
            donor_correct = answer_correct(donor['text'], [task['donor_answer']])
            rows.append({
                'id': task['id'], 'split': split, 'bundle': task['bundle'],
                'style': task['style'], 'family': task['family'],
                'source_text': source['text'], 'donor_text': donor['text'],
                'source_correct': source_correct, 'donor_correct': donor_correct,
                'source_class': ('correct' if source_correct else
                                 ('alternate_object' if task['family'] == 'composed'
                                  and answer_correct(source['text'], [task['donor_answer']]) else 'other')),
                'donor_class': ('correct' if donor_correct else
                                ('alternate_object' if task['family'] == 'composed'
                                 and answer_correct(donor['text'], [task['answer']]) else 'other')),
                'first_token_differs': bool(source['tokens']) and bool(donor['tokens'])
                and source['tokens'][0] != donor['tokens'][0],
                'source_arrays_sha256': source['arrays_sha256'], 'donor_arrays_sha256': donor['arrays_sha256'],
            })
        write_json(result/f'{split}-baselines.json', rows)
        summaries[split] = {style: counts(rows, style) for style in STYLES}
        write_json(result/f'{split}-summary.json', summaries[split])
        if split == 'discovery':
            selected = choose(summaries[split])
            write_json(result/'selection.json', {'style': selected, 'rule': 'eligible_copy_colon', 'seed': SEED})

    qualified = all(summaries[split][selected]['eligible'] >= 18
                    and summaries[split][selected]['source_copy'] >= 11
                    for split in ('discovery', 'confirmation'))
    write_json(result/'decision.json', {
        'status': 'qualified' if qualified else 'unqualified', 'qualified': qualified,
        'selected_style': selected, 'summary': summaries,
        'elapsed_seconds': time.monotonic()-started, 'peak_mlx_bytes': model.mx.get_peak_memory(),
        'scope': 'Prompt competence on supplied English tables; no mechanism or Voynich reading',
    })


if __name__ == '__main__':
    run()
