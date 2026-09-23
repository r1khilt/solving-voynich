"""CPU-only retrospective diagnosis of PATH-0004 clean two-hop baselines."""

from collections import Counter
from pathlib import Path
import json

from voynich.workspace.campaign import answer_correct, digest, write_json
from voynich.workspace.path4_tasks import path4_tasks


ROOT = Path('results/PATH-0004')
OUT = Path('results/PATH-0005-qualification')


def answer_class(text, expected, alternate):
    if answer_correct(text, [expected]):
        return 'expected'
    if answer_correct(text, [alternate]):
        return 'alternate'
    return 'other'


def run():
    rows = []
    input_hashes = {}
    for split in ('discovery', 'confirmation'):
        path = ROOT / f'{split}-baselines.json'
        input_hashes[str(path)] = digest(path)
        observed = json.loads(path.read_text())
        tasks = path4_tasks(split)
        if [r['id'] for r in observed] != [t['id'] for t in tasks]:
            raise ValueError(f'PATH-0004 task IDs/order changed: {split}')
        for baseline, task in zip(observed, tasks, strict=True):
            if baseline['family'] != task['family'] or baseline['bundle'] != task['bundle']:
                raise ValueError(f'PATH-0004 task metadata changed: {task["id"]}')
            if task['family'] != 'composed':
                continue
            source = answer_class(baseline['source_text'], task['answer'], task['donor_answer'])
            donor = answer_class(baseline['donor_text'], task['donor_answer'], task['answer'])
            if (source == 'expected') != baseline['source_correct']:
                raise ValueError(f'Source baseline label mismatch: {task["id"]}')
            if (donor == 'expected') != baseline['donor_correct']:
                raise ValueError(f'Donor baseline label mismatch: {task["id"]}')
            rows.append({
                'id': task['id'], 'split': split, 'bundle': task['bundle'],
                'template': task['template'], 'orientation': int(task['id'].split('/')[-2]),
                'query_slot': int(task['id'].split('/')[-1]),
                'source_expected': task['answer'], 'donor_expected': task['donor_answer'],
                'source_text': baseline['source_text'], 'donor_text': baseline['donor_text'],
                'source_class': source, 'donor_class': donor,
                'eligible': source == donor == 'expected' and baseline['first_token_differs'],
                'first_token_differs': baseline['first_token_differs'],
            })

    summary = {}
    for split in ('discovery', 'confirmation'):
        part = [r for r in rows if r['split'] == split]
        incorrect = [r[side+'_class'] for r in part for side in ('source', 'donor')
                     if r[side+'_class'] != 'expected']
        by_bundle = {}
        for bundle in range(6):
            group = [r for r in part if r['bundle'] == bundle]
            by_bundle[str(bundle)] = {
                'template': group[0]['template'],
                'eligible': sum(r['eligible'] for r in group),
                'source_correct': sum(r['source_class'] == 'expected' for r in group),
                'donor_correct': sum(r['donor_class'] == 'expected' for r in group),
                'both_wrong': sum(r['source_class'] != 'expected' and
                                  r['donor_class'] != 'expected' for r in group),
            }
        by_template = {}
        for template in (0, 1):
            group = [r for r in part if r['template'] == template]
            by_template[str(template)] = {
                'eligible': sum(r['eligible'] for r in group),
                'source_correct': sum(r['source_class'] == 'expected' for r in group),
                'donor_correct': sum(r['donor_class'] == 'expected' for r in group),
                'total': len(group),
            }
        by_slot = {}
        for slot in (0, 1):
            group = [r for r in part if r['query_slot'] == slot]
            by_slot[str(slot)] = {'eligible': sum(r['eligible'] for r in group), 'total': len(group)}
        summary[split] = {
            'pairs': len(part), 'eligible': sum(r['eligible'] for r in part),
            'both_correct': sum(r['source_class'] == r['donor_class'] == 'expected' for r in part),
            'both_correct_first_token_same': sum(r['source_class'] == r['donor_class'] == 'expected'
                                                 and not r['first_token_differs'] for r in part),
            'source_class': dict(Counter(r['source_class'] for r in part)),
            'donor_class': dict(Counter(r['donor_class'] for r in part)),
            'incorrect_total': len(incorrect),
            'incorrect_alternate': incorrect.count('alternate'),
            'incorrect_other': incorrect.count('other'),
            'by_bundle': by_bundle, 'by_template': by_template, 'by_query_slot': by_slot,
        }
    confirmation = summary['confirmation']
    confirmation['diagnostic_label'] = (
        'surface_competence_failure_concentrated'
        if confirmation['incorrect_total'] and
        confirmation['incorrect_alternate']/confirmation['incorrect_total'] >= .75
        else 'mixed_failure'
    )
    payload = {
        'status': 'retrospective_exploratory_only', 'model_inference': False,
        'input_sha256': input_hashes, 'task_source_sha256': digest('src/voynich/workspace/path4_tasks.py'),
        'registration_sha256': digest('docs/experiments/PATH-0005-qualification.md'),
        'summary': summary, 'rows': rows,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    write_json(OUT/'diagnosis.json', payload)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    run()
