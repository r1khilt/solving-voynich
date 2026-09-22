"""Independent integrity and label audit for the completed PATH-0001 run."""

from collections import defaultdict
import hashlib
import json
from pathlib import Path
import unicodedata


ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / 'results/PATH-0001'
CONDITIONS = ('identity', 'head_one', 'head_four', 'value_token_four',
              'qk_four', 'value_all_four', 'random_four', 'all_heads')


def load(name):
    return json.loads((RESULT/name).read_text())


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def normalize(text):
    value = text.strip().strip(' .!"\'`').casefold()
    return ''.join(c for c in unicodedata.normalize('NFD', value) if unicodedata.category(c) != 'Mn')


def main():
    inputs = load('inputs.json')
    for filename, expected in inputs['source_sha256'].items():
        if sha256(ROOT/filename) != expected:
            raise ValueError(f'Source changed: {filename}')
    if sha256(ROOT/'results/JSPACE-0001/inputs.json') != inputs['jspace_inputs_sha256']:
        raise ValueError('Model provenance changed')
    if sha256(ROOT/'outputs/PATH-0001/rendered-inputs.json') != inputs['rendered_inputs_sha256']:
        raise ValueError('Rendered inputs changed')
    scan, selection = load('discovery-head-scan.json'), load('selection.json')
    grouped = defaultdict(list)
    for row in scan:
        grouped[(row['layer'], row['head'])].append(row['margin_gain'])
    eligible_discovery = load('discovery-capability.json')['eligible_count']
    if len(scan) != eligible_discovery*4*32 or any(len(group) != eligible_discovery for group in grouped.values()):
        raise ValueError('Discovery scan incomplete')
    rank = sorted(grouped, key=lambda key: (-sum(grouped[key])/len(grouped[key]), key[0], key[1]))
    layer = rank[0][0]
    heads = [head for candidate_layer, head in rank if candidate_layer == layer][:4]
    if layer != selection['selected_layer'] or heads != selection['selected_heads']:
        raise ValueError('Head selection did not replay')

    rows = load('confirmation-rows.json')
    baselines = {r['id']: r for r in load('confirmation-baselines.json')}
    keys = {(r['task_id'], r['condition']) for r in rows}
    if len(rows) != 24*len(CONDITIONS) or len(keys) != len(rows):
        raise ValueError('Confirmation grid incomplete')
    if any(condition not in CONDITIONS for _, condition in keys):
        raise ValueError('Unknown condition')
    if any(r['eligible'] != baselines[r['task_id']]['eligible'] for r in rows):
        raise ValueError('Eligibility changed')
    tasks = {}
    for split in ('discovery', 'confirmation'):
        from voynich.workspace.path_tasks import path_tasks

        tasks.update({r['id']: r for r in path_tasks(split)})
    for row in rows:
        task = tasks[row['task_id']]
        if row['family'] == 'binding':
            expected = normalize(row['text']) == normalize(task['donor_answer'])
            if expected != row['desired_answer']:
                raise ValueError('Semantic output label differs from independent normalizer')
        else:
            expected = normalize(row['text']) == normalize(task['answer'])
            if expected != row['copy_preserved']:
                raise ValueError('Copy output label differs from independent normalizer')
        if row['condition'] == 'identity' and row['text'] != baselines[row['task_id']]['source_text']:
            raise ValueError('Identity response changed')

    summary, decision = load('summary.json'), load('decision.json')
    eligible = {key for key, item in baselines.items() if item['eligible']}
    for condition in CONDITIONS:
        semantic = [r for r in rows if r['condition'] == condition and r['task_id'] in eligible]
        copy = [r for r in rows if r['condition'] == condition and r['family'] == 'copy']
        stated = summary[condition]
        if (stated['semantic']['count'] != len(semantic)
                or stated['semantic']['switches'] != sum(r['desired_answer'] for r in semantic)
                or stated['copy']['count'] != len(copy)
                or stated['copy']['preserved'] != sum(r['copy_preserved'] for r in copy)):
            raise ValueError('Summary did not replay')
    if decision['confirmation_eligible'] != len(eligible):
        raise ValueError('Decision denominator did not replay')
    rates = {name: summary[name]['semantic']['switches']/len(eligible) for name in CONDITIONS}
    if rates != decision['rates']:
        raise ValueError('Decision rates did not replay')
    positive = rates['all_heads'] >= .5
    copy_rate = summary['value_token_four']['copy']['preserved']/summary['value_token_four']['copy']['count']
    supported = (positive and rates['value_token_four'] >= .5
                 and rates['value_token_four']-rates['random_four'] >= .2
                 and rates['value_token_four']-rates['qk_four'] >= .2 and copy_rate >= .95)
    expected_status = 'exploratory_supported' if supported else ('uninformative' if not positive else 'exploratory_failed')
    if decision['status'] != expected_status or decision['positive_control_passed'] != positive:
        raise ValueError('Decision verdict did not replay')
    by_id = {(r['task_id'], r['condition']): r for r in rows}
    for task_id in baselines:
        if abs(by_id[(task_id, 'random_four')]['edit_norm']-by_id[(task_id, 'head_four')]['edit_norm']) > 1e-3:
            raise ValueError('Random four-head control was not norm matched')
    report = {
        'passed': True, 'discovery_scan_rows': len(scan), 'confirmation_rows': len(rows),
        'eligible_confirmation': len(eligible), 'selected_layer': layer, 'selected_heads': heads,
        'per_condition': {condition: summary[condition]['semantic'] for condition in CONDITIONS},
        'dependent_bundle_count': len({baselines[key]['bundle'] for key in eligible}),
        'verdict': expected_status,
        'bundle_switches': {condition: summary[condition]['bundles'] for condition in CONDITIONS},
    }
    (RESULT/'independent-audit.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
