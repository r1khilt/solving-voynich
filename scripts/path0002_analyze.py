"""Independent output, source, and decision audit for PATH-0002."""

import hashlib
import json
from pathlib import Path
import unicodedata

import numpy as np

from voynich.workspace.path2_tasks import path2_tasks

ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT/'results/PATH-0002'
OUTPUT = ROOT/'outputs/PATH-0002'
LAYERS = (15, 19, 23, 27, 31)
CONDITIONS = ('identity', 'last_donor', 'value_donor', 'other_earlier_donor',
              'earlier_donor', 'all_donor', 'value_random')


def load(filename):
    return json.loads((RESULT/filename).read_text())


def sha256(filename):
    digest = hashlib.sha256()
    with filename.open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def normalize(text):
    text = text.strip().strip(' .!"\'`').casefold()
    return ''.join(c for c in unicodedata.normalize('NFD', text) if unicodedata.category(c) != 'Mn')


def bootstrap(rows, eligible, first, second):
    by_key = {(r['task_id'], r['condition']): r for r in rows if r['layer'] == 23 and r['task_id'] in eligible}
    bundles = {index: [ident for ident in sorted(eligible) if int(ident.split('/')[1]) == index]
               for index in range(12)}
    rng = np.random.default_rng(51052)
    values = []
    for _ in range(10000):
        chosen = rng.integers(0, 12, size=12)
        differences = [int(by_key[(ident, first)]['desired_answer'])-int(by_key[(ident, second)]['desired_answer'])
                       for bundle in chosen for ident in bundles[int(bundle)]]
        if differences:
            values.append(float(np.mean(differences)))
    return len(values), float(np.quantile(values, .025)), float(np.quantile(values, .975))


def main():
    manifest = load('inputs.json')
    for relative, expected in manifest['source_sha256'].items():
        if sha256(ROOT/relative) != expected:
            raise ValueError(f'Scientific source changed: {relative}')
    if sha256(ROOT/'configs/jspace0001.json') != manifest['config_sha256']:
        raise ValueError('Configuration changed')
    if sha256(ROOT/'results/JSPACE-0001/inputs.json') != manifest['jspace_inputs_sha256']:
        raise ValueError('Model provenance changed')
    if sha256(OUTPUT/'rendered-inputs.json') != manifest['rendered_inputs_sha256']:
        raise ValueError('Rendered inputs changed')
    task_list = path2_tasks()
    if canonical_hash(task_list) != manifest['tasks_sha256']:
        raise ValueError('Task list changed')
    rendered = json.loads((OUTPUT/'rendered-inputs.json').read_text())
    changed = {}
    for task in task_list:
        source = rendered[canonical_hash(task['prompt'])]
        donor = rendered[canonical_hash(task['donor_prompt'])]
        if len(source) != len(donor):
            raise ValueError('Paired token lengths differ')
        positions = [index for index, (a, b) in enumerate(zip(source, donor)) if a != b]
        if len(positions) < 2 or positions[-1] >= len(source)-8:
            raise ValueError('Changed token positions overlap query suffix')
        changed[task['id']] = positions
    if canonical_hash(changed) != manifest['changed_positions_sha256']:
        raise ValueError('Value-token position classification changed')
    tasks = {r['id']: r for r in task_list}
    base = {r['id']: r for r in load('baselines.json')}
    if len(base) != 72:
        raise ValueError('Missing baseline')
    checked_arrays = set()
    for ident, item in base.items():
        task = tasks[ident]
        if (item['source_correct'] != (normalize(item['source_text']) == normalize(task['answer']))
                or item['donor_correct'] != (normalize(item['donor_text']) == normalize(task['donor_answer']))):
            raise ValueError('Baseline label mismatch')
        for side in ('source', 'donor'):
            prompt = task['prompt'] if side == 'source' else task['donor_prompt']
            filename = OUTPUT/'baselines'/(canonical_hash(prompt)+'.npz')
            if filename not in checked_arrays:
                if sha256(filename) != item[f'{side}_arrays_sha256']:
                    raise ValueError('Baseline array checksum mismatch')
                checked_arrays.add(filename)
    eligible = {ident for ident, r in base.items() if r['family'] == 'binding' and r['source_correct']
                and r['donor_correct'] and r['first_token_differs']}
    baseline_copy = {ident for ident, r in base.items() if r['family'] == 'copy' and r['source_correct']}
    rows = load('rows.json')
    keys = {(r['task_id'], r['layer'], r['condition']) for r in rows}
    if len(rows) != 72*5*7 or len(keys) != len(rows):
        raise ValueError('Intervention grid incomplete')
    if keys != {(ident, layer, condition) for ident in tasks for layer in LAYERS for condition in CONDITIONS}:
        raise ValueError('Wrong intervention key set')
    by_key = {(r['task_id'], r['layer'], r['condition']): r for r in rows}
    for r in rows:
        ident = r['task_id']
        task = tasks[ident]
        if r['eligible'] != (ident in eligible):
            raise ValueError('Eligibility drift')
        if r['family'] == 'binding':
            if r['desired_answer'] != (normalize(r['text']) == normalize(task['donor_answer'])):
                raise ValueError('Semantic label mismatch')
        elif r['copy_preserved'] != (normalize(r['text']) == normalize(task['answer'])):
            raise ValueError('Copy label mismatch')
        if r['condition'] == 'identity' and r['text'] != base[ident]['source_text']:
            raise ValueError('Identity response mismatch')
    for ident in tasks:
        for layer in LAYERS:
            value = by_key[(ident, layer, 'value_donor')]['field_norm']
            random = by_key[(ident, layer, 'value_random')]['field_norm']
            if abs(value-random) > .002:
                raise ValueError('Random field norm did not match')

    summary = load('summary.json')
    decision = load('decision.json')
    for layer in LAYERS:
        for condition in CONDITIONS:
            members = [by_key[(ident, layer, condition)] for ident in tasks]
            semantic = [r for r in members if r['task_id'] in eligible]
            copy = [r for r in members if r['task_id'] in baseline_copy]
            record = summary[str(layer)][condition]
            if (record['semantic'] != {'count': len(semantic), 'switches': sum(r['desired_answer'] for r in semantic)}
                    or record['copy'] != {'count': len(copy), 'preserved': sum(r['copy_preserved'] for r in copy)}):
                raise ValueError('Summary mismatch')
            for template in range(3):
                group = [r for r in semantic if r['template'] == template]
                if record['by_template'][str(template)] != {'count': len(group), 'switches': sum(r['desired_answer'] for r in group)}:
                    raise ValueError('Template summary mismatch')
            for bundle in range(12):
                group = [r for r in semantic if r['bundle'] == bundle]
                if record['by_bundle'][str(bundle)] != {'count': len(group), 'switches': sum(r['desired_answer'] for r in group)}:
                    raise ValueError('Bundle summary mismatch')
    primary = summary['23']
    rates = {c: primary[c]['semantic']['switches']/len(eligible) for c in CONDITIONS}
    copy_rates = {c: primary[c]['copy']['preserved']/len(baseline_copy) for c in CONDITIONS}
    adequate = rates['all_donor'] >= .9 and copy_rates['all_donor'] >= .95
    value = (adequate and rates['value_donor'] >= .5
             and rates['value_donor']-rates['value_random'] >= .2 and copy_rates['value_donor'] >= .95)
    bypass = (adequate and rates['earlier_donor'] >= .5
              and rates['earlier_donor']-rates['last_donor'] >= .25 and copy_rates['earlier_donor'] >= .95)
    if (decision['rates'] != rates or decision['copy_rates'] != copy_rates
            or decision['value_route_supported'] != value or decision['earlier_bypass_supported'] != bypass
            or decision['status'] != ('exploratory_scored' if adequate else 'uninformative_positive_control')):
        raise ValueError('Decision mismatch')
    for first, second, field in (('value_donor', 'value_random', 'value_minus_random_bundle_bootstrap'),
                                 ('earlier_donor', 'last_donor', 'earlier_minus_last_bundle_bootstrap')):
        n, low, high = bootstrap(rows, eligible, first, second)
        stored = decision[field]
        if n != stored['resamples'] or abs(low-stored['lower_95']) > 1e-12 or abs(high-stored['upper_95']) > 1e-12:
            raise ValueError('Bundle bootstrap mismatch')
    if decision['all_donor_max_first_logit_error'] >= .002 or not decision['all_donor_argmax_all']:
        raise ValueError('Runtime numerical control reports failure')
    report = {
        'passed': True, 'rows': len(rows), 'baseline_arrays_checked': len(checked_arrays),
        'eligible_semantic': len(eligible), 'baseline_copy': len(baseline_copy),
        'eligible_bundles': len({tasks[ident]['bundle'] for ident in eligible}),
        'primary_rates': rates, 'copy_rates': copy_rates,
        'value_route_supported': value, 'earlier_bypass_supported': bypass,
    }
    (RESULT/'independent-audit.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
