"""Validate and archive the completed EXP-0009 fixed causal-map experiment.

Run after every concurrent campaign job has stopped. Raw prefixes and NPZ files
remain in ignored outputs; this archive contains compact summaries and hashes.
"""

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np

from voynich.causal_mapping import MappingSite, confirmation_decision, select_site, summarize_scores
from voynich.runtime import digest, write_json


def archive(source, destination):
    source, destination = Path(source), Path(destination)
    if destination.exists() and any(destination.iterdir()):
        raise ValueError('Archive destination must be new or empty')
    if (source/'failure.json').exists():
        raise ValueError('Failure record exists; inspect rather than archive as success')
    manifest = json.loads((source/'manifest.json').read_text())
    completion = json.loads((source/'completion.json').read_text())
    if manifest['environment']['git_dirty'] or manifest['test_evaluated'] or completion['test_evaluated']:
        raise ValueError('Source or holdout integrity failure')
    revision = manifest['environment']['git_commit']
    for path, expected in manifest['environment']['source_sha256'].items():
        committed = subprocess.check_output(['git', 'show', f'{revision}:{path}'])
        if hashlib.sha256(committed).hexdigest() != expected:
            raise ValueError(f'Registered source digest does not match committed bytes: {path}')
    if completion['jobs_completed'] != 7 or len(completion['results']) != 7:
        raise ValueError('Incomplete seven-model campaign')
    for key, file_hashes in manifest['corpus_identities'].items():
        for name, expected in file_hashes.items():
            if digest(Path('data/processed')/key/name) != expected:
                raise ValueError(f'Corpus drift: {key}/{name}')
    for path, expected in manifest['overlap_audit_inputs_sha256'].items():
        if digest(path) != expected:
            raise ValueError(f'Overlap-audit corpus drift: {path}')
    datasets = {}
    raw = {}
    for kind, splits in manifest['data'].items():
        datasets[kind] = {}
        for split, spec in splits.items():
            if digest(spec['path']) != spec['sha256']:
                raise ValueError('Prepared data digest mismatch')
            datasets[kind][split] = json.loads(Path(spec['path']).read_text())
            if len(datasets[kind][split]) != spec['pairs']:
                raise ValueError('Prepared data count mismatch')
            raw[spec['path']] = {'sha256': spec['sha256'], 'bytes': Path(spec['path']).stat().st_size,
                                'content': 'Raw inputs and scoring metadata; excluded from archive'}
        if kind == 'synthetic':
            prefixes = [{row[field] for row in rows for field in ('recipient', 'donor', 'wrong')}
                        for rows in datasets[kind].values()]
            if prefixes[0] & prefixes[1]:
                raise ValueError('Synthetic split prefix overlap')
        else:
            leaves = [{row['group'] for row in rows} for rows in datasets[kind].values()]
            if leaves[0] & leaves[1]:
                raise ValueError('Manuscript physical leaf overlap')
    copied = [source/'manifest.json', source/'completion.json']
    controls = []
    compact = []
    for summary in completion['results']:
        path = Path(summary['report_path'])
        if digest(path) != summary['report_sha256']:
            raise ValueError('Completed report digest mismatch')
        report = json.loads(path.read_text())
        if digest(report['checkpoint']) != report['checkpoint_sha256']:
            raise ValueError('Frozen checkpoint digest mismatch')
        if not report['frozen_backbone_verified'] or report['test_evaluated']:
            raise ValueError('Frozen backbone/holdout control failure')
        label = f"{report['kind']}-seed{report['seed']}"
        selection_path = source/f'{label}-selection.json'
        selection = json.loads(selection_path.read_text())
        if selection['checkpoint_sha256'] != report['checkpoint_sha256'] or selection['discovery'] != report['discovery']:
            raise ValueError('Persisted discovery/selection mismatch')
        sites = []
        for name in report['discovery']['conditions']:
            if not name.endswith('/donor'):
                continue
            layer, component, region = name.removesuffix('/donor').split('-')
            kind = 'head' if component.startswith('head') else component
            sites.append(MappingSite(int(layer[1:]), kind, region, int(component[4:]) if kind == 'head' else None))
        if len(sites) != 48:
            raise ValueError('Incomplete 48-site discovery map')
        selected = select_site(report['discovery'], sites)
        if asdict(selected) != selection['selected'] or asdict(selected) != report['decision']['selected_site']:
            raise ValueError('Selection was not generated by the frozen discovery rule')
        if selection_path.stat().st_mtime_ns > path.stat().st_mtime_ns:
            raise ValueError('Selection artifact was saved after confirmation report')
        kind = 'manuscript' if report['kind'] == 'manuscript' else 'synthetic'
        arrays_by_split = {}
        for split in ('discovery', 'confirmation'):
            controls.extend(report[split]['controls'].values())
            if max(report[split]['controls'].values()) > 1e-5:
                raise ValueError('Numerical intervention control failed')
            if report[split]['horizons'] != [1, 2, 4, 8]:
                raise ValueError('Unexpected scoring horizons')
            array_path = source/f'{label}-{split}-per-example.npz'
            with np.load(array_path) as stored:
                arrays = {name: stored[name] for name in stored.files}
            arrays_by_split[split] = arrays
            for condition, array in arrays.items():
                computed = summarize_scores(array, datasets[kind][split])
                expected = report[split]['conditions'][condition]
                if computed != expected:
                    raise ValueError(f'Per-example aggregation mismatch: {label}/{split}/{condition}')
            if len([name for name in arrays if name.endswith('/donor')]) != 48:
                raise ValueError('Incomplete site map')
            np.testing.assert_allclose(arrays['recipient'][:, 1:], arrays['late_readout'][:, 1:], atol=1e-5, rtol=0)
            raw[str(array_path)] = {'sha256': digest(array_path), 'bytes': array_path.stat().st_size,
                                    'content': 'Per-example 48-site scores and controls; excluded from archive'}
        decision = confirmation_decision(report['confirmation'], selected, datasets[kind]['confirmation'], arrays_by_split['confirmation'])
        if decision != report['decision']:
            raise ValueError('Registered threshold decision mismatch')
        copied.extend([path, selection_path])
        compact.append({'kind': report['kind'], 'seed': report['seed'], 'selected': selected.name,
                        **decision, 'elapsed_seconds': report['elapsed_seconds']})
    expected_synthetic = all(item['meets_registered_practical_threshold'] for item in compact if item['kind'] == 'synthetic')
    expected_manuscript = all(item['meets_registered_practical_threshold'] for item in compact if item['kind'] == 'manuscript')
    if completion['synthetic_all_trained_seeds_pass'] != expected_synthetic or completion['manuscript_all_seeds_pass'] != expected_manuscript:
        raise ValueError('Across-seed decision mismatch')
    destination.mkdir(parents=True, exist_ok=True)
    for path in copied:
        shutil.copyfile(path, destination/path.name)
    archive_report = {'experiment': 'EXP-0009', 'source_commit': manifest['environment']['git_commit'],
                      'copied_files': {path.name: digest(path) for path in copied},
                      'raw_artifact_manifest': raw, 'models': compact,
                      'maximum_numerical_control_error': max(controls),
                      'scope': 'No raw text, tensor arrays or checkpoints copied. All manuscript scores are validation only.'}
    write_json(destination/'archive_audit.json', archive_report)
    return archive_report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', default='outputs/EXP-0009')
    parser.add_argument('--destination', default='results/EXP-0009')
    args = parser.parse_args()
    audit = archive(args.source, args.destination)
    print(json.dumps({'source_commit': audit['source_commit'], 'models': len(audit['models']),
                      'maximum_numerical_control_error': audit['maximum_numerical_control_error']}))
