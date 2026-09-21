"""Archive bounded campaign provenance and sampled resource use, without weights."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', default='outputs/CAMPAIGN-0001')
    parser.add_argument('--destination', default='results/CAMPAIGN-0001')
    parser.add_argument('--capture-completion', action='store_true',
                        help='Capture source integrity before making any tracked post-run edits')
    args = parser.parse_args()
    source, destination = Path(args.source), Path(args.destination)
    if args.capture_completion:
        spec = importlib.util.spec_from_file_location('campaign_runner', Path('scripts/run_parallel_campaign.py'))
        runner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(runner)
        status = json.loads((source/'status.json').read_text())
        manifest = json.loads((source/'manifest.json').read_text())
        if status['state'] == 'running' or any(r['state'] == 'running' for r in status['jobs'].values()):
            raise ValueError('Wait until all campaign jobs stop')
        current = runner.source_state(Path.cwd())
        if current != manifest['source']:
            raise ValueError('Source changed or worktree is dirty')
        samples = [json.loads(line) for line in (source/'resources.jsonl').read_text().splitlines()]
        if any(s['memory_sampling_error'] for s in samples):
            raise ValueError('Resource sampling failed')
        integrity = {'campaign': manifest['campaign'], 'source_unchanged_and_clean': True,
                     'git_commit': current['git_commit'], 'status': status, 'sample_count': len(samples),
                     'nominal_sampling_seconds': 5,
                     'max_summed_process_rss_bytes': max(s['sum_process_group_rss_bytes'] for s in samples),
                     'memory_accounting': manifest['memory_accounting'],
                     'input_sha256': {p.name: digest(p) for p in [source/'manifest.json', source/'status.json',
                                      source/'resources.jsonl', *sorted(source.glob('*.log'))]},
                     'manuscript_final_test_scored': False, 'paid_research_api_calls': 0}
        (source/'completion_integrity.json').write_text(json.dumps(integrity, indent=2)+'\n')
        print(json.dumps(integrity, indent=2))
        return
    if destination.exists():
        raise ValueError('Use a new archive destination')
    manifest = json.loads((source/'manifest.json').read_text())
    status = json.loads((source/'status.json').read_text())
    integrity = json.loads((source/'completion_integrity.json').read_text())
    if status['state'] != 'completed' or not all(r['state'] == 'completed' for r in status['jobs'].values()):
        raise ValueError('This archive expects complete jobs; retain failures separately')
    if integrity['status'] != status or not integrity['source_unchanged_and_clean']:
        raise ValueError('Completion integrity mismatch')
    if manifest['source']['git_dirty'] or integrity['git_commit'] != manifest['source']['git_commit']:
        raise ValueError('Source provenance mismatch')
    for name, expected in integrity['input_sha256'].items():
        if digest(source/name) != expected:
            raise ValueError(f'Raw campaign output drift: {name}')
    commit = integrity['git_commit']
    for name, expected in manifest['source']['source_sha256'].items():
        saved = subprocess.check_output(['git', 'show', f'{commit}:{name}'])
        if hashlib.sha256(saved).hexdigest() != expected:
            raise ValueError(f'Registered commit mismatch: {name}')
    samples = [json.loads(line) for line in (source/'resources.jsonl').read_text().splitlines()]
    if any(s['memory_sampling_error'] for s in samples):
        raise ValueError('Resource sampling failure')
    if max(s['sum_process_group_rss_bytes'] for s in samples) != integrity['max_summed_process_rss_bytes']:
        raise ValueError('Resource maximum mismatch')
    destination.mkdir(parents=True)
    for name in ('manifest.json', 'status.json', 'completion_integrity.json'):
        shutil.copyfile(source/name, destination/name)
    compact = samples[::6]
    if compact[-1] != samples[-1]:
        compact.append(samples[-1])
    peak = max(samples, key=lambda s: s['sum_process_group_rss_bytes'])
    report = {'nominal_original_sampling_seconds': 5, 'nominal_compact_sampling_seconds': 30,
              'observed_peak_sample': peak, 'samples': compact,
              'raw_sha256': digest(source/'resources.jsonl'),
              'accounting': manifest['memory_accounting']}
    (destination/'resource_samples.json').write_text(json.dumps(report, indent=2)+'\n')
    benchmark = Path('outputs/EXP-0010/benchmark.json')
    shutil.copyfile(benchmark, destination/'context-artificial-benchmark.json')
    blind = Path('/private/tmp/voynich-exp0008-benchmark.log')
    value = json.loads(blind.read_text().strip().splitlines()[-1])
    (destination/'blind-artificial-benchmark.json').write_text(json.dumps({
        'result': value, 'raw_stdout_sha256': digest(blind),
        'scope': 'Artificial uniform IDs; dirty implementation hardware probe, not scientific model evaluation.',
        'command': '.venv/bin/python -m voynich.blind_recovery benchmark --device mps --size large --batch 32 --context 256 --steps 20',
    }, indent=2)+'\n')
    print(json.dumps({'source_commit': commit, 'elapsed_seconds': status['elapsed_seconds'],
                      'max_summed_process_rss_bytes': integrity['max_summed_process_rss_bytes'],
                      'archived_resource_samples': len(compact)}))


if __name__ == '__main__':
    main()
