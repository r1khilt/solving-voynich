"""Run a registered finite local campaign in parallel with scoped process cleanup."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
CHILD_ENV = {
    'PYTORCH_MPS_HIGH_WATERMARK_RATIO': '0.23',
    'PYTORCH_MPS_LOW_WATERMARK_RATIO': '0.18',
    'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2', 'OPENBLAS_NUM_THREADS': '2',
    'VECLIB_MAXIMUM_THREADS': '2', 'PYTHONUNBUFFERED': '1',
}


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temp.replace(path)


def validate_plan(plan):
    jobs = plan.get('jobs', [])
    if not 1 <= len(jobs) <= 3:
        raise ValueError('A campaign has one to three simultaneous jobs')
    for field, cap in [('max_seconds', 28800), ('rss_ceiling_gib', 48)]:
        value = plan.get(field)
        if not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 < value <= cap:
            raise ValueError(f'Invalid {field}; maximum {cap}')
    names = []
    for job in jobs:
        name = job.get('name', '')
        if not name or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789-_' for c in name):
            raise ValueError('Job name must be a safe lowercase identifier')
        names.append(name)
        command = job.get('command')
        if not isinstance(command, list) or not command or any(not isinstance(c, str) or not c for c in command):
            raise ValueError('Commands must be nonempty argument arrays, never shell strings')
        seconds = job.get('max_seconds')
        if not isinstance(seconds, (int, float)) or not math.isfinite(seconds) or not 0 < seconds <= plan['max_seconds']:
            raise ValueError('Invalid job time budget')
    if len(set(names)) != len(names):
        raise ValueError('Job identifiers must be unique')


def source_state(root):
    def git(*args):
        return subprocess.check_output(['git', *args], cwd=root, text=True).strip()
    paths = git('ls-files').splitlines()
    hashes = {}
    for name in paths:
        if name.startswith(('src/', 'scripts/', 'configs/', 'docs/experiments/')) or name in {'pyproject.toml', 'uv.lock'}:
            hashes[name] = hashlib.sha256((root/name).read_bytes()).hexdigest()
    return {'git_commit': git('rev-parse', 'HEAD'), 'git_dirty': bool(git('status', '--porcelain')),
            'source_sha256': hashes}


def process_group_rss(groups):
    """Conservative process RSS accounting; not total physical or GPU allocation."""
    values = {group: 0 for group in groups}
    output = subprocess.check_output(['ps', '-axo', 'pgid=,rss='], text=True)
    for row in output.splitlines():
        group, rss_kib = map(int, row.split())
        if group in values:
            values[group] += rss_kib*1024
    return values


def terminate_group(process, grace_seconds=10):
    # Every child is started in its own session. No unrelated process is signaled.
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=grace_seconds)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=10)


def run_campaign(plan, output, *, root=ROOT, enforce_clean=True, poll_seconds=5):
    validate_plan(plan)
    root, output = Path(root), Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError('Use an empty campaign output directory')
    provenance = source_state(root) if enforce_clean else {'test_fixture': True}
    if enforce_clean and provenance['git_dirty']:
        raise ValueError('Commit and publish all registered source before execution')
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    manifest = {'campaign': plan.get('campaign'), 'started_utc': datetime.now(timezone.utc).isoformat(),
                'plan': plan, 'source': provenance, 'child_environment_overrides': CHILD_ENV,
                'memory_accounting': 'Process-group RSS sampled; shared pages can be counted more than once. MPS allocator cap is separate. No summing RSS with GPU memory.',
                'manuscript_test_allowed': False, 'paid_services_allowed': False}
    write_json(output/'manifest.json', manifest)
    records, running, handles = {}, {}, []
    interrupted = None
    previous_handlers = {}

    def handle_signal(signum, _frame):
        nonlocal interrupted
        interrupted = signal.Signals(signum).name

    for signum in [signal.SIGTERM, signal.SIGINT]:
        previous_handlers[signum] = signal.signal(signum, handle_signal)
    samples = (output/'resources.jsonl').open('w')
    status = {'state': 'running', 'elapsed_seconds': 0., 'jobs': records}
    try:
        for job in plan['jobs']:
            log = (output/f"{job['name']}.log").open('w')
            handles.append(log)
            record = {'state': 'starting', 'command': job['command'], 'max_seconds': job['max_seconds'],
                      'started_elapsed_seconds': time.monotonic()-started, 'peak_rss_bytes': 0}
            records[job['name']] = record
            try:
                child = subprocess.Popen(job['command'], cwd=root, env={**os.environ, **CHILD_ENV},
                                         stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                running[job['name']] = child
                record.update(state='running', pid=child.pid)
            except OSError as exc:
                record.update(state='launch_failed', error=str(exc), returncode=None)
        while running:
            elapsed = time.monotonic()-started
            try:
                rss = process_group_rss([p.pid for p in running.values()])
                memory_error = None
            except (OSError, subprocess.SubprocessError, ValueError) as exc:
                rss, memory_error = {}, str(exc)
            combined = sum(rss.values())
            sample = {'elapsed_seconds': elapsed, 'sum_process_group_rss_bytes': combined,
                      'rss_by_job': {name: rss.get(p.pid) for name, p in running.items()},
                      'memory_sampling_error': memory_error}
            samples.write(json.dumps(sample)+'\n')
            samples.flush()
            for name, child in list(running.items()):
                record = records[name]
                record['peak_rss_bytes'] = max(record['peak_rss_bytes'], rss.get(child.pid, 0))
                reason = None
                if child.poll() is None:
                    if interrupted:
                        reason = 'interrupted_'+interrupted
                    elif combined > plan['rss_ceiling_gib']*2**30:
                        reason = 'campaign_rss_ceiling'
                    elif elapsed >= plan['max_seconds']:
                        reason = 'campaign_deadline'
                    elif elapsed-record['started_elapsed_seconds'] >= record['max_seconds']:
                        reason = 'job_deadline'
                    if reason:
                        terminate_group(child)
                code = child.poll()
                if code is not None:
                    record.update(state=reason or ('completed' if code == 0 else 'failed'),
                                  returncode=code, finished_elapsed_seconds=time.monotonic()-started)
                    del running[name]
                    print(json.dumps({'job': name, 'state': record['state'], 'returncode': code}), flush=True)
            status['elapsed_seconds'] = time.monotonic()-started
            write_json(output/'status.json', status)
            if running:
                time.sleep(poll_seconds)
    finally:
        for name, child in running.items():
            terminate_group(child)
            records[name].update(state='supervisor_cleanup', returncode=child.returncode)
        samples.close()
        for handle in handles:
            handle.close()
        for signum, previous in previous_handlers.items():
            signal.signal(signum, previous)
        status['elapsed_seconds'] = time.monotonic()-started
        status['state'] = 'completed' if records and all(r['state'] == 'completed' for r in records.values()) else 'incomplete'
        write_json(output/'status.json', status)
    return status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', default='configs/campaign-0001.json')
    parser.add_argument('--output', default='outputs/CAMPAIGN-0001')
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    plan = json.loads(Path(args.plan).read_text())
    validate_plan(plan)
    print(json.dumps({'plan': plan, 'environment': CHILD_ENV, 'execute': args.execute}, indent=2), flush=True)
    if args.execute:
        result = run_campaign(plan, args.output)
        print(json.dumps(result, indent=2), flush=True)
        sys.exit(0 if result['state'] == 'completed' else 1)


if __name__ == '__main__':
    main()
