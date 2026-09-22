"""Finite sequential follow-up for an already running registered calibration.

No model is loaded here. Child processes release their model before the next
starts. Failures stop the queue; negative scientific verdicts do not.
"""

import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from .campaign import digest, load_config, write_json


def process_exists(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def run(fit_pid):
    config = load_config('configs/jspace0001.json')
    output, result = Path(config['output']), Path(config['results'])
    directory = output / 'pipeline'
    directory.mkdir(exist_ok=True)
    with (directory / 'lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state_path = directory / 'state.json'
        if state_path.exists():
            raise FileExistsError('Pipeline already launched; inspect its state before any manual recovery')
        sources = sorted(Path('src/voynich/workspace').glob('*.py'))
        hashes = {str(path): digest(path) for path in sources}
        manifest = {'source_sha256': hashes, 'config_sha256': digest('configs/jspace0001.json'),
                    'calibration_inputs_sha256': digest(result / 'inputs.json'),
                    'fit_pid': fit_pid, 'launched_utc': datetime.now(timezone.utc).isoformat(),
                    'source_revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                    'policy': 'Sequential bounded children, no retry, no model or threshold adaptation, halt on execution failure'}
        write_json(directory / 'inputs.json', manifest)
        state = {'phase': 'waiting_for_calibration', 'stages': [], 'manifest_sha256': digest(directory / 'inputs.json')}

        def update(phase):
            state['phase'] = phase
            state['updated_utc'] = datetime.now(timezone.utc).isoformat()
            write_json(state_path, state)
            print(json.dumps({'phase': phase, 'time': state['updated_utc']}), flush=True)

        def verify():
            for path, expected in hashes.items():
                if digest(path) != expected:
                    raise ValueError(f'Queued study source changed: {path}')
            if digest('configs/jspace0001.json') != manifest['config_sha256'] or digest(result / 'inputs.json') != manifest['calibration_inputs_sha256']:
                raise ValueError('Queued study inputs changed')

        def stage(name, module, args=(), *, timeout, required):
            verify()
            update(name)
            command = [sys.executable, '-m', 'voynich.workspace.'+module, *args]
            started = time.monotonic()
            env = dict(os.environ, MLX_ENABLE_TF32='0', PYTHONPATH='src')
            with (directory / (name+'.log')).open('w') as log:
                completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=env, timeout=timeout, check=False)
            record = {'name': name, 'command': command, 'seconds': time.monotonic()-started,
                      'exit_code': completed.returncode, 'log_sha256': digest(directory / (name+'.log'))}
            state['stages'].append(record)
            write_json(state_path, state)
            if completed.returncode:
                raise RuntimeError(f'{name} exited {completed.returncode}; see its retained log')
            for path in required:
                if not Path(path).exists():
                    raise RuntimeError(f'{name} did not complete its required artifact: {path}')

        try:
            update('waiting_for_calibration')
            waiting = time.monotonic()
            while process_exists(fit_pid):
                if time.monotonic()-waiting > config['fit_seconds_cap']+300:
                    raise TimeoutError('Calibration wait cap; existing process was not killed')
                time.sleep(15)
            fit = json.loads((result / 'fit.json').read_text())
            if not fit['complete'] or fit['completed'] != config['calibration_prompts']:
                raise RuntimeError('Calibration did not complete; no causal scoring launched')
            stage('lens_analysis', 'lens_analysis', timeout=300, required=[result / 'lens-stability-512.json'])
            stage('precision_audit', 'precision_audit', timeout=600, required=[result / 'head-transport-sensitivity.json'])
            stage('causal_qualification', 'causal_campaign', ('freeze',), timeout=300, required=[result / 'evaluation-inputs.json'])
            stage('development', 'causal_campaign', ('development',), timeout=config['evaluation_seconds_cap']+120,
                  required=[result / 'development.json', result / 'selection.json'])
            selection = json.loads((result / 'selection.json').read_text())
            if selection['selected_layer'] is not None:
                used = json.loads((output / 'evaluation-progress.json').read_text())['seconds']
                stage('final', 'causal_campaign', ('final',), timeout=max(1, config['evaluation_seconds_cap']-used)+120,
                      required=[result / 'final.json', result / 'decision.json'])
            else:
                state['final_skipped'] = 'No development layer met the registered copy-preservation gate'
                write_json(state_path, state)
            stage('causal_analysis', 'causal_analysis', timeout=300, required=[result / 'causal-audit.json'])
            stage('neurons', 'neuron_campaign', timeout=2820, required=['results/NEURON-0001/results.json', 'results/NEURON-0001/decision.json'])
            stage('neuron_analysis', 'neuron_analysis', timeout=300, required=['results/NEURON-0001/natural-response-analysis.json'])
            update('complete')
        except Exception as error:
            state['error'] = f'{type(error).__name__}: {error}'
            update('stopped_on_error')
            raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fit-pid', required=True, type=int)
    args = parser.parse_args()
    if args.fit_pid <= 1 or args.fit_pid == os.getpid():
        raise ValueError('Expected the existing calibration process ID')
    run(args.fit_pid)


if __name__ == '__main__':
    main()
