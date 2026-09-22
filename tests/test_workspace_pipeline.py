import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from voynich.workspace import pipeline


def setup_queue(tmp_path, monkeypatch, *, fail=None, complete_fit=True):
    monkeypatch.chdir(tmp_path)
    for directory in ('outputs/JSPACE-0001', 'results/JSPACE-0001', 'configs', 'src/voynich/workspace'):
        Path(directory).mkdir(parents=True)
    config = {'experiment': 'JSPACE-0001', 'output': 'outputs/JSPACE-0001', 'results': 'results/JSPACE-0001',
              'fit_seconds_cap': 14400, 'evaluation_seconds_cap': 7200, 'calibration_prompts': 512}
    pipeline.write_json('configs/jspace0001.json', config)
    pipeline.write_json('results/JSPACE-0001/inputs.json', {'test': True})
    pipeline.write_json('results/JSPACE-0001/fit.json', {'complete': complete_fit, 'completed': 512})
    monkeypatch.setattr(pipeline, 'process_exists', lambda pid: False)
    monkeypatch.setattr(pipeline.subprocess, 'check_output', lambda *a, **k: 'test-revision')
    calls = []

    def child(command, **kwargs):
        module = command[2].rsplit('.', 1)[-1]
        name = command[3] if module == 'causal_campaign' else module
        calls.append(name)
        assert kwargs['timeout'] > 0
        assert kwargs['env']['MLX_ENABLE_TF32'] == '0'
        if name == fail:
            return SimpleNamespace(returncode=3)
        artifacts = {'lens_analysis': ['results/JSPACE-0001/lens-stability-512.json'],
                     'precision_audit': ['results/JSPACE-0001/head-transport-sensitivity.json'],
                     'freeze': ['results/JSPACE-0001/evaluation-inputs.json'],
                     'development': ['results/JSPACE-0001/development.json'],
                     'causal_analysis': ['results/JSPACE-0001/causal-audit.json'],
                     'neuron_campaign': ['results/NEURON-0001/results.json', 'results/NEURON-0001/decision.json'],
                     'neuron_analysis': ['results/NEURON-0001/natural-response-analysis.json']}
        for path in artifacts[name]:
            pipeline.write_json(path, {})
        if name == 'development':
            pipeline.write_json('results/JSPACE-0001/selection.json', {'selected_layer': None})
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(pipeline.subprocess, 'run', child)
    return calls


def test_negative_selection_skips_final_but_runs_independent_neuron_study(tmp_path, monkeypatch):
    calls = setup_queue(tmp_path, monkeypatch)
    pipeline.run(99)
    assert calls == ['lens_analysis', 'precision_audit', 'freeze', 'development', 'causal_analysis', 'neuron_campaign', 'neuron_analysis']
    state = json.loads(Path('outputs/JSPACE-0001/pipeline/state.json').read_text())
    assert state['phase'] == 'complete' and 'final_skipped' in state
    with pytest.raises(FileExistsError):
        pipeline.run(99)


def test_numerical_failure_stops_queue_without_retry(tmp_path, monkeypatch):
    calls = setup_queue(tmp_path, monkeypatch, fail='freeze')
    with pytest.raises(RuntimeError, match='exited 3'):
        pipeline.run(99)
    assert calls == ['lens_analysis', 'precision_audit', 'freeze']
    state = json.loads(Path('outputs/JSPACE-0001/pipeline/state.json').read_text())
    assert state['phase'] == 'stopped_on_error'


def test_incomplete_calibration_never_starts_causal_scoring(tmp_path, monkeypatch):
    calls = setup_queue(tmp_path, monkeypatch, complete_fit=False)
    with pytest.raises(RuntimeError, match='Calibration did not complete'):
        pipeline.run(99)
    assert calls == []
