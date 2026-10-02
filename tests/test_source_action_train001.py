import copy
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from scripts import audit_source_action_train001 as audit
from scripts import run_blind_channel_dev004 as storage
from scripts import run_latin_source_model001 as resources
from scripts import run_source_action_train001 as run
from voynich import joint_key_training as sampling
from voynich.recurrent_latin_source import ALPHABET
from voynich.source_action_proposal import ReadingEnvironment, SourceActionConfig, SourceActionProposal
from voynich.source_action_training import control_metrics, control_records, greedy_reading


@pytest.fixture(autouse=True)
def backend():
    fast = torch.backends.mha.get_fastpath_enabled()
    yield
    torch.backends.mha.set_fastpath_enabled(fast)


def fixture(monkeypatch, tmp_path):
    for module in (run, storage, resources):
        monkeypatch.setattr(module, 'ROOT', tmp_path)
    monkeypatch.setattr(run, 'OUT', tmp_path/'results'/'new')
    monkeypatch.setattr(run, 'BULK', tmp_path/'outputs'/'new')
    monkeypatch.setattr(run.old, 'OUT', tmp_path/'results'/'parent')
    monkeypatch.setattr(run.old, 'require_frozen', lambda *args: None)
    monkeypatch.setattr(run, 'require_admission', lambda: None)
    monkeypatch.setattr(run, 'limit_resources', lambda *args: None)
    monkeypatch.setattr(sampling, 'MIN_LENGTH', 8)
    monkeypatch.setattr(sampling, 'MAX_LENGTH', 16)
    monkeypatch.setattr(run, 'VALIDATION_COUNT', 2)
    monkeypatch.setattr(run, 'VALIDATION_SEED', 17)
    monkeypatch.setattr(run, 'CONTROL_COUNT', 1)
    monkeypatch.setattr(run, 'CONTROL_SEED', 19)
    train, valid = [ALPHABET*30], [ALPHABET[::-1]*30]
    parent = {'identity': {'training_text_sha256': 'fixture', 'source_roles': 'synthetic-only'}}
    parent['training'] = run.save_new(tmp_path/'outputs'/'parent-train.json.gz',
        {'role': 'training_pool', 'texts': train}, compressed=True)
    parent['validation_source'] = run.save_new(tmp_path/'outputs'/'parent-valid.json.gz',
        {'role': 'source_validation', 'texts': valid}, compressed=True)
    previous = [sampling.EpisodeSampler(valid).sample(np.random.default_rng(23))]
    monkeypatch.setattr(run.old, 'load_prepared', lambda: (parent, train, valid, previous))
    monkeypatch.setattr(run.old, 'load_inputs', lambda _: (train, valid, parent['identity']))
    for name in ('inputs', 'input-audit', 'audit'):
        run.save_new(run.old.OUT/(name+'.json'), {'status': 'PASS', 'fixture': True})
    return run.OUT, run.BULK


def test_prepare_roles_fresh_keys_exact_replay_exclusivity_and_hash_corruption(monkeypatch, tmp_path):
    out, _ = fixture(monkeypatch, tmp_path)
    run.prepare('fixture-freeze')
    run.audit_inputs('fixture-freeze')
    manifest, train, valid, episodes = run.load_prepared()
    assert train != valid and len(episodes) == 2
    assert len({sampling.dictionary_code(e[1]) for e in episodes}) == 2
    assert json.loads((out/'input-audit.json').read_text())['status'] == 'PASS'
    with pytest.raises(FileExistsError):
        run.prepare('fixture-freeze')
    with (tmp_path/manifest['training']['path']).open('ab') as handle:
        handle.write(b'corrupt')
    with pytest.raises(ValueError, match='identity'):
        run.load_prepared()


def test_real_four_cpu_microfits_all_checkpoints_ledger_controls_and_completion_audit(monkeypatch, tmp_path):
    out, bulk = fixture(monkeypatch, tmp_path)
    run.prepare('fixture-freeze')
    run.audit_inputs('fixture-freeze')
    monkeypatch.setattr(run, 'DEVICE', 'cpu')
    monkeypatch.setattr(run, 'STEPS', 3)
    monkeypatch.setattr(run, 'CHECKPOINTS', (0, 1, 3))
    monkeypatch.setattr(run, 'SEEDS', (11, 13))
    arms = tuple(f'{role}-{seed}' for seed in run.SEEDS for role in ('binding', 'no-binding'))
    monkeypatch.setattr(run, 'ARMS', arms)
    config = SourceActionConfig(width=8, heads=2, encoder_layers=1, decoder_layers=1, max_records=2)
    monkeypatch.setattr(run, 'CONFIG', config)
    monkeypatch.setattr(run, 'PARAMETERS', sum(p.numel() for p in SourceActionProposal(config).parameters()))
    monkeypatch.setattr(run, 'schedule', lambda step, total: 1e-4)
    processes = []
    for arm in arms:
        run.fit(arm, 'fixture-freeze')
        log = bulk/(arm+'.log')
        log.write_text('CPU synthetic integration fixture, no scientific allocation\n')
        processes.append({'arm': arm, 'returncode': 0, 'outer_timeout': False, 'log': run.artifact(log)})
    run.save_new(out/'campaign.json', {'freeze': 'fixture-freeze', 'processes': processes,
                                      'wall_seconds': 0., 'paid_spend_usd': 0})
    audit.audit()
    result = json.loads((out/'audit.json').read_text())
    assert result['all_fits_complete'] and len(result['outcomes']) == 4
    assert not result['decision']['qualified_historical_or_general_inverse_or_neural_circuit']
    value = json.loads((out/(arms[0]+'.json')).read_text())
    assert value['updates'] == 3 and value['episodes_seen'] == 12
    manifest, texts, valid, episodes = run.load_prepared()
    prefixes, _ = audit.ledger(value, texts, episodes, manifest)
    assert prefixes[3][0] == value['data_sha256']
    row = value['selected']
    model = audit.checkpoint(row['weights'], row, value)
    score = run.load_archive(row['validation'])['score']
    audit.replay(model, score, episodes, sampling.EpisodeSampler(valid))
    broken = copy.deepcopy(score)
    broken['joint_target_path_logq'][0] += 1
    with pytest.raises(AssertionError):
        audit.replay(model, broken, episodes, sampling.EpisodeSampler(valid))
    broken = copy.deepcopy(score)
    broken['recovery']['exact_records'] += 1
    with pytest.raises(AssertionError):
        audit.independent_recovery(broken, episodes, sampling.EpisodeSampler(valid))
    with pytest.raises(FileExistsError):
        run.fit(arms[0], 'fixture-freeze')
    with pytest.raises(FileExistsError):
        audit.audit()


def test_campaign_terminal_failures_are_preserved_and_never_retried(monkeypatch, tmp_path):
    out, _ = fixture(monkeypatch, tmp_path)
    called = []

    def child(argv, **kwargs):
        assert argv[2:5] == ['-m', 'scripts.run_source_action_train001', 'fit']
        called.append(argv)
        if len(called) == 2:
            raise subprocess.TimeoutExpired(argv, 1)
        if len(called) == 3:
            raise OSError('fixture missing process')
        return subprocess.CompletedProcess(argv, 0 if len(called) == 1 else 1)

    monkeypatch.setattr(run.subprocess, 'run', child)
    run.campaign('fixture-freeze')
    result = json.loads((out/'campaign.json').read_text())
    assert len(called) == 4
    assert [r['returncode'] for r in result['processes']] == [0, None, None, 1]
    assert result['processes'][1]['outer_timeout']
    with pytest.raises(FileExistsError):
        run.campaign('fixture-freeze')
    assert len(called) == 4


def test_actual_module_entrypoint_imports_from_repository_root():
    result = subprocess.run([sys.executable, '-m', 'scripts.run_source_action_train001', '--help'],
        cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert 'audit-inputs' in result.stdout and 'campaign' in result.stdout


def test_controls_preserve_shuffle_counts_and_nonidentifiability_is_explicit():
    sampler = sampling.EpisodeSampler([ALPHABET*30])
    episode = sampler.sample(np.random.default_rng(29))
    controls = control_records([episode], count=1, seed=31)
    assert controls == control_records([episode], count=1, seed=31)
    assert [sorted(r) for r in controls[0]['records']] == [sorted(r) for r in episode[0]]
    ambiguous = controls[-1]
    env = ReadingEnvironment(ambiguous['records'])
    key = (0,)*23
    a = env.teaching_trace([(0,)*64, (0,)*64], key)
    b = env.teaching_trace([(1,)*64, (1,)*64], key)
    assert a[2].texts != b[2].texts
    torch.manual_seed(37)
    model = SourceActionProposal(SourceActionConfig(width=8, heads=2, encoder_layers=1, decoder_layers=1))
    prediction = greedy_reading(model, env)
    result = control_metrics([prediction], [ambiguous])
    assert not result['language_or_semantic_classification_claimed']
    broken = copy.deepcopy(prediction)
    broken['texts'] = [[22], [22]]
    with pytest.raises(ValueError, match='mismatch'):
        control_metrics([broken], [ambiguous])
