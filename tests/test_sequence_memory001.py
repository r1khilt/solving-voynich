import json
import subprocess
from types import SimpleNamespace

import pytest

from scripts import benchmark_sequence_memory001 as runner
from scripts.benchmark_sequence_memory001 import artificial_records, check_memory


def test_artificial_input_inventory_is_deterministic_and_contains_all_lengths():
    texts = artificial_records()
    assert texts == artificial_records()
    assert len(texts) == 4096
    assert set(map(len, texts)) == set(range(180, 271))
    assert len(set(texts)) == 4096


@pytest.mark.parametrize('driver,host', [(2*1024**3+1, 0), (0, 4*1024**3+1)])
def test_guard_records_both_memory_domains(driver, host):
    with pytest.raises(MemoryError):
        check_memory({'driver_bytes': driver, 'peak_rss_bytes': host})
    check_memory({'driver_bytes': 2*1024**3, 'peak_rss_bytes': 4*1024**3})


def test_parent_retains_failure_timeout_and_success_without_retry(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, 'OUT', tmp_path/'results')
    monkeypatch.setattr(runner, 'BULK', tmp_path/'outputs')
    monkeypatch.setattr(runner, 'require_frozen', lambda *_: None)
    monkeypatch.setattr(runner, 'artifact', lambda p: {'path': str(p), 'bytes': p.stat().st_size})
    def save(p, data):
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open('x') as stream:
            json.dump(data, stream)
    monkeypatch.setattr(runner, 'save_new', save)
    calls = []
    def run(command, **kwargs):
        arm = command[command.index('--arm')+1]
        calls.append(arm)
        assert kwargs['timeout'] == 195 and not kwargs['check']
        if arm == 'variable8':
            raise subprocess.TimeoutExpired(command, 195)
        return SimpleNamespace(returncode=1 if arm == 'variable64' else 0)
    monkeypatch.setattr(runner.subprocess, 'run', run)
    runner.campaign('frozen')
    result = json.loads((runner.OUT/'campaign.json').read_text())
    assert calls == list(runner.ARMS)
    assert [r['returncode'] for r in result['processes']] == [1, None, 0]
    assert [r['outer_timeout'] for r in result['processes']] == [False, True, False]
    with pytest.raises(FileExistsError):
        runner.campaign('frozen')
    assert calls == list(runner.ARMS)
