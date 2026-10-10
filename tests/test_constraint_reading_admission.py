"""Artificial original-shape source admission/controller/full audit and corruption."""
from copy import deepcopy
import gzip
import hashlib
import json

import pytest

from scripts import run_constraint_reading_admit001 as run
from scripts import audit_constraint_reading_admit001 as auditor
from tests.test_window_reading_admission import fixture


def test_full_artificial_dispatch_source_controller_audit_and_exclusivity(tmp_path, monkeypatch):
    source, sampler, base = fixture()
    monkeypatch.setattr(run, 'ROOT', tmp_path)
    monkeypatch.setattr(run, 'OUT', tmp_path/'results'/run.EXP)
    monkeypatch.setattr(run, 'BULK', tmp_path/'outputs'/run.EXP)
    monkeypatch.setattr(run, 'PATHS', ('fixture.txt',))
    monkeypatch.setattr(run, 'TINY_SEEDS', (96521,))
    monkeypatch.setattr(run, 'TINY_SWEEPS', 4)
    monkeypatch.setattr(run, 'SOURCE_SWEEPS', 32)
    monkeypatch.setattr(run.training.old, 'require_frozen', lambda *_: None)
    monkeypatch.setattr(run.training, 'limit_resources', lambda *_: None)
    (tmp_path/'fixture.txt').write_text('Entirely artificial source and records.\n')
    def artifact(path):
        data = path.read_bytes()
        return {'path': str(path.relative_to(tmp_path)), 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}
    def save(path, value, compressed=False):
        path.parent.mkdir(parents=True, exist_ok=True)
        data = json.dumps(value, sort_keys=True, indent=2).encode()
        if compressed:
            data = gzip.compress(data, mtime=0)
        with path.open('xb') as stream:
            stream.write(data)
        return artifact(path)
    def load(spec):
        path = tmp_path/spec['path']
        assert artifact(path) == spec
        return json.loads(gzip.decompress(path.read_bytes()))
    monkeypatch.setattr(run.training, 'artifact', artifact)
    monkeypatch.setattr(run.training, 'save_new', save)
    monkeypatch.setattr(run.prior.prior, 'load_archive', load)
    monkeypatch.setattr(run.prior.prior, 'load_source', lambda: (source, {'counts': {'artificial': True}}))
    frames = []
    for phase in run.prior.PHASES:
        spec = save(tmp_path/'inputs'/f'{phase}.json.gz', {'records': list(map(list, sampler.env.records)),
            'base': run.prior.prior.prior.reading_receipt(base)}, compressed=True)
        frames.append({'case': 0, 'kind': 'artificial', 'phase': phase, 'archive': spec})
    monkeypatch.setattr(run, 'require_prior', lambda: {'cells': frames, 'source_arrays': run.prior.prior.array_identity(source)})
    assert all(run.admission('artificial-only').values())
    totals = auditor.audit('artificial-only')
    assert totals == {'local_attempts': 288, 'exchange_attempts': 108, 'cold_references': 72}
    result = json.loads((run.OUT/'result.json').read_text())
    assert len(result['cells']) == 4
    first = result['cells'][0]
    saved = load(first['archive'])
    expert, tiny_sampler, tiny_base, seed = list(run.tiny_cases())[0]
    corrupted = deepcopy(saved)
    corrupted['trace'][0]['local'][0]['info']['forward'] = '1/9999999'
    with pytest.raises(AssertionError):
        auditor.replay(expert, tiny_sampler, tiny_base, corrupted, seed=seed, sweeps=run.TINY_SWEEPS)
    with pytest.raises(FileExistsError):
        run.admission('artificial-only')
    with pytest.raises(FileExistsError):
        auditor.audit('artificial-only')
