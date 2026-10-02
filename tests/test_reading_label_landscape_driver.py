"""Actual all-pairs controller/auditor schema exercised only on an artificial source."""
import gzip
import hashlib
import json
from types import SimpleNamespace

import numpy as np

from scripts import audit_reading_label_landscape001 as audit
from scripts import run_reading_label_landscape001 as run
from voynich.joint_key_training import EpisodeSampler
from voynich.source_action_training import action_episode


def test_complete_driver_and_independent_audit_on_artificial_source(tmp_path, monkeypatch):
    def artifact(path):
        raw = path.read_bytes()
        return {'path': str(path.relative_to(tmp_path)), 'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw)}

    def save(path, value, compressed=False):
        path.parent.mkdir(parents=True, exist_ok=True)
        raw = json.dumps(value, sort_keys=True).encode()
        if compressed:
            raw = gzip.compress(raw)
        with path.open('xb') as handle:
            handle.write(raw)
        return artifact(path)

    def load(spec):
        path = tmp_path/spec['path']
        assert artifact(path) == spec
        return json.loads(gzip.decompress(path.read_bytes()))

    monkeypatch.setattr(run, 'ROOT', tmp_path)
    monkeypatch.setattr(run, 'OUT', tmp_path/'results'/'landscape')
    monkeypatch.setattr(run, 'BULK', tmp_path/'outputs'/'landscape')
    monkeypatch.setattr(run, 'PATHS', ['fixture.json'])
    monkeypatch.setattr(run.training, 'artifact', artifact)
    monkeypatch.setattr(run.training, 'save_new', save)
    monkeypatch.setattr(run.training, 'limit_resources', lambda *_: None)
    monkeypatch.setattr(run.training.old, 'require_frozen', lambda *_: None)
    monkeypatch.setattr(run.recovery, 'CASES', 3)
    monkeypatch.setattr(run.recovery, 'POSITIVE', 2)
    (tmp_path/'fixture.json').write_text('artificial only')
    p = np.full((1, 23), 1/23, dtype=np.float64)
    t = np.zeros((1, 23), dtype=np.uint32)
    p.flags.writeable = t.flags.writeable = False
    source = SimpleNamespace(probabilities=p, transitions=t, state=lambda _: 0)
    monkeypatch.setattr(run, 'load_source', lambda: (source, {}))
    monkeypatch.setattr(run, 'load_archive', load)
    episode_sampler = EpisodeSampler(['a'*400])
    episode = episode_sampler.make([{'segment': 0, 'start': 0, 'length': 64}]*2, [7]*23)
    env, trace = action_episode(episode_sampler, episode)
    prediction = {'status': 'complete_path', 'actions': list(trace[1]), 'key': list(trace[2].key),
                  'texts': list(map(list, trace[2].texts))}
    initial = save(tmp_path/'outputs'/'initial.json.gz', {'prediction': prediction}, compressed=True)
    cells = []
    for case in range(3):
        for seed in run.recovery.SEEDS:
            for arm in run.recovery.ARMS:
                spec = save(tmp_path/'outputs'/f'{case}-{seed}-{arm}.json.gz',
                            {'final': {'actions': list(trace[1])}}, compressed=True)
                cells.append({'seed': seed, 'case': case, 'arm': arm, 'archive': spec})
    monkeypatch.setattr(run, 'require_prior', lambda: {'cells': cells})
    allocated = [('positive' if i < 2 else 'null', i, env.records) for i in range(3)]
    monkeypatch.setattr(run.recovery.admitted, 'load_inputs', lambda: (
        {}, [episode]*2, episode_sampler, allocated, [{'archive': initial}]*3))
    assert run.measure('artificial-landscape-fixture') == {'states': 15, 'pairs': 3795}
    assert audit.audit('artificial-landscape-fixture') == 3795
    result = json.loads((run.OUT/'result.json').read_text())
    assert len(result['cells']) == 15
    assert all(c['summary']['uphill_pairs'] == 0 and c['summary']['best_uphill_pair'] is None
               for c in result['cells'])
    for metrics in result['known_answer_metrics'].values():
        assert metrics['base']['exact_records'] == metrics['best_by_target']['exact_records'] == 4
        assert all(p['label_orbit_reachable'] for p in metrics['orbit_obstructions'])
