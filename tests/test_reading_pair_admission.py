"""Artificial full-schema controller/auditor fixture; no original source archive."""
import gzip
import hashlib
import json
import math
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import run_reading_pair_admit001 as run
from scripts.audit_reading_pair_admit001 import audit, direct_eligible, independent_point
from voynich.reading_pair_rewrite import eligible_triplets, reading_actions
from voynich.reading_regrowth import SourceRegrowth
from voynich.source_action_proposal import ReadingEnvironment, ReadingState


def test_quantile_allocation_is_fixed_and_bounded():
    for degree in range(120):
        indices = run.selected_indices(degree)
        assert len(indices) == min(degree, 8)
        assert tuple(sorted(set(indices))) == indices
        assert all(0 <= i < degree for i in indices)
    for degree in (-1, False, 2.5):
        with pytest.raises(ValueError):
            run.selected_indices(degree)


def test_original_shape_controller_and_independent_auditor_schema(tmp_path, monkeypatch):
    p = np.array([[1/32]*22+[10/32]], dtype=np.float64)
    t = np.zeros((1, 23), dtype=np.uint32)
    p.flags.writeable = t.flags.writeable = False
    source = SimpleNamespace(probabilities=p, transitions=t, state=lambda _: 0)
    monkeypatch.setattr(run, 'ROOT', tmp_path)
    monkeypatch.setattr(run, 'OUT', tmp_path/'results'/run.EXP)
    monkeypatch.setattr(run, 'BULK', tmp_path/'outputs'/run.EXP)
    monkeypatch.setattr(run, 'CASES', 2)

    def artifact(path):
        return {'path': str(path.relative_to(tmp_path)), 'bytes': path.stat().st_size,
                'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}

    def save_new(path, value, compressed=False):
        path.parent.mkdir(parents=True, exist_ok=True)
        data = json.dumps(value, sort_keys=True).encode()
        if compressed:
            data = gzip.compress(data, mtime=0)
        with path.open('xb') as stream:
            stream.write(data)
        return artifact(path)

    def load(spec):
        path = tmp_path/spec['path']
        assert artifact(path) == spec
        return json.loads(gzip.decompress(path.read_bytes()))

    monkeypatch.setattr(run, 'training', SimpleNamespace(
        old=SimpleNamespace(require_frozen=lambda *_: None), artifact=artifact, save_new=save_new,
        limit_resources=lambda *_: None,
        resource_report=lambda *_: {'wall_seconds': .1, 'cpu_seconds': .1, 'peak_rss_bytes': 1024}))
    monkeypatch.setattr(run, 'load_archive', load)
    monkeypatch.setattr(run, 'load_source', lambda: (source, None))

    def point(source, sampler, path):
        n, d = independent_point(source, sampler.env, path.state, lambda: None)
        return {'log_probability': math.log(n)-math.log(d)+sum(k >= 0 for k in path.state.key)*math.log(42),
                'maximum_total_log_delta': 0., 'maximum_joint_log_delta': 0.}

    monkeypatch.setattr(run.prior.recovery.admitted, 'point_check', point)
    frames = []
    for case in range(2):
        env = ReadingEnvironment(((0, 1, 0, 1), (0, 1)), rows=23, glyphs=6)
        state = ReadingState((-1, 0, 1)+(-1,)*20, (4, 2), ((1, 2, 1, 2), (1, 2)))
        path = SourceRegrowth(source, env).path(forced_actions=reading_actions(env, state))
        assert direct_eligible(env, state) == eligible_triplets(env, state)
        for phase in run.PHASES:
            spec = save_new(tmp_path/f'inputs/{case}-{phase}.gz',
                            {'records': list(map(list, env.records)), 'base': run.prior.reading_receipt(path)}, True)
            frames.append({'case': case, 'kind': 'artificial', 'phase': phase, 'archive': spec})
    monkeypatch.setattr(run, 'require_prior', lambda: {'cells': frames, 'source_arrays': run.array_identity(source)})
    for name in run.PATHS:
        path = tmp_path/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(name)
    assert all(run.admission('artificial-only').values())
    assert audit('artificial-only') == 32
    result = json.loads((run.OUT/'result.json').read_text())
    assert len(result['cells']) == 4 and result['total_points'] == 32
    assert result['no_chain_gold_metrics_neural_calls_or_optimizer']
    with pytest.raises(FileExistsError):
        run.admission('artificial-only')
