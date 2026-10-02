"""Actual greedy-schema adapter and full-size-law receipt boundaries."""
import math
from fractions import Fraction

import pytest

from scripts.audit_reading_regrowth_admit001 import independent_dense_log_target
from scripts.run_reading_regrowth_admit001 import configuration, initial_receipt, literal_replay
from tests.test_reading_regrowth import fixture_source
from voynich.reading_regrowth import SourceRegrowth
from voynich.regrowth_trace import log_target, path_receipt
from voynich.source_action_proposal import ReadingEnvironment


def test_real_prior_greedy_schema_and_independent_density():
    env = ReadingEnvironment(((0, 1), (0, 1)), rows=2, glyphs=2)
    source = fixture_source(True)
    sampler = SourceRegrowth(source, env)
    path = sampler.path(forced_actions=(1, 1))
    old = {'status': 'complete_path', 'key': list(path.state.key),
           'texts': [list(t) for t in path.state.texts], 'actions': list(path.actions),
           'greedy_path_under_model_log_probability': -999}
    assert literal_replay(env, initial_receipt(env, old)) == path.state
    assert literal_replay(env, path_receipt(path)) == path.state
    assert abs(log_target(sampler, path)-independent_dense_log_target(source, env, path.state)) < 1e-12
    assert math.isfinite(path.policy_log_probability())
    assert configuration('independence').root_mass == Fraction(1)
    assert configuration('regrowth').root_mass == Fraction(1, 8)
    with pytest.raises(ValueError):
        configuration('other')
    with pytest.raises(AssertionError):
        initial_receipt(env, {**old, 'status': 'dead_end'})


def test_runner_and_audit_entire_json_pipeline_on_artificial_source(tmp_path, monkeypatch):
    """Exercise all194 cells, actual receipt schema and audit before scientific launch."""
    import gzip
    import hashlib
    import json
    from types import SimpleNamespace

    import numpy as np

    from scripts import audit_reading_regrowth_admit001 as audit
    from scripts import run_reading_regrowth_admit001 as run

    def artifact(path):
        raw = path.read_bytes()
        return {'path': str(path.relative_to(tmp_path)), 'bytes': len(raw),
                'sha256': hashlib.sha256(raw).hexdigest()}

    def save(path, value, compressed=False):
        raw = (json.dumps(value, sort_keys=True, allow_nan=False)+'\n').encode()
        if compressed:
            raw = gzip.compress(raw, mtime=0)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('xb') as stream:
            stream.write(raw)
        return artifact(path)

    def load(spec):
        path = tmp_path/spec['path']
        assert artifact(path) == spec
        return json.loads(gzip.decompress(path.read_bytes()))

    monkeypatch.setattr(run, 'OUT', tmp_path/'results'/'admission')
    monkeypatch.setattr(run, 'BULK', tmp_path/'outputs'/'admission')
    monkeypatch.setattr(run, 'PATHS', ['fixture.json'])
    monkeypatch.setattr(run.prior.training, 'ROOT', tmp_path)
    monkeypatch.setattr(run.prior.training, 'OUT', tmp_path/'results'/'training')
    monkeypatch.setattr(run.prior.training, 'artifact', artifact)
    monkeypatch.setattr(run.prior.training, 'save_new', save)
    monkeypatch.setattr(run.prior.training, 'limit_resources', lambda *_: None)
    monkeypatch.setattr(run.prior.training.old, 'require_frozen', lambda *_: None)
    monkeypatch.setattr(run.theory, 'ROOT', tmp_path)
    monkeypatch.setattr(run.theory, 'artifact', artifact)
    save(tmp_path/'fixture.json', {'artificial': True})
    save(run.prior.training.OUT/'inputs.json', {'artificial': True})
    finite = save(tmp_path/'results'/run.theory.EXP/'result.json', {'status': 'PASS_exact_finite_regrowth_kernel'})
    save(tmp_path/'results'/run.theory.EXP/'audit.json', {
        'result': finite, 'status': 'PASS_receipt_hash_and_arithmetic_closure'})
    p = np.full((1, 23), 1/23, dtype=np.float64)
    t = np.zeros((1, 23), dtype=np.uint32)
    p.flags.writeable = t.flags.writeable = False
    source = SimpleNamespace(probabilities=p, transitions=t, state=lambda _: 0,
        alphabet=tuple(chr(65+i) for i in range(23)),
        reference=SimpleNamespace(state=lambda _: 0, row=lambda _: p[0]))
    selected = {'counts': artifact(tmp_path/'fixture.json'), 'inputs': []}
    monkeypatch.setattr(run, 'load_source', lambda: (source, selected))
    monkeypatch.setattr(audit, 'load_source', lambda: (source, selected))
    monkeypatch.setattr(run, 'load_archive', load)
    monkeypatch.setattr(audit, 'load_archive', load)
    records = ((0, 1, 0, 1),)*2
    env = ReadingEnvironment(records, rows=23, glyphs=6)
    key = (env.pool.index((0, 1)),)+(0,)*22
    truth = env.teaching_trace(((0, 0),)*2, key)
    prior_prediction = {'status': 'complete_path', 'key': list(truth[2].key),
        'texts': [list(t) for t in truth[2].texts], 'actions': list(truth[1])}
    init = save(tmp_path/'outputs'/'initial.json.gz', {'records': [list(r) for r in records],
        'prediction': prior_prediction}, compressed=True)
    allocated = [('positive' if i < 64 else 'null', i, records) for i in range(97)]
    initials = [{'case': i, 'kind': kind, 'paired_case': paired, 'archive': init}
                for i, (kind, paired, _) in enumerate(allocated)]
    monkeypatch.setattr(run, 'load_inputs', lambda: ({}, list(range(64)), None, allocated, initials))
    monkeypatch.setattr(run, 'action_episode', lambda *_: (env, truth))
    monkeypatch.setattr(audit, 'action_episode', lambda *_: (env, truth))
    summary = run.admit('artificial-fixture')
    assert all(s['attempts'] == 776 for s in summary.values())
    assert audit.audit('artificial-fixture') == 'PASS_registered_trace_rng_literal_density_and_source_replay'
    result = json.loads((run.OUT/'audit.json').read_text())
    assert result['cells'] == 194 and result['attempts'] == 1552
    assert result['maximum_independent_dense_joint_log_delta'] <= 1e-9
