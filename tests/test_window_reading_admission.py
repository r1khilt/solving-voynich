"""Original23row/42unit shape using an entirely artificial contextual source."""
from copy import deepcopy
import gzip
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import run_window_reading_admit001 as run
from scripts import audit_window_reading_admit001 as auditor
from scripts.audit_window_reading_admit001 import audit_table
from voynich.reading_regrowth import SourceRegrowth
from voynich.source_action_proposal import ReadingEnvironment


def fixture():
    p = np.array([np.roll(np.arange(1, 24), i) for i in range(23)], dtype=np.float64)/276
    t = np.array([[(i+j+1) % 23 for j in range(23)] for i in range(23)], dtype=np.uint32)
    p.flags.writeable = t.flags.writeable = False
    source = SimpleNamespace(probabilities=p, transitions=t, state=lambda _: 0)
    env = ReadingEnvironment(((0, 1, 0, 0), (1, 0, 1)), rows=23, glyphs=6)
    sampler = SourceRegrowth(source, env)
    state, actions = env.initial, []
    while (r := env.selected_record(state)) is not None:
        action = 2*env.records[r][state.offsets[r]]
        actions.append(action)
        state = env.advance(state, action)
    return source, sampler, sampler.path(forced_actions=tuple(actions))


def test_original_shape_all_family_targets_rng_replay_and_corruption():
    source, sampler, base = fixture()
    ranks = run.indices(sum(2*len(r)-1 for r in sampler.env.records))
    assert len(ranks) == 4
    for j, rank in enumerate(ranks):
        table = run.table(sampler, base, rank, 96401+j)
        checked = audit_table(source, sampler, base, table, 96401+j)
        assert checked['valid_windows'] and checked['candidates'] == len(table['candidates'])
        corrupted = deepcopy(table)
        corrupted['candidates'][0]['ratio_hex'][0] = '0xdeadbeef'
        with pytest.raises(AssertionError):
            audit_table(source, sampler, base, corrupted, 96401+j)


def test_selected_window_quantiles_and_invalid_boundary_identity():
    assert run.indices(1) == (0,) and run.indices(2) == (0, 1)
    assert run.indices(12) == (0, 3, 6, 9)
    for count in (False, 0, -1):
        with pytest.raises(ValueError):
            run.indices(count)
    source, sampler, _ = fixture()
    # Bound row2 emits the first observed pair; its interior endpoint is invalid.
    full_key = (0, 1, sampler.env.pool.index((0, 1)))+(0,)*20
    _, actions, _ = sampler.env.teaching_trace(((2, 0, 0), (1, 0, 1)), full_key)
    base = sampler.path(forced_actions=actions)
    table = run.table(sampler, base, 0, 96409)
    assert not table['valid_endpoints'] and not table['candidates'] and table['selected_index'] is None
    assert audit_table(source, sampler, base, table, 96409) == {'valid_windows': 0, 'candidates': 0, 'changed_draws': 0}


def test_full_artificial_controller_audit_receipts_exclusivity_and_reporting_failure(tmp_path, monkeypatch):
    source, sampler, base = fixture()

    def artifact(path):
        raw = path.read_bytes()
        return {'path': str(path.relative_to(tmp_path)), 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}

    def save(path, value, compressed=False):
        path.parent.mkdir(parents=True, exist_ok=True)
        raw = json.dumps(value, sort_keys=True, indent=2).encode()
        if compressed:
            raw = gzip.compress(raw, mtime=0)
        with path.open('xb') as out:
            out.write(raw)
        return artifact(path)

    def load(spec):
        path = tmp_path/spec['path']
        assert artifact(path) == spec
        return json.loads(gzip.decompress(path.read_bytes()))

    monkeypatch.setattr(run, 'ROOT', tmp_path)
    monkeypatch.setattr(run, 'OUT', tmp_path/'results'/'test-admission')
    monkeypatch.setattr(run, 'BULK', tmp_path/'outputs'/'test-admission')
    monkeypatch.setattr(run, 'PATHS', ('fixture.txt',))
    monkeypatch.setattr(run, 'CASES', 1)
    monkeypatch.setattr(run.training, 'save_new', save)
    monkeypatch.setattr(run.training, 'artifact', artifact)
    monkeypatch.setattr(run.training, 'limit_resources', lambda *_: None)
    monkeypatch.setattr(run.training.old, 'require_frozen', lambda *_: None)
    monkeypatch.setattr(run.prior, 'load_source', lambda: (source, {}))
    monkeypatch.setattr(run.prior, 'load_archive', load)
    (tmp_path/'fixture.txt').write_text('Entirely artificial source and observations.\n')
    frames = []
    for phase in run.PHASES:
        spec = save(tmp_path/'inputs'/f'{phase}.json.gz', {'records': list(map(list, sampler.env.records)),
            'base': run.prior.prior.reading_receipt(base)}, compressed=True)
        frames.append({'case': 0, 'kind': 'artificial', 'phase': phase, 'archive': spec})
    previous = {'source_arrays': run.prior.array_identity(source), 'cells': frames}
    monkeypatch.setattr(run, 'require_prior', lambda: previous)
    assert all(run.admission('fixture').values())
    totals = auditor.audit('fixture')
    assert totals['valid_windows'] == 8 and totals['candidates'] > 8
    with pytest.raises(FileExistsError):
        run.admission('fixture')
    with pytest.raises(FileExistsError):
        auditor.audit('fixture')
    result = json.loads((run.OUT/'result.json').read_text())
    result['candidates'] += 1
    bad = tmp_path/'results'/'bad-report'
    save(bad/'result.json', result)
    monkeypatch.setattr(run, 'OUT', bad)
    with pytest.raises(AssertionError):
        auditor.audit('fixture')
    assert (bad/'audit-failure.json').exists() and not (bad/'audit.json').exists()
