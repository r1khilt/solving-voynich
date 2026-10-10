"""Artificial qualification/closure allocation, corruption and exclusive outputs."""
from copy import deepcopy
import json

import pytest

from scripts import check_constraint_shape001 as run
from scripts import audit_constraint_shape001 as closure


def test_artificial_shape_qualification_controller_closure_and_corruption(tmp_path, monkeypatch):
    monkeypatch.setattr(run, 'ROOT', tmp_path)
    monkeypatch.setattr(run, 'PATHS', ('fixture.txt',))
    monkeypatch.setattr(run, 'PANELS', ((3, False, ((0, 1),)),))
    monkeypatch.setattr(run.training.old, 'require_frozen', lambda *_: None)
    monkeypatch.setattr(run.training, 'limit_resources', lambda *_: None)
    monkeypatch.setattr(run.training, 'resource_report', lambda *_: {
        'wall_seconds': .1, 'cpu_seconds': .1, 'peak_rss_bytes': 1024, 'paid_spend_usd': 0})
    (tmp_path/'fixture.txt').write_text('Entirely artificial source, shape and dictionary space.\n')
    # Original artifact implementation has a different ROOT; adapt only the IO root.
    import hashlib
    def artifact(path):
        raw = path.read_bytes()
        return {'path': str(path.relative_to(tmp_path)), 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}
    def save(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('xb') as stream:
            stream.write(json.dumps(value, sort_keys=True).encode())
        return artifact(path)
    monkeypatch.setattr(run.training, 'artifact', artifact)
    monkeypatch.setattr(run.training, 'save_new', save)
    old = tmp_path/'inputs'/'prior-source'
    save(old/'result.json', {'status': 'PASS_constraint_dispatch_original_source_cost_and_cold_conformance'})
    save(old/'audit.json', {'status': 'PASS_full_constraint_dispatch_rng_source_and_cold_reference_replay',
                           'result': artifact(old/'result.json')})
    monkeypatch.setattr(run.prior, 'OUT', old)
    monkeypatch.setattr(run.prior, 'require_prior', lambda: None)
    totals = run.check('artificial-only')
    assert totals['states'] == 12 and totals['enumerated_visited_dictionary_assignments'] == 252
    assert totals['exact_exchange_flux_checks'] == 396
    assert closure.audit('artificial-only') == 'PASS_collapsed_shape_receipt_hash_and_arithmetic_closure'
    result = json.loads((tmp_path/'results'/run.EXP/'result.json').read_text())
    bad = deepcopy(result)
    bad['panels'][0]['summed_dictionary_terms'] += 1
    bad['totals']['summed_dictionary_terms'] += 1
    with pytest.raises(AssertionError):
        closure.validate(bad, 'artificial-only')
    with pytest.raises(FileExistsError):
        run.check('artificial-only')
    with pytest.raises(FileExistsError):
        closure.audit('artificial-only')
