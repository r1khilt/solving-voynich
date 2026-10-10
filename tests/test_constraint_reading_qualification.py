"""Artificial prospective checker/closure schema, arithmetic corruption and exclusivity."""
from copy import deepcopy
import hashlib
import json

import pytest

from scripts import check_constraint_reading001 as run
from scripts import audit_constraint_reading001 as closure


def test_full_artificial_qualification_controller_closure_and_corruption(tmp_path, monkeypatch):
    monkeypatch.setattr(run, 'ROOT', tmp_path)
    monkeypatch.setattr(run, 'PATHS', ('fixture.txt',))
    monkeypatch.setattr(run, 'PANELS', ((3, False, ((0, 1),)),))
    monkeypatch.setattr(run, 'require_frozen', lambda *_: None)
    monkeypatch.setattr(run, 'limit_resources', lambda *_: None)
    monkeypatch.setattr(run, 'resource_report', lambda *_: {
        'wall_seconds': .1, 'cpu_seconds': .1, 'peak_rss_bytes': 1024, 'paid_spend_usd': 0})
    (tmp_path/'fixture.txt').write_text('Entirely artificial source, states and observations.\n')
    def artifact(path):
        data = path.read_bytes()
        return {'path': str(path.relative_to(tmp_path)), 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}
    def save(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('xb') as stream:
            stream.write(json.dumps(value, sort_keys=True).encode())
        return artifact(path)
    monkeypatch.setattr(run, 'artifact', artifact)
    monkeypatch.setattr(run, 'save_new', save)
    totals = run.check('artificial-only')
    assert totals['states'] == 252 and totals['hard_states'] == 9
    assert totals['exact_exchange_flux_checks'] == 129276
    assert totals['wrong_prior_flux_failures'] > 0 and totals['wrong_uncorrected_proposal_flux_failures'] > 0
    assert closure.audit('artificial-only') == 'PASS_constraint_receipt_hash_and_arithmetic_closure'
    result = json.loads((tmp_path/'results'/run.EXP/'result.json').read_text())
    bad = deepcopy(result)
    bad['panels'][0]['independent_target_checks'] += 1
    bad['totals']['independent_target_checks'] += 1
    with pytest.raises(AssertionError):
        closure.validate(bad, 'artificial-only')
    with pytest.raises(FileExistsError):
        run.check('artificial-only')
    with pytest.raises(FileExistsError):
        closure.audit('artificial-only')
