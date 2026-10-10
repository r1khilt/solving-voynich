"""Small artificial preparation panel and exclusive controller/closure schemas."""
import hashlib
import json
from decimal import localcontext

import pytest

from scripts import check_tempered_reading001 as run
from scripts.audit_tempered_reading001 import audit, validate


def test_one_artificial_finite_panel_and_product_composition():
    with localcontext() as context:
        context.prec = 100
        panel = run.finite_panel(3, True, ((0, 1), (0, 1)))
    assert panel['states'] == 39 and panel['zero_eligibility_states'] > 0
    assert panel['wrong_powered_selector_flux_failures'] > 0
    assert panel['wrong_powered_suffix_flux_failures'] > 0
    assert panel['wrong_warm_prior_flux_failures'] > 0
    assert panel['reversed_swap_flux_failures'] > 0
    assert run.product_sweep_panel() == {'product_states': 27, 'exact_product_stationarity_equations': 54}


def test_controller_schema_exclusivity_and_corruption(tmp_path, monkeypatch):
    monkeypatch.setattr(run, 'ROOT', tmp_path)
    monkeypatch.setattr(run, 'require_frozen', lambda *_: None)
    monkeypatch.setattr(run, 'limit_resources', lambda *_: None)
    monkeypatch.setattr(run, 'resource_report', lambda *_: {'wall_seconds': .1, 'cpu_seconds': .1,
                                                         'peak_rss_bytes': 1024, 'paid_spend_usd': 0})
    for name in run.PATHS:
        path = tmp_path/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(name)

    def artifact(path):
        return {'path': str(path.relative_to(tmp_path)), 'bytes': path.stat().st_size,
                'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}

    def save_new(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('x') as stream:
            json.dump(value, stream)
        return artifact(path)

    def artificial(rows, context, records):
        return dict(rows=rows, contextual=context, records=list(map(list, records)), degrees=list(run.DEGREES),
            states=1, fast_score_reference_states=1, structural_components=1, label_components=1,
            regrowth_complete_components=1, failed_suffix_components=1, exact_power_flux_checks=3,
            wrong_powered_selector_flux_failures=1, wrong_powered_suffix_flux_failures=1,
            wrong_warm_prior_flux_failures=1, replica_exchange_checks=3, reversed_swap_flux_failures=1,
            zero_eligibility_states=1, kernel_stationarity_equations=16, maximum_decimal_residual='0')

    monkeypatch.setattr(run, 'artifact', artifact)
    monkeypatch.setattr(run, 'save_new', save_new)
    monkeypatch.setattr(run, 'finite_panel', artificial)
    monkeypatch.setattr(run, 'qualification_decisions', lambda: {'independent_radical_decisions': 6144,
        'two_block_boundary_and_abort_witnesses': 16})
    monkeypatch.setattr(run, 'product_sweep_panel', lambda: {'product_states': 27,
        'exact_product_stationarity_equations': 54})
    assert run.check('artificial-only')['exact_power_flux_checks'] == 24
    assert audit('artificial-only') == 'PASS_receipt_hash_and_arithmetic_closure'
    out = tmp_path/'results'/run.EXP
    result = json.loads((out/'result.json').read_text())
    result['panels'][0]['exact_power_flux_checks'] += 1
    with pytest.raises(AssertionError):
        validate(result, 'artificial-only')
    with pytest.raises(FileExistsError):
        run.check('artificial-only')
    with pytest.raises(FileExistsError):
        audit('artificial-only')
