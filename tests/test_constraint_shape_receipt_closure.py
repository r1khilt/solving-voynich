"""Preserved receipts only: no mathematical panel reenumeration or source scoring."""
from copy import deepcopy
import hashlib
import json

import pytest

from scripts import close_constraint_shape001_receipt as close


def receipts():
    origin = close.original.ROOT/'results'/close.original.EXP
    return (origin/'failure.json').read_bytes(), (origin/'started.json').read_bytes()


def semantic_check(monkeypatch, value, start):
    raw = json.dumps(value, sort_keys=True).encode()
    monkeypatch.setattr(close, 'FAILURE_SHA', hashlib.sha256(raw).hexdigest())
    return close.validate_saved(raw, start)


def test_saved_all_panels_absent_00_zeros_and_corrected_010_dimensions():
    failure, start = receipts()
    checked = close.validate_saved(failure, start)
    assert checked['totals']['enumerated_visited_dictionary_assignments'] == 24000
    assert checked['totals']['exact_exchange_flux_checks'] == 66780
    assert len(checked['restored_zero_counters']) == 8
    assert {r['panel_index'] for r in checked['restored_zero_counters']} == {1,5,9,13}
    original = json.loads(failure)
    for index in (1,5,9,13):
        assert not close.ZEROS & original['panels'][index].keys()
        assert all(checked['panels'][index][field] == 0 for field in close.ZEROS)
    assert close.dimensions(3, [[0,1,0]]) == (45,24,2430)
    assert close.dimensions(4, [[0,1,0]]) == (96,60,7416)


@pytest.mark.parametrize('corruption', ('missing_nonzero', 'extra_missing', 'wrong_count',
    'wrong_010', 'wrong_panel', 'wrong_error', 'wrong_type', 'record_type', 'resource', 'extra_metadata'))
def test_semantic_corruption_rejected_even_with_fixture_hash(monkeypatch, corruption):
    failure, start = receipts()
    value = deepcopy(json.loads(failure))
    if corruption == 'missing_nonzero':
        del value['panels'][0]['hard_incompatible_exchange_rejections']
    elif corruption == 'extra_missing':
        del value['panels'][1]['states']
    elif corruption == 'wrong_count':
        value['panels'][0]['wrong_proposal_flux_failures'] += 1
    elif corruption == 'wrong_010':
        value['panels'][3]['enumerated_visited_dictionary_assignments'] += 36
    elif corruption == 'wrong_panel':
        value['panels'][1], value['panels'][3] = value['panels'][3], value['panels'][1]
    elif corruption == 'wrong_error':
        value['error'] = "AssertionError('failed mathematical law')"
    elif corruption == 'wrong_type':
        value['panels'][0]['hard_states'] = True
    elif corruption == 'record_type':
        value['panels'][0]['records'][0][0] = False
    elif corruption == 'resource':
        value['resources']['cpu_seconds'] = close.original.CPU+1
    else:
        value['undisclosed_new_gate'] = True
    with pytest.raises(AssertionError):
        semantic_check(monkeypatch, value, start)


def test_original_hash_and_start_freeze_metadata_corruption(monkeypatch):
    failure, start = receipts()
    for bad_failure, bad_start in ((failure+b' ',start), (failure,start+b' ')):
        with pytest.raises(AssertionError):
            close.validate_saved(bad_failure, bad_start)
    value = json.loads(start)
    value['freeze'] = 'not_the_original_freeze'
    raw = json.dumps(value).encode()
    monkeypatch.setattr(close, 'START_SHA', hashlib.sha256(raw).hexdigest())
    with pytest.raises(AssertionError):
        close.validate_saved(failure, raw)


def test_temporary_complete_closure_io_freezes_resources_and_exclusivity(tmp_path, monkeypatch):
    failure, start = receipts()
    monkeypatch.setattr(close, 'ROOT', tmp_path)
    monkeypatch.setattr(close, 'PATHS', ('fixture.txt',))
    calls = []
    monkeypatch.setattr(close.training.old, 'require_frozen', lambda freeze, paths: calls.append((freeze, paths)))
    monkeypatch.setattr(close.training, 'limit_resources', lambda *_: None)
    original_resource = json.loads(failure)['resources']
    resource = {**original_resource, 'wall_seconds': .1, 'cpu_seconds': .1, 'peak_rss_bytes': 1024}
    monkeypatch.setattr(close.training, 'resource_report', lambda *_: resource)
    def artifact(path):
        raw = path.read_bytes()
        return {'path': str(path.relative_to(tmp_path)), 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}
    def save(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('xb') as handle:
            handle.write(json.dumps(value, sort_keys=True).encode())
        return artifact(path)
    monkeypatch.setattr(close.training, 'artifact', artifact)
    monkeypatch.setattr(close.training, 'save_new', save)
    (tmp_path/'fixture.txt').write_text('Temporary receipt-only fixture, no scientific rerun.\n')
    origin = tmp_path/'results'/close.original.EXP
    origin.mkdir(parents=True)
    (origin/'failure.json').write_bytes(failure)
    (origin/'started.json').write_bytes(start)
    assert close.close('temporary_closure_fixture') == 'PASS_saved_shape_panel_receipt_closure_original_run_failed'
    out = tmp_path/'results'/close.EXP
    value = json.loads((out/'closure.json').read_text())
    assert value['original_status'] == 'FAIL' and value['no_sampling_reenumeration_source_scoring_or_training']
    assert value['original_failure'] == artifact(origin/'failure.json')
    assert value['inputs'] == [artifact(tmp_path/'fixture.txt')]
    assert calls == [('temporary_closure_fixture', close.PATHS), (close.ORIGINAL_FREEZE, close.original.PATHS)]*2
    assert not (origin/'result.json').exists() and (origin/'failure.json').read_bytes() == failure
    with pytest.raises(FileExistsError):
        close.close('temporary_closure_fixture')
