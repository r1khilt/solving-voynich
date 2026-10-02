"""Publication closure cannot waive or misbind either empirical audit."""

import hashlib
import json

import pytest

from scripts import summarize_tempered_recovery_diag001 as post


@pytest.fixture
def publication(tmp_path, monkeypatch):
    run, parent = post.run, post.run.parent
    out, original = tmp_path/'diagnostic', tmp_path/'parent'
    out.mkdir()
    original.mkdir()
    monkeypatch.setattr(run, 'ROOT', tmp_path)
    monkeypatch.setattr(run, 'OUT', out)
    monkeypatch.setattr(parent, 'OUT', original)
    monkeypatch.setattr(run, 'PATHS', ('diagnostic-source',))
    monkeypatch.setattr(parent, 'PATHS', ('parent-source',))
    checked = []
    monkeypatch.setattr(parent, 'require_frozen', lambda freeze, paths: checked.append((freeze, paths)))
    def artifact(path):
        value = path.read_bytes()
        return {'path': str(path.relative_to(tmp_path)), 'bytes': len(value), 'sha256': hashlib.sha256(value).hexdigest()}
    monkeypatch.setattr(parent, 'artifact', artifact)
    monkeypatch.setattr(post.subprocess, 'run', lambda *args, **kwargs: None)
    def save(path, value):
        path.write_text(json.dumps(value))
    resources = {'wall_seconds': 1., 'cpu_seconds': 1., 'peak_rss_bytes': 1024}
    (tmp_path/'analysis.gz').write_bytes(b'private-fixture-never-read')
    shared = {'source': {}, 'source_arrays': {}, 'native_build': {}}
    save(original/'result.json', {'freeze': 'parent-freeze', **shared, 'prediction_seal': {},
                                  'summary': {'combined_qualification_gate': False}})
    save(original/'audit.json', {'status': 'PASS_full_fresh_tempered_recovery_replay',
        'result': artifact(original/'result.json'), 'calls': 64, 'literal_final_key_checks': 6784,
        'independent_fixed_final_key_source_checks': 424})
    save(original/'post-outcome.json', {'status': 'PASS_closed_tempered_recovery_publication',
        'result': artifact(original/'result.json'), 'audit': artifact(original/'audit.json')})
    save(out/'result.json', {'freeze': 'diagnostic-freeze', **shared,
        'parent_result': artifact(original/'result.json'), 'parent_prediction_seal': {},
        'analysis': artifact(tmp_path/'analysis.gz'), 'banks': [{}]*32, 'oracles': [{}]*4, 'resources': resources})
    save(out/'audit.json', {'status': 'PASS_full_known_answer_diagnostic_replay',
        'result': artifact(out/'result.json'), 'positive_bank_cases': 32, 'generating_key_cases': 4,
        'independent_known_key_source_checks': 4, 'maximum_known_key_source_log_delta': 1e-12, 'resources': resources})
    save(out/'closed-check001.json', {'status': 'PASS_diagnostic_artifacts_and_independent_arithmetic_parent_audit_pending',
        'result': artifact(out/'result.json'), 'diagnostic_audit': artifact(out/'audit.json'),
        'parent_result': artifact(original/'result.json'), 'all32bank_inventory_and_mass_arithmetic_checked': True})
    return out, original, checked, save


def test_requires_both_bound_audits_and_freezes(publication):
    _, _, checked, _ = publication
    value = post.summarize_closed()
    assert value['both_original_replays_pass'] and value['original_recovery_gate_still_fail']
    assert value['actual_corpus_compiler_sampler_or_source_calls'] == 0
    assert checked == [('diagnostic-freeze', ('diagnostic-source',)), ('parent-freeze', ('parent-source',))]


@pytest.mark.parametrize('directory', ['diagnostic', 'parent'])
def test_missing_or_failed_audit_never_qualifies(publication, directory):
    out, parent, _, save = publication
    path = (out if directory == 'diagnostic' else parent)/'audit.json'
    value = json.loads(path.read_text())
    value['status'] = 'FAIL'
    save(path, value)
    with pytest.raises(AssertionError):
        post.summarize_closed()
    path.unlink()
    with pytest.raises(FileNotFoundError):
        post.summarize_closed()


def test_modified_result_or_private_archive_refused(publication):
    out, _, _, save = publication
    path = out/'result.json'
    value = json.loads(path.read_text())
    save(path, {**value, 'changed_after_audit': True})
    with pytest.raises(AssertionError):
        post.summarize_closed()
    save(path, value)
    (post.run.ROOT/'analysis.gz').write_bytes(b'modified')
    with pytest.raises(AssertionError):
        post.summarize_closed()


def test_failure_marker_blocks_publication(publication):
    out, _, _, _ = publication
    (out/'audit-failure.json').write_text('{}')
    with pytest.raises(AssertionError):
        post.summarize_closed()
