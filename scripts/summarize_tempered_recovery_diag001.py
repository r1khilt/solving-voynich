"""Close publication only after BOTH unchanged empirical replay audits pass."""

import argparse
import json
import subprocess

from scripts import run_tempered_recovery_diag001 as run


def summarize_closed():
    parent = run.parent
    for directory in (run.OUT, parent.OUT):
        assert not (directory/'failure.json').exists()
        assert not (directory/'audit-failure.json').exists()
    result = json.loads((run.OUT/'result.json').read_text())
    audit = json.loads((run.OUT/'audit.json').read_text())
    original = json.loads((parent.OUT/'result.json').read_text())
    original_audit = json.loads((parent.OUT/'audit.json').read_text())
    post = json.loads((parent.OUT/'post-outcome.json').read_text())
    earlier = json.loads((run.OUT/'closed-check001.json').read_text())
    parent.require_frozen(result['freeze'], run.PATHS)
    parent.require_frozen(original['freeze'], parent.PATHS)
    assert audit['status'] == 'PASS_full_known_answer_diagnostic_replay'
    assert original_audit['status'] == 'PASS_full_fresh_tempered_recovery_replay'
    assert post['status'] == 'PASS_closed_tempered_recovery_publication'
    for value in (audit, earlier):
        assert value['result'] == parent.artifact(run.OUT/'result.json')
    assert earlier['diagnostic_audit'] == parent.artifact(run.OUT/'audit.json')
    assert earlier['all32bank_inventory_and_mass_arithmetic_checked']
    assert earlier['status'] == 'PASS_diagnostic_artifacts_and_independent_arithmetic_parent_audit_pending'
    for value in (original_audit, post):
        assert value['result'] == parent.artifact(parent.OUT/'result.json')
    assert post['audit'] == parent.artifact(parent.OUT/'audit.json')
    assert result['parent_result'] == earlier['parent_result'] == parent.artifact(parent.OUT/'result.json')
    assert result['parent_prediction_seal'] == original['prediction_seal']
    assert result['source'] == original['source'] and result['source_arrays'] == original['source_arrays']
    assert result['native_build'] == original['native_build']
    assert parent.artifact(run.ROOT/result['analysis']['path']) == result['analysis']
    subprocess.run(['git', 'check-ignore', '--quiet', result['analysis']['path']],
                   cwd=run.ROOT, check=True)
    assert len(result['banks']) == audit['positive_bank_cases'] == 32
    assert len(result['oracles']) == audit['generating_key_cases'] == 4
    assert audit['independent_known_key_source_checks'] == 4 and audit['maximum_known_key_source_log_delta'] <= 1e-7
    assert original_audit['calls'] == 64 and original_audit['literal_final_key_checks'] == 6784
    assert original_audit['independent_fixed_final_key_source_checks'] == 424
    assert not original['summary']['combined_qualification_gate']
    for resource in (result['resources'], audit['resources']):
        assert resource['wall_seconds'] <= run.WALL and resource['cpu_seconds'] <= run.CPU
        assert resource['peak_rss_bytes'] <= 2*1024**3
    return {'status': 'PASS_joint_search_and_known_answer_diagnostic_publication',
        'result': parent.artifact(run.OUT/'result.json'), 'audit': parent.artifact(run.OUT/'audit.json'),
        'parent_result': parent.artifact(parent.OUT/'result.json'),
        'parent_audit': parent.artifact(parent.OUT/'audit.json'),
        'parent_post_outcome': parent.artifact(parent.OUT/'post-outcome.json'),
        'earlier_pending_publication_check': parent.artifact(run.OUT/'closed-check001.json'),
        'frozen_diagnostic_paths': len(run.PATHS), 'frozen_parent_paths': len(parent.PATHS),
        'both_original_replays_pass': True, 'original_recovery_gate_still_fail': True,
        'actual_corpus_compiler_sampler_or_source_calls': 0, 'gold_contents_opened': False,
        'analysis_contents_opened': False, 'paid_spend_usd': 0,
        'historical_or_neural_or_inverse_qualification': False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save', action='store_true')
    args = parser.parse_args()
    result = summarize_closed()
    if args.save:
        run.parent.save_new(run.OUT/'post-outcome.json', result)
    print(json.dumps(result, indent=2))
