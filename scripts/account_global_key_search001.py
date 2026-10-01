"""All-case fit-only publication accounting; no transfer or answer access."""
import json
import math
import signal
import time
from dataclasses import asdict

from scripts.global_key_read001_common import terminal_inventory
from scripts.run_global_key_search001 import OUT, ROOT, PATHS, NAMES, admitted, config_for
from scripts.run_blind_channel_dev001 import checked_artifact, require_frozen
from scripts.run_blind_channel_dev004 import load_archive, resource_report, save_new
from scripts.run_latin_source_model001 import artifact, limit_resources


def account():
    limit_resources(600, 500)
    wall, cpu = time.monotonic(), time.process_time()
    campaign_path = OUT/'campaign.json'
    campaign = json.loads(campaign_path.read_text())
    started = json.loads((OUT/'campaign-started.json').read_text())
    terminal_inventory(campaign, started)
    require_frozen(campaign['freeze'], PATHS)
    panel, _ = admitted(campaign['freeze'])
    assert [r['case'] for r in campaign['results']] == list(NAMES)
    reports, identities = {}, 0
    for process in campaign['results']:
        name = process['case']
        assert json.loads((OUT/f'{name}-process.json').read_text()) == process
        if process['fit_returncode'] != 0 or process['audit_returncode'] != 0:
            reports[name] = {'process': process, 'status': 'failure_retained'}
            continue
        path = OUT/f'{name}.json'
        fit = json.loads(path.read_text())
        audit_path = OUT/f'{name}-audit.json'
        audit = json.loads(audit_path.read_text())
        assert fit['status'] == 'complete_fit_only' and fit['case'] == name and fit['freeze'] == campaign['freeze']
        assert fit['fit'] == panel['cases'][name]['fit'] and not fit['transfer_or_answer_files_opened']
        assert audit['status'] == 'PASS' and audit['case'] == name and not audit['transfer_or_answer_files_opened']
        assert audit['result'] == artifact(path) and audit['auditor'] == artifact(ROOT/'scripts/audit_global_key_search001.py')
        assert fit['config'] == json.loads(json.dumps(asdict(config_for(name))))
        assert fit['graph_caps'] == {'nodes': 10_000_000, 'edges': 40_000_000}
        for field in ('fit', 'old_fit', 'old_bank', 'parent', 'warm', 'search', 'reader_bank', 'progress'):
            assert artifact(ROOT/fit[field]['path']) == fit[field]
            identities += 1
        old_fit = checked_artifact(fit['old_fit'])
        assert old_fit['bank'] == fit['old_bank'] and old_fit['parent'] == fit['parent']
        bank, previous = load_archive(fit['reader_bank']), load_archive(fit['old_bank'])
        assert bank['whole_old_bank_preserved'] and not bank['visit_counts_used_as_weights']
        assert not bank['global_optimality_claimed'] and bank['old_bank_size'] == len(previous['bank'])
        assert len(bank['bank']) == bank['bank_size'] and fit['reader_summary'] == {k: v for k, v in bank.items() if k != 'bank'}
        assert len({tuple(row['units']) for row in bank['bank']}) == bank['bank_size']
        for i, row in enumerate(bank['bank']):
            assert row['index'] == i and len(row['units']) == 23
            assert all(1 <= len(u) <= 2 and not set(u)-set('ABCDEF') for u in row['units'])
        for row, old in zip(bank['bank'][:len(previous['bank'])], previous['bank'], strict=True):
            assert row['units'] == old['units'] and row['model_bits'] == old['model_bits']
            assert row['log_weight_unnormalized'] == old['log_weight_unnormalized']
        finite = [row for row in bank['bank'] if row['log_weight'] is not None]
        assert len(finite) == bank['finite_keys']
        high = max(row['log_weight_unnormalized'] for row in finite)
        total = high+math.log(math.fsum(math.exp(row['log_weight_unnormalized']-high) for row in finite))
        assert all(abs(row['log_weight']-(row['log_weight_unnormalized']-total)) <= 1e-7 for row in finite)
        best = max((i for i, row in enumerate(bank['bank']) if row['log_weight'] is not None), key=lambda i: bank['bank'][i]['log_weight_unnormalized'])
        assert bank['best_index'] == best and bank['best_units'] == bank['bank'][best]['units']
        warm = checked_artifact(fit['warm'])
        best_score = bank['bank'][best]['log_weight_unnormalized']
        previous_best = previous['bank'][previous['best_index']]['log_weight_unnormalized']
        best_warm = max(row['log_weight_unnormalized'] for row in warm['scores'])
        assert abs(fit['fit_objective_gain_nats']-(best_score-previous_best)) <= 1e-7
        assert abs(fit['gain_vs_best_warm_nats']-(best_score-best_warm)) <= 1e-7
        assert abs(fit['search_gain_vs_best_warm_nats']-(fit['search_summary']['best_score']-best_warm)) <= 1e-7
        assert best_score+1e-7 >= fit['search_summary']['best_score']
        work = fit['unique_full_fit_scores_including_warm_and_neighborhood']
        assert audit['full_fit_scores_accounted'] == work <= 151214
        assert audit['reference_records'] == 4*audit['reference_keys'] <= 40 and audit['maximum_delta'] <= 1e-7
        for resource in (fit['resources'], audit['resources']):
            assert resource['peak_rss_bytes'] <= 4*1024**3
        reports[name] = {'status': 'complete_fit_audit', 'fit': artifact(path), 'audit': artifact(audit_path),
            'new_keys_scored': work, 'new_native_record_marginals': 4*work,
            'reader_bank_size': bank['bank_size'], 'reader_finite_keys': bank['finite_keys'],
            'whole_old_bank_size': len(previous['bank']),
            'gain_vs_old_bits': fit['fit_objective_gain_nats']/math.log(2),
            'gain_vs_best_warm_bits': fit['gain_vs_best_warm_nats']/math.log(2),
            'sampler_gain_vs_best_warm_bits': fit['search_gain_vs_best_warm_nats']/math.log(2),
            'neighborhood_gain_bits': (best_score-fit['search_summary']['best_score'])/math.log(2),
            'stop_reason': fit['search_summary']['stop_reason'],
            'proposals': fit['search_summary']['proposals'], 'exchanges': fit['search_summary']['exchange_counts'],
            'reference_records': audit['reference_records'], 'maximum_reference_delta': audit['maximum_delta'],
            'fit_resources': fit['resources'], 'audit_resources': audit['resources']}
        if resource_report(wall, cpu)['peak_rss_bytes'] > 4*1024**3:
            raise MemoryError('Fixed all-case accounting RSS cap')
    complete = [r for r in reports.values() if r['status'] == 'complete_fit_audit']
    result = {'status': 'PASS' if len(complete) == 32 else 'FAIL_incomplete_fits_retained',
        'freeze': campaign['freeze'], 'campaign': artifact(campaign_path), 'cases': reports,
        'all_32_terminal_processes_accounted': True, 'input_archive_bindings': identities,
        'completed_cases': len(complete), 'failed_cases': 32-len(complete),
        'new_case_key_scores': sum(r['new_keys_scored'] for r in complete),
        'new_native_record_marginals': sum(r['new_native_record_marginals'] for r in complete),
        'reader_casewise_keys': sum(r['reader_bank_size'] for r in complete),
        'reader_casewise_finite_keys': sum(r['reader_finite_keys'] for r in complete),
        'reference_records': sum(r['reference_records'] for r in complete),
        'maximum_reference_delta': max((r['maximum_reference_delta'] for r in complete), default=0.),
        'fit_summed_wall_seconds': sum(r['fit_resources']['wall_seconds'] for r in complete),
        'fit_summed_cpu_seconds': sum(r['fit_resources']['cpu_seconds'] for r in complete),
        'audit_summed_wall_seconds': sum(r['audit_resources']['wall_seconds'] for r in complete),
        'audit_summed_cpu_seconds': sum(r['audit_resources']['cpu_seconds'] for r in complete),
        'no_transfer_or_answer_access': True, 'no_recovery_or_decipherment_claim': True,
        'same_author_alternate_accounting_not_agent_review': True,
        'audit_code': artifact(ROOT/'scripts/account_global_key_search001.py'), 'resources': resource_report(wall, cpu)}
    signal.alarm(0)
    save_new(OUT/'fit-accounting.json', result)
    print(json.dumps({k: v for k, v in result.items() if k not in ('cases', 'audit_code', 'campaign')}), flush=True)


if __name__ == '__main__':
    account()
