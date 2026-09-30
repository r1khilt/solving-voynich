"""Independent fit inventories and no-gold prediction audits for all fresh cases."""
import argparse
import gzip
import json
import math
import signal
import time

from scripts import audit_blind_channel_dev004 as initial_reference
from scripts import summarize_blind_channel_dev003_search as search_audit
from scripts.audit_key_bank_expand001 import inventory
from scripts.audit_key_bank_read001 import audit_reading, close, logvalue
from scripts.audit_key_bank_read002 import audit_evidence
from scripts.audit_suffix_reader001 import infer_reference, reading_score
from scripts.confirm002_common import BULK, NAMES, OUT, PATHS, ROOT, fit_inputs, fit_status, observed, verify
from scripts.run_blind_channel_confirm001 import inference_rows, sources
from scripts.run_blind_channel_dev001 import checked_artifact, digest
from scripts.run_blind_channel_dev004 import load_archive, resource_report, save_new
from scripts.run_blind_channel_dev005 import trace_accounting
from scripts.run_latin_source_model001 import artifact, limit_resources
from voynich.compact_suffix_reference import LiteralCountView
from voynich.compact_suffix_source import CompactSuffixSource
from voynich.finite_state_channel_fit import CodingContext


def literal_source():
    selected = checked_artifact(artifact(ROOT / 'results/LATIN-SOURCE-COMPACT-001/large.json'))
    verify(selected['counts'])
    compact = CompactSuffixSource.load(ROOT/selected['counts']['path'])
    return {'alphabet': compact.alphabet, 'order': 12, 'tau': selected['selected']['tau'],
            'counts': LiteralCountView(compact)}


def audit_points(raw, records, parent, bank, points, rho):
    checks = 0
    for arm, units in (('original', parent['units']), ('fit_selected', bank['best_units'])):
        rows = points[arm]
        if len(rows) != len(records):
            raise ValueError('Incomplete point records')
        for record, row in zip(records, rows, strict=True):
            total, best, _ = infer_reference(raw, units, record, rho)
            close(logvalue(row['log_likelihood']), total)
            close(logvalue(row['joint_log_probability']), best)
            close(reading_score(raw, units, record, row['plaintext'], rho), best)
            checks += 1
    return checks


def audit_fit(name):
    if name not in NAMES:
        raise ValueError('Unregistered case')
    path = OUT/f'{name}-fit.json'
    result = json.loads(path.read_text())
    panel, _ = fit_inputs(result['data_freeze'])
    save_new(OUT/f'{name}-fit-audit-started.json', {'case': name, 'result': artifact(path), 'start_unix': time.time()})
    limit_resources(650, 600)
    wall, cpu = time.monotonic(), time.process_time()
    if (result['status'] != 'complete_fit_only' or result['fit'] != panel['cases'][name]['fit']
            or result['source_freeze'] != panel['source_freeze'] or result['case'] != name
            or result['source'] != artifact(ROOT/'results/LATIN-SOURCE-COMPACT-001/large.json')
            or result['native_benchmark'] != artifact(ROOT/'results/NATIVE-SUFFIX-001/result.json')):
        raise ValueError('Fit result/source identity mismatch')
    for field in ('fit', 'parent', 'source', 'native_benchmark', 'bank', 'progress'):
        verify(result[field])
    data, parent = observed(result['fit'], 4), checked_artifact(result['parent'])
    if (parent['case'] != name or parent['fit'] != result['fit'] or parent['data_freeze'] != result['data_freeze']
            or parent['source_freeze'] != result['source_freeze']):
        raise ValueError('Parent input identity mismatch')
    stage1, stage2 = load_archive(parent['stage1_trace']), load_archive(parent['stage2_trace'])
    compact1 = load_archive(parent['stage1_summary'])
    accounting1 = json.loads(json.dumps(search_audit.validate_case(compact1, stage1, data['context'])))
    _, _, models = sources()
    if (compact1['source_hashes'] != {p: digest((ROOT/p).read_bytes()) for p in PATHS}
            or compact1['source_sha256'] != digest(json.dumps(models['order1'], sort_keys=True).encode())
            or compact1['input_sha256'] != result['fit']['sha256'] or compact1['full_output'] != parent['stage1_trace']
            or compact1['case_id'] != name or compact1['source_freeze'] != panel['source_freeze']
            or stage1['config'] != {'seed': panel['cases'][name]['search_seed'], 'restarts': 16,
                'max_sweeps': 80, 'max_seconds': 300., 'batch_size': 256, 'max_units': 256, 'improvement_tolerance_bits': 1e-8}
            or stage2['config'] != {'max_sweeps': 20, 'max_seconds': 300., 'tolerance': 1e-8}
            or stage2['initial_units'] != stage1['units']):
        raise ValueError('Initial source hashes/configuration/binding mismatch')
    accounting2 = trace_accounting(stage2, models['order3'], CodingContext(**data['context']))
    if (parent['stage1_accounting'] != accounting1 or parent['stage2_accounting'] != accounting2
            or parent['stage1_units'] != stage1['units'] or parent['units'] != stage2['units']
            or parent['stage1_score'] != stage1['score'] or parent['score'] != stage2['score']
            or parent['frequency_units'] != stage1['trace'][0]['units']
            or stage1['config']['seed'] != panel['cases'][name]['search_seed']):
        raise ValueError('Initial trajectory/selection mismatch')
    initial_delta = 0.
    for label, value in (('order1', stage1), ('order3', stage2)):
        total = math.fsum(initial_reference.infer_record(models[label], value['units'], r,
                         data['context']['stop_probability'])['log_likelihood'] for r in data['records'])
        close(total, value['score']['log_likelihood'])
        initial_delta = max(initial_delta, abs(total-value['score']['log_likelihood']))
    bank = load_archive(result['bank'])
    if any(len(row['record_log_likelihoods']) != 4 for row in bank['bank']):
        raise ValueError('Incomplete fit record inventory')
    events = inventory(bank, parent['units'], data['context'])
    actual = [json.loads(line) for line in gzip.decompress((ROOT/result['progress']['path']).read_bytes()).splitlines()]
    if actual != events:
        raise ValueError('Expanded progress/candidate inventory differs')
    summary = {k: v for k, v in bank.items() if k not in ('bank', 'rounds')}
    summary['rounds'] = [{**{k: v for k, v in r.items() if k != 'member_indices'},
                          'member_count': len(r['member_indices'])} for r in bank['rounds']]
    if any(result[k] != v for k, v in summary.items()):
        raise ValueError('Expanded compact summary differs')
    raw = literal_source()
    indices = sorted({0, bank['best_index'], len(bank['bank'])//2, len(bank['bank'])-1,
                      *(r['selected_index'] for r in bank['rounds'])})
    count, delta = 0, 0.
    for index in indices:
        row = bank['bank'][index]
        for j, record in enumerate(data['records']):
            total, _, nodes = infer_reference(raw, row['units'], record, data['context']['stop_probability'])
            target = logvalue(row['record_log_likelihoods'][j])
            close(total, target)
            if math.isfinite(total):
                delta = max(delta, abs(total-target))
            if nodes != row['record_nodes'][j]:
                raise ValueError('Independent string-state counts differ')
            count += 1
    save_new(OUT/f'{name}-fit-audit.json', {'status': 'PASS', 'result': artifact(path),
        'auditor': artifact(ROOT/'scripts/audit_blind_channel_confirm002.py'),
        'initial_selected_record_replays': 8, 'initial_maximum_delta': initial_delta,
        'candidates_accounted': bank['bank_size'], 'complete_rounds': bank['complete_rounds'],
        'sampled_indices': indices, 'independent_records': count, 'maximum_delta': delta,
        'all_candidate_likelihoods_independently_replayed': False, 'no_transfer_or_gold_opened': True,
        'resources': resource_report(wall, cpu)})
    signal.alarm(0)
    print(json.dumps({'case': name, 'status': 'fit_audit_pass', 'records': count}), flush=True)


def audit_predictions():
    campaign_path = OUT / 'campaign.json'
    campaign = json.loads(campaign_path.read_text())
    panel, _, admitted, _ = fit_status(campaign['freeze'])
    if [p['case'] for p in campaign['results']] != list(NAMES):
        raise ValueError('Campaign cases not fully accounted')
    limit_resources(2100, 1800)
    wall, cpu = time.monotonic(), time.process_time()
    selected = json.loads((ROOT / 'results/LATIN-SOURCE-COMPACT-001/large.json').read_text())
    if artifact(ROOT / selected['counts']['path']) != selected['counts']:
        raise ValueError('Source archive changed')
    compact = CompactSuffixSource.load(ROOT / selected['counts']['path'])
    raw = {'alphabet': compact.alphabet, 'order': 12, 'tau': selected['selected']['tau'], 'counts': LiteralCountView(compact)}
    _, source3, initial_models = sources()
    reports = {}
    for process in campaign['results']:
        name = process['case']
        report = {'process': process}
        evidence_path, result_path = OUT / f'{name}-evidence.json', OUT / f'{name}.json'
        if admitted[name] is None:
            if process['returncode'] == 0:
                result = json.loads(result_path.read_text())
                if result['status'] != 'fit_unavailable' or result['readings'] is not None or result['freeze'] != campaign['freeze']:
                    raise ValueError('Unavailable fitting incorrectly represented')
            reports[name] = {**report, 'status': 'fit_unavailable'}
            continue
        if not evidence_path.exists():
            if process['returncode'] == 0:
                raise ValueError('Successful case lacks evidence')
            reports[name] = {**report, 'status': 'process_failure_without_evidence'}
            continue
        header = json.loads(evidence_path.read_text())
        if (header['freeze'] != campaign['freeze'] or header['case'] != name
                or header['bank'] != admitted[name]['bank']
                or header['transfer'] != panel['cases'][name]['transfer']
                or header['parent'] != admitted[name]['parent']):
            raise ValueError('Evidence binding mismatch')
        bank, evidence = load_archive(header['bank']), load_archive(header['evidence'])
        report.update(audit_evidence(bank, evidence))
        if header['summary'] != {k: v for k, v in evidence.items() if k not in ('per_key', 'active_bank_indices')}:
            raise ValueError('Evidence compact summary mismatch')
        progress = [json.loads(line) for line in (BULK / f'{name}-evidence-progress.jsonl').read_text().splitlines()]
        if progress != evidence['per_key']:
            raise ValueError('Evidence progress mismatch')
        observed = checked_artifact(header['transfer'])
        parent = checked_artifact(header['parent'])
        records, rho = observed['records'], observed['context']['stop_probability']
        # Baseline independently reconstructed from the saved quantized counts.
        model = parent['baseline']
        stop = model['stop_count'] / model['stop_denominator']
        probs = dict(zip(model['glyph_alphabet'], (c/model['denominator'] for c in model['counts']), strict=True))
        iid = len(records) * math.log(stop) + sum(map(len, records)) * math.log1p(-stop)
        iid += math.fsum(math.log(probs[c]) for record in records for c in record)
        close(evidence['iid_log_likelihood'], iid)
        close(logvalue(evidence['gain_bits_vs_iid']), (logvalue(evidence['log_likelihood']) - iid)/math.log(2))
        per_key = {r['bank_index']: r for r in evidence['per_key']}
        indices = sorted({0, bank['best_index'], evidence['active_bank_indices'][len(per_key)//2], evidence['active_bank_indices'][-1]})
        delta, checks = 0., 0
        for index in indices:
            # Zero fitting weight at original index is allowed; audit original separately.
            expected = per_key.get(index)
            totals = []
            for j, record in enumerate(records):
                total, _, nodes = infer_reference(raw, bank['bank'][index]['units'], record, rho)
                totals.append(total)
                if expected is not None:
                    target = logvalue(expected['record_log_likelihoods'][j])
                    close(total, target)
                    if math.isfinite(total):
                        delta = max(delta, abs(total-target))
                    if nodes != expected['nodes'][j]:
                        raise ValueError('Independent graph counts differ')
                checks += 1
            if index == 0:
                if bank['bank'][0]['units'] != parent['units']:
                    raise ValueError('Original key index changed')
                close(logvalue(evidence['original_log_likelihood']), math.fsum(totals))
            if index == bank['best_index']:
                close(logvalue(evidence['fit_selected_log_likelihood']), math.fsum(totals))
        report.update(evidence=artifact(evidence_path), backward_records=checks, maximum_delta=delta,
                      status='evidence_complete', sampled_bank_indices=indices)
        # Partial completed baselines survive later failures and are audited too.
        points_path, frequency_path = OUT/f'{name}-points.json', OUT/f'{name}-frequency.json'
        if points_path.exists():
            points = json.loads(points_path.read_text())
            if points['freeze'] != campaign['freeze'] or points['transfer'] != header['transfer']:
                raise ValueError('Point baseline input binding mismatch')
            report['point_record_replays'] = audit_points(raw, records, parent, bank, load_archive(points['readings']), rho)
        if frequency_path.exists():
            frequency = json.loads(frequency_path.read_text())
            if (frequency['freeze'] != campaign['freeze'] or frequency['transfer'] != header['transfer']
                    or frequency['parent'] != header['parent']):
                raise ValueError('Frequency baseline input binding mismatch')
            expected, _ = inference_rows(source3, initial_models['order3'], parent['frequency_units'], records, rho)
            if load_archive(frequency['readings']) != expected:
                raise ValueError('Frequency baseline replay mismatch')
            report['frequency_record_replays'] = len(expected)
        if process['returncode'] == 0:
            result = json.loads(result_path.read_text())
            if any(result[k] != header[k] for k in ('case', 'freeze', 'transfer', 'bank', 'parent', 'evidence', 'summary')):
                raise ValueError('Final output binding mismatch')
            if not panel['cases'][name]['positive']:
                if result['status'] != 'complete_evidence_only_by_design' or result['readings'] is not None:
                    raise ValueError('Null scope differs from registration')
            else:
                if result['status'] != 'complete':
                    raise ValueError('Successful positive lacks complete inference')
                frequency = json.loads((OUT / f'{name}-frequency.json').read_text())
                if (frequency['freeze'] != campaign['freeze'] or frequency['transfer'] != header['transfer']
                        or frequency['parent'] != header['parent'] or result['frequency_order3'] != frequency['readings']
                        or frequency['independent_maximum_delta'] > 1e-7):
                    raise ValueError('Frequency baseline binding mismatch')
                load_archive(frequency['readings'])
                reading = load_archive(result['readings'])
                report['reading'] = audit_reading(compact.alphabet, records, bank, reading)
                close(logvalue(reading['mixture']['log_likelihood']), logvalue(evidence['log_likelihood']))
                for expected, actual in zip(evidence['per_key'], reading['mixture']['per_key'], strict=True):
                    close(logvalue(expected['log_likelihood']), logvalue(actual['log_likelihood']))
                points = json.loads((OUT / f'{name}-points.json').read_text())
                if points['readings'] != result['points'] or load_archive(result['points']) != reading['points']:
                    raise ValueError('Saved point output mismatch')
                progress = [json.loads(line) for line in (BULK / f'{name}-progress.jsonl').read_text().splitlines()]
                expected = [{'active_index': i, 'bank_index': original, **{k: v[k] for k in ('nodes', 'edges', 'expanded')}}
                            for i, (original, v) in enumerate(zip(reading['active_bank_indices'], reading['mixture']['per_key'], strict=True))]
                if progress != expected:
                    raise ValueError('Reading progress mismatch')
            report['result'] = artifact(result_path)
        reports[name] = report
        print(json.dumps({'case': name, 'backward_records': checks, 'delta': delta}), flush=True)
    signal.alarm(0)
    save_new(OUT / 'audit.json', {'status': 'PASS', 'campaign': artifact(campaign_path),
        'auditor': artifact(ROOT / 'scripts/audit_blind_channel_confirm002.py'), 'cases': reports,
        'no_gold_opened': True, 'resources': resource_report(wall, cpu)})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('fit', 'predictions'))
    parser.add_argument('--case')
    args = parser.parse_args()
    if args.mode == 'fit':
        audit_fit(args.case)
    else:
        audit_predictions()
