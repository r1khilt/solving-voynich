"""No-gold all-case evidence arithmetic and sampled independent backward audit."""
import json
import math
import signal
import time

from scripts.audit_key_bank_read001 import audit_reading, close, logsum, logvalue
from scripts.audit_suffix_reader001 import infer_reference
from scripts.run_blind_channel_dev001 import checked_artifact
from scripts.run_blind_channel_dev004 import load_archive, resource_report, save_new
from scripts.run_key_bank_read002 import BULK, MANIFEST, NAMES, OUT, ROOT, admission
from scripts.run_latin_source_model001 import artifact, limit_resources
from voynich.compact_suffix_reference import LiteralCountView
from voynich.compact_suffix_source import CompactSuffixSource


def audit_evidence(bank, value):
    active = [i for i, row in enumerate(bank['bank']) if row['log_weight'] is not None]
    rows = value['per_key']
    if (value['active_bank_indices'] != active or [r['bank_index'] for r in rows] != active
            or not value['all_positive_weights_retained']):
        raise ValueError('Evidence key inventory mismatch')
    close(logsum(bank['bank'][i]['log_weight'] for i in active), 0.)
    supported = 0
    for row in rows:
        if any(len(row[k]) != value['records'] for k in ('record_log_likelihoods', 'nodes', 'edges')):
            raise ValueError('Record inventory mismatch')
        close(logvalue(row['log_likelihood']), math.fsum(map(logvalue, row['record_log_likelihoods'])))
        supported += row['log_likelihood'] is not None
    total = logsum(bank['bank'][r['bank_index']]['log_weight'] + logvalue(r['log_likelihood']) for r in rows)
    close(logvalue(value['log_likelihood']), total)
    if supported != value['supported_keys']:
        raise ValueError('Supported key count mismatch')
    return {'active_keys': len(active), 'supported_keys': supported}


def main():
    campaign_path = OUT / 'campaign.json'
    campaign = json.loads(campaign_path.read_text())
    admitted, _ = admission(campaign['freeze'])
    if [p['case'] for p in campaign['results']] != list(NAMES):
        raise ValueError('Campaign cases not fully accounted')
    limit_resources(900, 900)
    wall, cpu = time.monotonic(), time.process_time()
    selected = json.loads((ROOT / 'results/LATIN-SOURCE-COMPACT-001/large.json').read_text())
    if artifact(ROOT / selected['counts']['path']) != selected['counts']:
        raise ValueError('Source archive changed')
    compact = CompactSuffixSource.load(ROOT / selected['counts']['path'])
    raw = {'alphabet': compact.alphabet, 'order': 12, 'tau': selected['selected']['tau'], 'counts': LiteralCountView(compact)}
    manifest = json.loads((ROOT / MANIFEST).read_text())
    reports = {}
    for process in campaign['results']:
        name = process['case']
        report = {'process': process}
        evidence_path, result_path = OUT / f'{name}-evidence.json', OUT / f'{name}.json'
        if not evidence_path.exists():
            if process['returncode'] == 0:
                raise ValueError('Successful case lacks evidence')
            reports[name] = {**report, 'status': 'process_failure_without_evidence'}
            continue
        header = json.loads(evidence_path.read_text())
        if (header['freeze'] != campaign['freeze'] or header['case'] != name
                or header['bank'] != admitted[name]['bank']
                or header['transfer'] != manifest['cases'][name]['artifacts']['transfer']
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
        if process['returncode'] == 0:
            result = json.loads(result_path.read_text())
            if any(result[k] != header[k] for k in ('case', 'freeze', 'transfer', 'bank', 'parent', 'evidence', 'summary')):
                raise ValueError('Final output binding mismatch')
            if name.endswith('-shuffle'):
                if result['status'] != 'complete_evidence_only_by_design' or result['readings'] is not None:
                    raise ValueError('Null scope differs from registration')
            else:
                if result['status'] != 'complete':
                    raise ValueError('Successful positive lacks complete inference')
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
        'auditor': artifact(ROOT / 'scripts/audit_key_bank_read002.py'), 'cases': reports,
        'no_gold_opened': True, 'resources': resource_report(wall, cpu)})


if __name__ == '__main__':
    main()
