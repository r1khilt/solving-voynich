"""Post-prediction-freeze evaluation on exposed gold; never fits or reruns keys."""
import argparse
import json
import math
import signal
import time

from scripts.run_blind_channel_dev001 import checked_artifact, require_frozen
from scripts.run_blind_channel_dev004 import load_archive, metrics, resource_report, save_new
from scripts.run_key_bank_read001 import MANIFEST, NAMES, OUT, PATHS, ROOT, admission, path_score
from scripts.run_latin_source_model001 import artifact, limit_resources
from scripts.run_neural_reader001 import edit_reference
from voynich.compact_suffix_source import CompactSuffixSource
from voynich.dense_suffix_adapter import DenseSuffixAdapter
from voynich.fresh_reader_panel import encode_known

ARMS = ('original', 'fit_selected', 'joint', 'mixture')


def arm_rows(arm, points, reading, count):
    if arm in ('original', 'fit_selected'):
        rows = points.get(arm)
        available = rows is not None
    else:
        available = reading is not None and reading['status'] == 'complete'
        chosen = None if not available else reading[arm]
        texts = None if chosen is None else chosen['plaintexts']
        rows = None if texts is None else [{'plaintext': t} for t in texts]
    rows = rows if rows is not None else [{'plaintext': None} for _ in range(count)]
    if len(rows) != count:
        raise ValueError('Incomplete record list')
    return rows, available


def truth_diagnosis(source, records, truth, bank, reading, rho):
    if bank is None or reading is None or reading['status'] != 'complete':
        return {'status': 'bank_or_inference_unavailable'}
    compatible = [i for i, row in enumerate(bank['bank']) if row['log_weight'] is not None
                  and [encode_known(t, source.alphabet, row['units']) for t in truth] == list(records)]
    if not compatible:
        return {'status': 'true_tuple_outside_positive_weight_bank_support', 'supporting_keys': 0}
    q = path_score(source, truth, rho)
    weights = [bank['bank'][i]['log_weight'] for i in compatible]
    peak = max(weights)
    joint_score = q + peak
    mixture_score = q + peak + math.log(math.fsum(math.exp(w - peak) for w in weights))
    tolerance = 1e-8 * max(1, sum(map(len, records)))
    joint = reading['joint']
    mixture = reading['mixture']
    if (joint is None or mixture['plaintexts'] is None
            or joint_score > joint['log_probability'] + tolerance):
        raise ValueError('Known true candidate contradicts complete per-key MAP inference')
    gap = mixture_score - mixture['joint_log_probability']
    if gap > tolerance and mixture['floating_bound_separated']:
        raise ValueError('Known true candidate contradicts separated mixture bound')
    upper = mixture['unseen_log_probability_upper_bound']
    if gap > tolerance and (upper is None or mixture_score > upper + tolerance):
        raise ValueError('Known true candidate violates unseen-candidate bound')
    exact = tuple(mixture['plaintexts']) == tuple(truth)
    status = ('exact_true_tuple' if exact else 'known_better_true_tuple_missed' if gap > tolerance
              else 'score_tie_or_numerical_ambiguity' if abs(gap) <= tolerance else 'model_prefers_wrong_tuple')
    return {'status': status, 'supporting_keys': len(compatible), 'true_source_log_probability': q,
            'true_joint_key_text_log_probability': joint_score, 'true_marginal_text_log_probability': mixture_score,
            'true_minus_returned_marginal_nats': gap}


def evaluate(freeze):
    admitted, fit_campaign = admission(freeze)
    campaign = json.loads((OUT / 'campaign.json').read_text())
    if len(campaign['results']) != len(NAMES) or {r['case'] for r in campaign['results']} != set(NAMES):
        raise ValueError('Prediction processes not fully accounted')
    paths = PATHS + [str((OUT / 'campaign.json').relative_to(ROOT)), str((OUT / 'campaign-started.json').relative_to(ROOT))]
    for name in NAMES:
        paths += [str(p.relative_to(ROOT)) for p in (OUT / f'{name}.json', OUT / f'{name}-points.json', OUT / f'{name}-failure.json') if p.exists()]
    require_frozen(freeze, paths)
    save_new(OUT / 'evaluation-started.json', {'freeze': freeze, 'start_unix': time.time()})
    limit_resources(600, 600)
    wall, cpu = time.monotonic(), time.process_time()
    source_result = json.loads((ROOT / 'results/LATIN-SOURCE-COMPACT-001/large.json').read_text())
    if artifact(ROOT / source_result['counts']['path']) != source_result['counts']:
        raise ValueError('Source counts changed')
    source = DenseSuffixAdapter(CompactSuffixSource.load(ROOT / source_result['counts']['path']), source_result['selected']['tau'])
    manifest = json.loads((ROOT / MANIFEST).read_text())
    old = json.loads((ROOT / 'results/KEY-SOURCE-DIAG-001/evaluation.json').read_text())
    reports, edit_checks, baseline_replays = {}, 0, 0
    for name in NAMES:
        spec = manifest['cases'][name]
        observed = checked_artifact(spec['artifacts']['transfer'])
        positive = not name.endswith('-shuffle')
        truth = checked_artifact(spec['artifacts']['answer'])['plaintext']['transfer'] if positive else None
        if truth is not None and (len(truth) != 2 or any(len(t) != 224 for t in truth)):
            raise ValueError('Original evaluation denominator changed')
        points_file, reading_file = OUT / f'{name}-points.json', OUT / f'{name}.json'
        points = load_archive(json.loads(points_file.read_text())['readings']) if points_file.exists() else {}
        reading = load_archive(json.loads(reading_file.read_text())['readings']) if reading_file.exists() else None
        bank = None if admitted[name] is None else load_archive(admitted[name]['bank'])
        report = {'fit_bank_available': bank is not None,
                  'inference_status': 'process_failure' if reading is None else reading['status'], 'arms': {}}
        for arm in ARMS:
            rows, available = arm_rows(arm, points, reading, len(observed['records']))
            value = metrics(rows, truth)
            value['available'] = available
            if truth is not None:
                reference = [edit_reference(r['plaintext'] or '', t) for r, t in zip(rows, truth, strict=True)]
                if value['record_edits'] != reference:
                    raise ValueError('Independent edit mismatch')
                edit_checks += len(rows)
            report['arms'][arm] = value
        if 'original' in points:
            previous = load_archive(old['archives']['statistical-large'][name])['learned']['transfer']
            for current, prior in zip(points['original'], previous, strict=True):
                if any(current[k] != prior[k] for k in ('plaintext', 'log_likelihood', 'joint_log_probability', 'reachable_nodes', 'edges')):
                    raise ValueError('Original large-source baseline did not reproduce')
                baseline_replays += 1
        if positive:
            report['truth_diagnosis'] = truth_diagnosis(source, observed['records'], truth, bank, reading, 1 / 225)
        if reading is not None and reading['mixture'] is not None:
            report['mixture_search'] = {k: reading['mixture'][k] for k in
                ('candidate_tuples', 'floating_bound_separated', 'unseen_log_probability_upper_bound', 'joint_log_probability', 'log_likelihood')}
        reports[name] = report
    positive = [reports[f'B-key{i}'] for i in range(1, 9)]
    totals = {arm: {field: sum(c['arms'][arm][field] for c in positive)
                    for field in ('edits', 'exact_records', 'supported_records', 'gold_characters')}
              for arm in ARMS}
    conditional = {arm: {'cases': sum(c['arms'][arm]['available'] for c in positive),
                         **{field: sum(c['arms'][arm][field] for c in positive if c['arms'][arm]['available'])
                            for field in ('edits', 'gold_characters')}} for arm in ARMS}
    clauses = {'all_positive_inference_complete': all(c['inference_status'] == 'complete' for c in positive),
               'all_positive_records_supported': totals['mixture']['supported_records'] == 16,
               'thirty_percent_fewer_edits_vs_fit_point': (totals['fit_selected']['edits'] > 0
                                                          and totals['mixture']['edits'] <= .7 * totals['fit_selected']['edits']),
               'every_positive_key_at_most_five_percent': all(c['arms']['mixture']['edits'] <= .05 * 448 for c in positive)}
    result = {'experiment': EXP, 'freeze': freeze, 'cases': reports, 'totals': totals,
              'completed_case_metrics_conditional': conditional, 'development_clauses': clauses,
              'development_gate': 'PASS' if all(clauses.values()) else 'FAIL',
              'marginal_minus_joint_edits': totals['mixture']['edits'] - totals['joint']['edits'],
              'independent_record_edit_checks': edit_checks, 'baseline_records_reproduced': baseline_replays,
              'fit_campaign_status': fit_campaign['status'], 'prediction_campaign_status': campaign['status'],
              'not_fresh_qualification_or_null_discrimination': True, 'resources': resource_report(wall, cpu)}
    save_new(OUT / 'evaluation.json', result)
    signal.alarm(0)
    print(json.dumps({'totals': totals, 'development_gate': result['development_gate']}), flush=True)


EXP = 'KEY-BANK-READ-001'
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--freeze', required=True)
    evaluate(parser.parse_args().freeze)
