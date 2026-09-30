"""Evaluate exposed answers only after expanded predictions and audit are frozen."""
import argparse
import json
import math
import signal
import time

from scripts.audit_key_bank_read001 import logvalue
from scripts.evaluate_key_bank_read001 import ARMS, arm_rows, truth_diagnosis
from scripts.run_blind_channel_dev001 import checked_artifact, require_frozen
from scripts.run_blind_channel_dev004 import load_archive, metrics, resource_report, save_new
from scripts.run_key_bank_read002 import EXP, MANIFEST, NAMES, OUT, ROOT, admission, load_source
from scripts.run_latin_source_model001 import artifact, limit_resources
from scripts.run_neural_reader001 import edit_reference


def decisions(reports, totals, previous):
    positives = [reports[n] for n in NAMES if not n.endswith('-shuffle')]
    nulls = [reports[n] for n in NAMES if n.endswith('-shuffle')]
    reading = {'all_positive_predictions_complete': all(r['inference_status'] == 'complete' for r in positives),
               'all_sixteen_positive_records_supported': totals['mixture']['supported_records'] == 16,
               'strictly_fewer_edits_than_previous_mixture': totals['mixture']['edits'] < previous,
               'every_positive_key_at_most_five_percent': all(r['arms']['mixture']['edits'] <= .05*448 for r in positives)}
    screen = {'all_sixteen_evidences_available': all(r['evidence'] is not None for r in reports.values()),
              'all_eight_positives_beat_frozen_iid': all(r['evidence'] is not None and logvalue(r['evidence']['gain_bits_vs_iid']) > 0 for r in positives),
              'no_shuffle_beats_frozen_iid': all(r['evidence'] is not None and logvalue(r['evidence']['gain_bits_vs_iid']) <= 0 for r in nulls)}
    return reading, screen


def evaluate(freeze):
    admitted, _ = admission(freeze)
    campaign = json.loads((OUT / 'campaign.json').read_text())
    audit = json.loads((OUT / 'audit.json').read_text())
    if (audit['status'] != 'PASS' or audit['campaign'] != artifact(OUT / 'campaign.json')
            or audit['auditor'] != artifact(ROOT / 'scripts/audit_key_bank_read002.py')
            or [r['case'] for r in campaign['results']] != list(NAMES)):
        raise ValueError('Complete prediction accounting and audit required')
    paths = [str(p.relative_to(ROOT)) for p in OUT.glob('*.json')]
    require_frozen(freeze, paths)
    save_new(OUT / 'evaluation-started.json', {'freeze': freeze, 'start_unix': time.time()})
    limit_resources(600, 600)
    wall, cpu = time.monotonic(), time.process_time()
    source = load_source()
    manifest = json.loads((ROOT / MANIFEST).read_text())
    previous = json.loads((ROOT / 'results/KEY-BANK-READ-001/evaluation.json').read_text())
    reports, edit_checks, original_checks = {}, 0, 0
    for name in NAMES:
        observed = checked_artifact(manifest['cases'][name]['artifacts']['transfer'])
        evidence_path = OUT / f'{name}-evidence.json'
        evidence = json.loads(evidence_path.read_text())['summary'] if evidence_path.exists() else None
        report = {'evidence': evidence}
        if not name.endswith('-shuffle'):
            truth = checked_artifact(manifest['cases'][name]['artifacts']['answer'])['plaintext']['transfer']
            if len(truth) != 2 or any(len(t) != 224 for t in truth):
                raise ValueError('Original evaluation denominator changed')
            result_path, points_path = OUT / f'{name}.json', OUT / f'{name}-points.json'
            result = json.loads(result_path.read_text()) if result_path.exists() else None
            reading = None if result is None else load_archive(result['readings'])
            points = load_archive(json.loads(points_path.read_text())['readings']) if points_path.exists() else {}
            report.update(inference_status='process_failure' if reading is None else reading['status'], arms={})
            for arm in ARMS:
                rows, available = arm_rows(arm, points, reading, 2)
                value = metrics(rows, truth)
                reference = [edit_reference(r['plaintext'] or '', t) for r, t in zip(rows, truth, strict=True)]
                if reference != value['record_edits']:
                    raise ValueError('Independent edit mismatch')
                edit_checks += 2
                value['available'] = available
                report['arms'][arm] = value
            if points:
                old_manifest = json.loads((ROOT / f'results/KEY-BANK-READ-001/{name}-points.json').read_text())
                old = load_archive(old_manifest['readings'])['original']
                if points['original'] != old:
                    raise ValueError('Original point baseline did not reproduce exactly')
                original_checks += 2
            bank = load_archive(admitted[name]['bank'])
            report['truth_diagnosis'] = truth_diagnosis(source, observed['records'], truth, bank, reading, 1/225)
            if reading is not None:
                report['mixture_search'] = {k: reading['mixture'][k] for k in
                    ('candidate_tuples', 'floating_bound_separated', 'unseen_log_probability_upper_bound', 'joint_log_probability', 'log_likelihood')}
        reports[name] = report
    positive = [reports[n] for n in NAMES if not n.endswith('-shuffle')]
    totals = {arm: {field: sum(r['arms'][arm][field] for r in positive)
                    for field in ('edits', 'exact_records', 'supported_records', 'gold_characters')} for arm in ARMS}
    conditional = {arm: {'cases': sum(r['arms'][arm]['available'] for r in positive),
                         'edits': sum(r['arms'][arm]['edits'] for r in positive if r['arms'][arm]['available']),
                         'gold_characters': sum(r['arms'][arm]['gold_characters'] for r in positive if r['arms'][arm]['available'])} for arm in ARMS}
    reading, screen = decisions(reports, totals, previous['totals']['mixture']['edits'])
    pairs = {}
    for i in range(1, 9):
        name = f'B-key{i}'
        a, b = reports[name]['evidence'], reports[name+'-shuffle']['evidence']
        ga = None if a is None else logvalue(a['gain_bits_vs_iid'])
        gb = None if b is None else logvalue(b['gain_bits_vs_iid'])
        pairs[name] = {'difference_bits': ga-gb if ga is not None and gb is not None and math.isfinite(ga) and math.isfinite(gb) else None,
                       'positive_strictly_higher': None if ga is None or gb is None else ga > gb}
    result = {'experiment': EXP, 'freeze': freeze, 'cases': reports, 'totals': totals,
              'completed_case_metrics_conditional': conditional, 'reading_clauses': reading, 'screen_clauses': screen,
              'reading_development_gate': 'PASS' if all(reading.values()) else 'FAIL',
              'iid_screen_development_gate': 'PASS' if all(screen.values()) else 'FAIL',
              'positive_minus_paired_shuffle_gain_bits': pairs,
              'previous_mixture_edits': previous['totals']['mixture']['edits'],
              'marginal_minus_joint_edits': totals['mixture']['edits'] - totals['joint']['edits'],
              'independent_record_edit_checks': edit_checks, 'original_point_records_reproduced': original_checks,
              'historically_exposed_not_fresh_qualification': True, 'resources': resource_report(wall, cpu)}
    save_new(OUT / 'evaluation.json', result)
    signal.alarm(0)
    print(json.dumps({'totals': totals, 'reading_gate': result['reading_development_gate'], 'screen_gate': result['iid_screen_development_gate']}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--freeze', required=True)
    evaluate(parser.parse_args().freeze)
