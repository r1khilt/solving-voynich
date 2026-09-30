"""One post-freeze evaluation, fixed denominators and independent edit/oracle checks."""
import argparse
import json
import signal
import time
from collections import Counter

from scripts.audit_blind_channel_confirm002 import literal_source
from scripts.audit_key_bank_read001 import close, logvalue
from scripts.audit_suffix_reader001 import infer_reference, reading_score
from scripts.confirm002_common import EXP, NAMES, OUT, ROOT, fit_status, load_source, observed
from scripts.evaluate_key_bank_read001 import ARMS, arm_rows, truth_diagnosis
from scripts.run_blind_channel_dev001 import checked_artifact, require_frozen
from scripts.run_blind_channel_dev004 import load_archive, metrics, resource_report, save_new
from scripts.run_latin_source_model001 import artifact, limit_resources
from scripts.run_neural_reader001 import edit_reference
from voynich.fresh_reader_panel import encode_known
from voynich.sparse_suffix_source import decode

ALL_ARMS = (*ARMS, 'frequency_order3', 'oracle_large')


def decisions(reports):
    """All 32 cases and all 7,168 true letters remain in qualification."""
    if len(reports) != 32 or sum(r['positive'] for r in reports.values()) != 16:
        raise ValueError('Incomplete fixed qualification panel')
    positive = [r for r in reports.values() if r['positive']]
    negative = [r for r in reports.values() if not r['positive']]
    complete = all(r['fit_complete'] and r['prediction_complete'] for r in reports.values())
    for row in positive:
        if any(row['arms'][arm]['gold_characters'] != 448 for arm in ALL_ARMS):
            raise ValueError('Full denominator changed')
    errors = [r['arms']['mixture']['edits'] for r in positive]
    oracle = [r['arms']['oracle_large']['edits'] for r in positive]
    frequency = [r['arms']['frequency_order3']['edits'] for r in positive]
    recovery = {
        'all_32_fits_and_predictions_complete_audited': complete,
        'all_32_positive_records_supported': all(r['arms']['mixture']['supported_records'] == 2 for r in positive),
        'macro_cer_at_most_002': sum(errors) <= .02 * 7168,
        'every_key_cer_at_most_005': all(e <= .05 * 448 for e in errors),
        'every_key_oracle_excess_at_most_002': all(e-o <= .02 * 448 for e, o in zip(errors, oracle, strict=True)),
        'every_key_strictly_beats_frequency_order3': all(e < f for e, f in zip(errors, frequency, strict=True)),
    }
    screen = {
        'all_32_fits_and_predictions_complete_audited': complete,
        'all_32_evidences_available': all(r['evidence'] is not None for r in reports.values()),
        'all_16_positives_beat_frozen_iid': all(r['evidence'] is not None and logvalue(r['evidence']['gain_bits_vs_iid']) > 0 for r in positive),
        'no_shuffle_beats_frozen_iid': all(r['evidence'] is not None and logvalue(r['evidence']['gain_bits_vs_iid']) <= 0 for r in negative),
    }
    return recovery, screen


def checked_metrics(rows, truth):
    value = metrics(rows, truth)
    edits = [edit_reference(row['plaintext'] or '', target) for row, target in zip(rows, truth, strict=True)]
    if edits != value['record_edits']:
        raise ValueError('Independent full-grid edit distance mismatch')
    return value


def oracle_rows(source, raw, units, records, rho):
    rows, delta = [], 0.
    for text in records:
        row = decode(source, units, text, rho, max_nodes=500_000).to_dict()
        total, best, _ = infer_reference(raw, units, text, rho)
        close(logvalue(row['log_likelihood']), total)
        close(logvalue(row['joint_log_probability']), best)
        if row['plaintext'] is not None:
            if encode_known(row['plaintext'], source.alphabet, units) != text:
                raise ValueError('Oracle literal encoding mismatch')
            score = reading_score(raw, units, text, row['plaintext'], rho)
            close(score, best)
            delta = max(delta, abs(logvalue(row['log_likelihood'])-total), abs(score-best))
        rows.append(row)
    return rows, delta


def evaluate(freeze):
    panel, _, admitted, fit_campaign = fit_status(freeze)
    campaign = json.loads((OUT/'campaign.json').read_text())
    audit = json.loads((OUT/'audit.json').read_text())
    if (audit['status'] != 'PASS' or audit['campaign'] != artifact(OUT/'campaign.json')
            or audit['auditor'] != artifact(ROOT/'scripts/audit_blind_channel_confirm002.py')
            or [r['case'] for r in campaign['results']] != list(NAMES)
            or set(audit['cases']) != set(NAMES)):
        raise ValueError('Audited terminal predictions required before gold access')
    require_frozen(freeze, [str(p.relative_to(ROOT)) for p in OUT.glob('*.json')])
    save_new(OUT/'evaluation-started.json', {'freeze': freeze, 'start_unix': time.time()})
    limit_resources(1500, 1200)
    wall, cpu = time.monotonic(), time.process_time()
    try:
        source, raw = load_source(), literal_source()
        reports, edits_checked, oracle_checked, oracle_delta = {}, 0, 0, 0.
        for process in campaign['results']:
            name, spec = process['case'], panel['cases'][process['case']]
            evidence_path, result_path = OUT/f'{name}-evidence.json', OUT/f'{name}.json'
            result = json.loads(result_path.read_text()) if result_path.exists() else None
            complete = (process['returncode'] == 0 and result is not None
                        and result['status'] in ('complete', 'complete_evidence_only_by_design'))
            report = {'positive': spec['positive'], 'pair_index': spec['pair_index'],
                      'fit_complete': admitted[name] is not None, 'prediction_complete': complete,
                      'evidence': json.loads(evidence_path.read_text())['summary'] if evidence_path.exists() else None}
            if spec['positive']:
                data = observed(spec['transfer'], 2)
                answer = checked_artifact(spec['answer'])  # First solver-evaluation gold access.
                truth = answer['plaintext']['transfer']
                if (answer['positive'] is not True or answer['pair_index'] != spec['pair_index']
                        or len(truth) != 2 or any(len(t) != 224 for t in truth)
                        or [encode_known(t, source.alphabet, answer['units']) for t in truth] != data['records']):
                    raise ValueError('Gold denominator/encoding/binding mismatch')
                reading = load_archive(result['readings']) if complete else None
                points_path, frequency_path = OUT/f'{name}-points.json', OUT/f'{name}-frequency.json'
                points = load_archive(json.loads(points_path.read_text())['readings']) if points_path.exists() else {}
                frequency = load_archive(json.loads(frequency_path.read_text())['readings']) if frequency_path.exists() else None
                oracle, delta = oracle_rows(source, raw, answer['units'], data['records'], 1/225)
                oracle_delta, oracle_checked = max(oracle_delta, delta), oracle_checked+2
                report['arms'] = {}
                for arm in ALL_ARMS:
                    if arm in ARMS:
                        rows, available = arm_rows(arm, points, reading, 2)
                    else:
                        rows = oracle if arm == 'oracle_large' else frequency
                        available = rows is not None
                        rows = rows if available else [{'plaintext': None}, {'plaintext': None}]
                    report['arms'][arm] = {**checked_metrics(rows, truth), 'available': available}
                    edits_checked += 2
                bank = load_archive(admitted[name]['bank']) if admitted[name] is not None else None
                report['truth_diagnosis'] = truth_diagnosis(source, data['records'], truth, bank, reading, 1/225)
                if bank is not None:
                    report['literal_key_row_matches'] = {label: sum(a == b for a, b in zip(units, answer['units'], strict=True))
                        for label, units in [('initial', bank['bank'][0]['units']), ('fit_selected', bank['best_units'])]}
                report['source_letter_exposure'] = {role: dict(Counter(''.join(texts)))
                                                   for role, texts in answer['plaintext'].items()}
                if reading is not None:
                    report['mixture_search'] = {k: reading['mixture'][k] for k in ('candidate_tuples', 'floating_bound_separated',
                        'unseen_log_probability_upper_bound', 'joint_log_probability', 'log_likelihood')}
            reports[name] = report
            print(json.dumps({'case': name, 'evaluated': True}), flush=True)
        positive = [r for r in reports.values() if r['positive']]
        totals = {arm: {key: sum(r['arms'][arm][key] for r in positive)
                       for key in ('edits', 'gold_characters', 'exact_records', 'supported_records')} for arm in ALL_ARMS}
        conditional = {arm: {'cases': sum(r['arms'][arm]['available'] for r in positive),
                             **{k: sum(r['arms'][arm][k] for r in positive if r['arms'][arm]['available'])
                                for k in ('edits', 'gold_characters')}} for arm in ALL_ARMS}
        recovery, screen = decisions(reports)
        result = {'experiment': EXP, 'freeze': freeze, 'cases': reports, 'totals': totals,
            'completed_case_metrics_conditional': conditional, 'recovery_clauses': recovery, 'screen_clauses': screen,
            'recovery_gate': 'PASS' if all(recovery.values()) else 'FAIL',
            'iid_screen_gate': 'PASS' if all(screen.values()) else 'FAIL',
            'independent_record_edit_checks': edits_checked, 'independent_oracle_records': oracle_checked,
            'maximum_oracle_delta': oracle_delta, 'fit_campaign': artifact(OUT/'fit-campaign.json'),
            'prediction_campaign': artifact(OUT/'campaign.json'), 'prediction_audit': artifact(OUT/'audit.json'),
            'fit_campaign_status': fit_campaign['status'], 'fresh_keys_and_passages_not_new_authors': True,
            'structured_null_or_historical_decipherment_qualified': False, 'resources': resource_report(wall, cpu)}
        save_new(OUT/'evaluation.json', result)
        print(json.dumps({'totals': totals, 'recovery': result['recovery_gate'], 'screen': result['iid_screen_gate']}), flush=True)
    except Exception as error:
        signal.alarm(0)
        save_new(OUT/'evaluation-failure.json', {'error': str(error), 'type': type(error).__name__,
                                               'resources': resource_report(wall, cpu)})
        raise
    finally:
        signal.alarm(0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--freeze', required=True)
    evaluate(parser.parse_args().freeze)
