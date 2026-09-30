"""Exposed-panel objective and discarded-restart diagnosis; no key optimization."""
import argparse
import gc
import json
import math
import signal
import time

from scripts.audit_blind_channel_dev004 import infer_record, literal_model_bits
from scripts.audit_blind_channel_confirm002 import literal_source
from scripts.audit_suffix_reader001 import infer_reference
from scripts.confirm002_common import PATHS as ORIGINAL_PATHS
from scripts.confirm002_common import ROOT, fit_status, load_source, observed
from scripts.run_blind_channel_confirm001 import sources
from scripts.run_blind_channel_dev001 import checked_artifact, require_frozen
from scripts.run_blind_channel_dev004 import load_archive, resource_report, save_new
from scripts.run_key_bank_expand001 import score_fit
from scripts.run_latin_source_model001 import artifact, limit_resources
from voynich.batched_unit_channel import batch_log_likelihood
from voynich.fresh_reader_panel import encode_known
from voynich.higher_order_unit_channel import decode_units
from voynich.native_suffix_marginal import NativeMarginal

EXP = 'BLIND-CHANNEL-CONFIRM-002-DIAG-A'
OUT = ROOT / 'results' / EXP
PREVIOUS = ROOT / 'results/BLIND-CHANNEL-CONFIRM-002'
PATHS = sorted(set(ORIGINAL_PATHS + [
    'scripts/diagnose_blind_channel_confirm002a.py',
    'tests/test_blind_channel_confirm002a.py',
    'docs/experiments/BLIND-CHANNEL-CONFIRM-002-DIAG-A.md',
    'results/BLIND-CHANNEL-CONFIRM-002/evaluation.json',
    'results/BLIND-CHANNEL-CONFIRM-002/evaluation-audit.json',
]))


def endpoints(trace):
    """Keep the accepted terminal state of each previously scored restart."""
    states = {}
    for event in trace:
        restart = event['restart']
        if event['event'] == 'restart':
            if restart in states:
                raise ValueError('Duplicate restart')
            states[restart] = {'units': list(event['units']), 'score': event['score'],
                               'stop_reason': 'no_sweep', 'method': event['method']}
        elif event['event'] == 'sweep':
            if restart not in states:
                raise ValueError('Sweep without restart')
            state = states[restart]
            if state['units'] != event['parent_units'] or state['score'] != event['parent_score']:
                raise ValueError('Restart state discontinuity')
            if event['accepted']:
                state['units'], state['score'] = list(event['selected_units']), event['selected_score']
            state['stop_reason'] = event['stop_reason']
        else:
            raise ValueError('Unknown trace event')
    return states


def objective(log_likelihood, model_bits):
    if log_likelihood is None or log_likelihood == -math.inf:
        return None
    if not math.isfinite(log_likelihood) or type(model_bits) is not int or model_bits < 0:
        raise ValueError('Invalid objective terms')
    return model_bits - log_likelihood / math.log(2)


def compare(learned_bits, true_bits):
    if true_bits is None:
        raise ValueError('Generating key must support its own observations')
    difference = None if learned_bits is None else learned_bits - true_bits
    return {'learned_minus_true_total_bits': difference,
            'true_key_is_known_better_candidate': difference is None or difference > 1e-7}


def run(freeze):
    require_frozen(freeze, PATHS)
    panel, benchmark, _, _ = fit_status(freeze)
    require_frozen(freeze, [str(p.relative_to(ROOT)) for p in PREVIOUS.glob('*.json')])
    evaluation = checked_artifact(artifact(PREVIOUS / 'evaluation.json'))
    audit = checked_artifact(artifact(PREVIOUS / 'evaluation-audit.json'))
    if audit['status'] != 'PASS' or audit['evaluation'] != artifact(PREVIOUS / 'evaluation.json'):
        raise ValueError('Published audited evaluation required')
    save_new(OUT / 'started.json', {'freeze': freeze, 'evaluation': artifact(PREVIOUS / 'evaluation.json'),
                                  'start_unix': time.time()})
    limit_resources(900, 750)
    wall, cpu = time.monotonic(), time.process_time()
    try:
        source1, source3, old_models = sources()
        source = load_source()
        raw = literal_source()
        scorer = NativeMarginal(source, benchmark['build'])
        rows, deltas = {}, []
        large_records = reference_records = small_records = 0
        for name in sorted(panel['cases']):
            spec = panel['cases'][name]
            if not spec['positive']:
                continue
            data = observed(spec['fit'], 4)
            answer = checked_artifact(spec['answer'])
            units = tuple(answer['units'])
            truth = answer['plaintext']['fit']
            if len(truth) != 4 or any(len(t) != 224 for t in truth):
                raise ValueError('Original fitting denominator changed')
            if [encode_known(t, source.alphabet, units) for t in truth] != data['records']:
                raise ValueError('Generating key/text does not match original fitting data')
            parent_path, fit_path = PREVIOUS / f'{name}-parent.json', PREVIOUS / f'{name}-fit.json'
            parent, fit = checked_artifact(artifact(parent_path)), checked_artifact(artifact(fit_path))
            if parent['fit'] != spec['fit'] or fit['fit'] != spec['fit']:
                raise ValueError('Saved fit identity changed')
            trace = load_archive(parent['stage1_trace'])
            terminal = endpoints(trace['trace'])
            if len(terminal) != parent['stage1_accounting']['scored_initializations'] or len(terminal) > 16:
                raise ValueError('Restart inventory mismatch')
            del trace
            gc.collect()
            candidates = {'generating': units, 'stage1_selected': tuple(parent['stage1_units']),
                          'stage2_selected': tuple(parent['units']), 'expanded_selected': tuple(fit['best_units'])}
            candidates.update({f'restart-{i:02d}': tuple(row['units']) for i, row in terminal.items()
                               if row['score'] is not None})
            if len(candidates) > 20:
                raise ValueError('Registered finite candidate cap exceeded')
            values, cache = {}, {}
            for label, candidate in candidates.items():
                if candidate not in cache:
                    cache[candidate] = score_fit(scorer, data, candidate)
                    large_records += 4
                value = cache[candidate]
                values[label] = {**value, 'total_bits': objective(value['fit_log_likelihood'], value['model_bits'])}
            gold_large = values['generating']
            for record, expected in zip(data['records'], gold_large['record_log_likelihoods'], strict=True):
                total, _, _ = infer_reference(raw, units, record, 1/225)
                delta = abs(total - expected)
                if not math.isfinite(delta) or delta > 1e-7:
                    raise ValueError('Generating-key native/reference disagreement')
                deltas.append(delta)
                reference_records += 1
            bank = load_archive(fit['bank'])
            for label, index in [('stage2_selected', 0), ('expanded_selected', bank['best_index'])]:
                previous = bank['bank'][index]
                actual = values[label]
                if (actual['model_bits'] != previous['model_bits']
                        or abs(actual['fit_log_likelihood'] - previous['fit_log_likelihood']) > 1e-7):
                    raise ValueError('Unchanged learned-key native score mismatch')
            del bank
            code = literal_model_bits(units, data['context'])
            small = {}
            for label in ('order1', 'order3'):
                likelihoods = []
                for record in data['records']:
                    expected = infer_record(old_models[label], units, record, 1/225)['log_likelihood']
                    actual = (float(batch_log_likelihood(source1, [units], [record], 1/225)[0])
                              if label == 'order1' else decode_units(source3, units, record, 1/225).log_likelihood)
                    delta = abs(actual - expected)
                    if not math.isfinite(delta) or delta > 1e-7:
                        raise ValueError('Generating-key original-source replay mismatch')
                    deltas.append(delta)
                    likelihoods.append(expected)
                    small_records += 1
                total = math.fsum(likelihoods)
                learned = parent['stage1_score'] if label == 'order1' else parent['score']
                cost = objective(total, code)
                small[label] = {'record_log_likelihoods': likelihoods, 'true_key_total_bits': cost,
                                'learned_total_bits': learned['total_bits'], **compare(learned['total_bits'], cost)}
            restart_labels = [k for k in values if k.startswith('restart-') and values[k]['total_bits'] is not None]
            best_restart = min(((values[k]['total_bits'], k) for k in restart_labels), default=(None, None))[1]
            best_cost = None if best_restart is None else values[best_restart]['total_bits']
            row = {
                'pair_index': spec['pair_index'], 'fit': spec['fit'], 'answer': spec['answer'],
                'parent': artifact(parent_path), 'fit_result': artifact(fit_path),
                'stage1_trace': parent['stage1_trace'], 'restart_endpoints': terminal,
                'original_source_comparisons': small, 'large_scores': values,
                'large_selected_comparison': compare(values['expanded_selected']['total_bits'], gold_large['total_bits']),
                'best_restart_by_large_fit_objective': best_restart,
                'best_restart_total_bits': best_cost,
                'best_restart_minus_original_stage1_selected_bits': None if best_cost is None else best_cost-values['stage1_selected']['total_bits'],
                'best_restart_minus_expanded_selected_bits': None if best_cost is None else best_cost-values['expanded_selected']['total_bits'],
                'restart_literal_row_matches': {f'restart-{i:02d}': sum(a == b for a, b in zip(v['units'], units, strict=True))
                                                for i, v in terminal.items()},
                'fresh_evaluation_case': evaluation['cases'][name],
            }
            rows[name] = row
            save_new(OUT / f'{name}.json', row)
            resources = resource_report(wall, cpu)
            if resources['peak_rss_bytes'] > 4*1024**3:
                raise MemoryError('Registered sampled4GiB host guard')
            print(json.dumps({'case': name, 'gold_better': row['large_selected_comparison'],
                              'restart_gain_vs_expanded_bits': None if best_cost is None else values['expanded_selected']['total_bits']-best_cost}), flush=True)
            del terminal, cache, values
            gc.collect()
        if len(rows) != 16:
            raise ValueError('All original positive cases must remain in diagnosis')
        save_new(OUT / 'result.json', {
            'experiment': EXP, 'freeze': freeze, 'status': 'complete_exposed_diagnostic',
            'cases': {name: artifact(OUT / f'{name}.json') for name in rows},
            'large_native_record_scores': large_records, 'large_generating_key_reference_records': reference_records,
            'original_source_independent_record_checks': small_records, 'maximum_reference_delta_nats': max(deltas, default=0.),
            'no_key_optimization_or_transfer_decoding': True, 'fresh_confirmation_not_repaired': True,
            'resources': resource_report(wall, cpu),
        })
    except Exception as error:
        signal.alarm(0)
        save_new(OUT / 'failure.json', {'type': type(error).__name__, 'error': str(error),
                                      'resources': resource_report(wall, cpu)})
        raise
    finally:
        signal.alarm(0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--freeze', required=True)
    run(parser.parse_args().freeze)
