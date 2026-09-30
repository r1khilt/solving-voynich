"""Frozen neural reranking of exposed candidate unions; never modifies CONFIRM-002."""
import argparse
import copy
import json
import math
import signal
import time

import torch

from scripts.audit_key_bank_read001 import close, logsum
from scripts.run_blind_channel_dev001 import checked_artifact, require_frozen
from scripts.run_blind_channel_dev004 import load_archive, metrics, resource_report, save_new
from scripts.run_key_bank_read002 import MANIFEST, PATHS as OLD_PATHS, admission as old_admission
from scripts.run_latin_source_model001 import ROOT, artifact, limit_resources
from scripts.run_neural_reader001 import PATHS as NEURAL_PATHS, edit_reference, load_neural, sources
from voynich.candidate_rerank import prepare_candidates, rank_candidates
from voynich.recurrent_latin_source import ALPHABET, score_records

EXP = 'KEY-BANK-NEURAL-001'
OUT, BULK = ROOT/'results'/EXP, ROOT/'outputs'/EXP
NAMES = tuple(f'B-key{i}' for i in range(1, 9))
ARMS = ('neural-31103', 'neural-31109')
PATHS = sorted(set(OLD_PATHS+NEURAL_PATHS+[
    'src/voynich/candidate_rerank.py', 'scripts/run_key_bank_neural001.py',
    'tests/test_candidate_rerank.py', 'tests/test_key_bank_neural001.py',
    'docs/experiments/KEY-BANK-NEURAL-001.md',
    'results/KEY-BANK-READ-002/audit.json', 'results/KEY-BANK-READ-002/campaign.json',
    'results/KEY-BANK-READ-002/evaluation.json',
]+[f'results/KEY-BANK-READ-002/{name}.json' for name in NAMES]))


def inputs(freeze):
    require_frozen(freeze, PATHS)
    admitted, _ = old_admission(freeze)
    audit = json.loads((ROOT/'results/KEY-BANK-READ-002/audit.json').read_text())
    if (audit['status'] != 'PASS' or audit['campaign'] != artifact(ROOT/'results/KEY-BANK-READ-002/campaign.json')):
        raise ValueError('Original candidate predictions lack matching audit')
    manifest = json.loads((ROOT/MANIFEST).read_text())
    cases = {}
    for name in NAMES:
        path = ROOT/f'results/KEY-BANK-READ-002/{name}.json'
        header = json.loads(path.read_text())
        if (header['status'] != 'complete' or header['bank'] != admitted[name]['bank']
                or header['transfer'] != manifest['cases'][name]['artifacts']['transfer']
                or audit['cases'][name]['result'] != artifact(path)):
            raise ValueError('Original reading input identity mismatch')
        cases[name] = header
    return cases, sources()


def literal_support_audit(prepared, bank, records, indices):
    active = prepared['active_bank_indices']
    weights = [bank['bank'][i]['log_weight'] for i in active]
    z = logsum(weights)
    checks, delta = 0, 0.
    for index in sorted(set(indices)):
        row = prepared['candidates'][index]
        texts = [prepared['texts'][i] for i in row['text_indices']]
        supported = []
        for i, weight in zip(active, weights, strict=True):
            table = dict(zip(ALPHABET, bank['bank'][i]['units'], strict=True))
            if all(''.join(table[c] for c in t) == r for t, r in zip(texts, records, strict=True)):
                supported.append(weight-z)
            checks += 1
        expected = logsum(supported)
        close(expected, row['key_log_mass'])
        delta = max(delta, abs(expected-row['key_log_mass']))
    return {'literal_candidate_key_checks': checks, 'maximum_delta': delta}


def sequence_scores(model, texts, device):
    """Every complete record starts at BOS; never chunk a candidate history."""
    result = score_records(model, texts, device, batch=64, length=max(map(len, texts)))
    if result['chunks'] != len(texts) or result['characters'] != sum(map(len, texts)):
        raise ValueError('Candidate source histories were truncated')
    return [-bits*math.log(2)+len(text)*math.log1p(-1/225)+math.log(1/225)
            for text, bits in zip(texts, result['record_bits'], strict=True)]


def numerical_audit(model, prepared, scores, ranked):
    """Full-sequence CPU float64 sample, and independent stepwise winning paths."""
    chosen = prepared['candidates'][ranked['candidate_index']]['text_indices']
    runner = ranked['runner_up_index']
    sampled = {i*(len(prepared['texts'])-1)//15 for i in range(16)} | set(chosen)
    if runner is not None:
        sampled.update(prepared['candidates'][runner]['text_indices'])
    indices = sorted(sampled)
    reference = copy.deepcopy(model).to('cpu').double().eval()
    exact = sequence_scores(reference, [prepared['texts'][i] for i in indices], 'cpu')
    delta = max(abs(scores[i]-v) for i, v in zip(indices, exact, strict=True))
    if delta > 1e-3:
        raise ValueError('MPS versus float64 record-score gate failed')
    lookup = dict(zip(indices, exact, strict=True))
    stepwise_delta = 0.
    with torch.inference_mode():
        for index in chosen:
            state, previous, values = None, reference.bos, []
            for char in prepared['texts'][index]:
                logits, state = reference(torch.tensor([[previous]]), state)
                target = reference.config['alphabet'].index(char)
                values.append(float(torch.log_softmax(logits[0, 0], -1)[target]))
                previous = target
            value = math.fsum(values)+len(values)*math.log1p(-1/225)+math.log(1/225)
            stepwise_delta = max(stepwise_delta, abs(value-lookup[index]))
    if stepwise_delta > 1e-7:
        raise ValueError('Full-forward versus stepwise recurrence mismatch')
    winner_score = math.fsum(lookup[i] for i in chosen)+prepared['candidates'][ranked['candidate_index']]['key_log_mass']
    gap = None
    if runner is not None:
        row = prepared['candidates'][runner]
        gap = winner_score-math.fsum(lookup[i] for i in row['text_indices'])-row['key_log_mass']
        if gap < -2e-3:
            raise ValueError('Leading candidate order reversed beyond numerical tolerance')
    return {'sampled_record_indices': indices, 'maximum_mps_float64_delta': delta,
            'winning_record_stepwise_checks': len(chosen), 'maximum_stepwise_delta': stepwise_delta,
            'float64_winner_minus_mps_runner_nats': gap,
            'sampled_numerics_not_global_floating_certificate': True}


def predict(freeze):
    headers, selected = inputs(freeze)
    if not torch.backends.mps.is_available():
        raise RuntimeError('MPS required; launch with device access, no CPU fallback')
    save_new(OUT/'prediction-started.json', {'source_freeze': freeze, 'start_unix': time.time()})
    limit_resources(2400, 1800)
    wall, cpu = time.monotonic(), time.process_time()
    torch.set_num_threads(2)
    try:
        prepared, banks, records, reports = {}, {}, {}, {}
        for name in NAMES:
            header = headers[name]
            data, bank = checked_artifact(header['transfer']), load_archive(header['bank'])
            reading = load_archive(header['readings'])
            value = prepare_candidates(ALPHABET, data['records'], bank, reading)
            if len(value['texts']) > 20000 or len(value['candidates']) > 40000:
                raise RuntimeError('Predeclared whole-case candidate cap; no truncation')
            sample = {i*(len(value['candidates'])-1)//7 for i in range(8)} | {value['statistical_best_index']}
            check = literal_support_audit(value, bank, data['records'], sample)
            archive = save_new(BULK/f'{name}-candidates.json.gz', value, compressed=True)
            reports[name] = {'prepared': archive, 'original': artifact(ROOT/f'results/KEY-BANK-READ-002/{name}.json'),
                'transfer': header['transfer'], 'bank': header['bank'], 'support_audit': check,
                'candidates': len(value['candidates']), 'unique_records': len(value['texts']),
                'characters': sum(map(len, value['texts'])), 'predictions': {}}
            prepared[name], banks[name], records[name] = value, bank, data['records']
        for arm in ARMS:
            model = load_neural(selected[arm])
            for name in NAMES:
                value, scores = prepared[name], []
                for start in range(0, len(value['texts']), 64):
                    scores.extend(sequence_scores(model, value['texts'][start:start+64], 'mps'))
                    if resource_report(wall, cpu)['peak_rss_bytes'] > 4*1024**3 or torch.mps.driver_allocated_memory() > 2*1024**3:
                        raise MemoryError('Sampled 4GiB host / 2GiB MPS driver limit')
                ranked = rank_candidates(value, scores)
                numeric = numerical_audit(model, value, scores, ranked)
                support = literal_support_audit(value, banks[name], records[name], [ranked['candidate_index']])
                packed = save_new(BULK/f'{name}-{arm}.json.gz', {'record_scores': scores, **ranked}, compressed=True)
                result = {'case': name, 'arm': arm, 'source_freeze': freeze, 'source': selected[arm],
                    'prepared': reports[name]['prepared'], 'readings': packed, 'numerical_audit': numeric,
                    'support_audit': support, 'candidate_index': ranked['candidate_index'], 'margin_nats': ranked['margin_nats']}
                reports[name]['predictions'][arm] = save_new(OUT/f'{name}-{arm}.json', result)
                print(json.dumps({'case': name, 'arm': arm, 'unique_records': len(scores),
                                  'margin_nats': ranked['margin_nats'], 'numerical_delta': numeric['maximum_mps_float64_delta']}), flush=True)
            del model
            torch.mps.empty_cache()
        save_new(OUT/'predictions.json', {'status': 'complete', 'experiment': EXP, 'source_freeze': freeze,
            'sources': selected, 'cases': reports, 'no_answers_opened': True,
            'global_map_or_neural_evidence_claimed': False, 'resources': resource_report(wall, cpu)})
    except Exception as error:
        signal.alarm(0)
        save_new(OUT/'prediction-failure.json', {'error': str(error), 'type': type(error).__name__, 'resources': resource_report(wall, cpu)})
        raise
    finally:
        signal.alarm(0)


def evaluate(freeze):
    _, selected = inputs(freeze)
    require_frozen(freeze, [str(p.relative_to(ROOT)) for p in OUT.glob('*.json')])
    manifest = json.loads((ROOT/MANIFEST).read_text())
    complete_path = OUT/'predictions.json'
    complete = json.loads(complete_path.read_text()) if complete_path.exists() else None
    started = json.loads((OUT/'prediction-started.json').read_text())
    if complete is not None and (complete['sources'] != selected or set(complete['cases']) != set(NAMES)):
        raise ValueError('Complete source/case accounting differs')
    save_new(OUT/'evaluation-started.json', {'freeze': freeze, 'start_unix': time.time()})
    limit_resources(300, 240)
    wall, cpu = time.monotonic(), time.process_time()
    reports, checks = {}, 0
    old = json.loads((ROOT/'results/KEY-BANK-READ-002/evaluation.json').read_text())
    for name in NAMES:
        truth = checked_artifact(manifest['cases'][name]['artifacts']['answer'])['plaintext']['transfer']
        if len(truth) != 2 or any(len(t) != 224 for t in truth):
            raise ValueError('Fixed exposed denominator changed')
        reports[name] = {}
        for arm in ARMS:
            path = OUT/f'{name}-{arm}.json'
            result = checked_artifact(artifact(path)) if path.exists() else None
            if result is None:
                rows = [{'plaintext': None}]*2
                available = False
            else:
                if (result['case'] != name or result['arm'] != arm or result['source'] != selected[arm]
                        or result['source_freeze'] != started['source_freeze']
                        or result['prepared'] != artifact(BULK/f'{name}-candidates.json.gz')
                        or result['numerical_audit']['maximum_mps_float64_delta'] > 1e-3
                        or result['numerical_audit']['maximum_stepwise_delta'] > 1e-7):
                    raise ValueError('Prediction source/input/numerical audit differs')
                prepared, values = load_archive(result['prepared']), load_archive(result['readings'])
                rebuilt = rank_candidates(prepared, values['record_scores'])
                if rebuilt != {k: v for k, v in values.items() if k != 'record_scores'}:
                    raise ValueError('Saved reranking arithmetic changed')
                if complete is not None and complete['cases'][name]['predictions'][arm] != artifact(path):
                    raise ValueError('Final prediction binding differs')
                rows, available = [{'plaintext': t} for t in values['plaintexts']], True
            value = metrics(rows, truth)
            if value['record_edits'] != [edit_reference(r['plaintext'] or '', t) for r, t in zip(rows, truth, strict=True)]:
                raise ValueError('Independent edit mismatch')
            checks += 2
            reports[name][arm] = {**value, 'available': available}
    totals = {arm: {k: sum(reports[n][arm][k] for n in NAMES) for k in
                    ('edits', 'gold_characters', 'supported_records', 'exact_records')} for arm in ARMS}
    previous = old['totals']['mixture']['edits']
    clauses = {'all_sixteen_case_model_predictions_complete_audited': complete is not None and complete['status'] == 'complete',
               'both_seeds_strictly_improve_43_edit_baseline': all(totals[a]['edits'] < previous for a in ARMS),
               'both_seeds_support_all_records': all(totals[a]['supported_records'] == 16 for a in ARMS),
               'neither_seed_has_a_key_above_five_percent': all(reports[n][a]['edits'] <= .05*448 for n in NAMES for a in ARMS)}
    save_new(OUT/'evaluation.json', {'experiment': EXP, 'freeze': freeze, 'cases': reports, 'totals': totals,
        'baseline_edits': previous, 'development_clauses': clauses, 'development_gate': 'PASS' if all(clauses.values()) else 'FAIL',
        'independent_edit_checks': checks, 'not_fresh_or_null_qualification': True, 'resources': resource_report(wall, cpu)})
    signal.alarm(0)
    print(json.dumps({'totals': totals, 'clauses': clauses}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('predict', 'evaluate'))
    parser.add_argument('--freeze', required=True)
    args = parser.parse_args()
    (predict if args.mode == 'predict' else evaluate)(args.freeze)
