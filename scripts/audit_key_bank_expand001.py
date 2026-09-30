"""Complete expansion/weight audit plus sampled independent backward inference."""
import argparse
import gzip
import itertools
import json
import math
import signal
import time

from scripts.audit_suffix_reader001 import infer_reference
from scripts.run_blind_channel_dev001 import checked_artifact, require_frozen
from scripts.run_blind_channel_dev004 import load_archive, resource_report, save_new
from scripts.run_key_bank_expand001 import OUT, PATHS, ROOT
from scripts.run_latin_source_model001 import artifact, limit_resources
from voynich.compact_suffix_reference import LiteralCountView
from voynich.compact_suffix_source import CompactSuffixSource


def close(left, right):
    if left is None or right is None:
        if left != right:
            raise ValueError('Support differs')
        return 0.
    if not math.isfinite(left) or not math.isfinite(right) or abs(left - right) > 1e-7:
        raise ValueError('Numerical arithmetic mismatch')
    return abs(left - right)


def inventory(bank, parent, context):
    rows = bank['bank']
    keys = [tuple(r['units']) for r in rows]
    if not rows or keys[0] != tuple(parent) or len(keys) != len(set(keys)):
        raise ValueError('Invalid parent/key inventory')
    pool = tuple(''.join(s) for length in (1, 2) for s in itertools.product(context['glyph_alphabet'], repeat=length))
    weights = []
    for i, row in enumerate(rows):
        if row['index'] != i or len(row['units']) != len(parent) or any(u not in pool for u in row['units']):
            raise ValueError('Invalid indexed key family')
        def width(n):
            return (n-1).bit_length()
        code = (width(context.get('source_count', 1)) + width(context['max_states'])
                + len(parent)*(width(context['max_alternatives']) + width(context['max_emission_length']))
                + sum(map(len, row['units']))*width(len(context['glyph_alphabet'])))
        logs = row['record_log_likelihoods']
        if len(logs) != len(row['record_nodes']) or len(logs) != len(row['record_edges']):
            raise ValueError('Record inventory mismatch')
        total = None if None in logs else math.fsum(logs)
        weight = None if total is None else total-code*math.log(2)
        if row['model_bits'] != code:
            raise ValueError('Code cost differs')
        close(total, row['fit_log_likelihood'])
        close(weight, row['log_weight_unnormalized'])
        weights.append(-math.inf if weight is None else weight)
    if (bank['maximum_rounds'] != 4 or bank['tolerance_nats'] != 1e-6
            or bank['global_optimality_claimed'] or not 1 <= len(bank['rounds']) <= 4):
        raise ValueError('Search configuration differs')
    seen, events, current = {}, [], 0
    by_key = {key: i for i, key in enumerate(keys)}
    for round_index, report in enumerate(bank['rounds']):
        center = keys[current]
        neighborhood = dict.fromkeys([center])
        for i, j in itertools.combinations(range(len(center)), 2):
            key = list(center)
            key[i], key[j] = key[j], key[i]
            neighborhood[tuple(key)] = None
        for i in range(len(center)):
            for unit in pool:
                neighborhood[center[:i]+(unit,)+center[i+1:]] = None
        if any(key not in by_key for key in neighborhood):
            raise ValueError('Missing neighborhood key')
        members = [by_key[key] for key in neighborhood]
        for index in members:
            if index not in seen:
                if index != len(seen):
                    raise ValueError('New candidate ordering differs')
                seen[index] = None
                events.append({'kind': 'candidate', 'value': {k: v for k, v in rows[index].items() if k != 'log_weight'}})
        selected = max(members, key=lambda index: weights[index])
        gain = weights[selected]-weights[current]
        accepted = gain > 1e-6
        expected = {'round': round_index, 'parent_index': current, 'member_indices': members,
                    'selected_index': selected, 'gain_nats': gain, 'accepted': accepted,
                    'cumulative_unique_keys': len(seen), 'complete_neighborhood': True}
        if report != expected:
            raise ValueError('Round membership/selection/gain differs')
        events.append({'kind': 'round', 'value': report})
        if accepted:
            current = selected
        elif round_index != len(bank['rounds'])-1:
            raise ValueError('Search continued after local tolerance stop')
    reason = 'round_limit' if bank['rounds'][-1]['accepted'] else 'local_tolerance_stop_at_center'
    if reason == 'round_limit' and len(bank['rounds']) != 4:
        raise ValueError('Incomplete round budget incorrectly completed')
    best = max(range(len(rows)), key=lambda index: weights[index])
    if (len(seen) != len(rows) or bank['bank_size'] != len(rows) or bank['best_index'] != best
            or bank['best_units'] != rows[best]['units'] or bank['search_center_index'] != current
            or bank['stop_reason'] != reason or bank['complete_rounds'] != len(bank['rounds'])
            or bank['finite_keys'] != sum(math.isfinite(w) for w in weights)):
        raise ValueError('Terminal bank summary differs')
    high = max(weights)
    shifted = math.log(math.fsum(math.exp(w-high) for w in weights))
    close(bank['fit_log_normalizer_restricted_not_full_family_evidence'], high+shifted)
    for row, weight in zip(rows, weights, strict=True):
        close(row['log_weight'], None if weight == -math.inf else (weight-high)-shifted)
    close(bank['maximum_weight'], math.exp(-shifted))
    return events


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--case', required=True)
    args = parser.parse_args()
    result_path = OUT / f'{args.case}.json'
    result = json.loads(result_path.read_text())
    require_frozen(result['freeze'], PATHS)
    if result['status'] != 'complete_fit_only_expanded_bank':
        raise ValueError('No complete expanded bank')
    limit_resources(600, 600)
    wall, cpu = time.monotonic(), time.process_time()
    for field in ['fit', 'parent', 'source', 'native_benchmark', 'bank', 'progress']:
        if artifact(ROOT/result[field]['path']) != result[field]:
            raise ValueError('Artifact identity changed: '+field)
    bank, observed = load_archive(result['bank']), checked_artifact(result['fit'])
    if any(len(row['record_log_likelihoods']) != len(observed['records']) for row in bank['bank']):
        raise ValueError('Fitting record count differs')
    parent = checked_artifact(result['parent'])['units']
    events = inventory(bank, parent, observed['context'])
    actual = [json.loads(line) for line in gzip.decompress((ROOT/result['progress']['path']).read_bytes()).splitlines()]
    if events != actual:
        raise ValueError('Progress stream differs from complete candidate/round inventory')
    for field in ['bank_size', 'best_index', 'finite_keys', 'best_units', 'search_center_index',
                  'complete_rounds', 'maximum_rounds', 'stop_reason', 'maximum_weight']:
        if result[field] != bank[field]:
            raise ValueError('Compact result differs')
    old_path = ROOT / f'results/KEY-BANK-FIT-001/{args.case}.json'
    old_checks, old_delta = 0, 0.
    if old_path.exists():
        old_result = json.loads(old_path.read_text())
        old = load_archive(old_result['bank'])
        members = bank['rounds'][0]['member_indices']
        if old_result['fit'] != result['fit'] or len(members) != old['bank_size']:
            raise ValueError('Old first-bank identity differs')
        for index, prior in zip(members, old['bank'], strict=True):
            row = bank['bank'][index]
            if tuple(row['units']) != tuple(prior['units']) or row['model_bits'] != prior['model_bits']:
                raise ValueError('First-round candidate/code differs')
            if row['record_nodes'] != prior['record_nodes']:
                raise ValueError('First-round graph counts differ')
            for a, b in zip(row['record_log_likelihoods'], prior['record_log_likelihoods'], strict=True):
                old_delta = max(old_delta, close(a, b))
                old_checks += 1
    selected = checked_artifact(result['source'])
    if artifact(ROOT/selected['counts']['path']) != selected['counts']:
        raise ValueError('Source archive changed')
    compact = CompactSuffixSource.load(ROOT/selected['counts']['path'])
    raw = {'alphabet': compact.alphabet, 'order': 12, 'tau': selected['selected']['tau'], 'counts': LiteralCountView(compact)}
    indices = sorted({0, bank['best_index'], len(bank['bank'])//2, len(bank['bank'])-1,
                      *(r['selected_index'] for r in bank['rounds'])})
    count, delta = 0, 0.
    for index in indices:
        row = bank['bank'][index]
        for j, record in enumerate(observed['records']):
            total, _, nodes = infer_reference(raw, row['units'], record, observed['context']['stop_probability'])
            value = None if total == -math.inf else total
            delta = max(delta, close(value, row['record_log_likelihoods'][j]))
            if nodes != row['record_nodes'][j]:
                raise ValueError('Independent string-state node count differs')
            count += 1
    signal.alarm(0)
    save_new(OUT/f'{args.case}-audit.json', {'status': 'PASS', 'result': artifact(result_path),
        'auditor': artifact(ROOT/'scripts/audit_key_bank_expand001.py'), 'candidates_accounted': bank['bank_size'],
        'complete_rounds': bank['complete_rounds'], 'sampled_indices': indices, 'independent_records': count,
        'maximum_log_likelihood_delta': delta, 'original_bank_records_compared': old_checks,
        'original_bank_maximum_delta': old_delta, 'all_candidate_likelihoods_independently_replayed': False,
        'resources': resource_report(wall, cpu)})
    print(json.dumps({'case': args.case, 'records': count, 'delta': delta, 'original_records': old_checks}), flush=True)


if __name__ == '__main__':
    main()
