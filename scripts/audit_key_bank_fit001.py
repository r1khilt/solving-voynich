"""Complete-bank artifact/arithmetic audit and independent sampled fit replay.

No transfer or answer access. The complete candidate inventory is checked;
only declared sampled candidates get a second numerical likelihood evaluation.
"""
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
from scripts.run_key_bank_fit001 import OUT, PATHS, ROOT
from scripts.run_latin_source_model001 import artifact, limit_resources
from voynich.compact_suffix_reference import LiteralCountView
from voynich.compact_suffix_source import CompactSuffixSource


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--case', required=True)
    args = parser.parse_args()
    result_path = OUT / f'{args.case}.json'
    result = json.loads(result_path.read_text())
    require_frozen(result['freeze'], PATHS)
    if result['status'] != 'complete_fit_only_bank':
        raise ValueError('No complete bank available')
    limit_resources(300, 300)
    wall, cpu = time.monotonic(), time.process_time()
    for field in ('fit', 'parent', 'source', 'bank'):
        if artifact(ROOT / result[field]['path']) != result[field]:
            raise ValueError('Artifact identity changed: ' + field)
    bank = load_archive(result['bank'])
    observed = checked_artifact(result['fit'])
    parent = tuple(checked_artifact(result['parent'])['units'])
    context = observed['context']
    pool = [''.join(s) for n in range(1, context['max_emission_length'] + 1)
            for s in itertools.product(context['glyph_alphabet'], repeat=n)]
    expected = dict.fromkeys([parent])
    for left, right in itertools.combinations(range(len(parent)), 2):
        key = list(parent)
        key[left], key[right] = key[right], key[left]
        expected[tuple(key)] = None
    for i in range(len(parent)):
        for unit in pool:
            expected[parent[:i] + (unit,) + parent[i + 1:]] = None
    if [tuple(r['units']) for r in bank['bank']] != list(expected):
        raise ValueError('Incomplete or altered neighborhood')
    progress_path = ROOT / f'outputs/KEY-BANK-FIT-001/{args.case}-progress.jsonl.gz'
    trace = [json.loads(line) for line in gzip.decompress(progress_path.read_bytes()).splitlines()]
    if trace != [{k: v for k, v in row.items() if k != 'log_weight'} for row in bank['bank']]:
        raise ValueError('Final bank does not match the complete retained trace')
    log_values = []
    for i, row in enumerate(bank['bank']):
        def width(x):
            return (x - 1).bit_length()
        code = (width(context.get('source_count', 1)) + width(context['max_states'])
                + len(parent) * (width(context['max_alternatives']) + width(context['max_emission_length']))
                + sum(map(len, row['units'])) * width(len(context['glyph_alphabet'])))
        values = row['record_log_likelihoods']
        total = None if any(v is None for v in values) else math.fsum(values)
        expected_weight = None if total is None else total - code * math.log(2)
        if (row['index'] != i or row['model_bits'] != code or len(values) != 4
                or row['fit_log_likelihood'] != total or row['log_weight_unnormalized'] != expected_weight):
            raise ValueError('Candidate likelihood/code arithmetic differs')
        log_values.append(-math.inf if expected_weight is None else expected_weight)
    high = max(log_values)
    normalizer = high + math.log(math.fsum(math.exp(v - high) for v in log_values))
    best = log_values.index(high)
    if (best != result['best_index'] or best != bank['best_index']
            or bank['bank'][best]['units'] != result['best_units']
            or result['bank_size'] != len(expected) or bank['bank_size'] != len(expected)
            or result['finite_keys'] != sum(math.isfinite(v) for v in log_values)
            or abs(normalizer - bank['fit_log_normalizer_restricted_not_full_family_evidence']) > 1e-9):
        raise ValueError('Best key/normalizer/count mismatch')
    for row, value in zip(bank['bank'], log_values):
        if value == -math.inf:
            if row['log_weight'] is not None:
                raise ValueError('Unsupported candidate has mass')
        elif abs(row['log_weight'] - (value - normalizer)) > 1e-9:
            raise ValueError('Normalized key mass differs')
    selected = checked_artifact(result['source'])
    if artifact(ROOT / selected['counts']['path']) != selected['counts']:
        raise ValueError('Source count archive changed')
    compact = CompactSuffixSource.load(ROOT / selected['counts']['path'])
    raw = {'alphabet': compact.alphabet, 'order': 12, 'tau': selected['selected']['tau'],
           'counts': LiteralCountView(compact)}
    indices = sorted({0, best, len(expected) // 2, len(expected) - 1})
    maximum, count = 0., 0
    for index in indices:
        row = bank['bank'][index]
        for j, record in enumerate(observed['records']):
            total, _, nodes = infer_reference(raw, row['units'], record, context['stop_probability'])
            expected_value = row['record_log_likelihoods'][j]
            if nodes != row['record_nodes'][j] or math.isfinite(total) != (expected_value is not None):
                raise ValueError('Independent support/node count differs')
            if expected_value is not None:
                maximum = max(maximum, abs(total - expected_value))
            count += 1
    if maximum > 1e-7:
        raise ValueError('Independent likelihood mismatch')
    signal.alarm(0)
    save_new(OUT / f'{args.case}-audit.json', {
        'status': 'PASS', 'result': artifact(result_path),
        'auditor': artifact(ROOT / 'scripts/audit_key_bank_fit001.py'),
        'candidates_accounted': len(expected), 'sampled_indices': indices,
        'independent_records': count, 'maximum_log_likelihood_delta': maximum,
        'all_candidate_likelihoods_independently_replayed': False,
        'resources': resource_report(wall, cpu)})
    print(json.dumps({'case': args.case, 'candidates': len(expected), 'records': count, 'maximum_delta': maximum}))


if __name__ == '__main__':
    main()
