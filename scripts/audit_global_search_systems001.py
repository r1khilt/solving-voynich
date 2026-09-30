"""Alternate arithmetic and complete trace/binding accounting after one run."""
import argparse
import gzip
import json
import math
import time
from dataclasses import asdict

from scripts.benchmark_global_search_systems001 import BULK, CONFIG, OUT, PATHS, ROOT, workloads
from scripts.run_blind_channel_dev001 import require_frozen
from scripts.run_blind_channel_dev004 import resource_report, save_new
from scripts.run_latin_source_model001 import artifact, limit_resources


def log_score(row):
    return -math.inf if row['score'] is None else row['score']


def trace(search, work, context):
    bank, events = search['bank'], search['events']
    assert len(bank) == len(work) == search['scored_keys'] <= CONFIG.max_scored_keys
    assert len({tuple(row['units']) for row in bank}) == len(bank)
    def width(n): return (n-1).bit_length()
    header = width(context['source_count'])+width(context['max_states'])
    fields = len(context['source_alphabet'])*(width(context['max_alternatives'])+width(context['max_emission_length']))
    for i, (row, value) in enumerate(zip(bank, work, strict=True)):
        assert row['index'] == i and row['units'] == value['units']
        bits = header+fields+sum(len(u)*width(len(context['glyph_alphabet'])) for u in row['units'])
        assert bits == value['model_bits']
        logs = value['record_log_likelihoods']
        expected = None if None in logs else sum(logs)-bits*math.log(2)
        assert (expected is None) == (row['score'] is None)
        if expected is not None:
            assert abs(expected-row['score']) <= 1e-7
    assert log_score(bank[search['best_index']]) == max(map(log_score, bank))
    assert search['best_units'] == bank[search['best_index']]['units']
    assert search['best_score'] == log_score(bank[search['best_index']])
    assert search['temperatures'] == list(CONFIG.temperatures)
    schedule = []
    exchange_round = 0
    for iteration in range(CONFIG.iterations):
        schedule.extend(('proposal', iteration, replica) for replica in range(len(CONFIG.temperatures)))
        if (iteration+1) % CONFIG.swap_interval == 0:
            schedule.extend(('exchange', iteration, left) for left in range(
                exchange_round % 2, len(CONFIG.temperatures)-1, 2))
            exchange_round += 1
    assert [(e['event'], e['iteration'], e['replica'] if e['event'] == 'proposal' else e['left'])
            for e in events] == schedule[:len(events)]
    replicas = list(search['initial_replica_indices'])
    proposals = exchanges = self_proposals = accepted_exchanges = 0
    counts = {kind: {'proposed': 0, 'accepted': 0} for kind in ('replace', 'swap', 'block')}
    for event in events:
        if event['event'] == 'proposal':
            replica, old, new = event['replica'], event['previous_index'], event['candidate_index']
            assert replicas[replica] == old
            before, after = bank[old]['units'], bank[new]['units']
            changed = list(before)
            assert len(event['rows']) == len(event['new_units'])
            for row, unit in zip(event['rows'], event['new_units'], strict=True):
                changed[row] = unit
            assert changed == after
            if event['kind'] == 'replace':
                assert len(event['rows']) == 1 and old != new
            elif event['kind'] == 'swap':
                a, b = event['rows']
                assert after[a] == before[b] and after[b] == before[a]
            else:
                assert len(event['rows']) in CONFIG.block_sizes
            a, b = log_score(bank[old]), log_score(bank[new])
            ratio = -math.inf if b == -math.inf else min(0., (b-a)/CONFIG.temperatures[replica])
            threshold = -math.inf if event['log_acceptance'] is None else event['log_acceptance']
            assert ratio == threshold
            if event['accepted']:
                replicas[replica] = new
            proposals += 1
            self_proposals += int(old == new)
            counts[event['kind']]['proposed'] += 1
            counts[event['kind']]['accepted'] += int(event['accepted'])
        else:
            assert event['event'] == 'exchange'
            a, b = event['left'], event['right']
            assert b == a+1 and (event['iteration']+1) % CONFIG.swap_interval == 0
            assert [replicas[a], replicas[b]] == event['previous_indices']
            old_a, old_b = (log_score(bank[replicas[i]]) for i in (a, b))
            ratio = min(0., (old_b-old_a)*(1/CONFIG.temperatures[a]-1/CONFIG.temperatures[b]))
            threshold = event['log_acceptance']
            assert abs(ratio-threshold) <= 1e-12
            if event['accepted']:
                replicas[a], replicas[b] = replicas[b], replicas[a]
            exchanges += 1
            accepted_exchanges += int(event['accepted'])
        draw = -math.inf if event['log_uniform'] is None else event['log_uniform']
        assert event['accepted'] == (draw < threshold)
    assert replicas == search['final_replica_indices']
    assert proposals == search['proposals'] and self_proposals == search['self_proposals']
    assert search['completed_iterations'] == proposals//len(CONFIG.temperatures)
    assert search['completed_iterations'] <= CONFIG.iterations
    assert counts == search['proposal_counts']
    assert search['exchange_counts'] == {'proposed': exchanges, 'accepted': accepted_exchanges}
    assert search['cache_hits'] == 1+proposals-len(bank)  # One scored initializer, copied to replicas.
    assert not search['global_optimality_claimed'] and not search['stationary_distribution_claimed']
    return {'scored_keys': len(bank), 'proposal_events': proposals, 'exchange_events': exchanges,
            'fit_records': sum(len(row['record_log_likelihoods']) for row in work)}


def main():
    parser = argparse.ArgumentParser()
    parser.parse_args()
    result = json.loads((OUT/'result.json').read_text())
    require_frozen(result['freeze'], PATHS)
    wall, cpu = time.monotonic(), time.process_time()
    limit_resources(120, 110)
    specs = [result[k] for k in ('inputs', 'native_benchmark', 'source', 'source_counts', 'search', 'progress', 'complete_scores')]
    specs += result['partial_results']
    for spec in specs:
        assert artifact(ROOT/spec['path']) == spec
    inputs = json.loads((ROOT/result['inputs']['path']).read_text())
    assert inputs == json.loads(json.dumps(workloads()))
    packed = json.loads(gzip.decompress((ROOT/result['search']['path']).read_bytes()))
    assert result['config'] == json.loads(json.dumps(asdict(CONFIG)))
    assert result['search_summary'] == {k: v for k, v in packed['search'].items() if k not in ('bank', 'events')}
    counts = trace(packed['search'], packed['scores'], inputs['context'])
    progress = [json.loads(row) for row in gzip.decompress((BULK/'progress.jsonl.gz').read_bytes()).splitlines()]
    assert [row['value'] for row in progress if row['kind'] == 'score'] == packed['scores']
    assert [row['value'] for row in progress if row['kind'] == 'event'] == packed['search']['events']
    complete = json.loads((ROOT/result['complete_scores']['path']).read_text())
    assert len(complete['timings']) == 8 and len(complete['equivalence']) == 16
    maximum = max(row['delta'] for row in complete['equivalence'])
    for row in complete['equivalence']:
        a, b = row['native'], row['reference']
        assert a['nodes'] == b['nodes'] and a['edges'] == b['edges']
        assert (a['log_likelihood'] is None) == (b['log_likelihood'] is None)
        assert row['delta'] == (0. if a['log_likelihood'] is None else abs(a['log_likelihood']-b['log_likelihood']))
    for assigned, spec in zip((0, 11, 20, 23), result['partial_results'], strict=True):
        row = json.loads((ROOT/spec['path']).read_text())
        assert row['assigned'] == assigned and row['native']['status'] == row['python']['status']
        if row['native']['status'] == 'complete':
            a, b = row['native']['value'], row['python']['value']
            assert a['nodes'] == b['nodes'] and a['edges'] == b['edges']
            delta = 0. if a['log_likelihood_upper_bound'] is None else abs(a['log_likelihood_upper_bound']-b['log_likelihood_upper_bound'])
            maximum = max(maximum, delta)
    assert maximum <= 1e-7 and maximum == result['maximum_likelihood_delta']
    assert result['engineering_gate'] == 'PASS' and all(result['engineering_clauses'].values())
    save_new(OUT/'audit.json', {'status': 'PASS', 'result': artifact(OUT/'result.json'), 'auditor': artifact(
        ROOT/'scripts/audit_global_search_systems001.py'), 'bindings': len(specs), 'trace_counts': counts,
        'progress_rows': len(progress), 'maximum_delta': maximum, 'resources': resource_report(wall, cpu),
        'no_empirical_cipher_panel_opened': True, 'same_author_alternate_arithmetic_not_agent_review': True})


if __name__ == '__main__':
    main()
