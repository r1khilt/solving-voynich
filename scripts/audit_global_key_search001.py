"""Complete fit trace/code/binding accounting and sampled reverse-source replay."""
import argparse
import gc
import gzip
import json
import math
import time
from dataclasses import asdict

from scripts.audit_blind_channel_confirm002 import literal_source
from scripts.audit_suffix_reader001 import infer_reference
from scripts.confirm002_common import NAMES, observed
from scripts.diagnose_blind_channel_confirm002a import endpoints
from scripts.run_blind_channel_dev001 import require_frozen
from scripts.run_blind_channel_dev004 import load_archive, resource_report, save_new
from scripts.run_global_key_search001 import (OUT, PATHS, ROOT, MAX_NODES, config_for,
                                              admitted, reader_union)
from scripts.run_latin_source_model001 import artifact, limit_resources


def validate_trace(search, cfg):
    bank, events = search['bank'], search['events']
    assert len(bank) == search['scored_keys'] <= cfg.max_scored_keys
    assert len({tuple(r['units']) for r in bank}) == len(bank)
    assert search['temperatures'] == list(cfg.temperatures)
    def value(i): return -math.inf if bank[i]['score'] is None else bank[i]['score']
    assert all(row['index'] == i for i, row in enumerate(bank))
    assert value(search['best_index']) == max(value(i) for i in range(len(bank))) == search['best_score']
    assert bank[search['best_index']]['units'] == search['best_units']
    replicas = list(search['initial_replica_indices'])
    assert len(replicas) == len(cfg.temperatures) and all(math.isfinite(value(i)) for i in replicas)
    counts = {k: {'proposed': 0, 'accepted': 0} for k in ('replace', 'swap', 'block')}
    proposals = exchanges = exchange_accepted = self_moves = 0
    expected = []
    rounds = 0
    for iteration in range(cfg.iterations):
        expected.extend(('proposal', iteration, r) for r in range(len(replicas)))
        if (iteration+1) % cfg.swap_interval == 0:
            expected.extend(('exchange', iteration, left) for left in range(rounds % 2, len(replicas)-1, 2))
            rounds += 1
    assert [(e['event'], e['iteration'], e.get('replica', e.get('left'))) for e in events] == expected[:len(events)]
    for e in events:
        if e['event'] == 'proposal':
            replica, old, new = e['replica'], e['previous_index'], e['candidate_index']
            assert replicas[replica] == old and len(e['rows']) == len(e['new_units'])
            before, after = bank[old]['units'], bank[new]['units']
            changed = list(before)
            for row, unit in zip(e['rows'], e['new_units'], strict=True):
                changed[row] = unit
            assert changed == after
            if e['kind'] == 'replace':
                assert len(e['rows']) == 1 and old != new
            elif e['kind'] == 'swap':
                a, b = e['rows']
                assert after[a] == before[b] and after[b] == before[a]
            else:
                assert e['kind'] == 'block' and len(e['rows']) in cfg.block_sizes
            ratio = -math.inf if value(new) == -math.inf else min(0., (value(new)-value(old))/cfg.temperatures[replica])
            threshold = -math.inf if e['log_acceptance'] is None else e['log_acceptance']
            assert ratio == threshold
            if e['accepted']:
                replicas[replica] = new
            counts[e['kind']]['proposed'] += 1
            counts[e['kind']]['accepted'] += int(e['accepted'])
            proposals += 1
            self_moves += int(old == new)
        else:
            a, b = e['left'], e['right']
            assert b == a+1 and [replicas[a], replicas[b]] == e['previous_indices']
            ratio = min(0., (value(replicas[b])-value(replicas[a]))*(1/cfg.temperatures[a]-1/cfg.temperatures[b]))
            threshold = e['log_acceptance']
            assert abs(ratio-threshold) < 1e-12
            if e['accepted']:
                replicas[a], replicas[b] = replicas[b], replicas[a]
            exchanges += 1
            exchange_accepted += int(e['accepted'])
        draw = -math.inf if e['log_uniform'] is None else e['log_uniform']
        assert e['accepted'] == (draw < threshold)
    assert replicas == search['final_replica_indices'] and proposals == search['proposals']
    assert counts == search['proposal_counts'] and self_moves == search['self_proposals']
    assert search['completed_iterations'] == proposals//len(replicas)
    assert search['exchange_counts'] == {'proposed': exchanges, 'accepted': exchange_accepted}
    assert search['cache_hits'] == len(search['initial_replica_indices'])+proposals-len(bank)
    assert not search['global_optimality_claimed'] and not search['stationary_distribution_claimed']
    return {'proposals': proposals, 'exchanges': exchanges, 'search_keys': len(bank)}


def audit(name):
    if name not in NAMES:
        raise ValueError('Unregistered case')
    path = OUT/f'{name}.json'
    result = json.loads(path.read_text())
    require_frozen(result['freeze'], PATHS)
    panel, _ = admitted(result['freeze'])
    assert result['status'] == 'complete_fit_only' and result['case'] == name
    assert result['fit'] == panel['cases'][name]['fit'] and result['config'] == json.loads(json.dumps(asdict(config_for(name))))
    save_new(OUT/f'{name}-audit-started.json', {'result': artifact(path), 'start_unix': time.time()})
    limit_resources(900, 850)
    wall, cpu = time.monotonic(), time.process_time()
    for field in ('fit', 'old_fit', 'old_bank', 'parent', 'warm', 'search', 'reader_bank', 'progress'):
        assert artifact(ROOT/result[field]['path']) == result[field]
    data = observed(result['fit'], 4)
    parent = json.loads((ROOT/result['parent']['path']).read_text())
    prior = json.loads((ROOT/result['old_fit']['path']).read_text())
    assert parent['fit'] == prior['fit'] == result['fit'] and prior['bank'] == result['old_bank']
    legacy = load_archive(parent['stage1_trace'])
    terminal = endpoints(legacy['trace'])
    expected_warm_keys = list(dict.fromkeys([tuple(prior['best_units']),
        *(tuple(row['units']) for row in terminal.values() if row['score'] is not None)]))
    assert len(terminal) == parent['stage1_accounting']['scored_initializations'] <= 16
    del legacy, terminal
    gc.collect()
    packed, reader = load_archive(result['search']), load_archive(result['reader_bank'])
    search, work = packed['search'], packed['scores']
    counts = validate_trace(search, config_for(name))
    cache = {tuple(row['units']): row for row in work}
    assert len(cache) == len(work) == result['unique_full_fit_scores_including_warm_and_neighborhood']
    assert packed['warm'] == [cache[key] for key in expected_warm_keys]
    ranked = sorted(expected_warm_keys, key=lambda key: (-cache[key]['log_weight_unnormalized'], expected_warm_keys.index(key)))
    selected_old = tuple(prior['best_units'])
    starts = [selected_old]+[key for key in ranked if key != selected_old][:7]
    starts += [selected_old]*(8-len(starts))
    assert packed['warm_starts'] == [list(key) for key in starts]
    assert [search['bank'][i]['units'] for i in search['initial_replica_indices']] == packed['warm_starts']
    warm = json.loads((ROOT/result['warm']['path']).read_text())
    assert warm['starts'] == packed['warm_starts'] and warm['scores'] == packed['warm'] and warm['trace'] == parent['stage1_trace']
    assert warm['parent'] == result['parent'] and warm['old_fit'] == result['old_fit']
    context = data['context']
    def width(n): return (n-1).bit_length()
    header = width(context['source_count'])+width(context['max_states'])
    fields = len(context['source_alphabet'])*(width(context['max_alternatives'])+width(context['max_emission_length']))
    for row in work:
        assert len(row['record_log_likelihoods']) == len(row['record_nodes']) == len(row['record_edges']) == 4
        bits = header+fields+sum(map(len, row['units']))*width(len(context['glyph_alphabet']))
        assert bits == row['model_bits']
        logs = row['record_log_likelihoods']
        total = None if None in logs else sum(logs)
        assert (total is None) == (row['fit_log_likelihood'] is None)
        if total is not None:
            assert abs(total-row['fit_log_likelihood']) <= 1e-7
        expected = None if total is None else total-bits*math.log(2)
        assert (expected is None) == (row['log_weight_unnormalized'] is None)
        if expected is not None:
            assert abs(expected-row['log_weight_unnormalized']) <= 1e-7
    for row in search['bank']:
        assert row['score'] == cache[tuple(row['units'])]['log_weight_unnormalized']
    old = load_archive(result['old_bank'])
    visited = [cache[tuple(row['units'])] for row in search['bank']]
    neighborhood = [cache[tuple(key)] for key in packed['neighborhood_units']]
    assert reader == reader_union(old, visited, neighborhood)
    assert result['reader_summary'] == {k: v for k, v in reader.items() if k != 'bank'}
    reader_h = reader['bank'][reader['best_index']]['log_weight_unnormalized']
    old_h = old['bank'][old['best_index']]['log_weight_unnormalized']
    warm_h = max(row['log_weight_unnormalized'] for row in packed['warm'])
    assert abs(result['fit_objective_gain_nats']-(reader_h-old_h)) <= 1e-7
    assert abs(result['gain_vs_best_warm_nats']-(reader_h-warm_h)) <= 1e-7
    assert abs(result['search_gain_vs_best_warm_nats']-(search['best_score']-warm_h)) <= 1e-7
    assert result['search_summary'] == {k: v for k, v in search.items() if k not in ('bank', 'events')}
    with gzip.open(ROOT/result['progress']['path'], 'rt') as stream:
        score_index = event_index = 0
        for line in stream:
            event = json.loads(line)
            if event['kind'] == 'score':
                assert event['value'] == work[score_index]
                score_index += 1
            else:
                assert event['kind'] == 'event' and event['value'] == search['events'][event_index]
                event_index += 1
    assert score_index == len(work) and event_index == len(search['events'])
    # A distinct reverse/string-context algorithm, not native or forward replay.
    selected = [tuple(reader['best_units']), tuple(search['best_units']),
                *(tuple(search['bank'][i]['units']) for i in search['initial_replica_indices'])]
    selected = list(dict.fromkeys(selected))
    references = [(key, cache.get(key, next((r for r in reader['bank'] if tuple(r['units']) == key), None))) for key in selected]
    assert len(references) <= 10
    del packed, old, reader, search, visited, neighborhood
    gc.collect()
    raw = literal_source()
    maximum = 0.
    for key, row in references:
        for observed_record, expected in zip(data['records'], row['record_log_likelihoods'], strict=True):
            total, _, _ = infer_reference(raw, key, observed_record, context['stop_probability'], max_nodes=MAX_NODES)
            assert (total == -math.inf) == (expected is None)
            if expected is not None:
                delta = abs(total-expected)
                assert math.isfinite(delta) and delta <= 1e-7
                maximum = max(maximum, delta)
    assert resource_report(wall, cpu)['peak_rss_bytes'] <= 4*1024**3
    save_new(OUT/f'{name}-audit.json', {'status': 'PASS', 'case': name, 'result': artifact(path),
        'auditor': artifact(ROOT/'scripts/audit_global_key_search001.py'), 'trace_counts': counts,
        'full_fit_scores_accounted': len(work), 'reference_keys': len(references),
        'reference_records': len(references)*4, 'maximum_delta': maximum, 'resources': resource_report(wall, cpu),
        'transfer_or_answer_files_opened': False, 'same_author_not_independent_agent_review': True})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--case', required=True)
    audit(parser.parse_args().case)
