"""ONE shape qualification receipt/hash/arithmetic closure; no reenumeration."""
import argparse
import json
import signal
import time

from scripts import check_constraint_shape001 as run


def dimensions(rows, records):
    if len(records) == 2:
        return rows**2, rows*(rows-1), 36*rows**2-30*rows
    if len(records[0]) == 3:
        return (rows**3+2*rows**2, rows**3-rows,
                6*rows+108*rows*(rows-1)+216*rows*(rows-1)*(rows-2)+72*rows**2-48*rows)
    return rows**2+rows, rows**2+(rows if records[0] == [0, 0] else 0), 36*rows**2-24*rows


def validate(result, freeze):
    assert result['status'] == 'PASS_exact_collapsed_shape_targets_components_and_exchanges'
    assert result['freeze'] == freeze
    assert result['inputs'] == [run.training.artifact(run.ROOT/p) for p in run.PATHS]
    assert [(p['rows'], p['contextual'], p['records']) for p in result['panels']] == [
        (rows, context, list(map(list, records))) for rows, context, records in run.PANELS]
    assert result['epsilons'] == [str(e) for e in run.EPSILONS] and result['kernels'] == list(run.KERNELS)
    keys = set(result['panels'][0])-{'rows', 'contextual', 'records', 'state_sha256'}
    assert result['totals'] == {k: sum(p[k] for p in result['panels']) for k in sorted(keys)}
    assert all(v > 0 for v in result['totals'].values())
    for panel in result['panels']:
        states, hard, assignments = dimensions(panel['rows'], panel['records'])
        assert panel['states'] == panel['soft_connected_states'] == states
        assert panel['hard_states'] == panel['hard_bijection_checks'] == hard
        assert panel['enumerated_visited_dictionary_assignments'] == assignments
        assert panel['summed_dictionary_terms'] == len(run.EPSILONS)*assignments
        assert panel['independent_marginal_target_checks'] == len(run.EPSILONS)*states
        assert panel['component_normalizations'] == len(run.KERNELS)*states
        assert panel['stationarity_equations'] == len(run.KERNELS)*len(run.EPSILONS)*states
        assert panel['exact_exchange_flux_checks'] == hard*states+2*states**2
        assert panel['hard_incompatible_exchange_rejections'] == hard*(states-hard)
        assert len(panel['state_sha256']) == 64 and all(c in '0123456789abcdef' for c in panel['state_sha256'])
    r = result['resources']
    assert 0 <= r['wall_seconds'] <= run.WALL and 0 <= r['cpu_seconds'] <= run.CPU
    assert r['peak_rss_bytes'] <= run.HOST and r['paid_spend_usd'] == 0
    assert result['no_original_source_rng_neural_training_or_recovery'] is True
    assert result['independent_expert_review'] is False
    assert result['independent_direct_shapes_components_and_full_dictionary_sums'] is True


def audit(freeze):
    run.training.old.require_frozen(freeze, run.PATHS)
    out = run.ROOT/'results'/run.EXP
    run.training.save_new(out/'audit-started.json', {'freeze': freeze, 'no_retry': True})
    run.training.limit_resources(60, 50)
    wall, cpu = time.monotonic(), time.process_time()
    try:
        result = json.loads((out/'result.json').read_text())
        validate(result, freeze)
        resources = run.training.resource_report(wall, cpu)
        assert resources['wall_seconds'] <= 60 and resources['cpu_seconds'] <= 50
        assert resources['peak_rss_bytes'] <= run.HOST
        run.training.save_new(out/'audit.json', {'status': 'PASS_collapsed_shape_receipt_hash_and_arithmetic_closure',
            'freeze': freeze, 'result': run.training.artifact(out/'result.json'), 'resources': resources,
            'no_sampling_or_reenumeration': True, 'not_independent_expert_or_kernel_reproof': True})
        return 'PASS_collapsed_shape_receipt_hash_and_arithmetic_closure'
    except Exception as error:
        run.training.save_new(out/'audit-failure.json', {'error': repr(error), 'no_retry': True,
            'resources': run.training.resource_report(wall, cpu)})
        raise
    finally:
        signal.alarm(0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--freeze', required=True)
    print(audit(parser.parse_args().freeze))
