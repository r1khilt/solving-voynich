"""ONE finite receipt/hash/arithmetic closure; not a second enumeration or expert review."""
import argparse
import json
import signal
import time

from scripts import check_constraint_reading001 as run


def validate(result, freeze):
    assert result['status'] == 'PASS_exact_constraint_component_target_exchange_and_support_laws'
    assert result['freeze'] == freeze
    assert result['inputs'] == [run.artifact(run.ROOT/p) for p in run.PATHS]
    assert [(p['rows'], p['contextual'], p['records']) for p in result['panels']] == [
        (r, context, list(map(list, records))) for r, context, records in run.PANELS]
    assert result['epsilons'] == [str(e) for e in run.EPSILONS] and result['kernels'] == list(run.KERNELS)
    assert result['boundary_witnesses'] == {'invalid_endpoint_identities': 4, 'invalid_endpoint_payload_rejections': 4}
    keys = set(result['panels'][0])-{'rows', 'contextual', 'records', 'state_sha256'}
    assert result['totals'] == {k: sum(p[k] for p in result['panels']) for k in sorted(keys)}
    assert all(v > 0 for v in result['totals'].values())
    for p in result['panels']:
        r = p['rows']
        expected_states = 36*r*r-(30 if len(p['records']) == 2 else 24)*r
        expected_hard = r*(r-1) if len(p['records']) == 2 else r*r+(r if p['records'][0] == [0, 0] else 0)
        assert p['states'] == p['soft_connected_states'] == p['independent_distance_checks'] == expected_states
        assert p['hard_states'] == p['hard_bijection_checks'] == expected_hard
        assert p['independent_target_checks'] == len(run.EPSILONS)*expected_states
        assert p['component_normalizations'] == len(run.KERNELS)*expected_states
        assert p['stationarity_equations'] == len(run.KERNELS)*(expected_hard+3*expected_states)
        assert p['exact_exchange_flux_checks'] == expected_hard*expected_states+2*expected_states**2
        assert p['hard_inconsistent_exchange_rejections'] == expected_hard*(expected_states-expected_hard)
        assert len(p['state_sha256']) == 64 and all(x in '0123456789abcdef' for x in p['state_sha256'])
    r = result['resources']
    assert 0 <= r['wall_seconds'] <= run.WALL and 0 <= r['cpu_seconds'] <= run.CPU
    assert r['peak_rss_bytes'] <= run.HOST and r['paid_spend_usd'] == 0
    assert result['no_original_source_scoring_gold_training_chain_rng_or_recovery'] is True
    assert result['independent_expert_review'] is False
    assert result['preparation_enumerator_and_components_shared'] is True


def audit(freeze):
    run.require_frozen(freeze, run.PATHS)
    out = run.ROOT/'results'/run.EXP
    run.save_new(out/'audit-started.json', {'freeze': freeze, 'no_retry': True})
    run.limit_resources(60, 50)
    wall, cpu = time.monotonic(), time.process_time()
    try:
        result = json.loads((out/'result.json').read_text())
        validate(result, freeze)
        resources = run.resource_report(wall, cpu)
        assert resources['wall_seconds'] <= 60 and resources['cpu_seconds'] <= 50
        assert resources['peak_rss_bytes'] <= run.HOST
        run.save_new(out/'audit.json', {'status': 'PASS_constraint_receipt_hash_and_arithmetic_closure',
            'freeze': freeze, 'result': run.artifact(out/'result.json'), 'resources': resources,
            'no_sampling_or_reenumeration': True, 'not_independent_expert_or_kernel_reproof': True})
        return 'PASS_constraint_receipt_hash_and_arithmetic_closure'
    except Exception as error:
        run.save_new(out/'audit-failure.json', {'error': repr(error), 'no_retry': True,
            'resources': run.resource_report(wall, cpu)})
        raise
    finally:
        signal.alarm(0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--freeze', required=True)
    print(audit(parser.parse_args().freeze))
