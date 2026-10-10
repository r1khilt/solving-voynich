"""ONE receipt/hash/arithmetic closure; no second finite enumeration or expert."""
import argparse
import json
import signal
import time

from scripts import check_window_reading001 as run


def validate(result, freeze):
    assert result['status'] == 'PASS_exact_window_flux_conditional_support_and_rng'
    assert result['freeze'] == freeze
    assert result['inputs'] == [run.artifact(run.ROOT/p) for p in run.PATHS]
    assert [(p['rows'], p['contextual'], p['records']) for p in result['panels']] == [
        (r, c, list(map(list, records))) for r, c, records in run.PANELS]
    keys = set(result['panels'][0])-{'rows', 'contextual', 'records', 'windows', 'degrees'}
    assert result['totals'] == {key: sum(p[key] for p in result['panels']) for key in sorted(keys)}
    assert all(value > 0 for value in result['totals'].values())
    for panel in result['panels']:
        assert panel['degrees'] == list(run.DEGREES)
        assert panel['windows'] == sum(2*len(r)-1 for r in panel['records'])
        assert panel['states'] == panel['cold_stationarity_equations'] > 0
        assert panel['states'] == panel['heatbath_stationarity_equations']
        assert panel['augmented_components'] == panel['valid_components']+panel['invalid_boundary_components']
        assert panel['exact_power_flux_checks'] == len(run.DEGREES)*panel['augmented_components']
        assert panel['split_components'] == panel['merge_components'] > 0
        assert panel['seeded_steps'] == run.STEPS
        assert panel['seeded_heatbath_steps'] == run.STEPS
        assert panel['exact_incremental_target_checks'] == panel['valid_components']
        assert 0 < panel['independent_categorical_draws'] <= run.STEPS
        assert 0 < panel['independent_root_decisions'] <= panel['seeded_steps']
    assert result['categorical_checks'] == {'independent_categorical_decisions': 1024,
                                          'two_block_and_abort_witnesses': 4}
    r = result['resources']
    assert 0 <= r['wall_seconds'] <= run.WALL and 0 <= r['cpu_seconds'] <= run.CPU
    assert r['peak_rss_bytes'] <= run.HOST and r['paid_spend_usd'] == 0
    assert result['no_original_source_training_gold_or_recovery'] is True
    assert result['independent_expert_review'] is False
    assert result['alternate_map_and_root_quantiles_but_legacy_enumerator_and_reference_shared'] is True


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
        run.save_new(out/'audit.json', {'status': 'PASS_window_receipt_hash_and_arithmetic_closure',
            'freeze': freeze, 'result': run.artifact(out/'result.json'), 'resources': resources,
            'no_sampling_or_reenumeration': True, 'not_independent_expert_or_kernel_reproof': True})
        return 'PASS_window_receipt_hash_and_arithmetic_closure'
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
