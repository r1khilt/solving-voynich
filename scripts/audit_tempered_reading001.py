"""ONE hash/arithmetic closure of the finite qualification, without re-enumeration."""
import argparse
import json
import signal
import time
from decimal import Decimal

from scripts import check_tempered_reading001 as run


def validate(result, freeze):
    assert result['freeze'] == freeze
    assert result['status'] == 'PASS_exact_power_flux_radical_and_product_laws'
    assert result['inputs'] == [run.artifact(run.ROOT/p) for p in run.PATHS]
    assert [(p['rows'], p['contextual'], p['records']) for p in result['panels']] == [
        (r, c, [list(t) for t in records]) for r, c, records in run.PANELS]
    expected = {key: sum(p[key] for p in result['panels']) for key in result['panels'][0] if key not in
                ('rows', 'contextual', 'records', 'degrees', 'maximum_decimal_residual')}
    assert result['totals'] == expected and all(v > 0 for v in expected.values())
    for panel in result['panels']:
        assert panel['degrees'] == list(run.DEGREES)
        assert panel['fast_score_reference_states'] == panel['states'] > 0
        assert panel['replica_exchange_checks'] == 3*panel['states']**2
        assert panel['kernel_stationarity_equations'] == 16*panel['states']
        assert panel['exact_power_flux_checks'] == (panel['structural_components']+
            panel['label_components']+panel['regrowth_complete_components'])
        assert 0 <= Decimal(panel['maximum_decimal_residual']) < Decimal('1e-75')
    assert result['radical_checks'] == {'independent_radical_decisions': 6144,
                                       'two_block_boundary_and_abort_witnesses': 16}
    assert result['product_checks'] == {'product_states': 27, 'exact_product_stationarity_equations': 54}
    assert result['resources']['wall_seconds'] <= 1800 and result['resources']['cpu_seconds'] <= 1700
    assert result['resources']['peak_rss_bytes'] <= 512*1024**2
    assert result['resources']['paid_spend_usd'] == 0
    assert result['no_original_source_training_gold_or_recovery'] is True
    assert result['independent_expert_review'] is False


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
        assert resources['peak_rss_bytes'] <= 512*1024**2
        run.save_new(out/'audit.json', {'status': 'PASS_receipt_hash_and_arithmetic_closure', 'freeze': freeze,
            'result': run.artifact(out/'result.json'), 'resources': resources,
            'no_sampling_or_reenumeration': True, 'not_independent_expert_or_kernel_reproof': True})
        return 'PASS_receipt_hash_and_arithmetic_closure'
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
