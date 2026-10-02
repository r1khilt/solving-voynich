"""Single receipt closure: hashes, panel identities and counters, no re-enumeration."""
import argparse
import itertools
import json
import signal
import time

from scripts import check_reading_label_transport001 as run


def audit(freeze):
    run.require_frozen(freeze, run.PATHS)
    out = run.ROOT/'results'/run.EXP
    run.save_new(out/'audit-started.json', {'freeze': freeze, 'no_retry': True})
    run.limit_resources(60, 50)
    wall, cpu = time.monotonic(), time.process_time()
    try:
        result = json.loads((out/'result.json').read_text())
        assert result['freeze'] == freeze and result['status'] == 'PASS_exact_finite_label_and_fixed_mixture'
        assert result['inputs'] == [run.artifact(run.ROOT/p) for p in run.PATHS]
        assert [(p['rows'], p['contextual'], p['bias']) for p in result['panels']] == list(
            itertools.product((2, 3), (False, True), ('1', '6')))
        assert result['totals'] == {k: sum(p[k] for p in result['panels']) for k in result['totals']}
        assert all(v > 0 for v in result['totals'].values())
        assert all(p['balance_pairs_per_reversible_kernel'] == p['states']**2 for p in result['panels'])
        assert result['resources']['wall_seconds'] <= 120 and result['resources']['cpu_seconds'] <= 100
        assert result['resources']['peak_rss_bytes'] <= 512*1024**2
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
