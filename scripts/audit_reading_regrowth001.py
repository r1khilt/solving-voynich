"""One read-only finite receipt closure; no re-enumeration or sampling."""
import argparse
import itertools
import json
import signal
import time

from scripts import check_reading_regrowth001 as run
from scripts.run_blind_channel_dev004 import resource_report, save_new


def audit(freeze):
    run.require_frozen(freeze, run.PATHS)
    out = run.ROOT/'results'/run.EXP
    save_new(out/'audit-started.json', {'freeze': freeze, 'no_retry': True})
    run.limit_resources(60, 50)
    wall, cpu = time.monotonic(), time.process_time()
    try:
        result = json.loads((out/'result.json').read_text())
        assert result['freeze'] == freeze and result['status'] == 'PASS_exact_finite_regrowth_kernel'
        assert result['inputs'] == [run.artifact(run.ROOT/p) for p in run.PATHS]
        records = tuple(r for n in (1, 2) for r in itertools.product(range(2), repeat=n))
        expected = [(list(map(list, panel)), contextual, bias, root)
                    for panel, contextual, bias, root in itertools.product(itertools.product(records, repeat=2),
                    (False, True), ('1', '6'), ('0', '1/8', '1'))]
        assert [(r['records'], r['contextual'], r['bias'], r['root']) for r in result['panels']] == expected
        single = (*tuple(r for n in (1, 2, 3) for r in itertools.product(range(2), repeat=n)),
                  (0, 1, 0, 1), (1, 0, 1, 0))
        assert [(r['record'], r['root']) for r in result['failure_panels']] == [
            (list(record), root) for record, root in itertools.product(single, ('0', '1/8', '1'))]
        all_rows = result['panels']+result['failure_panels']
        assert result['totals'] == {k: sum(r[k] for r in all_rows) for k in result['totals']}
        assert all(v > 0 for v in result['totals'].values())
        assert all(r['balance_pairs'] == r['states']**2 for r in all_rows)
        assert all(r['omitted_cut_flux_failures'] == 0 for r in result['panels'] if r['root'] == '1')
        assert result['resources']['wall_seconds'] <= 600 and result['resources']['cpu_seconds'] <= 500
        assert result['resources']['peak_rss_bytes'] <= 1536*1024**2
        save_new(out/'audit.json', {'status': 'PASS_receipt_hash_and_arithmetic_closure', 'freeze': freeze,
            'result': run.artifact(out/'result.json'), 'no_sampling_or_reenumeration': True,
            'not_independent_expert_or_kernel_reproof': True, 'resources': resource_report(wall, cpu)})
        return 'PASS_receipt_hash_and_arithmetic_closure'
    except Exception as error:
        save_new(out/'audit-failure.json', {'error': repr(error), 'no_retry': True,
                                           'resources': resource_report(wall, cpu)})
        raise
    finally:
        signal.alarm(0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--freeze', required=True)
    print(audit(parser.parse_args().freeze))
