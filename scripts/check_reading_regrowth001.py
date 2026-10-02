"""One registered exact finite qualification of globally revisable readings."""
import argparse
import itertools
import signal
import time
from fractions import Fraction as F

from scripts.run_joint_key_train001 import require_frozen
from scripts.run_latin_source_model001 import ROOT, artifact, limit_resources
from scripts.run_blind_channel_dev004 import resource_report, save_new
from tests.test_reading_regrowth import finite_regrowth_panel

EXP = 'READING-REGROWTH-THEORY-001'
PATHS = ('src/voynich/reading_regrowth.py', 'src/voynich/source_action_proposal.py',
         'src/voynich/joint_key_proposal.py', 'tests/test_reading_regrowth.py',
         'scripts/check_reading_regrowth001.py', 'scripts/audit_reading_regrowth001.py',
         'scripts/run_joint_key_train001.py', 'scripts/run_blind_channel_dev001.py',
         'scripts/run_latin_source_model001.py', 'scripts/run_blind_channel_dev004.py',
         'docs/research/reading-regrowth-2026-10-02.md',
         'docs/experiments/READING-REGROWTH-THEORY-001.md')


def check(freeze):
    require_frozen(freeze, PATHS)
    out = ROOT/'results'/EXP
    save_new(out/'started.json', {'freeze': freeze, 'start_unix': time.time(), 'no_retry': True})
    limit_resources(600, 500)
    wall, cpu = time.monotonic(), time.process_time()
    try:
        records = tuple(r for n in (1, 2) for r in itertools.product(range(2), repeat=n))
        panels = []
        for panel, contextual, bias, root in itertools.product(itertools.product(records, repeat=2),
                (False, True), (F(1), F(6)), (F(0), F(1, 8), F(1))):
            panels.append({'records': panel, 'contextual': contextual, 'bias': str(bias), 'root': str(root),
                           **finite_regrowth_panel(panel, contextual, bias, root)})
        failures = []
        single = (*tuple(r for n in (1, 2, 3) for r in itertools.product(range(2), repeat=n)),
                  (0, 1, 0, 1), (1, 0, 1, 0))
        for record, root in itertools.product(single, (F(0), F(1, 8), F(1))):
            failures.append({'record': record, 'root': str(root),
                             **finite_regrowth_panel((record,), False, F(6), root, rows=1)})
        totals = {k: sum(r[k] for r in panels+failures) for k in ('states', 'balance_pairs',
            'omitted_cut_flux_failures', 'omitted_prior_flux_failures', 'inventory_changing_edges',
            'different_length_edges', 'failed_branch_selfloops')}
        assert len(panels) == 432 and len(failures) == 48
        assert all(v > 0 for v in totals.values())
        assert all(r['omitted_cut_flux_failures'] == 0 for r in panels if r['root'] == '1')
        resources = resource_report(wall, cpu)
        assert resources['peak_rss_bytes'] <= 1536*1024**2
        save_new(out/'result.json', {'status': 'PASS_exact_finite_regrowth_kernel', 'freeze': freeze,
            'inputs': [artifact(ROOT/p) for p in PATHS], 'panels': panels, 'failure_panels': failures,
            'totals': totals, 'resources': resources, 'paid_spend_usd': 0,
            'no_empirical_source_or_recovery': True, 'independent_expert_review': False})
        return totals
    except Exception as error:
        save_new(out/'failure.json', {'error': repr(error), 'no_retry': True,
                                     'resources': resource_report(wall, cpu)})
        raise
    finally:
        signal.alarm(0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--freeze', required=True)
    print(check(parser.parse_args().freeze))
