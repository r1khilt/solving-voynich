"""One fixed exact finite qualification, before any original-source label moves."""
import argparse
import itertools
import signal
import time
from fractions import Fraction as F

from scripts.run_blind_channel_dev004 import resource_report, save_new
from scripts.run_joint_key_train001 import require_frozen
from scripts.run_latin_source_model001 import ROOT, artifact, limit_resources
from tests.test_reading_label_transport import finite_label_panel

EXP = 'READING-LABEL-TRANSPORT-THEORY-001'
PATHS = ('src/voynich/reading_label_transport.py', 'src/voynich/reading_regrowth.py',
    'src/voynich/source_action_proposal.py', 'src/voynich/joint_key_proposal.py',
    'tests/test_reading_label_transport.py', 'scripts/check_reading_label_transport001.py',
    'scripts/audit_reading_label_transport001.py', 'scripts/run_joint_key_train001.py',
    'scripts/run_blind_channel_dev001.py', 'scripts/run_latin_source_model001.py',
    'scripts/run_blind_channel_dev004.py', 'docs/research/reading-label-transport-2026-10-02.md',
    'docs/experiments/READING-LABEL-TRANSPORT-THEORY-001.md')


def check(freeze):
    require_frozen(freeze, PATHS)
    out = ROOT/'results'/EXP
    save_new(out/'started.json', {'freeze': freeze, 'no_retry': True, 'start_unix': time.time()})
    limit_resources(120, 100)
    wall, cpu = time.monotonic(), time.process_time()
    try:
        panels = [{'rows': rows, 'contextual': context, 'bias': str(bias),
                   **finite_label_panel(rows, context, bias)}
                  for rows, context, bias in itertools.product((2, 3), (False, True), (F(1), F(6)))]
        assert len(panels) == 8
        totals = {k: sum(row[k] for row in panels) for k in panels[0] if k not in ('rows', 'contextual', 'bias')}
        assert all(v > 0 for v in totals.values())
        resources = resource_report(wall, cpu)
        assert resources['wall_seconds'] <= 120 and resources['cpu_seconds'] <= 100
        assert resources['peak_rss_bytes'] <= 512*1024**2
        save_new(out/'result.json', {'status': 'PASS_exact_finite_label_and_fixed_mixture', 'freeze': freeze,
            'inputs': [artifact(ROOT/p) for p in PATHS], 'panels': panels, 'totals': totals,
            'resources': resources, 'paid_spend_usd': 0, 'no_empirical_source_or_recovery': True,
            'independent_expert_review': False, 'fixed_mixture_controller_not_implemented': True})
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
