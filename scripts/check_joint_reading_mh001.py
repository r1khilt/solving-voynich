"""One frozen exact-arithmetic joint reading MH qualification, no corpus."""

import argparse
import itertools
import signal
import time

from scripts.run_joint_key_train001 import require_frozen
from scripts.run_latin_source_model001 import ROOT, artifact, limit_resources
from scripts.run_blind_channel_dev004 import resource_report, save_new
from tests.test_joint_reading_mh import finite_panel, proposal_law
from voynich.source_action_proposal import ReadingEnvironment

EXP = 'JOINT-READING-MH-THEORY-001'
PATHS = ('src/voynich/joint_reading_mh.py', 'src/voynich/source_action_proposal.py',
         'src/voynich/joint_key_proposal.py', 'tests/test_joint_reading_mh.py',
         'scripts/check_joint_reading_mh001.py', 'scripts/run_joint_key_train001.py',
         'scripts/run_latin_source_model001.py', 'scripts/run_blind_channel_dev004.py',
         'docs/research/joint-reading-independence-2026-10-02.md',
         'docs/experiments/JOINT-READING-MH-THEORY-001.md')


def check(freeze):
    require_frozen(freeze, PATHS)
    out = ROOT/'results'/EXP
    save_new(out/'started.json', {'freeze': freeze, 'start_unix': time.time(), 'no_retry': True})
    limit_resources(600, 500)
    wall, cpu = time.monotonic(), time.process_time()
    try:
        records = tuple(r for n in (1, 2) for r in itertools.product(range(2), repeat=n))
        panels = []
        for panel, contextual, weighted in itertools.product(itertools.product(records, repeat=2),
                                                            (False, True), (False, True)):
            panels.append({'records': panel, 'contextual': contextual, 'weighted': weighted,
                           **finite_panel(panel, contextual, weighted)})
        failure_panels = []
        for n in (1, 2, 3):
            for record in itertools.product(range(2), repeat=n):
                q, failed = proposal_law(ReadingEnvironment((record,), rows=1, glyphs=2))
                failure_panels.append({'record': record, 'complete_states': len(q), 'failed_mass': str(failed)})
        totals = {name: sum(r[name] for r in panels) for name in
                  ('states', 'full_joint_states', 'balance_pairs', 'full_balance_pairs', 'unused_witnesses',
                   'duplicate_witnesses', 'key_density_witnesses', 'omitted_prior_balance_failures',
                   'key_marginal_balance_failures')}
        assert len(panels) == 144 and len(failure_panels) == 14
        assert all(totals[k] > 0 for k in totals)
        assert any(r['failed_mass'] != '0' for r in failure_panels)
        assert any(r['distinct_used_inventories'] > 1 for r in panels)
        report = resource_report(wall, cpu)
        assert report['peak_rss_bytes'] <= 1536*1024**2
        value = {'status': 'PASS_finite_joint_reading_target_proposal_balance_and_minorization',
                 'freeze': freeze, 'inputs': [artifact(ROOT/p) for p in PATHS],
                 'panels': panels, 'single_row_failure_panels': failure_panels, 'totals': totals,
                 'maximum_float_acceptance_delta': max(r['max_acceptance_delta'] for r in panels),
                 'resources': report, 'paid_spend_usd': 0, 'no_empirical_or_model_access': True,
                 'expert_independent_review': False, 'trained_support_or_recovery_claim': False}
        saved = save_new(out/'result.json', value)
        assert saved['bytes'] <= 2*1024**2
        return saved
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
