"""ONE separately frozen closure of saved panels; original scientific run stays FAIL.

No panel enumeration, random draw, source scoring, or scientific retry occurs here.
The recorded counter table is post-outcome evidence, not a new success criterion.
"""
import argparse
from copy import deepcopy
import hashlib
import json
import math
import signal
import time

from scripts import check_constraint_shape001 as original

EXP = 'CONSTRAINT-SHAPE-THEORY-001-CLOSE'
ROOT, training = original.ROOT, original.training
ORIGINAL_FREEZE = 'ebd45a9d9ad34bbdfc9000cd552de1133c91ede4'
FAILURE_SHA = 'dde6f5dac474699d30c3d35cfa5fa91edb0df454b4fc45e9d6c0a4a5f88aa3ac'
START_SHA = 'fa85ab2e8a3af34a0854425d0a91d4fdbfbb39aafcc31b892577b8c17ba3fff5'
WALL, CPU, HOST = 60, 50, 512*1024**2
PATHS = tuple(sorted(set([*original.PATHS,
    'results/CONSTRAINT-SHAPE-THEORY-001/failure.json',
    'results/CONSTRAINT-SHAPE-THEORY-001/started.json',
    'scripts/close_constraint_shape001_receipt.py',
    'tests/test_constraint_shape_receipt_closure.py',
    'docs/experiments/CONSTRAINT-SHAPE-THEORY-001-CLOSE.md'])))
META = {'rows', 'contextual', 'records', 'state_sha256'}
ZEROS = {'hard_incompatible_exchange_rejections', 'hard_incompatible_local_rejections'}
COUNTERS = (
    'component_normalizations', 'direct_proposal_component_checks',
    'enumerated_visited_dictionary_assignments', 'exact_exchange_flux_checks',
    'exact_local_flux_checks', 'hard_bijection_checks',
    'hard_incompatible_exchange_rejections', 'hard_incompatible_local_rejections',
    'hard_states', 'independent_marginal_target_checks', 'reverse_component_checks',
    'soft_connected_states', 'states', 'stationarity_equations',
    'summed_dictionary_terms', 'wrong_explicit_key_exchange_flux_failures',
    'wrong_proposal_flux_failures', 'wrong_visited_prior_flux_failures')
# The eight preserved counter vectors repeat for constant/contextual sources.
# These literal observations bind every count; arithmetic below is independent of them.
SAVED_COUNTS = (
    (24,117,252,396,363,9,27,21,9,48,96,12,12,96,1008,216,198,216),
    (24,117,252,432,384,12,0,0,12,48,96,12,12,96,1008,270,216,240),
    (18,63,234,216,132,6,18,12,6,36,36,9,9,72,936,72,0,72),
    (90,693,2430,5130,1974,24,504,150,24,180,558,45,45,360,9720,3600,1068,1044),
    (40,272,480,1120,904,16,64,40,16,80,236,20,20,160,1920,544,480,528),
    (40,272,480,1200,944,20,0,0,20,80,236,20,20,160,1920,672,512,576),
    (32,144,456,704,360,12,48,24,12,64,96,16,16,128,1824,192,0,144),
    (192,2080,7416,24192,6516,60,2160,420,60,384,1792,96,96,768,29664,16832,3504,3480))


def dimensions(rows, records):
    """Independent partition/Stirling counting; U=6, no state enumeration."""
    if records == [[0], [1]]:
        return rows**2, rows*(rows-1), 36*rows**2-30*rows
    if records == [[0, 1, 0]]:
        # Each mixed-width partition contributes 6R+36R(R-1).
        return (rows**3+2*rows**2, rows**3-rows,
            6*rows+108*rows*(rows-1)+216*rows*(rows-1)*(rows-2)+72*rows**2-60*rows)
    assert records in ([[0, 1]], [[0, 0]])
    return rows**2+rows, rows**2+(rows if records == [[0, 0]] else 0), 36*rows**2-24*rows


def resource_check(value, wall, cpu):
    assert set(value) == {'wall_seconds', 'cpu_seconds', 'peak_rss_bytes', 'python',
        'platform', 'paid_spend_usd', 'thread_environment'}
    for field, cap in (('wall_seconds', wall), ('cpu_seconds', cpu), ('paid_spend_usd', 0)):
        assert type(value[field]) in (int, float) and math.isfinite(value[field])
        assert 0 <= value[field] <= cap
    assert type(value['peak_rss_bytes']) is int and 0 <= value['peak_rss_bytes'] <= HOST
    assert isinstance(value['python'], str) and value['python']
    assert isinstance(value['platform'], str) and value['platform']
    assert value['thread_environment'] == {
        'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': None}


def validate_saved(failure_raw, start_raw):
    """Hash original receipts, strictly restore only eight provably-zero fields."""
    assert hashlib.sha256(failure_raw).hexdigest() == FAILURE_SHA
    assert hashlib.sha256(start_raw).hexdigest() == START_SHA
    failure, start = json.loads(failure_raw), json.loads(start_raw)
    assert set(start) == {'freeze', 'no_retry', 'start_unix'}
    assert start['freeze'] == ORIGINAL_FREEZE and start['no_retry'] is True
    assert type(start['start_unix']) in (int, float) and math.isfinite(start['start_unix'])
    assert start['start_unix'] > 0
    assert set(failure) == {'error', 'panels', 'resources', 'no_retry'}
    assert failure['error'] == "KeyError('hard_incompatible_exchange_rejections')"
    assert failure['no_retry'] is True
    resource_check(failure['resources'], original.WALL, original.CPU)
    assert type(failure['panels']) is list and len(failure['panels']) == 16
    panels, restored = deepcopy(failure['panels']), []
    for index, (panel, allocation) in enumerate(zip(panels, original.PANELS, strict=True)):
        rows, contextual, records = allocation
        assert type(panel['rows']) is int and panel['rows'] == rows
        assert type(panel['contextual']) is bool and panel['contextual'] is contextual
        assert type(panel['records']) is list and all(type(record) is list and
            all(type(glyph) is int for glyph in record) for record in panel['records'])
        assert panel['records'] == list(map(list, records))
        assert type(panel['state_sha256']) is str and len(panel['state_sha256']) == 64
        assert all(c in '0123456789abcdef' for c in panel['state_sha256'])
        states, hard, assignments = dimensions(rows, panel['records'])
        missing = set(COUNTERS)-set(panel)
        # Exactly the four 00 panels have no incompatible states in either kernel.
        assert missing == (ZEROS if panel['records'] == [[0, 0]] and hard == states else set())
        assert set(panel) == META | (set(COUNTERS)-missing)
        for field in sorted(missing):
            panel[field] = 0
            restored.append({'panel_index': index, 'counter': field, 'value': 0,
                'reason': 'Every enumerated shape is hard compatible (H=S)'})
        assert all(type(panel[field]) is int and panel[field] >= 0 for field in COUNTERS)
        expected = SAVED_COUNTS[(0 if rows == 3 else 4)+index % 4]
        assert tuple(panel[field] for field in COUNTERS) == expected
        assert panel['states'] == panel['soft_connected_states'] == states
        assert panel['hard_states'] == panel['hard_bijection_checks'] == hard
        assert panel['enumerated_visited_dictionary_assignments'] == assignments
        assert panel['summed_dictionary_terms'] == 4*assignments
        assert panel['independent_marginal_target_checks'] == 4*states
        assert panel['component_normalizations'] == 2*states
        assert panel['stationarity_equations'] == 8*states
        assert panel['exact_exchange_flux_checks'] == hard*states+2*states**2
        assert panel['hard_incompatible_exchange_rejections'] == hard*(states-hard)
    assert len(restored) == 8
    totals = {field: sum(p[field] for p in panels) for field in COUNTERS}
    assert all(value > 0 for value in totals.values())
    assert (totals['states'], totals['hard_states'], totals['enumerated_visited_dictionary_assignments']) == (460,318,24000)
    return {'panels': panels, 'totals': totals, 'restored_zero_counters': restored,
        'original_resources': failure['resources']}


def close(freeze):
    training.old.require_frozen(freeze, PATHS)
    training.old.require_frozen(ORIGINAL_FREEZE, original.PATHS)
    origin = ROOT/'results'/original.EXP
    assert not any((origin/name).exists() for name in ('result.json', 'audit.json', 'audit-started.json', 'audit-failure.json'))
    out = ROOT/'results'/EXP
    training.save_new(out/'started.json', {'freeze': freeze, 'original_freeze': ORIGINAL_FREEZE, 'no_retry': True})
    training.limit_resources(WALL, CPU)
    wall, cpu = time.monotonic(), time.process_time()
    try:
        checked = validate_saved((origin/'failure.json').read_bytes(), (origin/'started.json').read_bytes())
        inputs = [training.artifact(ROOT/path) for path in PATHS]
        training.old.require_frozen(freeze, PATHS)
        training.old.require_frozen(ORIGINAL_FREEZE, original.PATHS)
        resources = training.resource_report(wall, cpu)
        resource_check(resources, WALL, CPU)
        training.save_new(out/'closure.json', {'status': 'PASS_saved_shape_panel_receipt_closure_original_run_failed',
            'freeze': freeze, 'original_freeze': ORIGINAL_FREEZE, 'original_status': 'FAIL',
            'original_error': "KeyError('hard_incompatible_exchange_rejections')",
            'original_failure': training.artifact(origin/'failure.json'),
            'original_started': training.artifact(origin/'started.json'), 'inputs': inputs,
            **checked, 'resources': resources, 'no_retry': True,
            'no_sampling_reenumeration_source_scoring_or_training': True,
            'not_independent_expert_or_kernel_reproof': True,
            'observed_counter_table_is_post_outcome_evidence': True,
            'corrected_length3_assignment_term': '72R^2-60R',
            'registered_assignment_total_was_incorrect': 24168})
        return 'PASS_saved_shape_panel_receipt_closure_original_run_failed'
    except Exception as error:
        training.save_new(out/'failure.json', {'error': repr(error), 'no_retry': True,
            'resources': training.resource_report(wall, cpu)})
        raise
    finally:
        signal.alarm(0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--freeze', required=True)
    print(close(parser.parse_args().freeze))
