"""One full-source bounded proposal admission; no posterior/recovery qualification."""
import argparse
import json
import math
import signal
import time
from fractions import Fraction

import torch

from scripts import check_reading_regrowth001 as theory
from scripts import run_source_action_cipher001 as prior
from scripts.benchmark_source_prefix_systems001 import array_identity, load_source
from scripts.run_blind_channel_dev004 import load_archive
from voynich.joint_reading_source import source_point_checked
from voynich.reading_regrowth import RegrowthConfig, SourceRegrowth
from voynich.regrowth_trace import chain, log_target
from voynich.source_action_proposal import ReadingEnvironment
from voynich.source_action_training import action_episode, control_records
from voynich.joint_key_training import EpisodeSampler

EXP = 'READING-REGROWTH-ADMIT-001'
OUT, BULK = prior.training.ROOT/'results'/EXP, prior.training.ROOT/'outputs'/EXP
ARMS, STEPS, SEED = ('independence', 'regrowth'), 8, 92721
WALL, CPU, HOST, PRIVATE = 1800, 1800, 2*1024**3, 128*1024**2
PATHS = sorted(set([*prior.PATHS, *theory.PATHS, 'src/voynich/regrowth_trace.py',
    'tests/test_regrowth_trace.py', 'tests/test_reading_regrowth_admission.py',
    'scripts/run_reading_regrowth_admit001.py',
    'scripts/audit_reading_regrowth_admit001.py', 'docs/experiments/READING-REGROWTH-ADMIT-001.md',
    'results/READING-REGROWTH-THEORY-001/result.json', 'results/READING-REGROWTH-THEORY-001/audit.json',
    'results/SOURCE-ACTION-CIPHER-001/result.json', 'results/SOURCE-ACTION-CIPHER-001/audit.json']))


def configuration(arm):
    if arm not in ARMS:
        raise ValueError('Registered arm required')
    return RegrowthConfig(root_mass=Fraction(1) if arm == 'independence' else Fraction(1, 8))


def literal_replay(env, receipt):
    state = env.initial
    for action in receipt['actions']:
        state = env.advance(state, action)
    assert list(state.key) == receipt['key'] and list(state.offsets) == receipt['offsets']
    assert [list(t) for t in state.texts] == receipt['texts']
    assert (env.selected_record(state) is None) == receipt['complete']
    if not receipt['complete']:
        assert not env.legal_actions(state)
    return state


def initial_receipt(env, prediction):
    """Explicit adapter for the previous GREEDY schema, with no density reuse."""
    assert prediction['status'] == 'complete_path'
    receipt = {'actions': prediction['actions'], 'key': prediction['key'], 'texts': prediction['texts'],
               'offsets': list(map(len, env.records)), 'complete': True}
    literal_replay(env, receipt)
    return receipt


def point_check(source, sampler, path):
    point = source_point_checked(source, path.state.texts)
    expected = point['log_probability']-sum(k >= 0 for k in path.state.key)*math.log(len(sampler.env.pool))
    delta = abs(log_target(sampler, path)-expected)
    assert delta <= 1e-9
    return {'joint_log_potential': log_target(sampler, path), 'reference_policy_log_probability':
            path.policy_log_probability(), 'maximum_joint_log_delta': delta, **point}


def load_inputs():
    manifest, _, valid, episodes = prior.training.load_prepared()
    sampler = EpisodeSampler(valid)
    controls = control_records(episodes, seed=92451)
    allocated = [('positive', i, e[0]) for i, e in enumerate(episodes)]
    allocated.extend((c['kind'], c['paired_case'], c['records']) for c in controls)
    result = json.loads((prior.OUT/'result.json').read_text())
    closure = json.loads((prior.OUT/'audit.json').read_text())
    assert result['status'] == 'PASS_complete_registered_cipher_route_and_history_diagnostic'
    assert closure['status'] == 'PASS_saved_records_source_policy_and_metric_closure'
    assert closure['result'] == prior.training.artifact(prior.OUT/'result.json')
    initial = [r for r in result['free_cases'] if r['mode'] == 'source_increment']
    assert len(allocated) == len(initial) == 97 and [r['case'] for r in initial] == list(range(97))
    return manifest, episodes, sampler, allocated, initial


def summarize(cells):
    result = {}
    for arm in ARMS:
        rows = [c for c in cells if c['arm'] == arm]
        assert len(rows) == 97 and [r['case'] for r in rows] == list(range(97))
        names = ('attempts', 'complete_proposals', 'failed_proposals', 'accepted', 'accepted_inventory_changes',
                 'accepted_key_changes', 'accepted_length_changes', 'root_cuts', 'acceptance_raw64_blocks')
        result[arm] = {k: sum(r['summary'][k] for r in rows) for k in names}
        result[arm]['positive_joint_log_potential_change_sum'] = math.fsum(
            r['summary']['joint_log_potential_change'] for r in rows if r['kind'] == 'positive')
        result[arm]['cells_with_accepted_inventory_change'] = sum(
            r['summary']['accepted_inventory_changes'] > 0 for r in rows)
        result[arm]['mean_cell_wall_seconds'] = math.fsum(r['wall_seconds'] for r in rows)/len(rows)
    return result


def admit(freeze):
    prior.training.old.require_frozen(freeze, PATHS)
    t = json.loads((theory.ROOT/'results'/theory.EXP/'result.json').read_text())
    a = json.loads((theory.ROOT/'results'/theory.EXP/'audit.json').read_text())
    assert t['status'] == 'PASS_exact_finite_regrowth_kernel'
    assert a['result'] == theory.artifact(theory.ROOT/'results'/theory.EXP/'result.json')
    assert a['status'] == 'PASS_receipt_hash_and_arithmetic_closure'
    prior.training.save_new(OUT/'started.json', {'freeze': freeze, 'start_unix': time.time(), 'no_retry': True})
    prior.training.limit_resources(WALL, CPU)
    wall, cpu = time.monotonic(), time.process_time()
    torch.set_num_threads(1)
    BULK.mkdir(parents=True, exist_ok=True)
    private_bytes, cells, gold = 0, [], []
    try:
        def guard():
            report = prior.training.resource_report(wall, cpu)
            if (report['wall_seconds'] > WALL or report['cpu_seconds'] > CPU or report['peak_rss_bytes'] > HOST
                    or private_bytes > PRIVATE):
                raise MemoryError('Registered source proposal admission bound exceeded; no retry')
            return report
        manifest, episodes, episode_sampler, allocated, initials = load_inputs()
        source, selected = load_source()
        arrays = array_identity(source)
        guard()
        for i, episode in enumerate(episodes):
            env, truth = action_episode(episode_sampler, episode)
            sampler = SourceRegrowth(source, env, configuration('regrowth'))
            path = sampler.path(forced_actions=truth[1], progress=guard)
            assert path.state == truth[2]
            gold.append({'case': i, **point_check(source, sampler, path)})
        for i, (kind, paired, records) in enumerate(allocated):
            prior_case = initials[i]
            assert prior_case['kind'] == kind and prior_case['paired_case'] == paired
            initial = load_archive(prior_case['archive'])
            assert initial['records'] == [list(r) for r in records]
            env = ReadingEnvironment(records, rows=23, glyphs=6)
            initial_receipt(env, initial['prediction'])
            for arm in ARMS:
                began = time.monotonic()
                sampler = SourceRegrowth(source, env, configuration(arm))
                value = chain(sampler, initial['prediction']['actions'], seed=SEED+i, steps=STEPS, progress=guard)
                value.update({'case': i, 'kind': kind, 'paired_case': paired, 'arm': arm,
                              'records': [list(r) for r in records], 'initial_archive': prior_case['archive']})
                for receipt in [value['initial'], *(p['candidate'] for p in value['proposals']), value['final']]:
                    literal_replay(env, receipt)
                final = sampler.path(forced_actions=tuple(value['final']['actions']))
                reference = point_check(source, sampler, final)
                spec = prior.training.save_new(BULK/f'{i}-{arm}.json.gz', value, compressed=True)
                private_bytes += spec['bytes']
                cells.append({'case': i, 'kind': kind, 'paired_case': paired, 'arm': arm, 'archive': spec,
                    'summary': value['summary'], 'final_source_reference': reference,
                    'wall_seconds': time.monotonic()-began})
                guard()
            print(json.dumps({'completed_cases': i+1, 'cases': 97}), flush=True)
        prior.training.old.require_frozen(freeze, PATHS)
        assert array_identity(source) == arrays
        value = {'status': 'PASS_full_source_regrowth_density_literal_and_cost_admission', 'freeze': freeze,
            'inputs': [prior.training.artifact(prior.training.ROOT/p) for p in PATHS], 'cells': cells,
            'gold_source_density_checks': gold, 'summary': summarize(cells), 'source_arrays': arrays,
            'source_counts': selected['counts'], 'source_inputs': selected['inputs'],
            'training_inputs': prior.training.artifact(prior.training.OUT/'inputs.json'), 'resources': guard(),
            'private_archive_bytes': private_bytes, 'paid_spend_usd': 0, 'exposed_development_only': True,
            'original_training_unchanged': True, 'posterior_convergence_or_recovery_claim': False}
        prior.training.save_new(OUT/'result.json', value)
        return value['summary']
    except Exception as error:
        prior.training.save_new(OUT/'failure.json', {'error': repr(error), 'no_retry': True, 'cells': cells,
            'gold_source_density_checks': gold, 'resources': prior.training.resource_report(wall, cpu)})
        raise
    finally:
        signal.alarm(0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--freeze', required=True)
    print(admit(parser.parse_args().freeze))
