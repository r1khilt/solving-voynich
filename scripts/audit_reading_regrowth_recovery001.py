"""One complete replay of a frozen recovery campaign, without extending it."""
import argparse
import json
import math
import signal
import time
from fractions import Fraction

import torch

from scripts import run_reading_regrowth_recovery001 as run
from scripts.audit_reading_regrowth_admit001 import independent_dense_log_target
from scripts.audit_source_action_cipher001 import integer_edits
from scripts.benchmark_source_prefix_systems001 import array_identity, load_source
from scripts.run_blind_channel_dev004 import load_archive
from voynich.reading_endpoint_metrics import reading_endpoint_metrics
from voynich.reading_regrowth import SourceRegrowth
from voynich.regrowth_trace import chain
from voynich.source_action_proposal import ReadingEnvironment, ReadingState
from voynich.source_action_training import action_episode

WALL, CPU, HOST = 21600, 21000, 2*1024**3


def linear_literal(env, receipt):
    """Independent fractional scheduler/literal decoder; original boundary validation."""
    key, offsets, texts = [-1]*env.rows, [0]*len(env.records), [[] for _ in env.records]
    for action in receipt['actions']:
        if type(action) is not int or not 0 <= action < 2*env.rows:
            raise ValueError('Bounded integer reading action required')
        unfinished = [i for i, record in enumerate(env.records) if offsets[i] < len(record)]
        if not unfinished:
            raise ValueError('Extra actions beyond completion')
        i = min(unfinished, key=lambda j: (Fraction(offsets[j], len(env.records[j])), j))
        row, length = action//2, action%2+1
        unit = env.records[i][offsets[i]:offsets[i]+length]
        if len(unit) != length:
            raise ValueError('Overlong emitted unit')
        if key[row] < 0:
            key[row] = env.pool.index(unit)
        elif env.pool[key[row]] != unit:
            raise ValueError('Shared binding disagrees with literal observation')
        texts[i].append(row)
        offsets[i] += length
    state = ReadingState(tuple(key), tuple(offsets), tuple(map(tuple, texts)))
    env.validate(state)
    if (list(state.key) != receipt['key'] or list(state.offsets) != receipt['offsets']
            or [list(t) for t in state.texts] != receipt['texts']):
        raise ValueError('Independent literal path disagrees with saved endpoint')
    complete = env.selected_record(state) is None
    if type(receipt['complete']) is not bool or receipt['complete'] != complete or (
            not complete and env.legal_actions(state)):
        raise ValueError('Complete or truly dead boundary required')
    return state


def audit(freeze):
    training = run.admitted.prior.training
    training.old.require_frozen(freeze, run.PATHS)
    training.save_new(run.OUT/'audit-started.json', {'freeze': freeze, 'no_retry': True,
                                                   'no_chain_extension': True})
    training.limit_resources(WALL, CPU)
    wall, cpu = time.monotonic(), time.process_time()
    torch.set_num_threads(1)
    try:
        def guard():
            value = training.resource_report(wall, cpu)
            if value['wall_seconds'] > WALL or value['cpu_seconds'] > CPU or value['peak_rss_bytes'] > HOST:
                raise MemoryError('Registered recovery audit resource bound exceeded; no retry')
            return value
        result = json.loads((run.OUT/'result.json').read_text())
        assert result['status'] == 'COMPLETE_registered_recovery_campaign' and result['freeze'] == freeze
        assert result['inputs'] == [training.artifact(run.ROOT/p) for p in run.PATHS]
        assert [(c['seed'], c['case'], c['arm']) for c in result['cells']] == run.expected_cells()
        sealed = result['sealed_cells']
        assert training.artifact(run.ROOT/sealed['path']) == sealed
        seal = json.loads((run.ROOT/sealed['path']).read_text())
        assert seal['cells'] == result['cells'] and seal['no_gold_metrics_yet'] and seal['freeze'] == freeze
        manifest, episodes, episode_sampler, allocated, initials = run.admitted.load_inputs()
        assert len(allocated) == run.CASES and len(episodes) == run.POSITIVE
        source, selected = load_source()
        assert result['source_arrays'] == array_identity(source)
        assert result['source_counts'] == selected['counts'] and result['source_inputs'] == selected['inputs']
        endpoints = {str(seed): {arm: [] for arm in run.ARMS} for seed in run.SEEDS}
        initial_endpoints, maximum, attempted = [], 0., 0
        for cell_no, cell in enumerate(result['cells']):
            seed, i, arm = cell['seed'], cell['case'], cell['arm']
            kind, paired, records = allocated[i]
            assert cell['kind'] == kind and cell['paired_case'] == paired
            saved = load_archive(cell['archive'])
            assert (saved['seed_base'], saved['case'], saved['arm'], saved['kind'], saved['paired_case']) == (
                seed, i, arm, kind, paired)
            assert saved['records'] == [list(r) for r in records] and saved['seed'] == seed+i
            assert saved['steps'] == run.MOVES[arm] and saved['initial_archive'] == initials[i]['archive']
            initial = load_archive(saved['initial_archive'])
            env = ReadingEnvironment(records, rows=23, glyphs=6)
            run.admitted.initial_receipt(env, initial['prediction'])
            sampler = SourceRegrowth(source, env, run.admitted.configuration(arm))
            replay = chain(sampler, initial['prediction']['actions'], seed=seed+i,
                           steps=run.MOVES[arm], progress=guard)
            for name in ('seed', 'steps', 'initial', 'proposals', 'final', 'final_rng_state', 'summary'):
                assert replay[name] == saved[name]
            assert saved['summary'] == cell['summary']
            del replay
            for receipt in [saved['initial'], *(p['candidate'] for p in saved['proposals']), saved['final']]:
                state = linear_literal(env, receipt)
                if receipt['complete']:
                    # Independent source-context accumulation, no use of saved coefficient terms.
                    expected = independent_dense_log_target(source, env, state)
                    path_numerator = math.prod(n for n, _ in receipt['source_terms'])*224**len(receipt['actions'])
                    path_denominator = (1 << sum(e for _, e in receipt['source_terms']))*225**(
                        len(receipt['actions'])+len(env.records))*len(env.pool)**sum(k >= 0 for k in state.key)
                    delta = abs(math.log(path_numerator)-math.log(path_denominator)-expected)
                    maximum = max(maximum, delta)
                    assert delta <= 1e-9
            final = sampler.path(forced_actions=tuple(saved['final']['actions']), progress=guard)
            assert run.admitted.point_check(source, sampler, final) == cell['final_source_reference']
            if i < run.POSITIVE:
                endpoints[str(seed)][arm].append(saved['final'])
                if seed == run.SEEDS[0] and arm == run.ARMS[0]:
                    initial_endpoints.append(saved['initial'])
            attempted += len(saved['proposals'])
            del saved
            print(json.dumps({'audited_cells': cell_no+1, 'replayed_attempts': attempted,
                              'last_case': i, 'seed': seed, 'arm': arm,
                              **guard()}), flush=True)
        initial_metrics = reading_endpoint_metrics(initial_endpoints, episodes, episode_sampler)
        metrics = {seed: {arm: reading_endpoint_metrics(receipts, episodes, episode_sampler)
                         for arm, receipts in arms.items()} for seed, arms in endpoints.items()}
        assert initial_metrics == result['initial_metrics'] and metrics == result['endpoint_metrics']
        # Independent integer DP checks all positive endpoint edit totals, not string-only interfaces.
        for seed, arms in endpoints.items():
            for arm, receipts in arms.items():
                for j, (receipt, episode) in enumerate(zip(receipts, episodes, strict=True)):
                    _, truth = action_episode(episode_sampler, episode)
                    errors = sum(integer_edits(t, p) for t, p in zip(truth[2].texts, receipt['texts'], strict=True))
                    assert errors == metrics[seed][arm]['cases'][j]['edits']
        assert run.gate_summary(metrics, initial_metrics) == result['gates']
        assert run.summaries(result['cells']) == result['summary']
        assert attempted == run.CASES*len(run.SEEDS)*sum(run.MOVES.values())
        assert result['private_archive_bytes'] == sum(c['archive']['bytes'] for c in result['cells']) <= run.PRIVATE
        assert result['resources']['wall_seconds'] <= run.WALL and result['resources']['cpu_seconds'] <= run.CPU
        assert result['resources']['peak_rss_bytes'] <= run.HOST
        training.old.require_frozen(freeze, run.PATHS)
        assert array_identity(source) == result['source_arrays']
        training.save_new(run.OUT/'audit.json', {'status': 'PASS_full_registered_recovery_replay',
            'freeze': freeze, 'result': training.artifact(run.OUT/'result.json'), 'attempts': attempted,
            'cells': len(result['cells']), 'inputs_checked': len(run.PATHS),
            'maximum_independent_joint_log_delta': maximum, 'no_chain_extension': True,
            'same_researcher_same_proposal_backend_not_expert_replication': True,
            'resources': guard(), 'paid_spend_usd': 0, 'gates': result['gates']})
        return result['gates']
    except Exception as error:
        training.save_new(run.OUT/'audit-failure.json', {'error': repr(error), 'no_retry': True,
            'resources': training.resource_report(wall, cpu)})
        raise
    finally:
        signal.alarm(0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--freeze', required=True)
    print(audit(parser.parse_args().freeze))
