"""One deterministic trace/RNG/source replay, without extending any chain."""
import argparse
import json
import math
import signal
import time

import torch

from scripts import run_reading_regrowth_admit001 as run
from scripts.benchmark_source_prefix_systems001 import array_identity, load_source
from scripts.run_blind_channel_dev004 import load_archive
from voynich.reading_regrowth import SourceRegrowth
from voynich.regrowth_trace import chain, path_receipt
from voynich.source_action_proposal import ReadingEnvironment
from voynich.source_action_training import action_episode

WALL, CPU, HOST = 1800, 1800, 2*1024**3


def independent_dense_log_target(source, env, state):
    terms = [-len(set(a for text in state.texts for a in text))*math.log(len(env.pool))]
    for text in state.texts:
        terms.extend((math.log(1/225), len(text)*math.log1p(-1/225)))
        context = source.state('')
        for row in text:
            terms.append(math.log(float(source.probabilities[context, row])))
            context = int(source.transitions[context, row])
    return math.fsum(terms)


def audit(freeze):
    run.prior.training.old.require_frozen(freeze, run.PATHS)
    save = run.prior.training.save_new
    save(run.OUT/'audit-started.json', {'freeze': freeze, 'no_retry': True,
        'no_new_search_or_chain_extension': True})
    run.prior.training.limit_resources(WALL, CPU)
    wall, cpu = time.monotonic(), time.process_time()
    torch.set_num_threads(1)
    try:
        def guard():
            value = run.prior.training.resource_report(wall, cpu)
            assert value['wall_seconds'] <= WALL and value['cpu_seconds'] <= CPU
            assert value['peak_rss_bytes'] <= HOST
            return value
        result = json.loads((run.OUT/'result.json').read_text())
        assert result['freeze'] == freeze and result['status'] == (
            'PASS_full_source_regrowth_density_literal_and_cost_admission')
        assert result['inputs'] == [run.prior.training.artifact(run.prior.training.ROOT/p) for p in run.PATHS]
        assert [(c['case'], c['arm']) for c in result['cells']] == [(i, arm) for i in range(97) for arm in run.ARMS]
        manifest, episodes, episode_sampler, allocated, initials = run.load_inputs()
        source, selected = load_source()
        assert array_identity(source) == result['source_arrays']
        assert selected['counts'] == result['source_counts'] and selected['inputs'] == result['source_inputs']
        maximum = 0.
        for i, (episode, saved) in enumerate(zip(episodes, result['gold_source_density_checks'], strict=True)):
            env, truth = action_episode(episode_sampler, episode)
            sampler = SourceRegrowth(source, env, run.configuration('regrowth'))
            path = sampler.path(forced_actions=truth[1], progress=guard)
            assert saved == {'case': i, **run.point_check(source, sampler, path)}
        for cell in result['cells']:
            i, arm = cell['case'], cell['arm']
            kind, paired, records = allocated[i]
            assert cell['kind'] == kind and cell['paired_case'] == paired
            saved = load_archive(cell['archive'])
            assert saved['case'] == i and saved['arm'] == arm and saved['records'] == [list(r) for r in records]
            assert saved['seed'] == run.SEED+i and saved['steps'] == run.STEPS
            assert saved['initial_archive'] == initials[i]['archive']
            initial = load_archive(saved['initial_archive'])
            env = ReadingEnvironment(records, rows=23, glyphs=6)
            run.initial_receipt(env, initial['prediction'])
            sampler = SourceRegrowth(source, env, run.configuration(arm))
            # Replay only the registered seed and EIGHT moves. No extension/adaptive choice.
            replay = chain(sampler, initial['prediction']['actions'], seed=run.SEED+i,
                           steps=run.STEPS, progress=guard)
            for name in ('seed', 'steps', 'initial', 'proposals', 'final', 'final_rng_state', 'summary'):
                assert replay[name] == saved[name]
            assert saved['summary'] == cell['summary']
            for receipt in [saved['initial'], *(p['candidate'] for p in saved['proposals']), saved['final']]:
                state = run.literal_replay(env, receipt)
                path = sampler.path(forced_actions=tuple(receipt['actions']), progress=guard)
                assert path_receipt(path) == receipt
                if path.complete:
                    numerator, denominator = sampler.target_integers(path)
                    difference = abs(math.log(numerator)-math.log(denominator)-
                                     independent_dense_log_target(source, env, state))
                    maximum = max(maximum, difference)
                    assert difference <= 1e-9
            final = sampler.path(forced_actions=tuple(saved['final']['actions']))
            assert run.point_check(source, sampler, final) == cell['final_source_reference']
            guard()
            if arm == run.ARMS[-1]:
                print(json.dumps({'audited_cases': i+1, 'cases': 97}), flush=True)
        assert result['summary'] == run.summarize(result['cells'])
        assert result['private_archive_bytes'] == sum(c['archive']['bytes'] for c in result['cells'])
        assert result['resources']['wall_seconds'] <= run.WALL and result['resources']['cpu_seconds'] <= run.CPU
        assert result['resources']['peak_rss_bytes'] <= run.HOST and result['private_archive_bytes'] <= run.PRIVATE
        run.prior.training.old.require_frozen(freeze, run.PATHS)
        assert array_identity(source) == result['source_arrays']
        save(run.OUT/'audit.json', {'status': 'PASS_registered_trace_rng_literal_density_and_source_replay',
            'freeze': freeze, 'result': run.prior.training.artifact(run.OUT/'result.json'),
            'inputs_checked': len(run.PATHS), 'cells': 194, 'attempts': 1552,
            'maximum_independent_dense_joint_log_delta': maximum, 'no_chain_extension': True,
            'same_researcher_same_proposal_replay_not_expert_review': True, 'resources': guard(),
            'paid_spend_usd': 0})
        return 'PASS_registered_trace_rng_literal_density_and_source_replay'
    except Exception as error:
        save(run.OUT/'audit-failure.json', {'error': repr(error), 'no_retry': True,
            'resources': run.prior.training.resource_report(wall, cpu)})
        raise
    finally:
        signal.alarm(0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--freeze', required=True)
    print(audit(parser.parse_args().freeze))
