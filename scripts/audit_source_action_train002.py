"""One full ledger/checkpoint/selection audit and CPUdouble reading replay."""

import gc
import gzip
import hashlib
import json
import math
import signal
import time

import numpy as np
import torch

from scripts import run_source_action_train002 as run
from voynich.joint_key_training import EpisodeSampler, dictionary_code, metadata_bytes
from voynich.source_action_proposal import ReadingEnvironment, SourceActionConfig, SourceActionProposal
from voynich.source_action_training import action_episode, control_metrics, control_records, recovery_metrics

WALL, CPU = 10_800, 9_000


def verify(spec):
    assert run.artifact(run.ROOT/spec['path']) == spec


def distance(left, right):
    """Independent integer-sequence dynamic program, not the bit-vector metric."""
    previous = list(range(len(right)+1))
    for i, a in enumerate(left, 1):
        current = [i]
        for j, b in enumerate(right, 1):
            current.append(min(previous[j]+1, current[-1]+1, previous[j-1]+(a != b)))
        previous = current
    return previous[-1]


def independent_recovery(score, episodes, sampler):
    actual = recovery_metrics(score['predictions'], episodes, sampler)
    assert actual == score['recovery']
    for prediction, episode, case in zip(score['predictions'], episodes, actual['cases'], strict=True):
        env, truth = action_episode(sampler, episode)
        state = env.initial
        for action in prediction['actions']:
            state = env.advance(state, action)
        if prediction['status'] == 'complete_path':
            expected = sum(distance(x, y) for x, y in zip(truth[2].texts, state.texts, strict=True))
        else:
            expected = sum(map(len, truth[2].texts))
        assert case['edits'] == expected
    return actual


def checkpoint(spec, row, value):
    verify(spec)
    content = torch.load(run.ROOT/spec['path'], map_location='cpu', weights_only=True)
    assert content['config'] == value['config'] and content['step'] == row['step']
    assert content['freeze'] == value['freeze'] and content['inputs'] == value['inputs']
    assert content['seed'] == value['seed']
    digest = hashlib.sha256()
    for name, tensor in content['state_dict'].items():
        assert tensor.dtype == torch.float32 and torch.isfinite(tensor).all()
        digest.update(name.encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.numpy().tobytes())
    assert digest.hexdigest() == row['weights_sha256']
    if row['step'] not in (0, value['selected']['step']):
        return None
    model = SourceActionProposal(SourceActionConfig(**value['config']))
    model.load_state_dict(content['state_dict'], strict=True)
    return model.double().eval()


@torch.no_grad()
def replay_prefix(model, env, prediction):
    state, snapshots = env.initial, []
    actions = tuple(prediction['actions'])
    assert actions
    for action in actions:
        snapshots.append(state)
        state = env.advance(state, action)
    packed = model.pack([env], [(tuple(snapshots), actions, state)], require_complete=False)
    logits = model(packed)[0]
    indices, choices = torch.arange(len(actions)), torch.tensor(actions)
    deficit = float((logits.max(-1).values-logits[indices, choices]).max())
    logq = float(logits.log_softmax(-1)[indices, choices].sum())
    return logq, deficit


@torch.no_grad()
def replay(model, score, episodes, sampler):
    targets, guesses, deficits = [], [], []
    for episode, prediction in zip(episodes, score['predictions'], strict=True):
        env, truth = action_episode(sampler, episode)
        packed = model.pack([env], [truth])
        logits = model(packed)[0].log_softmax(-1)
        target = packed[-1][0]
        value = logits[torch.arange(len(target)), target].sum().item()
        targets.append(value)
        guess, deficit = replay_prefix(model, env, prediction)
        guesses.append(guess)
        deficits.append(deficit)
    target_delta = max(abs(a-b) for a, b in zip(targets, score['joint_target_path_logq'], strict=True))
    guess_delta = max(abs(a-p['greedy_path_under_model_log_probability'])
                      for a, p in zip(guesses, score['predictions'], strict=True))
    assert max(target_delta, guess_delta) <= .01 and max(deficits) <= .0002
    assert abs(score['mean_whole_path_nll']+math.fsum(score['joint_target_path_logq'])/len(episodes)) <= 1e-10
    control_delta, control_deficit = 0., 0.
    for control, prediction in zip(score['controls']['records'], score['controls']['predictions'], strict=True):
        env = ReadingEnvironment(control['records'], rows=23, glyphs=6)
        guess, deficit = replay_prefix(model, env, prediction)
        control_delta = max(control_delta, abs(guess-prediction['greedy_path_under_model_log_probability']))
        control_deficit = max(control_deficit, deficit)
    assert control_delta <= .01 and control_deficit <= .0002
    return {'target_whole_path_logq_max_delta': target_delta, 'greedy_model_path_logq_max_delta': guess_delta,
            'cpu_greedy_choice_max_deficit': max(deficits), 'episodes': len(episodes),
            'control_model_path_logq_max_delta': control_delta, 'control_choice_max_deficit': control_deficit}


def ledger(value, texts, episodes, manifest):
    raw = manifest['old_forbidden_raw']+[dictionary_code(e[2]['raw_indices']) for e in episodes]
    canonical = manifest['old_forbidden_canonical']+[dictionary_code(e[1]) for e in episodes]
    sampler = EpisodeSampler(texts, forbidden_raw=raw, forbidden_canonical=canonical)
    rng, digest = np.random.default_rng(value['seed']+100003), hashlib.sha256()
    prefixes, count, letters, literal, canon = {}, 0, 0, set(), set()
    verify(value['ledger'])
    with gzip.open(run.ROOT/value['ledger']['path'], 'rb') as stream:
        for step, encoded in enumerate(stream, 1):
            assert step <= run.STEPS
            batch = [sampler.sample(rng) for _ in range(run.BATCH)]
            expected = metadata_bytes({'step': step, 'episodes': [e[2] for e in batch]})
            assert encoded == expected
            digest.update(encoded)
            count += run.BATCH
            letters += sum(w['length'] for e in batch for w in e[2]['windows'])
            literal.update(dictionary_code(e[2]['raw_indices']) for e in batch)
            canon.update(dictionary_code(e[1]) for e in batch)
            prefixes[step] = digest.hexdigest(), count, letters
    assert len(prefixes) == value['updates'] == run.STEPS
    assert digest.hexdigest() == value['data_sha256'] and count == value['episodes_seen']
    assert letters == value['source_letters'] == value['supervised_action_targets']
    assert sampler.rejected_keys == value['rejected_keys']
    return prefixes, {'distinct_raw_dictionaries': len(literal), 'distinct_canonical_dictionaries': len(canon)}


def gates(values):
    seed_rows = []
    for seed in run.SEEDS:
        a, b = (values.get(f'{role}-{seed}') for role in ('binding', 'no-binding'))
        if a is None or b is None:
            seed_rows.append({'seed': seed, 'memory_benefit_gate': False, 'missing_completed_fit': True})
            continue
        assert a['data_sha256'] == b['data_sha256'] and a['initial_weights_sha256'] == b['initial_weights_sha256']
        x, y = a['selected']['recovery'], b['selected']['recovery']
        fraction = x['used_matches_with_failed_readings_zero']/x['used_rows']
        control = y['used_matches_with_failed_readings_zero']/y['used_rows']
        gain = b['selected']['mean_whole_path_nll']-a['selected']['mean_whole_path_nll']
        gate = (gain >= 1. and fraction >= control+.05
            and x['edits_with_failed_readings_full_length'] <= .9*y['edits_with_failed_readings_full_length']
            and x['complete_readings'] >= y['complete_readings']
            and x['complete_used_keys'] >= 1 and x['exact_records'] >= 1)
        seed_rows.append({'seed': seed, 'whole_path_nll_gain': gain, 'used_row_fraction_gain': fraction-control,
                          'memory_benefit_gate': gate})
    return {'seed_comparisons': seed_rows, 'replicated_exploratory_memory_benefit': all(r['memory_benefit_gate'] for r in seed_rows),
            'qualified_historical_or_general_inverse_or_neural_circuit': False}


def audit():
    campaign = json.loads((run.OUT/'campaign.json').read_text())
    freeze = campaign['freeze']
    run.old.require_frozen(freeze, [*run.BASE_PATHS, *run.INPUT_PATHS])
    save = run.save_new
    save(run.OUT/'audit-started.json', {'freeze': freeze, 'start_unix': time.time(), 'no_retry': True})
    run.limit_resources(WALL, CPU)
    wall, cpu = time.monotonic(), time.process_time()
    torch.set_num_threads(2)
    torch.backends.mha.set_fastpath_enabled(False)
    outcomes, values = {}, {}
    try:
        manifest, texts, valid, episodes = run.load_prepared()
        sampler = EpisodeSampler(valid)
        assert [p['arm'] for p in campaign['processes']] == list(run.ARMS)
        assert campaign['wall_seconds'] <= run.CAMPAIGN_WALL+120 and campaign['paid_spend_usd'] == 0
        for process in campaign['processes']:
            arm = process['arm']
            verify(process['log'])
            result_path, failure_path = run.OUT/(arm+'.json'), run.OUT/(arm+'-failure.json')
            assert not (result_path.exists() and failure_path.exists())
            passed = process['returncode'] == 0 and not process['outer_timeout']
            assert passed == result_path.exists()
            if not passed:
                outcomes[arm] = {'status': 'retained_failure', 'process': process,
                    'partial_artifacts': [run.artifact(p) for p in sorted(run.BULK.glob(arm+'-*')) if p.is_file()]}
                continue
            value = json.loads(result_path.read_text())
            assert value['status'] == 'PASS' and value['arm'] == arm and value['freeze'] == freeze
            assert value['config'] == dict(run.CONFIG.__dict__, binding_input=arm.startswith('binding-'))
            assert value['cache_policy'] == run.CACHE_POLICY and value['fixed_query_horizon'] == run.STATIC_STEPS
            assert value['checkpoint_unused_cache_cleanup']
            assert value['parameters'] == run.PARAMETERS and value['inputs'] == run.artifact(run.OUT/'inputs.json')
            assert not value['reserved_authors_or_cipher_panels_opened'] and value['paid_spend_usd'] == 0
            assert value['environment']['device'] == run.DEVICE and value['environment']['torch_threads'] == 2
            if run.DEVICE == 'mps':
                assert value['environment']['mps_cpu_fallback'] == '0'
                assert value['environment']['omp_threads'] == value['environment']['openblas_threads'] == '1'
            prefixes, counts = ledger(value, texts, episodes, manifest)
            verify(value['trace'])
            trace = [json.loads(row) for row in (run.ROOT/value['trace']['path']).read_text().splitlines()]
            updates = [row for row in trace if row['stage'] == 'optimizer']
            assert [r['step'] for r in updates] == list(range(1, run.STEPS+1))
            for row in updates:
                assert (row['data_sha256'], row['episodes_seen'], row['source_letters']) == prefixes[row['step']]
                assert row['lr'] == run.schedule(row['step'], total=run.STEPS)
                assert all(math.isfinite(row[k]) and row[k] >= 0 for k in ('whole_path_nll', 'preclip_norm', 'step_wall_seconds'))
            for stage in ('packed', 'forward', 'backward', 'update', 'cleanup'):
                assert [r['step'] for r in trace if r['stage'] == stage] == list(range(1, run.STEPS+1))
            samples = [r for r in trace if 'driver_bytes' in r]
            assert max(r['driver_bytes'] for r in samples) <= run.DRIVER
            assert max(r['peak_rss_bytes'] for r in samples) <= run.HOST
            assert max(r['owned_bulk_bytes'] for r in samples) <= run.BULK_LIMIT
            assert value['resources']['wall_seconds'] <= run.FIT_WALL and value['resources']['cpu_seconds'] <= run.FIT_CPU
            assert value['resources']['peak_rss_bytes'] <= run.HOST
            assert [r['step'] for r in value['checkpoints']] == list(run.CHECKPOINTS)
            assert value['selected'] == min(value['checkpoints'], key=lambda r: (r['mean_whole_path_nll'], r['step']))
            replays = {}
            for row in value['checkpoints']:
                assert json.loads((run.OUT/f"{arm}-step{row['step']}.json").read_text()) == row
                verify(row['validation'])
                stored = run.load_archive(row['validation'])
                score = stored['score']
                assert score['episodes'] == len(episodes) and not score['target_length_normalization_used']
                assert row['mean_whole_path_nll'] == score['mean_whole_path_nll']
                assert row['recovery'] == independent_recovery(score, episodes, sampler)
                controls = control_records(episodes, count=run.CONTROL_COUNT, seed=run.CONTROL_SEED)
                assert score['controls']['records'] == json.loads(json.dumps(controls))
                assert row['controls'] == score['controls']['metrics'] == control_metrics(score['controls']['predictions'], controls)
                assert stored['short_cpu_double_legal_logit_delta'] <= .002
                model = checkpoint(row['weights'], row, value)
                if model is not None:
                    replays[str(row['step'])] = replay(model, score, episodes, sampler)
                    del model
                    gc.collect()
                if row['step'] == 0:
                    assert row['weights_sha256'] == value['initial_weights_sha256']
                assert run.resource_report(wall, cpu)['peak_rss_bytes'] <= run.HOST
            values[arm] = value
            outcomes[arm] = {'status': 'PASS', 'ledger': counts, 'cpu_initial_selected_replays': replays}
        resources = run.resource_report(wall, cpu)
        assert resources['peak_rss_bytes'] <= run.HOST
        assert resources['wall_seconds'] <= WALL and resources['cpu_seconds'] <= CPU
        save(run.OUT/'audit.json', {'status': 'PASS_full_static_reading_action_completion_audit',
            'campaign': run.artifact(run.OUT/'campaign.json'), 'outcomes': outcomes, 'decision': gates(values),
            'all_fits_complete': len(values) == len(run.ARMS), 'resources': resources,
            'full_optimizer_retraining_claimed': False, 'same_author_shared_pytorch_backend': True,
            'independent_agent_review': False, 'paid_spend_usd': 0})
    except Exception as error:
        signal.alarm(0)
        save(run.OUT/'audit-failure.json', {'error': repr(error), 'no_retry': True, 'resources': run.resource_report(wall, cpu)})
        raise
    finally:
        signal.alarm(0)


if __name__ == '__main__':
    audit()
