"""One frozen CPU checkpoint/input-dependence and first-error diagnostic."""

import argparse
import gc
import gzip
import json
import math
import signal
import time
from collections import Counter

import torch

from scripts import run_source_action_train002 as run
from scripts.benchmark_key_proposal_systems001 import weights_digest
from voynich.joint_key_training import EpisodeSampler
from voynich.source_action_diagnosis import action_groups, binding_packet, first_deviation
from voynich.source_action_proposal import SourceActionConfig, SourceActionProposal
from voynich.source_action_training import action_episode

EXP = 'SOURCE-ACTION-DIAG-001'
OUT, BULK = run.ROOT/'results'/EXP, run.ROOT/'outputs'/EXP
STEPS, MODES = (0, 1000), ('base', 'sham', 'erase', 'rotate')
WALL, CPU, HOST, BULK_CAP = 3600, 5000, 3*1024**3, 32*1024**2
PATHS = sorted(set([*run.BASE_PATHS, *run.INPUT_PATHS,
    'results/SOURCE-ACTION-TRAIN-002/binding-92403-step0.json',
    'results/SOURCE-ACTION-TRAIN-002/binding-92403-step1000.json',
    'results/SOURCE-ACTION-TRAIN-002/step1000-readonly-check001.json',
    'src/voynich/source_action_diagnosis.py', 'tests/test_source_action_diagnosis.py',
    'tests/test_source_action_proposal.py', 'scripts/run_source_action_diag001.py',
    'docs/research/source-action-causal-diagnosis-2026-10-02.md',
    'docs/experiments/SOURCE-ACTION-DIAG-001.md']))


def score_logits(logits, targets, groups):
    """Finite legal probabilities; report whole path and fixed group denominators."""
    values = logits.double()
    logq = values.log_softmax(-1)
    gold = logq[torch.arange(len(targets)), targets]
    correct = values.argmax(-1) == targets
    forced = torch.isfinite(values).sum(-1) == 1
    assert torch.isfinite(gold).all()
    result = {}
    for group in ('first_binding', 'reuse'):
        mask = torch.tensor([g == group for g in groups])
        result[group] = {'actions': int(mask.sum()), 'correct_argmax': int(correct[mask].sum()),
                         'forced_actions': int((mask & forced).sum()),
                         'nonforced_correct_argmax': int((correct & mask & ~forced).sum()),
                         'nll_sum': float(-gold[mask].sum())}
    return {'whole_path_nll': float(-gold.sum()), 'groups': result}, gold.tolist(), correct.tolist()


@torch.no_grad()
def diagnose(freeze):
    run.old.require_frozen(freeze, PATHS)
    save = run.save_new
    save(OUT/'started.json', {'freeze': freeze, 'start_unix': time.time(), 'no_retry': True})
    run.limit_resources(WALL, CPU)
    wall, cpu = time.monotonic(), time.process_time()
    torch.set_num_threads(2)
    torch.backends.mha.set_fastpath_enabled(False)
    BULK.mkdir(parents=True, exist_ok=True)
    checkpoints = []
    with (BULK/'trace.jsonl').open('x') as trace_file:
        def observe(stage, **extra):
            report = run.resource_report(wall, cpu)
            owned = sum(p.stat().st_size for p in BULK.glob('*') if p.is_file())
            if report['peak_rss_bytes'] > HOST or report['wall_seconds'] > WALL or owned > BULK_CAP:
                raise MemoryError('Registered CPU diagnostic resource cap exceeded; no retry')
            trace_file.write(json.dumps({'stage': stage, **extra, **report, 'owned_bytes': owned})+'\n')
            trace_file.flush()
        try:
            manifest, _, valid, episodes = run.load_prepared()
            assert len(episodes) == 64
            sampler = EpisodeSampler(valid)
            for step in STEPS:
                compact = run.OUT/f'binding-92403-step{step}.json'
                row = json.loads(compact.read_text())
                for name in ('weights', 'validation'):
                    assert run.artifact(run.ROOT/row[name]['path']) == row[name]
                score = json.loads(gzip.decompress((run.ROOT/row['validation']['path']).read_bytes()))['score']
                state = torch.load(run.ROOT/row['weights']['path'], map_location='cpu', weights_only=True)
                assert state['step'] == step and state['seed'] == 92403
                assert state['inputs'] == run.artifact(run.OUT/'inputs.json')
                assert state['freeze'] == '3ab56fb85b18231597b9fbc64c40c1fbff504408'
                model = SourceActionProposal(SourceActionConfig(**state['config'])).cpu().eval()
                model.load_state_dict(state['state_dict'], strict=True)
                model.requires_grad_(False)
                assert weights_digest(model) == row['weights_sha256']
                assert sum(p.numel() for p in model.parameters()) == 96_039_982
                del state
                gc.collect()
                cases, arrays = [], []
                max_delta, max_deficit, sham_delta = 0., 0., 0.
                observe('loaded_checkpoint', step=step)
                for i, (episode, prediction) in enumerate(zip(episodes, score['predictions'], strict=True)):
                    env, truth = action_episode(sampler, episode)
                    packed = model.pack([env], [truth])
                    groups, target = action_groups(truth), packed[-1][0]
                    deviation = first_deviation(env, truth, prediction, episode[1])
                    rows, data = {}, {}
                    base = None
                    for mode in MODES:
                        packet = packed if mode == 'base' else binding_packet(packed,
                            rows=model.config.rows, glyphs=model.config.glyphs, mode=mode)
                        logits = model(packet)[0].detach().cpu()
                        assert torch.equal(torch.isfinite(logits), packed[5][0])
                        assert torch.isfinite(logits[packed[5][0]]).all()
                        metrics, target_logq, argmax_correct = score_logits(logits, target, groups)
                        changed_slots = packet[1][0] != packed[1][0]
                        changed_queries = changed_slots.any(-1)
                        metrics['memory_changed_queries'] = int(changed_queries.sum())
                        metrics['memory_changed_slots'] = int(changed_slots.sum())
                        rows[mode] = metrics
                        data[mode] = {'target_logq': target_logq, 'argmax_correct': argmax_correct,
                                      'memory_changed_queries': changed_queries.tolist()}
                        if mode == 'base':
                            base = logits
                            delta = abs(metrics['whole_path_nll']+score['joint_target_path_logq'][i])
                            assert delta <= .01
                            max_delta = max(max_delta, delta)
                            # Before AND at the first divergence, history still equals
                            # truth: the full causal decoder must score the saved choice.
                            upto = min(len(prediction['actions']), deviation['shared_prefix_actions']+1)
                            chosen = torch.tensor(prediction['actions'][:upto])
                            deficit = float((base[:upto].max(-1).values-base[torch.arange(upto), chosen]).max())
                            assert deficit <= .0002
                            max_deficit = max(max_deficit, deficit)
                        elif mode == 'sham':
                            assert torch.equal(base, logits)
                        del logits, packet
                        observe('forward', step=step, case=i, mode=mode)
                    cases.append({'case': i, 'first_free_deviation': deviation, 'teacher_scores': rows,
                                  'truth_actions': len(target)})
                    arrays.append({'case': i, 'action_groups': groups, 'scores': data,
                                   'legal_action_counts': packed[5][0].sum(-1).tolist()})
                    del base, packed, target
                    print(json.dumps({'checkpoint': step, 'cases_complete': i+1}), flush=True)
                summary = {}
                for mode in MODES:
                    summary[mode] = {'mean_whole_path_nll': math.fsum(c['teacher_scores'][mode]['whole_path_nll'] for c in cases)/64,
                        'memory_changed_queries': sum(c['teacher_scores'][mode]['memory_changed_queries'] for c in cases),
                        'memory_changed_slots': sum(c['teacher_scores'][mode]['memory_changed_slots'] for c in cases),
                        'groups': {g: {k: sum(c['teacher_scores'][mode]['groups'][g][k] for c in cases)
                            for k in ('actions', 'correct_argmax', 'forced_actions',
                                      'nonforced_correct_argmax', 'nll_sum')} for g in ('first_binding', 'reuse')}}
                deviations = [c['first_free_deviation'] for c in cases]
                counts = {'exact_action_paths': sum(d['exact_actions'] for d in deviations),
                          'introduced_wrong_binding': sum(d.get('introduced_wrong_dictionary_binding', False) for d in deviations),
                          'teacher_groups': dict(Counter(d.get('teacher_group', 'exact') for d in deviations)),
                          'chosen_groups': dict(Counter(d.get('chosen_group', 'exact') for d in deviations))}
                spec = save(BULK/f'step{step}-per-action.json.gz', {'cases': arrays}, compressed=True)
                checkpoint = {'step': step, 'compact_input': run.artifact(compact),
                    'weights': row['weights'], 'validation': row['validation'], 'cases': cases,
                    'summary': summary, 'first_deviation_counts': counts, 'per_action': spec,
                    'maximum_cpu_float32_mps_path_nll_delta': max_delta,
                    'maximum_saved_greedy_shared_prefix_argmax_deficit': max_deficit, 'sham_logit_delta': sham_delta}
                checkpoints.append(checkpoint)
                save(OUT/f'step{step}.json', checkpoint)
                observe('saved_checkpoint', step=step)
                del model
                gc.collect()
            trace_file.flush()
            value = {'status': 'PASS_complete_cpu_feature_intervention_and_first_error_diagnostic',
                'freeze': freeze, 'inputs': [run.artifact(run.ROOT/p) for p in PATHS],
                'source_inputs': run.artifact(run.OUT/'inputs.json'), 'checkpoints': checkpoints,
                'resources': run.resource_report(wall, cpu), 'trace': run.artifact(BULK/'trace.jsonl'),
                'paid_spend_usd': 0, 'original_training_unchanged': True,
                'single_seed_exposed_development_only': True, 'circuit_or_recovery_qualification': False,
                'counterfactual_memory_is_not_a_literal_symbolic_state': True,
                'torch_version': torch.__version__, 'torch_threads': torch.get_num_threads()}
            save(OUT/'result.json', value)
            return value['status']
        except Exception as error:
            trace_file.flush()
            save(OUT/'failure.json', {'error': repr(error), 'no_retry': True,
                 'completed_checkpoints': checkpoints, 'resources': run.resource_report(wall, cpu),
                 'trace': run.artifact(BULK/'trace.jsonl')})
            raise
        finally:
            signal.alarm(0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--freeze', required=True)
    print(diagnose(parser.parse_args().freeze))
