"""One frozen full-size cipher-route/history diagnostic; unchanged training."""

import argparse
import gc
import gzip
import json
import math
import signal
import time

import torch

from scripts import run_source_action_sample001 as previous
from scripts import run_source_action_train002 as training
from scripts.benchmark_key_proposal_systems001 import weights_digest
from scripts.benchmark_source_prefix_systems001 import array_identity, load_source
from voynich.joint_key_training import EpisodeSampler
from voynich.joint_reading_source import source_point_checked
from voynich.recurrent_latin_source import ALPHABET
from voynich.source_action_cipher_diagnosis import (
    NEURAL_MODES, SOURCE_MODES, measure, neural_greedy_checked, neural_routes,
    source_greedy, source_teacher_logits,
)
from voynich.source_action_proposal import ReadingEnvironment, SourceActionConfig, SourceActionProposal
from voynich.source_action_training import action_episode, control_metrics, control_records, recovery_metrics

EXP = 'SOURCE-ACTION-CIPHER-001'
OUT, BULK = training.ROOT/'results'/EXP, training.ROOT/'outputs'/EXP
STEPS = (0, 1000, 4000)
SEED = 92671
WALL, CPU, HOST, PRIVATE = 3600, 5000, 4*1024**3, 64*1024**2
PATHS = sorted(set([*previous.PATHS,
    'results/SOURCE-ACTION-SAMPLE-001/failure.json',
    'results/SOURCE-ACTION-SAMPLE-001/closed-check-failure001.json',
    'results/SOURCE-ACTION-SAMPLE-001/publication-accounting001.json',
    'src/voynich/stochastic_reading_metrics.py', 'tests/test_stochastic_reading_metrics.py',
    'src/voynich/source_action_cipher_diagnosis.py',
    'tests/test_source_action_cipher_diagnosis.py', 'tests/test_source_action_cipher001.py',
    'scripts/run_source_action_cipher001.py', 'scripts/audit_source_action_cipher001.py',
    'docs/research/source-action-cipher-history-2026-10-02.md',
    'docs/experiments/SOURCE-ACTION-CIPHER-001.md']))


def teacher_summary(cases):
    """Fixed report reducer shared with miniature end-to-end qualification."""
    if not cases:
        raise ValueError('Every allocated teacher case required')
    result = {}
    for mode in cases[0]['scores']:
        result[mode] = {'mean_whole_path_nll': math.fsum(c['scores'][mode]['whole_path_nll'] for c in cases)/len(cases),
            'groups': {g: {k: (math.fsum(c['scores'][mode]['groups'][g][k] for c in cases)
                              if k.endswith('_sum') else sum(c['scores'][mode]['groups'][g][k] for c in cases))
                          for k in cases[0]['scores'][mode]['groups'][g]}
                       for g in ('first_binding', 'reuse')}}
    return result


def free_summary(cases, episodes, sampler, controls):
    result = {}
    for mode in dict.fromkeys(c['mode'] for c in cases):
        rows = [c for c in cases if c['mode'] == mode]
        assert len(rows) == 97 and [c['case'] for c in rows] == list(range(97))
        result[mode] = {'recovery': recovery_metrics([c['prediction'] for c in rows[:64]], episodes, sampler),
                       'controls': control_metrics([c['prediction'] for c in rows[64:]], controls)}
    return result


def public_free_cases(cases):
    """Keep actual actions/readings/logits in ignored, hash-bound archives."""
    return [{k: v for k, v in case.items() if k != 'prediction'} for case in cases]


@torch.no_grad()
def diagnose(freeze):
    training.old.require_frozen(freeze, PATHS)
    save, root = training.save_new, training.ROOT
    save(OUT/'started.json', {'freeze': freeze, 'start_unix': time.time(), 'no_retry': True})
    training.limit_resources(WALL, CPU)
    wall, cpu = time.monotonic(), time.process_time()
    torch.set_num_threads(2)
    torch.backends.mha.set_fastpath_enabled(False)
    BULK.mkdir(parents=True, exist_ok=True)
    checkpoints, source_cases, free = [], [], []
    private_bytes = 0
    with (BULK/'trace.jsonl').open('x') as trace:
        def guard():
            r = training.resource_report(wall, cpu)
            if r['peak_rss_bytes'] > HOST or r['wall_seconds'] > WALL or private_bytes+trace.tell() > PRIVATE:
                raise MemoryError('Registered diagnostic resource bound exceeded; no retry')
            return r
        def observe(stage, **extra):
            trace.write(json.dumps({'stage': stage, **extra, **guard()})+'\n')
            trace.flush()
        try:
            manifest, _, valid, episodes = training.load_prepared()
            sampler = EpisodeSampler(valid)
            tasks = [action_episode(sampler, e) for e in episodes]
            controls = control_records(episodes, seed=92451)
            allocated = [('positive', i, e[0]) for i, e in enumerate(episodes)]
            allocated.extend((c['kind'], c['paired_case'], c['records']) for c in controls)
            assert len(tasks) == 64 and len(allocated) == 97
            source, selected = load_source()
            original = json.loads((root/'results/TEMPERED-RECOVERY-DIAG-001/result.json').read_text())
            assert selected['counts'] == original['source']
            assert array_identity(source) == original['source_arrays']
            assert tuple(source.alphabet) == tuple(ALPHABET)
            assert selected['inputs'] == training.artifact(root/selected['inputs']['path'])
            source_audit = json.loads((root/'results/LATIN-SOURCE-COMPACT-001/large-audit.json').read_text())
            assert source_audit['status'] == 'PASS'
            assert source_audit['result'] == training.artifact(root/'results/LATIN-SOURCE-COMPACT-001/large.json')
            observe('loaded_source')
            for i, (env, truth) in enumerate(tasks):
                scores, arrays = {}, {}
                for mode in SOURCE_MODES:
                    logits = source_teacher_logits(source, env, truth, mode)
                    scores[mode], arrays[mode] = measure(logits, truth)
                    if mode == 'source_increment':
                        point = source_point_checked(source, truth[2].texts)['log_probability']
                        potential = float(logits[torch.arange(len(truth[1])), torch.tensor(truth[1])].sum())
                        expected = point-sum(k >= 0 for k in truth[2].key)*math.log(42)
                        assert abs(potential-expected) <= 1e-9
                    guard()
                spec = save(BULK/f'source-teacher-{i}.json.gz', {'case': i, 'arrays': arrays}, compressed=True)
                private_bytes += spec['bytes']
                source_cases.append({'case': i, 'scores': scores, 'arrays': spec})
                observe('source_teacher', case=i)
            save(OUT/'source-teacher.json', {'cases': source_cases, 'summary': teacher_summary(source_cases)})
            for step in STEPS:
                compact = training.OUT/f'binding-92403-step{step}.json'
                row = json.loads(compact.read_text())
                for name in ('weights', 'validation'):
                    assert training.artifact(root/row[name]['path']) == row[name]
                saved = json.loads(gzip.decompress((root/row['validation']['path']).read_bytes()))['score']
                content = torch.load(root/row['weights']['path'], map_location='cpu', weights_only=True)
                assert content['step'] == step and content['seed'] == 92403
                assert content['inputs'] == training.artifact(training.OUT/'inputs.json')
                assert content['freeze'] == '3ab56fb85b18231597b9fbc64c40c1fbff504408'
                assert all(t.dtype == torch.float32 and torch.isfinite(t).all()
                           for t in content['state_dict'].values())
                model = SourceActionProposal(SourceActionConfig(**content['config'])).cpu().eval()
                model.load_state_dict(content['state_dict'], strict=True)
                model.requires_grad_(False)
                assert weights_digest(model) == row['weights_sha256']
                assert sum(p.numel() for p in model.parameters()) == 96_039_982
                del content
                gc.collect()
                cases, max_delta = [], 0.
                observe('loaded_checkpoint', step=step)
                for i, (env, truth) in enumerate(tasks):
                    packed = model.pack([env], [truth])
                    assert packed[-2].all()  # one-case queries have no inactive padding
                    scores, arrays, route_counts, competitors = {}, {}, {}, None
                    base = None
                    for mode in NEURAL_MODES:
                        with neural_routes(model, mode, seed=SEED+i) as counts:
                            logits = model(packed)[0].detach().cpu()
                        assert counts == {'glyph_calls': 1, 'binding_calls': 1,
                                          'full_cross_calls': 4, 'cached_cross_calls': 0}
                        assert torch.equal(torch.isfinite(logits), packed[5][0])
                        scores[mode], arrays[mode] = measure(logits, truth, competitors)
                        if mode == 'base':
                            base = logits
                            competitors = arrays[mode]['fixed_competitor_actions']
                            delta = abs(scores[mode]['whole_path_nll']+saved['joint_target_path_logq'][i])
                            assert delta <= .01
                            max_delta = max(max_delta, delta)
                        elif mode == 'sham':
                            assert torch.equal(base, logits)
                            assert scores[mode] == scores['base'] and arrays[mode] == arrays['base']
                        route_counts[mode] = dict(counts)
                        del logits
                        observe('neural_teacher', step=step, case=i, mode=mode)
                    spec = save(BULK/f'step{step}-teacher-{i}.json.gz', {'case': i, 'step': step, 'arrays': arrays},
                                compressed=True)
                    private_bytes += spec['bytes']
                    cases.append({'case': i, 'scores': scores, 'arrays': spec, 'route_counts': route_counts})
                    del base, packed
                checkpoint = {'step': step, 'compact': training.artifact(compact), 'weights': row['weights'],
                              'validation': row['validation'], 'cases': cases, 'summary': teacher_summary(cases),
                              'maximum_cpu_float32_mps_path_nll_delta': max_delta}
                save(OUT/f'step{step}-teacher.json', checkpoint)
                checkpoints.append(checkpoint)
                if step == 4000:
                    for i, (kind, paired, records) in enumerate(allocated):
                        env = ReadingEnvironment(records, rows=23, glyphs=6)
                        base_prediction = None
                        for mode in NEURAL_MODES:
                            began = time.monotonic()
                            prediction, values, numeric = neural_greedy_checked(model, env, mode,
                                seed=SEED+i, observe=guard)
                            if mode == 'base':
                                base_prediction = prediction
                            elif mode == 'sham':
                                assert prediction == base_prediction
                            spec = save(BULK/f'free-{i}-{mode}.json.gz',
                                {'records': records, 'prediction': prediction, 'logits': values}, compressed=True)
                            private_bytes += spec['bytes']
                            free.append({'case': i, 'kind': kind, 'paired_case': paired, 'mode': mode,
                                         'prediction': prediction, 'numeric': numeric, 'archive': spec,
                                         'wall_seconds': time.monotonic()-began})
                            observe('neural_greedy', case=i, mode=mode)
                        print(json.dumps({'checkpoint': step, 'free_cases_complete': i+1}), flush=True)
                del model
                gc.collect()
            for i, (kind, paired, records) in enumerate(allocated):
                env = ReadingEnvironment(records, rows=23, glyphs=6)
                for mode in SOURCE_MODES:
                    began = time.monotonic()
                    prediction, values = source_greedy(source, env, mode)
                    spec = save(BULK/f'free-{i}-{mode}.json.gz',
                        {'records': records, 'prediction': prediction, 'actual_legal_logits': values}, compressed=True)
                    private_bytes += spec['bytes']
                    free.append({'case': i, 'kind': kind, 'paired_case': paired, 'mode': mode,
                                 'prediction': prediction, 'archive': spec, 'wall_seconds': time.monotonic()-began})
                    observe('source_greedy', case=i, mode=mode)
            summaries = free_summary(free, episodes, sampler, controls)
            observe('finished')
            training.old.require_frozen(freeze, PATHS)
            result = {'status': 'PASS_complete_registered_cipher_route_and_history_diagnostic', 'freeze': freeze,
                'inputs': [training.artifact(root/p) for p in PATHS], 'checkpoints': checkpoints,
                'source_teacher': {'cases': source_cases, 'summary': teacher_summary(source_cases)},
                'free_cases': public_free_cases(free), 'free_summary': summaries, 'source_counts': selected['counts'],
                'source_arrays': array_identity(source), 'shuffle_seed_base': SEED,
                'source_inputs': selected['inputs'], 'training_inputs': training.artifact(training.OUT/'inputs.json'),
                'torch_version': torch.__version__, 'torch_threads': torch.get_num_threads(),
                'parameters': 96_039_982, 'numerical_dtype': 'float32_neural_float64_source_and_scoring',
                'trace': training.artifact(BULK/'trace.jsonl'), 'resources': guard(),
                'private_archive_bytes': private_bytes, 'paid_spend_usd': 0,
                'original_training_unchanged': True, 'single_seed_exposed_development_only': True,
                'minimal_circuit_jspace_or_recovery_qualified': False}
            save(OUT/'result.json', result)
            return result['status']
        except Exception as error:
            save(OUT/'failure.json', {'error': repr(error), 'checkpoints': checkpoints,
                'source_teacher_cases': source_cases, 'free_cases': public_free_cases(free), 'no_retry': True,
                'resources': training.resource_report(wall, cpu), 'trace': training.artifact(BULK/'trace.jsonl')})
            raise
        finally:
            signal.alarm(0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--freeze', required=True)
    print(diagnose(parser.parse_args().freeze))
