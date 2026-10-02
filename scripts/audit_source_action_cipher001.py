"""One model-free closure: saved paths, arithmetic, source policies and hashes."""

import argparse
import gzip
import json
import math
import signal
import time
from collections import Counter

import numpy as np

from scripts import run_source_action_cipher001 as run
from scripts.benchmark_source_prefix_systems001 import array_identity, load_source
from voynich.joint_key_training import EpisodeSampler
from voynich.joint_reading_source import source_point_checked
from voynich.source_action_cipher_diagnosis import NEURAL_MODES, SOURCE_MODES, source_action_scores, source_teacher_logits, measure
from voynich.source_action_diagnosis import action_groups
from voynich.source_action_proposal import ReadingEnvironment
from voynich.source_action_training import action_episode, control_records

WALL, CPU, HOST = 600, 500, 3*1024**3
GROUPS = ('first_binding', 'reuse')


def close(a, b, tol=1e-8):
    assert math.isfinite(a) and math.isfinite(b) and abs(a-b) <= tol, (a, b)


def equivalent(a, b):
    if isinstance(a, dict):
        assert isinstance(b, dict) and a.keys() == b.keys()
        for key in a:
            equivalent(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert isinstance(b, (list, tuple)) and len(a) == len(b)
        for x, y in zip(a, b, strict=True):
            equivalent(x, y)
    elif isinstance(a, float):
        close(a, b)
    else:
        assert a == b


def integer_edits(a, b):
    """Independent integer-sequence DP; avoids string-only edit interfaces."""
    previous = list(range(len(b)+1))
    for i, x in enumerate(a, 1):
        current = [i]
        for j, y in enumerate(b, 1):
            current.append(min(current[-1]+1, previous[j]+1, previous[j-1]+(x != y)))
        previous = current
    return previous[-1]


def teacher_arithmetic(environment, truth, metrics, arrays):
    groups = action_groups(truth)
    assert arrays['groups'] == list(groups)
    assert all(len(v) == len(truth[1]) for v in arrays.values())
    computed = {g: {k: 0 for k in metrics['groups'][g]} for g in GROUPS}
    terms = {g: {k: [] for k in ('nll_sum', 'category_nll_sum', 'within_category_nll_sum', 'fixed_margin_sum')}
             for g in GROUPS}
    for j, (state, target, group) in enumerate(zip(truth[0], truth[1], groups, strict=True)):
        legal = environment.legal_actions(state)
        prediction = arrays['prediction'][j]
        assert prediction in legal and target in legal
        target_q, category_q = arrays['target_logq'][j], arrays['category_logq'][j]
        assert math.isfinite(target_q) and math.isfinite(category_q)
        assert target_q <= category_q+1e-10 and category_q <= 1e-10
        out = computed[group]
        out['actions'] += 1
        out['correct_actions'] += prediction == target
        out['correct_rows'] += prediction//2 == target//2
        out['correct_lengths'] += prediction%2 == target%2
        out['forced_actions'] += len(legal) == 1
        terms[group]['nll_sum'].append(-target_q)
        terms[group]['category_nll_sum'].append(-category_q)
        terms[group]['within_category_nll_sum'].append(-target_q+category_q)
        competitor = arrays['fixed_competitor_actions'][j]
        if len(legal) == 1:
            assert competitor is arrays['competitor_logq'][j] is arrays['target_minus_fixed_competitor'][j] is None
            close(target_q, 0.)
        else:
            assert competitor in legal and competitor != target
            value, margin = arrays['competitor_logq'][j], arrays['target_minus_fixed_competitor'][j]
            assert math.isfinite(value) and value <= 1e-10
            close(target_q-value, margin)
            out['fixed_margin_queries'] += 1
            terms[group]['fixed_margin_sum'].append(margin)
    for group in GROUPS:
        for key, values in terms[group].items():
            computed[group][key] = math.fsum(values)
        equivalent(computed[group], metrics['groups'][group])
    close(-math.fsum(arrays['target_logq']), metrics['whole_path_nll'])
    return computed


def path_arithmetic(environment, prediction, actual, reference=None):
    state = environment.initial
    assert len(actual) == len(prediction['actions']) and actual
    if reference is not None:
        assert len(reference) == len(actual)
    terms, ref_terms, deltas, deficits = [], [], [], []
    for j, (action, vector) in enumerate(zip(prediction['actions'], actual, strict=True)):
        legal = environment.legal_actions(state)
        assert len(vector) == len(legal) and np.isfinite(vector).all()
        assert action == legal[int(np.argmax(vector))]
        index = legal.index(action)
        v = np.asarray(vector, dtype=np.float64)
        normalizer = float(np.logaddexp.reduce(v))
        terms.append(float(v[index]-normalizer))
        if reference is not None:
            r = np.asarray(reference[j], dtype=np.float64)
            assert len(r) == len(legal) and np.isfinite(r).all()
            ref_terms.append(float(r[index]-np.logaddexp.reduce(r)))
            deltas.append(float(np.max(np.abs(v-r))))
            deficits.append(float(r.max()-r[index]))
        state = environment.advance(state, action)
    assert tuple(prediction['key']) == state.key and tuple(map(tuple, prediction['texts'])) == state.texts
    complete = environment.selected_record(state) is None
    assert prediction['status'] == ('complete_path' if complete else 'dead_end')
    assert complete or not environment.legal_actions(state)
    close(math.fsum(terms), prediction['greedy_path_under_model_log_probability'])
    if reference is None:
        return state, None
    numeric = {'maximum_legal_logit_delta': max(deltas),
        'maximum_path_logq_delta': abs(math.fsum(ref_terms)-prediction['greedy_path_under_model_log_probability']),
        'maximum_reference_argmax_deficit': max(deficits), 'reference_path_logq': math.fsum(ref_terms),
        'route_counts': {'glyph_calls': 2, 'binding_calls': len(actual)+1,
                         'full_cross_calls': 4, 'cached_cross_calls': 4*len(actual)}}
    assert numeric['maximum_legal_logit_delta'] <= .002
    assert numeric['maximum_path_logq_delta'] <= .01
    assert numeric['maximum_reference_argmax_deficit'] <= .0002
    return state, numeric


def paired_summary(cases):
    """Fixed all-case paired descriptions; no independent-token significance."""
    result = {}
    for mode in cases[0]['scores']:
        if mode == 'base':
            continue
        differences = [c['scores'][mode]['whole_path_nll']-c['scores']['base']['whole_path_nll'] for c in cases]
        result[mode] = {'mean_nll_change': math.fsum(differences)/len(cases),
            'mean_absolute_nll_change': math.fsum(map(abs, differences))/len(cases),
            'cases': len(cases), 'min_nll_change': min(differences), 'max_nll_change': max(differences)}
    return result


def audit(freeze):
    training, root = run.training, run.training.ROOT
    training.old.require_frozen(freeze, run.PATHS)
    training.save_new(run.OUT/'audit-started.json', {'freeze': freeze, 'no_retry': True, 'start_unix': time.time()})
    training.limit_resources(WALL, CPU)
    began, cpu = time.monotonic(), time.process_time()

    def guard():
        resources = training.resource_report(began, cpu)
        assert resources['wall_seconds'] <= WALL and resources['peak_rss_bytes'] <= HOST
        return resources

    def archive(spec):
        assert training.artifact(root/spec['path']) == spec
        guard()
        return json.loads(gzip.decompress((root/spec['path']).read_bytes()))

    try:
        assert not (run.OUT/'failure.json').exists()
        value = json.loads((run.OUT/'result.json').read_text())
        assert value['status'] == 'PASS_complete_registered_cipher_route_and_history_diagnostic'
        assert value['freeze'] == freeze and value['parameters'] == 96_039_982
        assert value['torch_threads'] == 2 and value['paid_spend_usd'] == 0
        equivalent(value['inputs'], [training.artifact(root/p) for p in run.PATHS])
        assert value['training_inputs'] == training.artifact(training.OUT/'inputs.json')
        assert value['source_inputs'] == training.artifact(root/value['source_inputs']['path'])
        r = value['resources']
        assert r['wall_seconds'] <= run.WALL and r['cpu_seconds'] <= run.CPU and r['peak_rss_bytes'] <= run.HOST
        assert training.artifact(root/value['trace']['path']) == value['trace']
        trace = [json.loads(line) for line in (root/value['trace']['path']).read_text().splitlines()]
        assert Counter(t['stage'] for t in trace) == Counter(loaded_source=1, source_teacher=64,
            loaded_checkpoint=3, neural_teacher=960, neural_greedy=485, source_greedy=194, finished=1)
        assert [(t['step'], t['case'], t['mode']) for t in trace if t['stage'] == 'neural_teacher'] == [
            (s, i, m) for s in run.STEPS for i in range(64) for m in NEURAL_MODES]
        for kind, modes in (('neural_greedy', NEURAL_MODES), ('source_greedy', SOURCE_MODES)):
            assert [(t['case'], t['mode']) for t in trace if t['stage'] == kind] == [(i, m) for i in range(97) for m in modes]
        assert all(t['peak_rss_bytes'] <= run.HOST and t['wall_seconds'] <= run.WALL and t['cpu_seconds'] <= run.CPU for t in trace)
        _, _, valid, episodes = training.load_prepared()
        sampler = EpisodeSampler(valid)
        tasks = [action_episode(sampler, e) for e in episodes]
        controls = control_records(episodes, seed=92451)
        allocated = [('positive', i, env.records) for i, (env, _) in enumerate(tasks)]
        allocated.extend((c['kind'], c['paired_case'], tuple(map(tuple, c['records']))) for c in controls)
        source, selected = load_source()
        assert selected['counts'] == value['source_counts'] and array_identity(source) == value['source_arrays']
        assert selected['inputs'] == value['source_inputs']
        original = json.loads((root/'results/TEMPERED-RECOVERY-DIAG-001/result.json').read_text())
        assert selected['counts'] == original['source'] and value['source_arrays'] == original['source_arrays']
        private_bytes = 0
        source_cases = value['source_teacher']['cases']
        assert [c['case'] for c in source_cases] == list(range(64))
        equivalent(value['source_teacher'], json.loads((run.OUT/'source-teacher.json').read_text()))
        for case, (env, truth) in zip(source_cases, tasks, strict=True):
            saved = archive(case['arrays'])
            private_bytes += case['arrays']['bytes']
            assert saved['case'] == case['case']
            for mode in SOURCE_MODES:
                teacher_arithmetic(env, truth, case['scores'][mode], saved['arrays'][mode])
                logits = source_teacher_logits(source, env, truth, mode)
                metrics, arrays = measure(logits, truth)
                equivalent(metrics, case['scores'][mode])
                equivalent(arrays, saved['arrays'][mode])
                if mode == 'source_increment':
                    selected_logits = [float(logits[j, a]) for j, a in enumerate(truth[1])]
                    expected = source_point_checked(source, truth[2].texts)['log_probability']-sum(k >= 0 for k in truth[2].key)*math.log(42)
                    close(math.fsum(selected_logits), expected)
        equivalent(value['source_teacher']['summary'], run.teacher_summary(source_cases))
        assert [c['step'] for c in value['checkpoints']] == list(run.STEPS)
        paired = {}
        for checkpoint in value['checkpoints']:
            step = checkpoint['step']
            equivalent(checkpoint, json.loads((run.OUT/f'step{step}-teacher.json').read_text()))
            assert checkpoint['maximum_cpu_float32_mps_path_nll_delta'] <= .01
            compact = json.loads((root/checkpoint['compact']['path']).read_text())
            for field in ('compact', 'weights', 'validation'):
                assert training.artifact(root/checkpoint[field]['path']) == checkpoint[field]
            assert checkpoint['weights'] == compact['weights'] and checkpoint['validation'] == compact['validation']
            mps = archive(checkpoint['validation'])['score']['joint_target_path_logq']
            cases = checkpoint['cases']
            assert [c['case'] for c in cases] == list(range(64))
            deltas = []
            for case, (env, truth) in zip(cases, tasks, strict=True):
                data = archive(case['arrays'])
                private_bytes += case['arrays']['bytes']
                assert data['case'] == case['case'] and data['step'] == step
                assert set(case['scores']) == set(NEURAL_MODES) and set(data['arrays']) == set(NEURAL_MODES)
                for mode in NEURAL_MODES:
                    teacher_arithmetic(env, truth, case['scores'][mode], data['arrays'][mode])
                    assert case['route_counts'][mode] == {'glyph_calls': 1, 'binding_calls': 1,
                        'full_cross_calls': 4, 'cached_cross_calls': 0}
                    assert data['arrays'][mode]['fixed_competitor_actions'] == data['arrays']['base']['fixed_competitor_actions']
                assert data['arrays']['base'] == data['arrays']['sham'] and case['scores']['base'] == case['scores']['sham']
                deltas.append(abs(case['scores']['base']['whole_path_nll']+mps[case['case']]))
            close(max(deltas), checkpoint['maximum_cpu_float32_mps_path_nll_delta'])
            equivalent(checkpoint['summary'], run.teacher_summary(cases))
            paired[str(step)] = paired_summary(cases)
        free = []
        assert len(value['free_cases']) == 679
        for case in value['free_cases']:
            kind, pair, records = allocated[case['case']]
            assert case['kind'] == kind and case['paired_case'] == pair and 'prediction' not in case
            data = archive(case['archive'])
            private_bytes += case['archive']['bytes']
            assert tuple(map(tuple, data['records'])) == records
            env = ReadingEnvironment(records, rows=23, glyphs=6)
            mode, prediction = case['mode'], data['prediction']
            if mode in NEURAL_MODES:
                values = data['logits']
                _, numeric = path_arithmetic(env, prediction, values['actual_legal_logits'], values['reference_legal_logits'])
                equivalent(numeric, case['numeric'])
            else:
                assert mode in SOURCE_MODES
                actual = data['actual_legal_logits']
                path_arithmetic(env, prediction, actual)
                state = env.initial
                for action, vector in zip(prediction['actions'], actual, strict=True):
                    logits = source_action_scores(source, env, state, mode)
                    equivalent(logits[list(env.legal_actions(state))].tolist(), vector)
                    state = env.advance(state, action)
            free.append({**case, 'prediction': prediction})
            if case['case'] < 64:
                truth = tasks[case['case']][1][2]
                metrics = value['free_summary'][mode]['recovery']['cases'][case['case']]
                finished = prediction['status'] == 'complete_path'
                texts = tuple(map(tuple, prediction['texts']))
                edits = sum(integer_edits(t, p) for t, p in zip(truth.texts, texts, strict=True)) if finished else sum(map(len, truth.texts))
                assert edits == metrics['edits']
        equivalent(run.free_summary(free, episodes, sampler, controls), value['free_summary'])
        indexed = {(c['case'], c['mode']): c['prediction'] for c in free}
        assert len(indexed) == 679
        assert all(indexed[i, 'base'] == indexed[i, 'sham'] for i in range(97))
        assert private_bytes == value['private_archive_bytes']
        assert private_bytes+value['trace']['bytes'] <= run.PRIVATE
        guard()
        training.old.require_frozen(freeze, run.PATHS)
        result = {'status': 'PASS_saved_records_source_policy_and_metric_closure', 'freeze': freeze,
            'result': training.artifact(run.OUT/'result.json'), 'inputs_checked': len(run.PATHS),
            'neural_teacher_forwards_recorded': 960, 'neural_paths_checked': 485, 'source_paths_checked': 194,
            'all_source_teacher_cases_checked': 64, 'paired_teacher_changes': paired,
            'saved_neural_teacher_arithmetic_only_not_logit_replay': True,
            'neural_model_calls': 0, 'optimizer_or_sampling_calls': 0,
            'source_lookup_reconstruction': True, 'independent_expert_review': False,
            'resources': guard(), 'paid_spend_usd': 0, 'minimal_circuit_or_historical_claim': False}
        training.save_new(run.OUT/'audit.json', result)
        return result['status']
    except Exception as error:
        training.save_new(run.OUT/'audit-failure.json', {'error': repr(error), 'no_retry': True,
            'resources': training.resource_report(began, cpu)})
        raise
    finally:
        signal.alarm(0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--freeze', required=True)
    print(audit(parser.parse_args().freeze))
