"""One fixed trained-policy stochastic-density/source-point admission on CPU."""

import argparse
import gc
import gzip
import json
import signal
import time
from dataclasses import asdict

import torch

from scripts import run_source_action_diag001 as diagnostic
from scripts import run_source_action_train002 as run
from scripts.benchmark_key_proposal_systems001 import weights_digest
from scripts.benchmark_source_prefix_systems001 import array_identity, load_source
from voynich.joint_key_training import EpisodeSampler
from voynich.joint_reading_mh import joint_reading_log_weight
from voynich.joint_reading_source import source_point_checked
from voynich.recurrent_latin_source import ALPHABET
from voynich.source_action_density_audit import sample_and_check
from voynich.source_action_proposal import ReadingEnvironment, SourceActionConfig, SourceActionProposal
from voynich.source_action_training import action_episode, control_metrics, control_records, recovery_metrics

EXP = 'SOURCE-ACTION-SAMPLE-001'
OUT, BULK = run.ROOT/'results'/EXP, run.ROOT/'outputs'/EXP
WALL, CPU, HOST, BULK_CAP = 3600, 5000, 4*1024**3, 64*1024**2
STEP, SEED, RHO = 4000, 92501, 1/225
PATHS = sorted(set([*diagnostic.PATHS,
    'results/SOURCE-ACTION-DIAG-001/result.json', 'results/SOURCE-ACTION-DIAG-001/closed-check001.json',
    'results/SOURCE-ACTION-TRAIN-002/binding-92403-step4000.json',
    'results/SOURCE-ACTION-TRAIN-002/step4000-readonly-check001.json',
    'src/voynich/joint_reading_mh.py', 'src/voynich/joint_reading_source.py',
    'src/voynich/source_action_density_audit.py', 'tests/test_source_action_density_audit.py',
    'src/voynich/compact_suffix_source.py', 'src/voynich/compact_suffix_adapter.py',
    'src/voynich/dense_suffix_adapter.py', 'src/voynich/source_prefix_inverse.py',
    'scripts/benchmark_source_prefix_systems001.py', 'scripts/run_source_action_sample001.py',
    'results/LATIN-SOURCE-COMPACT-001/large.json', 'results/LATIN-SOURCE-COMPACT-001/large-audit.json',
    'results/LATIN-SOURCE-COMPACT-001/large-inputs.json',
    'results/TEMPERED-RECOVERY-DIAG-001/result.json',
    'docs/research/joint-reading-independence-2026-10-02.md',
    'docs/research/trained-reading-density-admission-2026-10-02.md',
    'docs/experiments/SOURCE-ACTION-SAMPLE-001.md']))


@torch.no_grad()
def admission(freeze):
    run.old.require_frozen(freeze, PATHS)
    run.save_new(OUT/'started.json', {'freeze': freeze, 'start_unix': time.time(), 'no_retry': True})
    run.limit_resources(WALL, CPU)
    wall, cpu = time.monotonic(), time.process_time()
    torch.set_num_threads(2)
    torch.backends.mha.set_fastpath_enabled(False)
    BULK.mkdir(parents=True, exist_ok=True)
    cases, predictions, truth_scores = [], [], []
    with (BULK/'trace.jsonl').open('x') as trace:
        def observe(stage, **extra):
            resources = run.resource_report(wall, cpu)
            owned = sum(p.stat().st_size for p in BULK.glob('*') if p.is_file())
            if resources['peak_rss_bytes'] > HOST or resources['wall_seconds'] > WALL or owned > BULK_CAP:
                raise MemoryError('Fixed trained-sampler admission resource cap exceeded; no retry')
            trace.write(json.dumps({'stage': stage, **extra, **resources, 'owned_bytes': owned})+'\n')
            trace.flush()
        try:
            manifest, _, valid, episodes = run.load_prepared()
            sampler = EpisodeSampler(valid)
            compact = run.OUT/f'binding-92403-step{STEP}.json'
            row = json.loads(compact.read_text())
            for name in ('weights', 'validation'):
                assert run.artifact(run.ROOT/row[name]['path']) == row[name]
            saved_score = json.loads(gzip.decompress((run.ROOT/row['validation']['path']).read_bytes()))['score']
            content = torch.load(run.ROOT/row['weights']['path'], map_location='cpu', weights_only=True)
            assert content['step'] == STEP and content['seed'] == 92403
            assert content['inputs'] == run.artifact(run.OUT/'inputs.json')
            assert content['freeze'] == '3ab56fb85b18231597b9fbc64c40c1fbff504408'
            model = SourceActionProposal(SourceActionConfig(**content['config'])).cpu().eval()
            model.load_state_dict(content['state_dict'], strict=True)
            model.requires_grad_(False)
            assert weights_digest(model) == row['weights_sha256']
            assert sum(p.numel() for p in model.parameters()) == 96_039_982
            del content
            gc.collect()
            source, selected = load_source()
            source_audit = json.loads((run.ROOT/'results/LATIN-SOURCE-COMPACT-001/large-audit.json').read_text())
            assert selected['inputs'] == run.artifact(run.ROOT/selected['inputs']['path'])
            assert source_audit['status'] == 'PASS'
            assert source_audit['result'] == run.artifact(run.ROOT/'results/LATIN-SOURCE-COMPACT-001/large.json')
            arrays = array_identity(source)
            original = json.loads((run.ROOT/'results/TEMPERED-RECOVERY-DIAG-001/result.json').read_text())
            assert arrays == original['source_arrays'] and selected['counts'] == original['source']
            assert tuple(source.alphabet) == tuple(ALPHABET)
            observe('loaded')
            controls = control_records(episodes, seed=92451)
            assert json.loads(json.dumps(controls)) == saved_score['controls']['records']
            for i, episode in enumerate(episodes):
                env, truth = action_episode(sampler, episode)
                packet = model.pack([env], [truth])
                logits = model(packet)[0].double()
                logq = float(logits.log_softmax(-1)[torch.arange(len(truth[1])), packet[-1][0]].sum())
                assert abs(logq-saved_score['joint_target_path_logq'][i]) <= .01
                point = source_point_checked(source, truth[2].texts, rho=RHO)
                weight = joint_reading_log_weight(env, truth[2], source_log_probability=point['log_probability'],
                                                  proposal_log_probability=logq)
                truth_scores.append({'case': i, 'path_logq': logq, 'source': point, 'log_importance_weight': weight})
                del packet, logits
                observe('truth_density_reference', case=i)
            allocated = [('positive', i, e[0]) for i, e in enumerate(episodes)]
            allocated.extend((c['kind'], c['paired_case'], c['records']) for c in controls)
            assert len(allocated) == 97
            for i, (kind, paired, records) in enumerate(allocated):
                env = ReadingEnvironment(records, rows=23, glyphs=6)
                prediction, actual, reference, metrics = sample_and_check(model, env, seed=SEED+i)
                state = prediction['state']
                point, weight = None, None
                if prediction['status'] == 'complete_path':
                    point = source_point_checked(source, state.texts, rho=RHO)
                    weight = joint_reading_log_weight(env, state, source_log_probability=point['log_probability'],
                        proposal_log_probability=prediction['path_log_probability'])
                archive = run.save_new(BULK/f'case{i}.json.gz', {'records': records, 'actions': prediction['actions'],
                    'state': asdict(state), 'status': prediction['status'], 'seed': SEED+i,
                    'actual_legal_logits': actual, 'reference_legal_logits': reference,
                    'metrics': metrics, 'source': point, 'log_importance_weight': weight}, compressed=True)
                cases.append({'case': i, 'kind': kind, 'paired_case': paired, 'seed': SEED+i,
                    'status': prediction['status'], 'actions': len(prediction['actions']),
                    'used_rows': sum(k >= 0 for k in state.key), 'metrics': metrics, 'source': point,
                    'log_importance_weight': weight, 'failed_target_weight_is_zero': point is None,
                    'archive': archive})
                predictions.append({'status': prediction['status'], 'actions': prediction['actions'],
                                    'key': state.key, 'texts': state.texts})
                observe('sample_and_reference_complete', case=i)
                print(json.dumps({'case': i, 'kind': kind, 'status': prediction['status'], **metrics}), flush=True)
            recovery = recovery_metrics(predictions[:64], episodes, sampler)
            control_result = control_metrics(predictions[64:], controls)
            observe('finished')
            run.old.require_frozen(freeze, PATHS)
            result = {'status': 'PASS_trained_visited_density_and_original_source_point_admission',
                'freeze': freeze, 'inputs': [run.artifact(run.ROOT/p) for p in PATHS],
                'checkpoint': run.artifact(compact), 'weights': row['weights'], 'validation': row['validation'],
                'source_counts': selected['counts'], 'source_arrays': arrays, 'rho': RHO,
                'temperature': 1, 'base_seed': SEED, 'cases': cases, 'truth_reference_scores': truth_scores,
                'recovery': recovery, 'controls': control_result, 'resources': run.resource_report(wall, cpu),
                'trace': run.artifact(BULK/'trace.jsonl'), 'paid_spend_usd': 0,
                'original_training_unchanged': True, 'single_seed_exposed_development_only': True,
                'global_float_support_or_mixing_qualified': False, 'mh_or_evidence_estimation_performed': False,
                'historical_or_recovery_qualification': False, 'torch_version': torch.__version__,
                'torch_threads': torch.get_num_threads()}
            run.save_new(OUT/'result.json', result)
            return result['status']
        except Exception as error:
            run.save_new(OUT/'failure.json', {'error': repr(error), 'no_retry': True, 'cases': cases,
                'completed_truth_references': truth_scores, 'resources': run.resource_report(wall, cpu),
                'trace': run.artifact(BULK/'trace.jsonl')})
            raise
        finally:
            signal.alarm(0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--freeze', required=True)
    print(admission(parser.parse_args().freeze))
