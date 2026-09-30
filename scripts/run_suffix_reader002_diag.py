"""One answer-informed posterior/length diagnosis; no retuning or qualification."""
import argparse
import gc
import json
import math
import resource
import signal
import time

from scripts.audit_blind_channel_dev004 import edit_distance
from scripts.audit_suffix_posterior_diagnostics import replay
from scripts.audit_suffix_reader002 import reading
from scripts.run_blind_channel_dev001 import checked_artifact, digest, require_frozen
from scripts.run_blind_channel_dev004 import load_archive, resource_report, save_new
from scripts.run_suffix_reader002 import ARMS, PATHS, ROOT
from voynich.calibrated_suffix_source import DepthSource
from voynich.suffix_posterior_diagnostics import diagnose

EXP = 'SUFFIX-READER-002-DIAG-A'
OUT = ROOT / 'results' / EXP
INPUT = ROOT / 'results/SUFFIX-READER-002'
BINDINGS = [*PATHS, *[f'results/SUFFIX-READER-002/{f}.json' for f in
    ('calibration', 'panel_manifest', 'prediction_manifest', 'evaluation')],
    'src/voynich/suffix_posterior_diagnostics.py', 'scripts/audit_suffix_posterior_diagnostics.py',
    'scripts/run_suffix_reader002_diag.py', 'tests/test_suffix_posterior_diagnostics.py',
    'docs/experiments/SUFFIX-READER-002-DIAG-A.md']


def run(freeze):
    require_frozen(freeze, BINDINGS)
    resource.setrlimit(resource.RLIMIT_CPU, (1200, 1200))
    def timeout(*_):
        raise TimeoutError('Registered wall cap')
    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(1800)
    save_new(OUT / 'started.json', {'freeze': freeze, 'start_unix': time.time()})
    wall, cpu = time.monotonic(), time.process_time()
    calibration = json.loads((INPUT / 'calibration.json').read_text())
    panel_meta = json.loads((INPUT / 'panel_manifest.json').read_text())
    pred_meta = json.loads((INPUT / 'prediction_manifest.json').read_text())
    evaluation = json.loads((INPUT / 'evaluation.json').read_text())
    if (panel_meta['calibration_sha256'] != digest((INPUT / 'calibration.json').read_bytes())
            or pred_meta['panel_manifest_sha256'] != digest((INPUT / 'panel_manifest.json').read_bytes())
            or pred_meta['calibration_sha256'] != panel_meta['calibration_sha256']):
        raise ValueError('Input bindings differ')
    panel = checked_artifact(panel_meta['panel'])
    truth = checked_artifact(panel_meta['answers'])['cases']
    predictions = load_archive(pred_meta['predictions'])['cases']
    raw = load_archive(calibration['shared_counts_archive'])
    masses = calibration['selected']['masses']
    if len(truth) != 16 or set(truth) != set(panel['cases']) or set(truth) != set(predictions):
        raise ValueError('All sixteen cases required')
    rows, lengths, texts = [], [], {}
    max_delta = {field: 0. for field in ('log_likelihood', 'joint_log_probability',
                                        'entropy_bits', 'expected_length', 'length_variance')}
    vectors = ([256.] * 3, [256.] * 12, masses, masses[:3])
    for arm, vector in zip(ARMS, vectors, strict=True):
        model = DepthSource(raw['alphabet'], vector,
                            {h: r for h, r in raw['counts'].items() if len(h) <= len(vector)})
        view = {'alphabet': model.alphabet, 'masses': model.masses, 'counts': model.counts}
        for name, case in panel['cases'].items():
            for index, (observed, gold) in enumerate(zip(case['records'], truth[name], strict=True)):
                old = predictions[name][arm][index]
                answer = reading(view, case['units'], observed, gold, 1 / 225)
                if answer > old['joint_log_probability'] + 1e-7:
                    raise ValueError('Gold exceeds frozen exact MAP')
                for supplied in ((None, len(gold)) if arm == 'calibrated12' else (None,)):
                    actual = diagnose(model, case['units'], observed, 1 / 225,
                                      length=supplied, max_nodes=500_000)
                    ref = replay(view, case['units'], observed, 1 / 225,
                                 length=supplied, max_nodes=500_000)
                    for field in max_delta:
                        delta = abs(getattr(actual, field) - ref[field])
                        max_delta[field] = max(max_delta[field], delta)
                        if delta > (1e-5 if field == 'length_variance' else 1e-6):
                            raise ValueError(f'Reference differs: {field}')
                    if actual.path_count != ref['path_count']:
                        raise ValueError('Reference path counts differ')
                    path_score = reading(view, case['units'], observed, actual.plaintext, 1 / 225)
                    if abs(path_score - actual.joint_log_probability) > 1e-7:
                        raise ValueError('Returned path does not match score')
                    if actual.entropy_bits > math.log2(actual.path_count) + 1e-6:
                        raise ValueError('Entropy exceeds support bound')
                    if actual.entropy_bits < (actual.log_likelihood - actual.joint_log_probability) / math.log(2) - 1e-6:
                        raise ValueError('Entropy below min-entropy')
                    if supplied is None:
                        if (actual.plaintext != old['plaintext']
                                or abs(actual.log_likelihood - old['log_likelihood']) > 1e-7
                                or abs(actual.joint_log_probability - old['joint_log_probability']) > 1e-7):
                            raise ValueError('Unconstrained diagnostic changes frozen reading')
                    else:
                        if (len(actual.plaintext) != supplied
                                or actual.log_likelihood > old['log_likelihood'] + 1e-7
                                or actual.joint_log_probability > old['joint_log_probability'] + 1e-7
                                or actual.joint_log_probability < answer - 1e-7):
                            raise ValueError('Length counterfactual subset/score check failed')
                        texts.setdefault(name, []).append(actual.plaintext)
                    row = actual.to_dict()
                    del row['plaintext']
                    row.update({'case': name, 'record': index, 'arm': arm, 'supplied_length': supplied,
                        'edits': edit_distance(actual.plaintext, gold), 'exact': actual.plaintext == gold,
                        'decoded_length': len(actual.plaintext), 'gold_length': len(gold),
                        'gold_joint_log_probability': answer,
                        'map_probability': math.exp(actual.joint_log_probability - actual.log_likelihood),
                        'gold_posterior_surprisal_bits': (actual.log_likelihood - answer) / math.log(2),
                        'map_over_gold_bits': (actual.joint_log_probability - answer) / math.log(2)})
                    if supplied is not None:
                        row['posterior_mass_on_true_length'] = math.exp(actual.log_likelihood - old['log_likelihood'])
                    (rows if supplied is None else lengths).append(row)
            print(json.dumps({'completed_arm': arm, 'case': name}), flush=True)
        del model, view
        gc.collect()
    if len(rows) != 128 or len(lengths) != 32:
        raise ValueError('Incomplete panel')
    summary = {}
    for arm in ARMS:
        subset = [r for r in rows if r['arm'] == arm]
        if sum(r['edits'] for r in subset) != evaluation['total_edits'][arm]:
            raise ValueError('Frozen aggregate edit count differs')
        summary[arm] = {'edits': sum(r['edits'] for r in subset),
            'exact_records': sum(r['exact'] for r in subset),
            'model_expected_exact_records': math.fsum(r['map_probability'] for r in subset),
            'mean_entropy_bits': math.fsum(r['entropy_bits'] for r in subset) / 32,
            'mean_expected_length': math.fsum(r['expected_length'] for r in subset) / 32,
            'mean_gold_posterior_surprisal_bits': math.fsum(r['gold_posterior_surprisal_bits'] for r in subset) / 32}
    summary['calibrated12_true_length'] = {
        'edits': sum(r['edits'] for r in lengths), 'exact_records': sum(r['exact'] for r in lengths),
        'mean_entropy_bits': math.fsum(r['entropy_bits'] for r in lengths) / 32,
        'mean_posterior_mass_on_true_length': math.fsum(r['posterior_mass_on_true_length'] for r in lengths) / 32,
        'per_key_edits': {name: sum(r['edits'] for r in lengths if r['case'] == name) for name in truth}}
    artifact = save_new(ROOT / 'outputs' / EXP / 'length_oracle_readings.json.gz', texts, compressed=True)
    result = {'experiment': EXP, 'freeze': freeze,
        'status': 'answer_informed_posthoc_diagnosis_not_new_qualification',
        'input_evaluation_sha256': digest((INPUT / 'evaluation.json').read_bytes()),
        'unconstrained': rows, 'true_length': lengths, 'summary': summary,
        'independent_reverse_replays': 160, 'maximum_reference_deltas': max_delta,
        'length_oracle_readings': artifact, 'resources': resource_report(wall, cpu)}
    save_new(OUT / 'diagnostic.json', result)
    print(json.dumps({'summary': summary, 'maximum_reference_deltas': max_delta}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--freeze', required=True)
    args = parser.parse_args()
    try:
        run(args.freeze)
    except Exception as error:
        if not (OUT / 'failure.json').exists():
            save_new(OUT / 'failure.json', {'error': str(error), 'type': type(error).__name__, 'freeze': args.freeze})
        raise
