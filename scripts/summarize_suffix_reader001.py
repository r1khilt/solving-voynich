"""Post-evaluation descriptive decomposition; no fitting or new decoding."""
import json
import math
import resource
import time
from pathlib import Path

from scripts.audit_suffix_reader001 import reading_score
from scripts.run_blind_channel_dev001 import checked_artifact, digest
from scripts.run_blind_channel_dev004 import load_archive, resource_report, save_new
from voynich.sparse_suffix_source import SuffixSource

ROOT = Path(__file__).resolve().parents[1]
DIR = ROOT / 'results/SUFFIX-READER-001'


def decompose(raw, history, char):
    """Exact empirical-distribution mixture implied by recursive interpolation."""
    root = raw['counts']['']
    root_p = (root.get(char, 0) + .5) / (sum(root.values()) + .5 * len(raw['alphabet']))
    weights, contributions = [0.] * (raw['order'] + 1), [0.] * (raw['order'] + 1)
    remaining = 1.
    for depth in range(min(len(history), raw['order']), 0, -1):
        context = history[-depth:]
        if context in raw['counts']:
            row = raw['counts'][context]
            count = sum(row.values())
            weights[depth] = remaining * count / (count + raw['tau'])
            contributions[depth] = weights[depth] * row.get(char, 0) / count
            remaining *= raw['tau'] / (count + raw['tau'])
    weights[0], contributions[0] = remaining, remaining * root_p
    probability = math.fsum(contributions)
    assert abs(math.fsum(weights) - 1.) <= 1e-12 and probability > 0
    return weights, [p / probability for p in contributions], probability


def main():
    resource.setrlimit(resource.RLIMIT_CPU, (60, 60))
    started, cpu = time.monotonic(), time.process_time()
    evaluation = json.loads((DIR / 'evaluation.json').read_text())
    selection = json.loads((DIR / 'source_selection.json').read_text())
    manifest = json.loads((DIR / 'panel_manifest.json').read_text())
    prediction_manifest = json.loads((DIR / 'prediction_manifest.json').read_text())
    truth = checked_artifact(manifest['answers'])['cases']
    panel = checked_artifact(manifest['panel'])['cases']
    predictions = load_archive(prediction_manifest['predictions'])['cases']
    raw = load_archive(selection['source'])
    source = SuffixSource.from_dict(raw)
    totals = {kind: {'n': 0, 'prior': [0.] * (raw['order'] + 1), 'posterior': [0.] * (raw['order'] + 1)}
              for kind in ('truth_history', 'selected_reading_history')}
    maximum = 0.
    records = []
    for name in truth:
        for i, gold in enumerate(truth[name]):
            row = predictions[name]['selected'][i]
            gold_score = reading_score(raw, panel[name]['units'], panel[name]['records'][i], gold, 1 / 225)
            assert gold_score <= row['joint_log_probability'] + 1e-7
            records.append({'case': name, 'record': i, 'correct_map': row['plaintext'] == gold,
                            'map_minus_truth_joint_bits': (row['joint_log_probability'] - gold_score) / math.log(2),
                            'conditional_map_probability': math.exp(row['joint_log_probability'] - row['log_likelihood'])})
            for kind, text in (('truth_history', gold), ('selected_reading_history', row['plaintext'])):
                for j, char in enumerate(text):
                    weights, responsibilities, probability = decompose(raw, text[:j], char)
                    state = source.state(text[:j])
                    maximum = max(maximum, abs(source.probabilities[state, source.letters[char]] - probability))
                    target = totals[kind]
                    target['n'] += 1
                    for depth in range(raw['order'] + 1):
                        target['prior'][depth] += weights[depth]
                        target['posterior'][depth] += responsibilities[depth]
    assert maximum <= 1e-12
    for target in totals.values():
        for kind in ('prior', 'posterior'):
            target[kind] = [v / target['n'] for v in target[kind]]
        target['prior_mass_depth_at_least4'] = math.fsum(target['prior'][4:])
        target['posterior_mass_depth_at_least4'] = math.fsum(target['posterior'][4:])
    cases = evaluation['cases']
    result = {'experiment': 'SUFFIX-READER-001', 'status': 'posthoc_descriptive_not_new_qualification',
              'mixture': totals, 'records': records, 'maximum_probability_delta': maximum,
              'key_improved_equal_worse': [sum((r['selected']['edits'] < r['baseline']['edits'],
                   r['selected']['edits'] == r['baseline']['edits'], r['selected']['edits'] > r['baseline']['edits'])[i]
                  for r in cases.values()) for i in range(3)],
              'evaluation_sha256': digest((DIR / 'evaluation.json').read_bytes()),
              'resources': resource_report(started, cpu)}
    save_new(DIR / 'descriptive_supplement.json', result)
    print(json.dumps({'mixture':totals, 'keys_improved_equal_worse':result['key_improved_equal_worse'],
                      'maximum_probability_delta':maximum}))


if __name__ == '__main__':
    main()
