"""Recompute PATH-0009 candidate margins without its scorer functions."""

import json
from pathlib import Path

import numpy as np

from voynich.workspace.campaign import canonical_digest, digest, write_json
from voynich.workspace.path7_tasks import path7_tasks

ROOT = Path('results/PATH-0009')
BASE = Path('outputs/PATH-0007/baselines')


def audit():
    result = json.loads((ROOT/'results.json').read_text())
    for name, expected in result['source_sha256'].items():
        assert digest(name) == expected, name
    tasks = {row['id']: row for split in ('discovery', 'confirmation') for row in path7_tasks(split)}
    rows = result['rows']
    assert len(rows) == 96 and not result['missing_single_token_variant']
    assert len({row['id'] for row in rows}) == len(rows)
    for row in rows:
        task = tasks[row['id']]
        assert task['family'] == 'composed'
        assert row['expected'] == task['answer'] and row['alternative'] == task['donor_answer']
        local = BASE/(canonical_digest(task['prompt'])+'.npz')
        assert digest(local) == row['array_sha256']
        with np.load(local) as baseline:
            logits = baseline['first_logits'].astype(np.float64)
            generated = baseline['generated_tokens'].tolist()
        expected = np.logaddexp.reduce(logits[row['expected_token_ids']])
        alternate = np.logaddexp.reduce(logits[row['alternative_token_ids']])
        assert abs(row['margin']-float(expected-alternate)) < 1e-9
        assert row['expected_preferred'] == bool(expected > alternate)
        assert row['generated_object_token'] == (bool(generated) and
                generated[0] in set(row['expected_token_ids']+row['alternative_token_ids']))
    pairs = result['query_pairs']
    assert len(pairs) == 48
    for pair in pairs:
        matching = [row for row in rows if all(row[key] == pair[key]
                    for key in ('split', 'bundle', 'style', 'orientation'))]
        assert len(matching) == 2 and {row['query_slot'] for row in matching} == {0, 1}
        assert pair['both_expected_preferred'] == all(row['expected_preferred'] for row in matching)
        assert pair['query_margins'] == [next(row['margin'] for row in matching if row['query_slot'] == i)
                                         for i in (0, 1)]
    for split in ('discovery', 'confirmation'):
        for style in ('colon', 'arrow'):
            subset = [row for row in rows if row['split'] == split and row['style'] == style]
            related = [pair for pair in pairs if pair['split'] == split and pair['style'] == style]
            summary = result['summary'][split][style]
            assert summary['query_count'] == len(subset) == 24
            assert summary['query_expected_preferred'] == sum(row['expected_preferred'] for row in subset)
            assert summary['paired_query_count'] == len(related) == 12
            assert summary['paired_reversals'] == sum(pair['both_expected_preferred'] for pair in related)
            assert summary['generated_object_first'] == sum(row['generated_object_token'] for row in subset)
    report = {'passed': True, 'rows': len(rows), 'query_pairs': len(pairs),
              'scope': 'Retrospective exposed-data first-token candidate diagnostic only'}
    write_json(ROOT/'audit.json', report)
    return report


if __name__ == '__main__':
    print(json.dumps(audit(), indent=2))
