"""Retrospective PATH-0007 two-choice first-token logit diagnostic."""

import json
from pathlib import Path

import numpy as np
from scipy.special import logsumexp
from transformers import AutoTokenizer

from voynich.workspace.campaign import canonical_digest, digest, load_config, write_json
from voynich.workspace.path7_tasks import path7_tasks

ROOT = Path('results/PATH-0009')
RAW = Path('outputs/PATH-0007/baselines')


def variants(tokenizer, word):
    ids = set()
    for string in (word, ' '+word):
        encoded = tokenizer.encode(string, add_special_tokens=False)
        if len(encoded) == 1:
            ids.add(encoded[0])
    return sorted(ids)


def run():
    ROOT.mkdir(parents=True, exist_ok=True)
    if (ROOT/'results.json').exists():
        raise FileExistsError('No automatic repeat of PATH-0009')
    snapshot = load_config('configs/jspace0001.json')['model_snapshot']
    tokenizer = AutoTokenizer.from_pretrained(snapshot, trust_remote_code=False, local_files_only=True)
    rows = []
    missing = []
    for split in ('discovery', 'confirmation'):
        for task in path7_tasks(split):
            if task['family'] != 'composed':
                continue
            expected, alternative = variants(tokenizer, task['answer']), variants(tokenizer, task['donor_answer'])
            if not expected or not alternative:
                missing.append(task['id'])
                continue
            local = RAW/(canonical_digest(task['prompt'])+'.npz')
            with np.load(local) as arrays:
                logits = arrays['first_logits'].astype(np.float64)
                generated = arrays['generated_tokens'].tolist()
            expected_score = float(logsumexp(logits[expected]))
            alternate_score = float(logsumexp(logits[alternative]))
            row = {
                'id': task['id'], 'split': split, 'bundle': task['bundle'],
                'style': task['style'], 'orientation': int(task['id'].split('/')[-2]),
                'query_slot': int(task['id'].split('/')[-1]),
                'expected': task['answer'], 'alternative': task['donor_answer'],
                'expected_token_ids': expected, 'alternative_token_ids': alternative,
                'margin': expected_score-alternate_score,
                'expected_preferred': expected_score > alternate_score,
                'generated_object_token': bool(generated) and generated[0] in set(expected+alternative),
                'array_sha256': digest(local),
            }
            rows.append(row)
    groups = {}
    for row in rows:
        key = (row['split'], row['bundle'], row['style'], row['orientation'])
        groups.setdefault(key, []).append(row)
    pairs = []
    for (split, bundle, style, orientation), members in sorted(groups.items()):
        assert sorted(r['query_slot'] for r in members) == [0, 1]
        assert members[0]['expected'] == members[1]['alternative']
        pairs.append({'split': split, 'bundle': bundle, 'style': style, 'orientation': orientation,
                      'both_expected_preferred': all(r['expected_preferred'] for r in members),
                      'query_margins': [next(r['margin'] for r in members if r['query_slot'] == slot)
                                        for slot in (0, 1)]})
    summary = {split: {style: {
        'query_count': sum(r['split'] == split and r['style'] == style for r in rows),
        'query_expected_preferred': sum(r['split'] == split and r['style'] == style and r['expected_preferred']
                                        for r in rows),
        'paired_query_count': sum(r['split'] == split and r['style'] == style for r in pairs),
        'paired_reversals': sum(r['split'] == split and r['style'] == style and r['both_expected_preferred']
                                for r in pairs),
        'generated_object_first': sum(r['split'] == split and r['style'] == style and r['generated_object_token']
                                      for r in rows),
    } for style in ('colon', 'arrow')} for split in ('discovery', 'confirmation')}
    write_json(ROOT/'results.json', {
        'status': 'exploratory_exposed', 'source_sha256': {
            'docs/experiments/PATH-0009.md': digest('docs/experiments/PATH-0009.md'),
            'scripts/path0009_candidate_logits.py': digest('scripts/path0009_candidate_logits.py'),
            'results/PATH-0007/inputs.json': digest('results/PATH-0007/inputs.json'),
        },
        'tokenizer_snapshot': snapshot, 'missing_single_token_variant': missing,
        'rows': rows, 'query_pairs': pairs, 'summary': summary,
    })
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    run()
