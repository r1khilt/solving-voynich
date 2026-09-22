"""Audit dependence and positional confounds without scoring any model output."""

from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

from voynich.workspace.campaign import canonical_digest, digest, load_config, write_json
from voynich.workspace.tasks import build_tasks


def main():
    config = load_config('configs/jspace0001.json')
    manifest = json.loads(Path('results/JSPACE-0001/inputs.json').read_text())
    task_path = 'src/voynich/workspace/tasks.py'
    if digest(task_path) != manifest['source_sha256'][task_path]:
        raise ValueError('Frozen tasks changed')
    tokenizer = AutoTokenizer.from_pretrained(config['model_snapshot'], local_files_only=True, trust_remote_code=False)
    cache = {}

    def encode(prompt):
        if prompt not in cache:
            rendered = tokenizer.apply_chat_template([{'role': 'user', 'content': prompt}], tokenize=False,
                                                     add_generation_prompt=True, enable_thinking=False)
            cache[prompt] = tokenizer.encode(rendered, add_special_tokens=False)
        return cache[prompt]

    rows = []
    for task in build_tasks(seed=config['seed']):
        source, donor = encode(task['prompt']), encode(task['donor_prompt'])
        row = {k: task[k] for k in ('id', 'split', 'family', 'relation', 'pair_id', 'prompt_group', 'paraphrase_id')}
        row.update({'source_tokens': len(source), 'donor_tokens': len(donor),
                    'source_ids_sha256': canonical_digest(source), 'donor_ids_sha256': canonical_digest(donor)})
        rows.append(row)
    groups = defaultdict(list)
    for row in rows:
        groups[(row['split'], row['family'])].append(row)
    summaries = []
    for (split, family), members in groups.items():
        counts = Counter(r['prompt_group'] for r in members)
        lengths = np.array([r['source_tokens'] for r in members])
        differences = np.array([r['donor_tokens']-r['source_tokens'] for r in members])
        summaries.append({'split': split, 'family': family, 'records': len(members),
                          'unique_rendered_prompts': len(counts), 'prompt_multiplicities': dict(Counter(counts.values())),
                          'unordered_country_pairs': len({r['pair_id'] for r in members}),
                          'source_tokens_min_median_max': [int(lengths.min()), float(np.median(lengths)), int(lengths.max())],
                          'unequal_length_donor_fraction': float(np.mean(differences != 0)),
                          'absolute_length_difference_mean_max': [float(np.abs(differences).mean()), int(np.abs(differences).max())]})
    output = Path(config['output']) / 'input-confound-rows.json'
    write_json(output, rows)
    pilot_path = Path(config['output']) / 'pilot-rendered-inputs.json'
    pilot = json.loads(pilot_path.read_text())
    if any(encode(prompt) != ids for prompt, ids in pilot.items()):
        raise ValueError('Independent tokenizer rendering differs from actual pilot inputs')
    report = {'status': 'Input audit only; no model outputs or causal outcomes inspected', 'summaries': summaries,
              'pilot_rendering_exact_matches': len(pilot), 'pilot_inputs_sha256': digest(pilot_path),
              'rows_sha256': digest(output), 'script_sha256': digest(__file__),
              'inputs_sha256': digest('results/JSPACE-0001/inputs.json'),
              'limitations': [
                  'Rows sharing countries, facts, templates, and reverse directions are dependent; record counts are not independent sample sizes.',
                  'Different prompt lengths can change positional and attention context. Last-prefix edits do not isolate a position-invariant concept.',
                  'Cross-query neuron differences can transfer a mixture of country, landmark, and length effects; matched random edits alone cannot distinguish these.',
                  'Repeated copy prompts retain their registered record weights. No primary denominator or acceptance rule is changed after this audit.',
                  'A prompt-disjoint paraphrase holdout does not provide new world facts, languages, or independent training seeds.'
              ]}
    write_json(Path(config['results']) / 'input-confound-audit.json', report)
    for row in summaries:
        print(json.dumps(row), flush=True)


if __name__ == '__main__':
    main()
