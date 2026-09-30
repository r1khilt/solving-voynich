"""Describe the complete fixed reader panel without selecting or rerunning arms."""
import json

from scripts.run_blind_channel_dev004 import save_new
from scripts.run_neural_reader001 import ARMS, OUT, ROOT, artifact, verify


def summarize():
    path = OUT / 'evaluation.json'
    result = json.loads(path.read_text())
    verify(result['prediction_manifest'])
    if set(result['case_metrics']) != set(ARMS):
        raise ValueError('Complete six-arm result required')
    names = [f'B-key{i}' for i in range(1, 17)]
    rows = {}
    for arm in ARMS:
        cases = result['case_metrics'][arm]
        counts = [cases[name]['record_edits'] for name in names]
        if len(cases) != 32 or any(len(c) != 2 for c in counts):
            raise ValueError('Complete two-author positive/null panel required')
        edits = sum(sum(c) for c in counts)
        if edits != result['positive_total_edits'][arm]:
            raise ValueError('Reported total and per-record errors disagree')
        rows[arm] = {'edits': edits, 'cer': edits / 7168,
                     'exact_records': sum(cases[n]['exact_records'] for n in names),
                     'author_edits': [sum(c[i] for c in counts) for i in range(2)],
                     'per_key_edits': [sum(c) for c in counts],
                     'keys_above_5pct': [i + 1 for i, c in enumerate(counts) if 20 * sum(c) > 448],
                     'null_edits': result['null_total_edits'][arm]}
    diagnostic_counts = {}
    for arm, cases in result['search_source_diagnostics'].items():
        diagnostic_counts[arm] = {}
        for positive in (True, False):
            records = [row for n in names for row in cases[n if positive else n + '-shuffle']]
            diagnostic_counts[arm]['positive' if positive else 'null'] = {
                key: sum(row[key] for row in records) for key in (
                    'better_known_path_proves_search_gap', 'wrong_returned_path_beats_gold',
                    'narrow_search_finds_better_score', 'score_bound_flag_computed_not_interval_proof')}
    stages = {stage: json.loads((OUT / file).read_text())['resources']
              for stage, file in (('prepare', 'panel_manifest.json'), ('predict', 'prediction_manifest.json'),
                                  ('evaluate', 'evaluation.json'))}
    summary = {'evaluation': artifact(path), 'arms': rows, 'diagnostic_counts': diagnostic_counts,
               'primary_gates': result['primary_gates'], 'joint_pass': result['both_neural_seeds_pass'],
               'resources': stages, 'known_keys_supplied': True, 'unknown_key_recovery_tested': False}
    save_new(OUT / 'summary.json', summary)
    lines = ['# NEURAL-READER-001 results', '',
             f"Joint registered reader gate: **{'PASS' if summary['joint_pass'] else 'FAIL'}**.", '',
             'All six arms read the same 32 fresh Latin passages and 32 matched plaintext-permutation '
             'controls. Correct keys, language alphabet, cipher family and record boundaries were supplied. '
             'Each positive record has 224 true letters; the decoder receives a geometric length prior '
             'but is not told that exact length. Positive denominator: 7,168 letters.', '',
             '| Source / search | Positive edits | Edit rate | Exact /32 | Nepos edits /3584 | Apuleius edits /3584 | Null edits /7168 |',
             '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for arm, row in rows.items():
        lines.append(f"| {arm} | {row['edits']} | {100 * row['cer']:.4f}% | {row['exact_records']} | "
                     f"{row['author_edits'][0]} | {row['author_edits'][1]} | {row['null_edits']} |")
    lines += ['', 'The primary arms are both neural seeds at beam128; beam32 is diagnostic. '
              'No arm or source was selected from these errors. Null errors measure behavior on '
              'shuffled plaintext and do not define a language detector.', '', '## Registered gates', '']
    for arm, gate in summary['primary_gates'].items():
        reduction = gate['relative_edit_reduction']
        shown = 'undefined (zero baseline)' if reduction is None else f'{100 * reduction:.4f}%'
        lines += [f"**{arm}: {'PASS' if gate['pass'] else 'FAIL'}**. Edit reduction versus large statistical: {shown}. "
                  f"Keys above 5% error: {rows[arm]['keys_above_5pct']}.", '',
                  *[f"- {name}: {passed}" for name, passed in gate['gates'].items()], '']
    lines += ['## Search versus source diagnostics', '',
              'Counts below cover the 32 positive records. A higher-scoring gold path demonstrates '
              'a missed better candidate. A higher-scoring wrong returned path shows that this '
              'source objective prefers an incorrect alternative. These comparisons use full-forward '
              'replays and the frozen numerical tolerance; they are not exact posterior estimates.', '',
              '| Primary arm | Gold proves search gap | Wrong returned path beats gold | Beam32 beats beam128 score | Bound flag |',
              '| --- | ---: | ---: | ---: | ---: |']
    for arm, row in diagnostic_counts.items():
        d = row['positive']
        lines.append(f"| {arm} | {d['better_known_path_proves_search_gap']} | {d['wrong_returned_path_beats_gold']} | "
                     f"{d['narrow_search_finds_better_score']} | {d['score_bound_flag_computed_not_interval_proof']} |")
    lines += ['', 'Bound flags use floating scores and are not interval-arithmetic proofs. MAP optimizes '
              'exact-string probability rather than edit distance. Complete per-key errors, shuffled '
              'diagnostics and numerical margins remain in the tracked evaluation and summary.', '',
              'In this panel, neither gold paths nor the narrower search demonstrate a higher-scoring '
              'alternative to the primary positive readings; that is not proof of exact MAP search. '
              'Every erroneous primary positive reading scores above its true path. Wider search '
              'therefore cannot make the truth become the MAP answer on those records with this '
              'unchanged source. Neural A beam32 makes20edits versus23atbeam128, even though its '
              'score is not better: this illustrates the difference between path probability and '
              'edit accuracy. The primary result stays23.', '',
              '## Validation and limits', '',
              'Every returned path re-encodes exactly. Both statistical arms have separate backward '
              'marginal/MAP and path-score checks; neural paths have full-forward score checks. All384 '
              'edit counts agree with a separate full-grid calculation. Sources were fixed before '
              'author allocation, and predictions were published before answer scoring.', '',
              'This panel has two authors and16 sampled keys, not a population-wide guarantee. '
              'No unknown key or Voynich passage was decoded. Any future repair using these errors '
              'turns this panel into development data and needs a new allocation for qualification.', '',
              '## Resources', '']
    for stage, r in stages.items():
        lines.append(f"- {stage}: {r['wall_seconds']:.3f}s wall, {r['cpu_seconds']:.3f}s host CPU; "
                     f"peak RSS {r['peak_rss_bytes']} bytes.")
    lines += ['', 'Zero paid experiment API/cloud use. GPU time is reflected in elapsed time; '
              'local energy was not measured. Bulk readings and weights remain ignored with hashes.', '',
              '![Primary reader errors and all per-key outcomes](../../results/NEURAL-READER-001/reader-comparison.png)', '']
    target = ROOT / 'docs/experiments/NEURAL-READER-001-results.md'
    with target.open('x') as handle:
        handle.write('\n'.join(lines))
    print(json.dumps({'joint_pass': summary['joint_pass'], 'arms': rows, 'diagnostics': diagnostic_counts}))


if __name__ == '__main__':
    summarize()
