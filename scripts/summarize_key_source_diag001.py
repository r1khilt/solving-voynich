"""Result-only report for the frozen exposed-key diagnostic."""
import json
import math

from scripts.run_blind_channel_dev004 import save_new
from scripts.run_key_source_diag001 import ARMS, OUT, ROOT, artifact


def main():
    result = json.loads((OUT / 'evaluation.json').read_text())
    old = json.loads((ROOT / result['original_evaluation']['path']).read_text())
    if artifact(ROOT / result['original_evaluation']['path']) != result['original_evaluation']:
        raise ValueError('Original evaluation identity changed')
    if result['reading_count'] != 576 or result['new_key_fits'] != 0:
        raise ValueError('Diagnostic scope differs')
    totals = {}
    for arm in ARMS:
        totals[arm] = {}
        for key in ('learned', 'gold'):
            rows = [result['case_metrics'][arm][f'B-key{i}'][key]['transfer'] for i in range(1, 9)]
            if sum(r['gold_characters'] for r in rows) != 3584:
                raise ValueError('Changed positive transfer denominator')
            totals[arm][key] = {field: sum(r[field] for r in rows)
                                for field in ('edits', 'exact_records', 'supported_records')}
    historical = {key: sum(old['cases'][f'B-key{i}'][key]['transfer']['edits'] for i in range(1, 9))
                  for key in ('learned', 'oracle')}
    summary = {'evaluation': artifact(OUT / 'evaluation.json'), 'totals': totals,
               'historical_order3_edits': historical, 'positive_transfer_characters': 3584,
               'no_new_key_recovery_or_qualification': True}
    save_new(OUT / 'summary.json', summary)
    lines = ['# KEY-SOURCE-DIAG-001 results', '',
             '**The improved sources reduce reading errors but do not repair the old learned keys.** '
             'This is an exposed fixed-key diagnostic, with zero new key fits and no changed qualification.', '',
             '| Source | Unchanged learned-key edits /3584 | True-key edits /3584 | Learned supported /16 | True-key exact /16 |',
             '| --- | ---: | ---: | ---: | ---: |',
             f"| Historical old order3 | {historical['learned']} | {historical['oracle']} | 15 | 1 |"]
    for arm in ARMS:
        rows = totals[arm]
        lines.append(f"| {arm} | {rows['learned']['edits']} | {rows['gold']['edits']} | "
                     f"{rows['learned']['supported_records']} | {rows['gold']['exact_records']} |")
    lines += ['', 'All new arms use the same eight old positive keys and passages, plus all eight '
              'old glyph-shuffle controls. The historical row is the previously frozen result, not '
              'a new measurement or a pure data-size control. New small/large statistical arms share '
              'the same estimator and selection policy. The neural sources were selected on Pliny '
              'before this diagnostic, but the diagnostic itself is adaptively chosen and not fresh.', '',
              '## Exact statistical key preferences', '',
              'Positive gaps below mean the true key is a known better fitting candidate. Negative '
              'gaps mean the unchanged wrong key has a better exact marginal-plus-code objective. '
              'Neither establishes the global optimum.', '',
              '| Key | Small: learned minus true, bits | Large: learned minus true, bits | Large code-cost gap | Large data-cost gap |',
              '| --- | ---: | ---: | ---: | ---: |']
    for i in range(1, 9):
        a = result['case_metrics']['statistical-small'][f'B-key{i}']
        b = result['case_metrics']['statistical-large'][f'B-key{i}']
        gap = b['fit_key_comparison']['learned_minus_gold_total_bits']
        code = b['learned']['fit']['model_bits'] - b['gold']['fit']['model_bits']
        lines.append(f"| {i} | {a['fit_key_comparison']['learned_minus_gold_total_bits']:.6f} | "
                     f"{gap:.6f} | {code} | {gap - code:.6f} |")
    lines += ['', 'Keys4and8 now have known better true-key candidates under the large statistical '
              'source by291.920and112.376bits. No new search was run, so these keys are still not '
              'recovered. Key7still prefers the wrong key by15.557bits. Key6has a6-bit coding '
              'advantage for the shorter learned dictionary, with only0.050bits of additional data '
              'advantage; the resulting missing singleton still makes one transfer record unsupported.', '',
              'A new source therefore does not erase uncertainty about unseen assignments. Forkey6, '
              'the large statistical true-key reading is perfect, while the unchanged learned key '
              'incurs224errors from one unsupported record. No source can create a parse outside '
              'the fixed dictionary support. This is a dictionary limitation, not a new reading failure.', '',
              '## Neural estimates have a different meaning', '',
              'The table below compares found joint key/text scores in bits, not marginal likelihoods. '
              'Positive favors the found true-key candidate. Every pair of completion-bound intervals '
              'overlaps, so none of these differences certifies ordering of the exact joint MAP objectives.', '',
              '| Key | Neural A found true minus learned | Neural B found true minus learned |',
              '| --- | ---: | ---: |']
    for i in range(1, 9):
        values = [result['case_metrics'][arm][f'B-key{i}']['fit_key_comparison']
                  for arm in ('neural-31103', 'neural-31109')]
        if any(v['floating_bounds_favor_gold'] or v['floating_bounds_favor_learned'] for v in values):
            raise ValueError('Update report interpretation: some bounds separate')
        lines.append(f"| {i} | {values[0]['gold_minus_learned_returned_joint_nats'] / math.log(2):.6f} | "
                     f"{values[1]['gold_minus_learned_returned_joint_nats'] / math.log(2):.6f} |")
    r = result['resources']
    lines += ['', 'These candidate scores motivate further investigation, especially key7, but do '
              'not justify substituting a beam score for exact evidence or declaring a recovered key.', '',
              '## Validation, scope and next action', '',
              'All576readings completed. Statistical backward marginal/MAP/node and path checks '
              'agree within1.48e-12nats; neural incremental/full-forward paths within5.01e-5nats. '
              'Supported readings re-encode, Boolean support checks agree, and all384positive '
              'record edits match a separate full-grid calculation. Nulls have no true plaintext; '
              'their full scores and support remain in all16case reports per source. No null was dropped.', '',
              f"Elapsed{r['wall_seconds']:.3f}s, host CPU{r['cpu_seconds']:.3f}s, peak RSS{r['peak_rss_bytes']}bytes. "
              'The run exceeded the rough six-minute estimate but stayed within its fixed900second '
              'cap. Zero paid API/cloud use; no restart or key alteration.', '',
              'The next pipeline needs stronger search and explicit key uncertainty. The separately '
              'implemented [shared-key decoder](../research/shared-key-mixture-decoding-2026-09-30.md) '
              'has artificial checks only and has not been applied to these cases. It cannot fix '
              'a candidate bank that omits every useful key. A subsequent empirical bank policy and '
              'fresh recovery qualification must be frozen separately. The original confirmation '
              'FAIL and later supplied-key reader PASS retain their original meanings.', '']
    path = ROOT / 'docs/experiments/KEY-SOURCE-DIAG-001-results.md'
    with path.open('x') as stream:
        stream.write('\n'.join(lines))
    print(json.dumps(summary))


if __name__ == '__main__':
    main()
