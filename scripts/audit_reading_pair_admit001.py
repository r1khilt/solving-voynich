"""ONE original-source replay of literal maps, independent eligibility and target."""
import argparse
import itertools
import json
import math
import signal
import time

import torch

from scripts import run_reading_pair_admit001 as run
from scripts.audit_reading_label_landscape001 import independent_point
from scripts.audit_reading_regrowth_recovery001 import linear_literal
from voynich.reading_label_landscape import target_fingerprint
from voynich.source_action_proposal import ReadingEnvironment


def direct_eligible(env, state):
    """Full ordered triplet enumeration; no production eligibility list."""
    units = [None if k < 0 else env.pool[k] for k in state.key]
    pairs = {p for text in state.texts for p in zip(text, text[1:])}
    answer = []
    for a, b, c in itertools.permutations(range(env.rows), 3):
        merge = (units[a] is None and units[b] is not None and units[c] is not None
                 and len(units[b]) == len(units[c]) == 1 and (b, c) in pairs)
        split = (units[a] is not None and len(units[a]) == 2
                 and units[b] in (None, units[a][:1]) and units[c] in (None, units[a][1:])
                 and (b, c) not in pairs)
        if merge or split:
            answer.append((a, b, c))
    return tuple(answer)


def audit(freeze):
    run.training.old.require_frozen(freeze, run.PATHS)
    run.require_prior()
    run.training.save_new(run.OUT/'audit-started.json', {'freeze': freeze, 'no_retry': True})
    run.training.limit_resources(run.WALL, run.CPU)
    wall, cpu = time.monotonic(), time.process_time()
    torch.set_num_threads(1)
    try:
        def guard():
            r = run.training.resource_report(wall, cpu)
            if r['wall_seconds'] > run.WALL or r['cpu_seconds'] > run.CPU or r['peak_rss_bytes'] > run.HOST:
                raise MemoryError('Registered structural audit cap; no retry')
            return r

        result = json.loads((run.OUT/'result.json').read_text())
        assert result['freeze'] == freeze and result['status'] == 'PASS_original_source_structural_literal_ratio_and_cost'
        assert result['inputs'] == [run.training.artifact(run.ROOT/p) for p in run.PATHS]
        assert [(c['case'], c['phase']) for c in result['cells']] == [(i, p) for i in range(run.CASES) for p in run.PHASES]
        source, _ = run.load_source()
        assert run.array_identity(source) == result['source_arrays']
        points, private, largest_delta = 0, 0, 0.
        for cell in result['cells']:
            value = run.load_archive(cell['archive'])
            old = run.load_archive(value['input_archive'])
            assert value['base'] == old['base'] and value['records'] == old['records']
            assert (value['case'], value['kind'], value['phase']) == (cell['case'], cell['kind'], cell['phase'])
            env = ReadingEnvironment(value['records'], rows=23, glyphs=6)
            base = linear_literal(env, value['base'])
            old_n, old_d = independent_point(source, env, base, guard)
            assert target_fingerprint(old_n, old_d) == value['base_target']['target_fingerprint']
            eligible = direct_eligible(env, base)
            assert value['old_degree'] == cell['old_degree'] == len(eligible)
            indices = tuple(sorted({j*len(eligible)//8 for j in range(8)})) if eligible else ()
            assert [p['eligible_index'] for p in value['points']] == list(indices)
            assert cell['points'] == len(value['points'])
            for p in value['points']:
                triplet = eligible[p['eligible_index']]
                assert p['triplet'] == list(triplet)
                destination, branch, replacements = run.finite.direct_rewrite(env, base, triplet)
                actual = linear_literal(env, p['reading'])
                assert actual == destination and p['branch'] == branch and p['replacements'] == replacements
                assert tuple(p['reading']['actions']) == run.finite.direct_actions(env, actual)
                assert run.finite.direct_rewrite(env, actual, triplet)[0] == base
                reverse = direct_eligible(env, actual)
                assert triplet in reverse and p['new_degree'] == len(reverse)
                new_n, new_d = independent_point(source, env, actual, guard)
                assert p['target']['target_fingerprint'] == target_fingerprint(new_n, new_d)
                n, d = new_n*old_d*len(eligible), new_d*old_n*len(reverse)
                assert p['valid_ratio_fingerprint'] == target_fingerprint(n, d)
                largest_delta = max(largest_delta, abs(p['valid_log_ratio']-(math.log(n)-math.log(d))))
                expected = math.log(new_n)-math.log(new_d)
                independent_log = p['source_reference']['log_probability']-p['target']['visited']*math.log(42)
                largest_delta = max(largest_delta, abs(p['target']['log_target']-expected),
                                    abs(independent_log-expected))
                assert p['target']['length'] == sum(map(len, actual.texts))
                assert p['target']['visited'] == sum(k >= 0 for k in actual.key)
                assert p['wall_seconds'] >= 0 and p['cpu_seconds'] >= 0
                points += 1
            assert cell['branches'] == {b: sum(p['branch'] == b for p in value['points']) for b in ('merge', 'split')}
            assert cell['candidate_cpu_seconds'] == sum(p['cpu_seconds'] for p in value['points'])
            private += cell['archive']['bytes']
        assert points == result['total_points'] <= 1552 and largest_delta <= 1e-9
        assert private == result['private_archive_bytes'] <= run.PRIVATE
        mean = sum(c['candidate_cpu_seconds'] for c in result['cells'])/max(1, points)
        assert result['mean_candidate_cpu_seconds'] == mean
        assert result['gates'] == {'some_nonidentity_candidates': points > 0,
                                  'mean_candidate_cpu_under_two_seconds': mean <= 2}
        assert all(result['gates'].values())
        r = result['resources']
        assert r['wall_seconds'] <= run.WALL and r['cpu_seconds'] <= run.CPU and r['peak_rss_bytes'] <= run.HOST
        assert run.array_identity(source) == result['source_arrays']
        run.training.save_new(run.OUT/'audit.json', {'status': 'PASS_original_source_map_eligibility_target_and_ratio_replay',
            'freeze': freeze, 'result': run.training.artifact(run.OUT/'result.json'), 'points': points,
            'maximum_joint_log_delta': largest_delta, 'resources': guard(),
            'no_chain_gold_metrics_neural_or_optimizer_calls': True, 'independent_expert_review': False,
            'reference_policy_counts_not_independently_replayed': True})
        return points
    except Exception as error:
        run.training.save_new(run.OUT/'audit-failure.json', {'error': repr(error), 'no_retry': True,
                                                           'resources': run.training.resource_report(wall, cpu)})
        raise
    finally:
        signal.alarm(0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--freeze', required=True)
    print(audit(parser.parse_args().freeze))
