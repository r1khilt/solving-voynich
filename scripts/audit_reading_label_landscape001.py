"""One full independent transposition/target replay, no chain or panel extension."""
import argparse
import itertools
import json
import math
import signal
import time
from collections import Counter
from fractions import Fraction as F

import torch

from scripts import run_reading_label_landscape001 as run
from scripts.audit_reading_regrowth_recovery001 import linear_literal
from scripts.audit_source_action_cipher001 import integer_edits
from voynich.reading_endpoint_metrics import reading_endpoint_metrics
from voynich.reading_label_landscape import target_fingerprint
from voynich.source_action_proposal import ReadingEnvironment, ReadingState
from voynich.source_action_training import action_episode


def independent_point(source, env, state, progress):
    env.validate(state)
    assert env.selected_record(state) is None
    terms = []
    for text in state.texts:
        context = source.state('')
        for row in text:
            progress()
            n, d = float(source.probabilities[context, row]).as_integer_ratio()
            assert n > 0 and d > 0 and d & (d-1) == 0
            terms.append((n, d))
            context = int(source.transitions[context, row])
    rho = F(1, 225)
    length = sum(map(len, state.texts))
    return (math.prod(n for n, _ in terms)*(1-rho).numerator**length*rho.numerator**len(state.texts),
        math.prod(d for _, d in terms)*(1-rho).denominator**length*rho.denominator**len(state.texts)
        *len(env.pool)**sum(k >= 0 for k in state.key))


def transpose(state, actions, a, b):
    def rename(row):
        return b if row == a else a if row == b else row
    key = list(state.key)
    key[a], key[b] = key[b], key[a]
    return ReadingState(tuple(key), state.offsets,
        tuple(tuple(rename(row) for row in text) for text in state.texts)), tuple(
            2*rename(action//2)+action%2 for action in actions)


def independent_orbit(env, base, known):
    def partition(texts):
        seen, pattern = {}, []
        for text in texts:
            for row in text:
                if row not in seen:
                    seen[row] = len(seen)
                pattern.append(seen[row])
        return pattern
    same_lengths = tuple(map(len, base.texts)) == tuple(map(len, known.texts))
    same_units = same_lengths and [len(env.pool[base.key[a]]) for x in base.texts for a in x] == [
        len(env.pool[known.key[a]]) for x in known.texts for a in x]
    same_pattern = same_lengths and partition(base.texts) == partition(known.texts)
    reachable = same_lengths and same_units and same_pattern
    matching = Counter(k for k in base.key if k >= 0) & Counter(k for k in known.key if k >= 0)
    used = sum(k >= 0 for k in known.key)
    mapping = sorted(map(list, dict(zip((a for x in base.texts for a in x),
                    (a for x in known.texts for a in x), strict=True)).items())) if reachable else None
    return {'source_lengths_match': same_lengths, 'unit_lengths_match': same_units,
        'cross_record_equality_pattern_matches': same_pattern, 'label_orbit_reachable': reachable,
        'length_edit_lower_bound': sum(abs(len(x)-len(y)) for x, y in zip(base.texts, known.texts, strict=True)),
        'binding_match_ceiling': sum(matching.values()), 'known_used_rows': used,
        'inventory_can_cover_known_used_rows': sum(matching.values()) == used,
        'permutation_of_used_rows': mapping, 'oracle_only_no_proposal_or_density': True}


def audit(freeze):
    run.training.old.require_frozen(freeze, run.PATHS)
    prior = run.require_prior()
    run.training.save_new(run.OUT/'audit-started.json', {'freeze': freeze, 'no_retry': True,
                                                       'no_chain_or_panel_extension': True})
    run.training.limit_resources(run.WALL, run.CPU)
    wall, cpu = time.monotonic(), time.process_time()
    torch.set_num_threads(1)
    try:
        def guard():
            value = run.training.resource_report(wall, cpu)
            if value['wall_seconds'] > run.WALL or value['cpu_seconds'] > run.CPU or value['peak_rss_bytes'] > run.HOST:
                raise MemoryError('Registered all-pairs audit cap; no retry')
            return value
        result = json.loads((run.OUT/'result.json').read_text())
        assert result['freeze'] == freeze and result['status'] == 'COMPLETE_registered_all_pairs_development_diagnostic'
        assert result['inputs'] == [run.training.artifact(run.ROOT/p) for p in run.PATHS]
        assert result['total_pairs'] == run.recovery.CASES*len(run.PHASES)*253
        assert result['private_archive_bytes'] <= run.PRIVATE
        assert result['resources']['wall_seconds'] <= run.WALL and result['resources']['cpu_seconds'] <= run.CPU
        assert result['resources']['peak_rss_bytes'] <= run.HOST
        assert run.training.artifact(run.ROOT/result['sealed_cells']['path']) == result['sealed_cells']
        sealed = json.loads((run.ROOT/result['sealed_cells']['path']).read_text())
        assert sealed['cells'] == result['cells'] and sealed['all_pairs_complete_before_gold_metrics']
        assert sealed['freeze'] == freeze
        _, episodes, episode_sampler, allocated, initials = run.recovery.admitted.load_inputs()
        source, _ = run.load_source()
        assert result['source_arrays'] == run.array_identity(source)
        by_phase = {p: [] for p in run.PHASES}
        checked, private_bytes = 0, 0
        for case, (kind, paired, records) in enumerate(allocated):
            env = ReadingEnvironment(records, rows=23, glyphs=6)
            for phase, actions, input_archive in run.endpoint_frames(prior, initials, case):
                cell = result['cells'][case*len(run.PHASES)+run.PHASES.index(phase)]
                assert (cell['case'], cell['phase'], cell['kind']) == (case, phase, kind)
                saved = run.load_archive(cell['archive'])
                private_bytes += cell['archive']['bytes']
                assert (saved['case'], saved['phase'], saved['kind'], saved['paired_case']) == (case, phase, kind, paired)
                assert saved['records'] == list(map(list, records)) and saved['input_archive'] == input_archive
                assert saved['gold_metrics_not_yet_computed'] and saved['summary'] == cell['summary']
                assert saved['base']['actions'] == list(actions)
                base = linear_literal(env, saved['base'])
                old_n, old_d = independent_point(source, env, base, guard)
                old_log = math.log(old_n)-math.log(old_d)
                best_n, best_d, best, best_actions, best_pair = old_n, old_d, base, tuple(actions), None
                expected_pairs = list(itertools.combinations(range(env.rows), 2))
                assert [p['pair'] for p in saved['pairs']] == list(map(list, expected_pairs))
                for (a, b), point in zip(expected_pairs, saved['pairs'], strict=True):
                    state, transformed = transpose(base, actions, a, b)
                    n, d = independent_point(source, env, state, guard)
                    assert target_fingerprint(n, d) == point['target_fingerprint']
                    potential = math.log(n)-math.log(d)
                    assert potential == point['joint_log_potential'] and potential-old_log == point['log_potential_change']
                    assert point['exact_target_change_sign'] == int(n*old_d > d*old_n)-int(n*old_d < d*old_n)
                    assert point['key_changed'] == (state.key != base.key)
                    assert point['texts_changed'] == (state.texts != base.texts)
                    used = sum(k >= 0 for k in base.key)
                    assert point['pair_probability'] == str(F(int(base.key[a] >= 0)+int(base.key[b] >= 0), used*(env.rows-1)))
                    if n*best_d > best_n*d:
                        best_n, best_d, best, best_actions, best_pair = n, d, state, transformed, [a, b]
                    checked += 1
                assert linear_literal(env, saved['best_by_target']) == best
                assert saved['best_by_target']['actions'] == list(best_actions)
                pairs = saved['pairs']
                summary = {'pairs': len(pairs), 'uphill_pairs': sum(p['exact_target_change_sign'] > 0 for p in pairs),
                    'equal_target_pairs': sum(p['exact_target_change_sign'] == 0 for p in pairs),
                    'downhill_pairs': sum(p['exact_target_change_sign'] < 0 for p in pairs),
                    'zero_selection_probability_pairs': sum(p['pair_probability'] == '0' for p in pairs),
                    'key_unchanged_text_changed_pairs': sum(not p['key_changed'] and p['texts_changed'] for p in pairs),
                    'best_uphill_pair': best_pair, 'best_uphill_log_gain': math.log(best_n)-math.log(best_d)-old_log,
                    'no_uphill_label_pair': best_pair is None, 'reference_q_replay_calls': 0,
                    'no_gold_sampling_or_chain_density': True}
                assert summary == saved['summary']
                if kind == 'positive':
                    by_phase[phase].append(saved)
                print(json.dumps({'audited_states': case*len(run.PHASES)+run.PHASES.index(phase)+1,
                                  'audited_pairs': checked, **guard()}), flush=True)
        assert private_bytes == result['private_archive_bytes']
        for phase, positive in by_phase.items():
            expected = {}
            for name in ('base', 'best_by_target'):
                metrics = reading_endpoint_metrics([p[name] for p in positive], episodes, episode_sampler)
                assert metrics == result['known_answer_metrics'][phase][name]
                for p, e, score in zip(positive, episodes, metrics['cases'], strict=True):
                    env, truth = action_episode(episode_sampler, e)
                    state = linear_literal(env, p[name])
                    assert score['edits'] == sum(integer_edits(x, y) for x, y in zip(truth[2].texts, state.texts, strict=True))
            expected['orbit_obstructions'] = [independent_orbit(action_episode(episode_sampler, e)[0],
                linear_literal(action_episode(episode_sampler, e)[0], p['base']), action_episode(episode_sampler, e)[1][2])
                for p, e in zip(positive, episodes, strict=True)]
            assert expected['orbit_obstructions'] == result['known_answer_metrics'][phase]['orbit_obstructions']
        run.training.old.require_frozen(freeze, run.PATHS)
        run.training.save_new(run.OUT/'audit.json', {'status': 'PASS_independent_all_pairs_target_literal_and_orbit_replay',
            'freeze': freeze, 'result': run.training.artifact(run.OUT/'result.json'), 'pairs': checked,
            'inputs_checked': len(run.PATHS), 'resources': guard(), 'same_researcher_not_expert_replication': True,
            'no_chain_training_or_panel_extension': True})
        return checked
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
