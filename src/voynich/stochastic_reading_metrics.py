"""Literal recovery metrics for stochastic action paths, with no greedy schema.

This helper checks saved outcomes, not the correctness of supplied densities.
Failures get the full source-length penalty and zero recovered row matches.
"""

import math

from voynich.recurrent_latin_source import ALPHABET
from voynich.source_action_training import action_episode
from voynich.unit_channel_decision import edit_distance


def stochastic_recovery_metrics(predictions, episodes, sampler):
    if not episodes or len(predictions) != len(episodes):
        raise ValueError('Every allocated episode requires one recorded stochastic outcome')
    cases = []
    for prediction, episode in zip(predictions, episodes, strict=True):
        logq = prediction.get('path_log_probability')
        if isinstance(logq, bool) or not isinstance(logq, (int, float)) or not math.isfinite(logq) or logq > 0:
            raise ValueError('Finite stochastic action-path log probability required')
        env, truth = action_episode(sampler, episode)
        state = env.initial
        for action in prediction['actions']:
            state = env.advance(state, action)
        if state.key != tuple(prediction['key']) or state.texts != tuple(map(tuple, prediction['texts'])):
            raise ValueError('Recorded actions, dictionary and reading disagree')
        complete = env.selected_record(state) is None
        if prediction['status'] != ('complete_path' if complete else 'dead_end') or (not complete and env.legal_actions(state)):
            raise ValueError('Actual completion or literal dead end required')
        used = [r for r, k in enumerate(truth[2].key) if k >= 0]
        letters = sum(map(len, truth[2].texts))
        matches = sum(state.key[r] == episode[1][r] for r in used) if complete else 0
        errors = sum(edit_distance(''.join(ALPHABET[r] for r in a), ''.join(ALPHABET[r] for r in b))
                     for a, b in zip(truth[2].texts, state.texts, strict=True)) if complete else letters
        exact = sum(a == b for a, b in zip(truth[2].texts, state.texts, strict=True)) if complete else 0
        cases.append({'status': prediction['status'], 'edits': errors, 'true_letters': letters,
            'used_matches': matches, 'used_rows': len(used), 'exact_records': exact,
            'complete_used_key': complete and matches == len(used)})
    complete = sum(c['status'] == 'complete_path' for c in cases)
    result = {'episodes': len(episodes), 'complete_readings': complete,
              'failed_readings': len(episodes)-complete, 'cases': cases,
              'record_attempts': sum(len(e[0]) for e in episodes),
              'complete_used_keys': sum(c['complete_used_key'] for c in cases),
              'unused_rows_not_scored': True, 'failure_penalty_is_full_length': True,
              'density_kind': 'stochastic_action_path'}
    for name, key in (('edits_with_failed_readings_full_length', 'edits'), ('true_letters', 'true_letters'),
                      ('used_matches_with_failed_readings_zero', 'used_matches'),
                      ('used_rows', 'used_rows'), ('exact_records', 'exact_records')):
        result[name] = sum(c[key] for c in cases)
    return result
