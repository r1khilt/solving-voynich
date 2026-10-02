"""Literal reading accuracy without assigning a sampling density to chain states."""
from voynich.recurrent_latin_source import ALPHABET
from voynich.source_action_training import action_episode
from voynich.unit_channel_decision import edit_distance


def reading_endpoint_metrics(receipts, episodes, sampler):
    if not episodes or len(receipts) != len(episodes):
        raise ValueError('One literal endpoint for every allocated episode required')
    cases = []
    for receipt, episode in zip(receipts, episodes, strict=True):
        env, trace = action_episode(sampler, episode)
        state = env.initial
        for action in receipt['actions']:
            state = env.advance(state, action)
        if (list(state.key) != list(receipt['key']) or list(state.offsets) != list(receipt['offsets'])
                or [list(t) for t in state.texts] != [list(t) for t in receipt['texts']]):
            raise ValueError('Endpoint actions, dictionary, offsets and reading disagree')
        complete = env.selected_record(state) is None
        if type(receipt['complete']) is not bool or complete != receipt['complete'] or (
                not complete and env.legal_actions(state)):
            raise ValueError('Literal completion or true death required')
        used = [a for a, k in enumerate(trace[2].key) if k >= 0]
        letters = sum(map(len, trace[2].texts))
        matched = sum(state.key[a] == episode[1][a] for a in used) if complete else 0
        errors = sum(edit_distance(''.join(ALPHABET[a] for a in t), ''.join(ALPHABET[a] for a in p))
                     for t, p in zip(trace[2].texts, state.texts, strict=True)) if complete else letters
        exact = sum(t == p for t, p in zip(trace[2].texts, state.texts, strict=True)) if complete else 0
        cases.append({'complete': complete, 'edits': errors, 'true_letters': letters, 'used_matches': matched,
            'used_rows': len(used), 'exact_records': exact, 'complete_used_key': complete and matched == len(used)})
    result = {k: sum(c[k] for c in cases) for k in ('edits', 'true_letters', 'used_matches', 'used_rows',
        'exact_records', 'complete_used_key', 'complete')}
    result.update({'episodes': len(episodes), 'record_attempts': sum(len(e[0]) for e in episodes), 'cases': cases,
        'unused_rows_not_scored': True, 'failure_penalty_is_full_length': True, 'no_state_density_claim': True})
    return result


def competence_gate(score):
    """Frozen ambitious inverse-competence thresholds, never a historical gate."""
    return (score['used_rows'] > 0 and score['true_letters'] > 0 and score['episodes'] > 0
        and score['record_attempts'] > 0 and score['used_matches']*10 >= score['used_rows']*9
        and score['edits']*10 <= score['true_letters']
        and score['exact_records']*2 >= score['record_attempts']
        and score['complete_used_key']*2 >= score['episodes'])
