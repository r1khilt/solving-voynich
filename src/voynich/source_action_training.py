"""Auditable reading supervision and free-running metrics on shared-key episodes.

Whole-path probabilities and deterministic decoder output laws are distinct.
Failed proposals count as full-length reading failures, never partial credit.
"""

import math

import numpy as np
import torch

from voynich.joint_key_proposal import canonicalize_records, unit_pool
from voynich.recurrent_latin_source import ALPHABET
from voynich.source_action_cache import _PathCache
from voynich.source_action_proposal import ReadingEnvironment
from voynich.unit_channel_decision import edit_distance


def action_episode(sampler, episode):
    """Only provenance windows/key select supervised targets; all features are past-only."""
    cipher, key, metadata = episode
    if len(metadata['windows']) != 2 or len(cipher) != 2 or tuple(metadata['canonical_key']) != tuple(key):
        raise ValueError('Two-record canonical episode provenance required')
    texts = []
    for window in metadata['windows']:
        segment, start, length = (window[name] for name in ('segment', 'start', 'length'))
        if (any(type(v) is not int for v in (segment, start, length))
                or not 0 <= segment < len(sampler.records) or start < 0 or length < 1
                or start+length > len(sampler.records[segment])):
            raise ValueError('Source window cannot cross its audited segment')
        texts.append(tuple(map(int, sampler.records[segment][start:start+length])))
    environment = ReadingEnvironment(cipher, rows=len(key), glyphs=6)
    trace = environment.teaching_trace(texts, key)
    return environment, trace


@torch.no_grad()
def greedy_reading(model, environment):
    """Greedy search, not a sample from its reported base model path density."""
    was_training = model.training
    model.eval()
    try:
        cache, actions, logq = _PathCache(model, environment), [], 0.
        for _ in range(sum(map(len, environment.records))):
            if environment.selected_record(cache.state) is None:
                break
            legal = environment.legal_actions(cache.state)
            if not legal:
                return {'status': 'dead_end', 'key': cache.state.key, 'texts': cache.state.texts,
                        'actions': tuple(actions), 'greedy_path_under_model_log_probability': logq}
            scores = cache.scores().detach().cpu().double()
            if not torch.isfinite(scores[list(legal)]).all():
                raise FloatingPointError('Nonfinite legal-action scores; no repaired greedy path')
            # Sorted legal action list and torch argmax give the fixed lowest tie.
            action = int(scores.argmax())
            logq += float(scores.log_softmax(-1)[action])
            actions.append(action)
            cache.advance(action)
        if environment.selected_record(cache.state) is not None:
            raise AssertionError('Non-erasing path exceeded its glyph horizon')
        return {'status': 'complete_path', 'key': cache.state.key, 'texts': cache.state.texts,
                'actions': tuple(actions), 'greedy_path_under_model_log_probability': logq}
    finally:
        model.train(was_training)


def control_records(episodes, *, count=16, seed=92251):
    """Fixed matched shuffle/IID nulls plus a deliberately nonidentifiable cipher."""
    rng = np.random.default_rng(seed)
    controls = []
    for i, episode in enumerate(episodes[:count]):
        cipher = episode[0]
        for kind in ('within_record_shuffle', 'iid_glyphs'):
            records = []
            for record in cipher:
                if kind == 'within_record_shuffle':
                    # Retain each symbol's first occurrence so canonical glyph
                    # names cannot become an accidental control confound.
                    cut = max(record.index(g) for g in set(record))+1
                    values = (*record[:cut], *map(int, rng.permutation(record[cut:])))
                else:
                    values = tuple(map(int, rng.integers(0, 6, size=len(record))))
                records.append(values)
            if kind == 'iid_glyphs':
                records = list(canonicalize_records([''.join('ABCDEF'[g] for g in r)
                                                    for r in records], 'ABCDEF').records)
            controls.append({'kind': kind, 'paired_case': i, 'records': records})
    # Any all-single-zero dictionary makes different source strings produce
    # this same observation: complete output cannot prove the source meaning.
    controls.append({'kind': 'unrecoverable_ambiguity', 'paired_case': None,
                     'records': [(0,)*64, (0,)*64]})
    return controls


def control_metrics(predictions, controls):
    if len(predictions) != len(controls):
        raise ValueError('All allocated controls must be reported')
    groups = {}
    for prediction, control in zip(predictions, controls, strict=True):
        env = ReadingEnvironment(control['records'], rows=23, glyphs=6)
        state = env.initial
        for action in prediction['actions']:
            state = env.advance(state, action)
        if state.key != tuple(prediction['key']) or state.texts != tuple(map(tuple, prediction['texts'])):
            raise ValueError('Control literal action/key/reading mismatch')
        complete = env.selected_record(state) is None
        if prediction['status'] != ('complete_path' if complete else 'dead_end'):
            raise ValueError('Control termination differs from literal replay')
        if not complete and env.legal_actions(state):
            raise ValueError('Control truncated before a true dead end')
        group = groups.setdefault(control['kind'], {'cases': 0, 'complete_paths': 0, 'dead_ends': 0})
        group['cases'] += 1
        group['complete_paths' if complete else 'dead_ends'] += 1
    return {'groups': groups, 'language_or_semantic_classification_claimed': False,
            'conditional_path_probability_is_not_cipher_evidence': True}


def recovery_metrics(predictions, episodes, sampler):
    if len(predictions) != len(episodes) or not episodes:
        raise ValueError('Every allocated validation episode must be reported')
    pool = unit_pool(6)
    rows, errors, letters, exact_records, exact_keys, complete = 0, 0, 0, 0, 0, 0
    used_matches, cases = 0, []
    for prediction, episode in zip(predictions, episodes, strict=True):
        env, trace = action_episode(sampler, episode)
        truth, used = trace[2].texts, tuple(i for i, k in enumerate(trace[2].key) if k >= 0)
        # Replay all chosen actions independently from the returned strings/key.
        state = env.initial
        for action in prediction['actions']:
            state = env.advance(state, action)
        if state.key != tuple(prediction['key']) or state.texts != tuple(map(tuple, prediction['texts'])):
            raise ValueError('Predicted actions/partial dictionary/plaintext differ')
        finished = env.selected_record(state) is None
        if prediction['status'] != ('complete_path' if finished else 'dead_end'):
            raise ValueError('Reported ending must be literal completion or actual dead end')
        if not finished and env.legal_actions(state):
            raise ValueError('An arbitrarily truncated prefix is not a dead end')
        if not math.isfinite(prediction['greedy_path_under_model_log_probability']):
            raise ValueError('Finite model path density required even for failures')
        source_letters = sum(map(len, truth))
        matched = sum(state.key[i] == episode[1][i] for i in used) if finished else 0
        if finished:
            for text, record in zip(state.texts, env.records, strict=True):
                assert tuple(g for row in text for g in pool[state.key[row]]) == record
            edits = sum(edit_distance(''.join(ALPHABET[a] for a in t), ''.join(ALPHABET[a] for a in p))
                        for t, p in zip(truth, state.texts, strict=True))
            exact = sum(t == p for t, p in zip(truth, state.texts, strict=True))
        else:
            edits, exact = source_letters, 0
        case = {'status': prediction['status'], 'edits': edits, 'true_letters': source_letters,
                'used_matches': matched, 'used_rows': len(used), 'exact_records': exact,
                'complete_used_key': finished and matched == len(used)}
        cases.append(case)
        complete += finished
        errors += edits
        letters += source_letters
        rows += len(used)
        used_matches += matched
        exact_records += exact
        exact_keys += case['complete_used_key']
    return {'episodes': len(episodes), 'complete_readings': complete, 'failed_readings': len(episodes)-complete,
        'edits_with_failed_readings_full_length': errors, 'true_letters': letters,
        'exact_records': exact_records, 'record_attempts': 2*len(episodes),
        'used_matches_with_failed_readings_zero': used_matches, 'used_rows': rows,
        'complete_used_keys': exact_keys, 'cases': cases,
        'unused_rows_not_scored': True, 'failure_penalty_is_full_length': True}


@torch.no_grad()
def validation(model, episodes, sampler, *, batch=4, progress=lambda _: None, free=True):
    """Select by mean WHOLE-path loss; greedy recovery separately reported."""
    if not episodes or type(batch) is not int or batch < 1:
        raise ValueError('Bounded positive validation batch and episodes required')
    was_training = model.training
    model.eval()
    logq, predictions = [], []
    try:
        for start in range(0, len(episodes), batch):
            subset = episodes[start:start+batch]
            items = [action_episode(sampler, episode) for episode in subset]
            packed = model.pack([i[0] for i in items], [i[1] for i in items])
            logits = model(packed).detach().cpu().double()
            targets, active = packed[-1].cpu(), packed[-2].cpu()
            terms = logits.log_softmax(-1).gather(-1, targets[..., None]).squeeze(-1)
            values = terms.masked_fill(~active, 0.).sum(-1)
            if not torch.isfinite(values).all():
                raise FloatingPointError('Nonfinite validation target-path density')
            logq.extend(values.tolist())
            if free:
                predictions.extend(greedy_reading(model, item[0]) for item in items)
            progress({'validation_episodes': start+len(subset)})
        result = {'episodes': len(episodes), 'joint_target_path_logq': logq,
                  'mean_whole_path_nll': -math.fsum(logq)/len(episodes),
                  'target_length_normalization_used': False, 'predictions': predictions}
        if free:
            result['recovery'] = recovery_metrics(predictions, episodes, sampler)
        return result
    finally:
        model.train(was_training)
