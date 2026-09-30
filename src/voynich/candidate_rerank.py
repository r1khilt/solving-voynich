"""Fixed whole-record candidate reranking with one shared key across records.

Changing the source invalidates the proposal source's unseen-candidate bound.
These routines make no global MAP, normalized posterior, or evidence claim.
"""
import math

from voynich.kbest_suffix import _matcher


def logsum(values):
    values = list(values)
    high = max(values, default=-math.inf)
    return high if high == -math.inf else high+math.log(math.fsum(math.exp(v-high) for v in values))


def prepare_candidates(alphabet, records, bank, reading):
    active = [i for i, row in enumerate(bank['bank']) if row['log_weight'] is not None]
    if reading['active_bank_indices'] != active or len(reading['mixture']['per_key']) != len(active):
        raise ValueError('Candidate/key inventory differs')
    weights = [bank['bank'][i]['log_weight'] for i in active]
    if not weights or any(not math.isfinite(w) for w in weights) or abs(logsum(weights)) > 1e-7:
        raise ValueError('Invalid fitting weights')
    normalizer = logsum(weights)
    weights = [w-normalizer for w in weights]
    keys = [bank['bank'][i]['units'] for i in active]
    if len(set(map(tuple, keys))) != len(keys):
        raise ValueError('Duplicate candidate keys')
    candidates = {}
    for row in reading['mixture']['per_key']:
        for texts, score in row['readings']:
            texts = tuple(texts)
            if len(texts) != len(records) or any(not t or set(t)-set(alphabet) for t in texts):
                raise ValueError('Invalid whole-record tuple')
            if not math.isfinite(score) or (texts in candidates and abs(score-candidates[texts]) > 1e-8):
                raise ValueError('Inconsistent source score')
            candidates[texts] = score
    if not candidates or len(candidates) != reading['mixture']['candidate_tuples']:
        raise ValueError('Incomplete candidate union')
    texts = sorted({text for pair in candidates for text in pair})
    indices = {text: i for i, text in enumerate(texts)}
    match, cache, rows = _matcher(alphabet, keys), {}, []
    for pair, score in candidates.items():
        mask = match(records, pair)
        if not mask:
            raise ValueError('Candidate has no globally consistent fitting key')
        if mask not in cache:
            cache[mask] = logsum(w for i, w in enumerate(weights) if mask & (1 << i))
        rows.append({'text_indices': [indices[t] for t in pair], 'key_log_mass': cache[mask],
                     'statistical_source_log_probability': score})
    best = max(range(len(rows)), key=lambda i: rows[i]['statistical_source_log_probability']+rows[i]['key_log_mass'])
    score = rows[best]['statistical_source_log_probability']+rows[best]['key_log_mass']
    if (tuple(texts[i] for i in rows[best]['text_indices']) != tuple(reading['mixture']['plaintexts'])
            or abs(score-reading['mixture']['joint_log_probability']) > 1e-7):
        raise ValueError('Original statistical decision did not reproduce')
    return {'texts': texts, 'candidates': rows, 'statistical_best_index': best,
            'active_bank_indices': active, 'unique_support_masks': len(cache),
            'statistical_score': score, 'source_bound_cannot_transfer_to_new_model': True}


def rank_candidates(prepared, record_log_probabilities):
    if (len(record_log_probabilities) != len(prepared['texts'])
            or any(not math.isfinite(v) for v in record_log_probabilities)):
        raise ValueError('Complete finite neural record scores required')
    scores = [math.fsum(record_log_probabilities[i] for i in row['text_indices'])+row['key_log_mass']
              for row in prepared['candidates']]
    ordered = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
    best = ordered[0]
    return {'candidate_index': best, 'plaintexts': [prepared['texts'][i] for i in prepared['candidates'][best]['text_indices']],
            'score': scores[best], 'runner_up_index': ordered[1] if len(ordered) > 1 else None,
            'margin_nats': scores[best]-scores[ordered[1]] if len(ordered) > 1 else None,
            'candidate_scores': scores, 'global_map_claimed': False, 'neural_evidence_computed': False}
