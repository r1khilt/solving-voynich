"""Keep the union of complete neighborhoods along bounded best-improvement search.

All evaluated keys remain in the final finite bank, including zero-mass rows.
This is neither a global optimizer nor a full-family posterior approximation
with an error guarantee. An interrupted round must not produce a completed bank.
"""
import math

from voynich.local_key_bank import one_move_bank


def expand_key_bank(score_key, parent, glyphs, *, max_rounds=4, tolerance=1e-6,
                    progress=None, round_progress=None):
    if (type(max_rounds) is not int or not 1 <= max_rounds <= 16
            or isinstance(tolerance, bool) or not math.isfinite(tolerance) or tolerance < 0):
        raise ValueError('Invalid bounded expansion settings')
    parent, glyphs = tuple(parent), tuple(glyphs)
    # Validate the entire family before calling the scorer.
    one_move_bank(parent, glyphs)
    rows, indices, rounds = [], {}, []

    def evaluate(key):
        key = tuple(key)
        if key not in indices:
            value = score_key(key)
            if not isinstance(value, dict) or any(k in value for k in ('index', 'units', 'log_weight')):
                raise ValueError('Scorer returned invalid candidate fields')
            weight = value['log_weight_unnormalized']
            if weight is not None and (isinstance(weight, bool) or not math.isfinite(weight)):
                raise ValueError('Numerical failure cannot become zero-mass support')
            row = {'index': len(rows), 'units': list(key), **value}
            indices[key] = len(rows)
            rows.append(row)
            if progress is not None:
                progress(dict(row))
        return indices[key]

    current = evaluate(parent)
    if rows[current]['log_weight_unnormalized'] is None:
        raise ValueError('Original fitting key must support all fitting records')
    reason = 'round_limit'
    for round_index in range(max_rounds):
        neighborhood = one_move_bank(rows[current]['units'], glyphs)
        members = [evaluate(key) for key in neighborhood]
        supported = [i for i in members if rows[i]['log_weight_unnormalized'] is not None]
        selected = max(supported, key=lambda i: rows[i]['log_weight_unnormalized'])
        gain = rows[selected]['log_weight_unnormalized'] - rows[current]['log_weight_unnormalized']
        accepted = gain > tolerance
        report = {'round': round_index, 'parent_index': current, 'member_indices': members,
                  'selected_index': selected, 'gain_nats': gain, 'accepted': accepted,
                  'cumulative_unique_keys': len(rows), 'complete_neighborhood': True}
        rounds.append(report)
        if round_progress is not None:
            round_progress(dict(report))
        if not accepted:
            reason = 'local_tolerance_stop_at_center'
            break
        current = selected
    finite = [row for row in rows if row['log_weight_unnormalized'] is not None]
    best = max(finite, key=lambda r: r['log_weight_unnormalized'])
    high = best['log_weight_unnormalized']
    shifted = [r['log_weight_unnormalized'] - high for r in finite]
    if any(not math.isfinite(value) for value in shifted):
        raise ArithmeticError('Relative log-weight range exceeds floating representation')
    shifted_normalizer = math.log(math.fsum(math.exp(value) for value in shifted))
    normalizer = high + shifted_normalizer
    for row in rows:
        weight = row['log_weight_unnormalized']
        row['log_weight'] = None if weight is None else (weight - high) - shifted_normalizer
    return {'bank': rows, 'bank_size': len(rows), 'finite_keys': len(finite),
            'best_index': best['index'], 'best_units': best['units'],
            'search_center_index': current, 'rounds': rounds, 'stop_reason': reason,
            'complete_rounds': len(rounds), 'maximum_rounds': max_rounds, 'tolerance_nats': tolerance,
            'fit_log_normalizer_restricted_not_full_family_evidence': normalizer,
            'maximum_weight': math.exp(best['log_weight']), 'all_positive_weights_retained_in_log_domain': True,
            'global_optimality_claimed': False}
