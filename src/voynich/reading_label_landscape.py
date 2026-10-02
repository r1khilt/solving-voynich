"""Every unordered label transposition, without Gold or sampling/chain claims."""
import hashlib
import itertools

from voynich.reading_label_transport import pair_probability, score_label_transport
from voynich.regrowth_trace import log_target


def target_fingerprint(numerator, denominator):
    """Exact unreduced target integers, framed without decimal digit limits."""
    digest = hashlib.sha256()
    for integer in (numerator, denominator):
        if type(integer) is not int or integer <= 0:
            raise ValueError('Positive exact target integers required')
        data = integer.to_bytes((integer.bit_length()+7)//8, 'big')
        digest.update(len(data).to_bytes(8, 'big'))
        digest.update(data)
    return digest.hexdigest()


def complete_label_landscape(sampler, path, *, progress=lambda: None):
    """Return all point scores and the first strictly best uphill neighbor.

    Reference-q replay is absent. Choosing by the frozen target here is only a
    one-hop diagnostic; no best trajectory, optimization run or MH transition.
    """
    old_n, old_d = sampler.target_integers(path)
    old_log = log_target(sampler, path)
    best_n, best_d, best, best_pair = old_n, old_d, None, None
    pairs = []
    for a, b in itertools.combinations(range(sampler.env.rows), 2):
        point = score_label_transport(sampler, path, a, b, progress=progress)
        product_new, product_old = point.numerator*old_d, point.denominator*old_n
        sign = int(product_new > product_old)-int(product_new < product_old)
        if point.numerator*best_d > best_n*point.denominator:
            best_n, best_d, best, best_pair = point.numerator, point.denominator, point, [a, b]
        potential = point.log_potential()
        pairs.append({'pair': [a, b], 'target_fingerprint': target_fingerprint(point.numerator, point.denominator),
            'joint_log_potential': potential, 'log_potential_change': potential-old_log,
            'exact_target_change_sign': sign, 'key_changed': point.state.key != path.state.key,
            'texts_changed': point.state.texts != path.state.texts,
            'pair_probability': str(pair_probability(path, a, b))})
    summary = {'pairs': len(pairs), 'uphill_pairs': sum(p['exact_target_change_sign'] > 0 for p in pairs),
        'equal_target_pairs': sum(p['exact_target_change_sign'] == 0 for p in pairs),
        'downhill_pairs': sum(p['exact_target_change_sign'] < 0 for p in pairs),
        'zero_selection_probability_pairs': sum(p['pair_probability'] == '0' for p in pairs),
        'key_unchanged_text_changed_pairs': sum(not p['key_changed'] and p['texts_changed'] for p in pairs),
        'best_uphill_pair': best_pair, 'best_uphill_log_gain': best.log_potential()-old_log if best else 0.,
        'no_uphill_label_pair': best is None, 'reference_q_replay_calls': 0,
        'no_gold_sampling_or_chain_density': True}
    return pairs, best, summary
