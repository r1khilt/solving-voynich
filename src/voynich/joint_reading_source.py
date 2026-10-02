"""Unchanged full-record source law: every record resets and has geometric EOS."""

import math


def source_point_checked(source, texts, *, rho=1/225):
    """Direct dense score versus lazy sparse rows at independently rebuilt histories.

    This is a point reading probability, never a dictionary likelihood marginal.
    Source must expose the original compact lazy reference alongside dense rows.
    """
    if isinstance(rho, bool) or not math.isfinite(rho) or not 0 < rho < 1:
        raise ValueError('Fixed proper geometric stop probability required')
    texts = tuple(tuple(t) for t in texts)
    if not texts or any(not t or any(type(a) is not int or not 0 <= a < len(source.alphabet)
                                     for a in t) for t in texts):
        raise ValueError('Nonempty complete integer source records required')
    direct, scalar = [], []
    for text in texts:
        direct.extend((math.log(rho), len(text)*math.log1p(-rho)))
        # Scalar branch counts continuation factors explicitly.
        scalar.append(math.log(rho))
        state, history = source.state(''), ''
        for row in text:
            direct.append(math.log(float(source.probabilities[state, row])))
            other = source.reference.state(history)
            if state != other:
                raise ArithmeticError('Dense and independently rebuilt sparse states differ')
            scalar.extend((math.log1p(-rho), math.log(float(source.reference.row(other)[row]))))
            history += source.alphabet[row]
            state = int(source.transitions[state, row])
    a, b = math.fsum(direct), math.fsum(scalar)
    if not math.isfinite(a) or a > 0 or abs(a-b) > 1e-9:
        raise ArithmeticError('Full source law differs from sparse history replay')
    return {'log_probability': a, 'maximum_total_log_delta': abs(a-b)}
