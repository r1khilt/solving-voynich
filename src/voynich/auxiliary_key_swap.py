"""Coupled label transposition of a dictionary and all its source readings.

This terminal-temperature, uniform-prior joint move preserves the observation.
It cannot modify the inventory of emission units and is not a mixing guarantee.
No sampler or empirical experiment is started by these helpers.
"""

from __future__ import annotations

import math
from numbers import Real


def swap_joint_state(key,texts,alphabet,a,b):
    """Apply one involution, retaining duplicate units and every record reset."""
    if isinstance(key,(str,bytes)) or isinstance(texts,(str,bytes)):
        raise ValueError('Separate dictionary units and whole source records required')
    key,texts,alphabet = tuple(key),tuple(texts),tuple(alphabet)
    if (not 2<=len(alphabet)<=64 or len(set(alphabet))!=len(alphabet) or len(key)!=len(alphabet)
            or any(not isinstance(c,str) or len(c)!=1 for c in alphabet)
            or any(not isinstance(u,str) or not u for u in key)
            or not texts or any(not isinstance(t,str) or any(c not in alphabet for c in t) for t in texts)
            or type(a) is not int or type(b) is not int or not 0<=a<len(key) or not 0<=b<len(key) or a==b):
        raise ValueError('Valid complete joint state and distinct source rows required')
    candidate = list(key)
    candidate[a],candidate[b] = candidate[b],candidate[a]
    translated = tuple(t.translate(str.maketrans({alphabet[a]:alphabet[b],alphabet[b]:alphabet[a]})) for t in texts)
    return tuple(candidate),translated


def auxiliary_swap_log_acceptance(old_log_q,new_log_q,*,beta,uniform_prior):
    """Point-score acceptance is valid here ONLY at beta1/uniform row prior.

    With nonuniform priors or intermediate likelihood temperatures, additional
    prior/marginal-likelihood factors are required; this API refuses those uses.
    State-independent transposition selection and positive source are assumed.
    """
    if (isinstance(beta,bool) or not isinstance(beta,Real) or beta!=1
            or uniform_prior is not True or any(isinstance(v,bool) or not isinstance(v,Real)
            or not math.isfinite(v) or v>0 for v in (old_log_q,new_log_q))):
        raise ValueError('Finite joint scores, beta1 and explicit uniform prior required')
    return min(0.,float(new_log_q)-float(old_log_q))
