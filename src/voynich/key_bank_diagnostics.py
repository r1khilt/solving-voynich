"""Known-answer inventory and full-key posterior-support diagnostics.

No decoder, source or archives are opened. Bounds assume a shared iid uniform
dictionary prior and complete likelihoods under the same records/source/EOS.
Floating results are not interval certificates or reading-marginal bounds.
"""

import math
from collections import Counter


def _key(key,pool_size,rows=None):
    if type(pool_size) is not int or not 1<=pool_size<=4096 or isinstance(key,(str,bytes)):
        raise ValueError('Bounded integer code dictionary required')
    key = tuple(key)
    if (not 1<=len(key)<=64 or rows is not None and len(key)!=rows
            or any(type(k) is not int or not 0<=k<pool_size for k in key)):
        raise ValueError('Matching bounded integer code dictionary required')
    return key


def _used(used_rows,rows):
    used = tuple(used_rows)
    if len(set(used))!=len(used) or any(type(i) is not int or not 0<=i<rows for i in used):
        raise ValueError('Distinct valid used source rows required')
    return used


def _log_probability(value):
    if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or value>0:
        raise ValueError('Finite nonpositive complete log probability required')
    return float(value)


def inventory_ceiling(key,known_key,used_rows,*,pool_size):
    known = _key(known_key,pool_size)
    key,used = _key(key,pool_size,len(known)),_used(used_rows,len(known))
    required,available = Counter(known[i] for i in used),Counter(key)
    ceiling = sum(min(n,available[u]) for u,n in required.items())
    return {'used_rows':len(used),'current_used_matches':sum(key[i]==known[i] for i in used),
        'permutation_used_match_ceiling':ceiling,'minimum_inventory_replacements':len(used)-ceiling,
        'inventory_permits_complete_used_mapping':ceiling==len(used)}


def full_key_bank_bounds(keys,log_likelihoods,known_source_log_probability,used_rows,*,pool_size,
                         known_key_log_likelihood=None):
    """Distinct-key event mass, not particle multiplicity or plaintext mass.

    One verified compatible source tuple with M used rows supplies the lower
    bound Q/ U^M. Its unused-row completions are counted even if they are absent
    from this bank. The optional known complete key supplies a second lower
    bound; use their maximum, never their sum.
    """
    keys,scores = tuple(keys),tuple(log_likelihoods)
    if not 1<=len(keys)<=128 or len(scores)!=len(keys):
        raise ValueError('One to128 complete bank keys and scores required')
    first = _key(keys[0],pool_size)
    used = _used(used_rows,len(first))
    q = _log_probability(known_source_log_probability)
    unique = {}
    for key,score in zip(keys,scores,strict=True):
        key,score = _key(key,pool_size,len(first)),_log_probability(score)
        if key in unique and unique[key]!=score:
            raise ArithmeticError('Duplicate full keys have inconsistent likelihoods')
        unique[key] = score
    high = max(unique.values())
    total = high+math.log(math.fsum(math.exp(v-high) for v in unique.values()))
    lower_in_likelihood_units = q+(len(first)-len(used))*math.log(pool_size)
    if known_key_log_likelihood is not None:
        known = _log_probability(known_key_log_likelihood)
        if known<q-1e-7:
            raise ArithmeticError('Known compatible path exceeds whole-key marginal')
        lower_in_likelihood_units = max(lower_in_likelihood_units,known)
    upper = min(0.,total-lower_in_likelihood_units)
    return {'particles':len(keys),'distinct_full_keys':len(unique),'distinct_key_log_likelihood_sum':total,
        'known_reading_log_evidence_lower':q-len(used)*math.log(pool_size),
        'combined_log_evidence_lower':lower_in_likelihood_units-len(first)*math.log(pool_size),
        'full_key_bank_log_posterior_mass_upper':upper,
        'bank_supported_law_tv_lower_float':-math.expm1(upper),
        'bank_supported_law_forward_kl_lower_nats':-upper,
        'unused_rows':len(first)-len(used),'reading_marginal_bound_claimed':False,
        'interval_arithmetic_certificate':False}
