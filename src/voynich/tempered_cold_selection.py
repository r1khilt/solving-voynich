"""Select the earliest exact-target maximum among visited cold readings only."""
import math
from fractions import Fraction as F

from voynich.reading_label_landscape import target_fingerprint


def receipt_target(receipt, *, pool_size=42, stop=F(1, 225)):
    if receipt['complete'] is not True or len(receipt['actions']) != len(receipt['source_terms']):
        raise ValueError('Full complete reference receipt required')
    terms, length, records = receipt['source_terms'], len(receipt['actions']), len(receipt['texts'])
    continuation = 1-stop
    numerator = math.prod(n for n, _ in terms)*continuation.numerator**length*stop.numerator**records
    denominator = (1 << sum(e for _, e in terms))*continuation.denominator**length*stop.denominator**records
    denominator *= pool_size**sum(k >= 0 for k in receipt['key'])
    return numerator, denominator


def select_cold(trace):
    """No warm candidate, Gold, final-state-only restriction or floating ranking."""
    best, selected = trace['initial'], -1
    n, d = receipt_target(best)
    for row in trace['trace']:
        candidate = row['retained'][0]
        cn, cd = receipt_target(candidate)
        if cn*d > n*cd:
            best, selected, n, d = candidate, row['index'], cn, cd
    return {'reading': best, 'selected_sweep': selected, 'target_sha256': target_fingerprint(n, d),
            'log_target': math.log(n)-math.log(d), 'cold_states_only': True, 'earliest_exact_target_maximum': True}
